import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_hf_easyedit import (  # noqa: E402
    build_easyedit_request,
    iter_chunks,
    load_eval_hparams,
    load_model,
    normalize_metric_pairs,
    parse_record,
    print_time,
    resolve_base_model_path,
    set_seed,
)


def ensure_tokenizer_padding(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"


def maybe_resize_embeddings(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and input_embeddings.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def load_requests(args: argparse.Namespace) -> List[Dict[str, Any]]:
    source_path = args.requests_path or args.data_path
    with open(source_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    if not isinstance(source_data, list):
        raise ValueError(f"Expected {source_path} to contain a JSON list")

    parsed: List[Dict[str, Any]] = []
    skipped = 0
    for idx, record in enumerate(source_data):
        item = parse_record(record, idx)
        if item is None:
            skipped += 1
            continue
        parsed.append(item)

    if args.requests_path:
        data = parsed
    elif args.num_samples > 0 and len(parsed) > args.num_samples:
        data = random.sample(parsed, args.num_samples)
    else:
        data = parsed

    print(f"Loaded {len(data)} usable samples from {source_path}")
    print(f"Sample source: {'requests_path' if args.requests_path else 'data_path'}")
    if not args.requests_path:
        print(f"Sampling seed: {args.seed}")
    print(f"Skipped invalid records: {skipped}")
    return [build_easyedit_request(item) for item in data]


def format_teacher_forcing_prompt(
    tokenizer: AutoTokenizer,
    hparams: SimpleNamespace,
    prompt: str,
    locality: bool,
) -> str:
    if locality or not getattr(hparams, "use_chat_template", False):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def build_kl_items(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for case_index, request in enumerate(requests):
        if request.get("prompt") and request.get("target_new"):
            items.append(
                {
                    "case_index": case_index,
                    "case_id": request.get("case_id"),
                    "kind": "rewrite",
                    "metric_key": "rewrite",
                    "pair_index": 0,
                    "prompt": request["prompt"],
                    "target": request["target_new"],
                }
            )

        if request.get("rephrase_prompt") and request.get("target_new"):
            items.append(
                {
                    "case_index": case_index,
                    "case_id": request.get("case_id"),
                    "kind": "rephrase",
                    "metric_key": "rephrase",
                    "pair_index": 0,
                    "prompt": request["rephrase_prompt"],
                    "target": request["target_new"],
                }
            )

        locality = request.get("locality")
        if not isinstance(locality, dict):
            continue

        for locality_key, payload in locality.items():
            if not isinstance(payload, dict):
                continue
            pairs = normalize_metric_pairs(
                payload.get("prompt"),
                payload.get("ground_truth"),
                request.get("subject"),
            )
            for pair_index, (prompt, ground_truth) in enumerate(pairs):
                items.append(
                    {
                        "case_index": case_index,
                        "case_id": request.get("case_id"),
                        "kind": "locality",
                        "metric_key": locality_key,
                        "pair_index": pair_index,
                        "prompt": prompt,
                        "target": ground_truth,
                    }
                )

    return items


def model_input_device(model: AutoModelForCausalLM) -> torch.device:
    return next(model.parameters()).device


def forward_logits(model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    try:
        outputs = model(**batch, use_cache=False)
    except TypeError:
        outputs = model(**batch)
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs.logits


def position_kl(
    base_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    if base_logits.size(-1) != edited_logits.size(-1):
        raise ValueError(
            f"KL vocab size mismatch: base={base_logits.size(-1)} edited={edited_logits.size(-1)}"
        )
    if base_logits.device != edited_logits.device:
        base_logits = base_logits.to(edited_logits.device)

    base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
    edited_log_probs = F.log_softmax(edited_logits.float(), dim=-1)

    if direction == "base_to_edit":
        return F.kl_div(
            edited_log_probs,
            base_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
    if direction == "edit_to_base":
        return F.kl_div(
            base_log_probs,
            edited_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
    raise ValueError(f"Unsupported KL direction: {direction}")


def non_target_vocab_kl(
    base_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    if base_logits.size(-1) != edited_logits.size(-1):
        raise ValueError(
            f"KL vocab size mismatch: base={base_logits.size(-1)} edited={edited_logits.size(-1)}"
        )
    if base_logits.size(0) != target_token_ids.numel():
        raise ValueError(
            f"Expected one target token id per KL row, got rows={base_logits.size(0)} "
            f"ids={target_token_ids.numel()}"
        )
    if base_logits.device != edited_logits.device:
        base_logits = base_logits.to(edited_logits.device)

    target_token_ids = target_token_ids.to(edited_logits.device).long().reshape(-1, 1)
    vocab_size = int(edited_logits.size(-1))
    if bool(((target_token_ids < 0) | (target_token_ids >= vocab_size)).any().item()):
        raise ValueError(f"Target token id falls outside vocab size {vocab_size}")

    base_masked_logits = base_logits.float().clone()
    edited_masked_logits = edited_logits.float().clone()
    base_masked_logits.scatter_(1, target_token_ids, float("-inf"))
    edited_masked_logits.scatter_(1, target_token_ids, float("-inf"))

    base_log_probs = F.log_softmax(base_masked_logits, dim=-1)
    edited_log_probs = F.log_softmax(edited_masked_logits, dim=-1)
    excluded_token_mask = torch.zeros_like(base_log_probs, dtype=torch.bool)
    excluded_token_mask.scatter_(1, target_token_ids, True)

    if direction == "base_to_edit":
        log_ratio = (base_log_probs - edited_log_probs).masked_fill(excluded_token_mask, 0.0)
        return (base_log_probs.exp() * log_ratio).sum(dim=-1)
    if direction == "edit_to_base":
        log_ratio = (edited_log_probs - base_log_probs).masked_fill(excluded_token_mask, 0.0)
        return (edited_log_probs.exp() * log_ratio).sum(dim=-1)
    raise ValueError(f"Unsupported KL direction: {direction}")


def summarize_values(values: torch.Tensor) -> Tuple[Optional[float], float, int]:
    if values.numel() == 0:
        return None, 0.0, 0
    values = values.float()
    total = float(values.sum().item())
    count = int(values.numel())
    return total / count, total, count


def make_empty_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_index": int(item["case_index"]),
        "case_id": item.get("case_id"),
        "kind": item["kind"],
        "metric_key": item["metric_key"],
        "pair_index": int(item["pair_index"]),
        "prompt": item["prompt"],
        "target": item["target"],
        "prompt_kl": None,
        "prompt_kl_sum": 0.0,
        "prompt_kl_count": 0,
        "target_kl": None,
        "target_kl_sum": 0.0,
        "target_kl_count": 0,
        "non_target_kl": None,
        "non_target_kl_sum": 0.0,
        "non_target_kl_count": 0,
    }


def make_entry(
    item: Dict[str, Any],
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    token_kl: torch.Tensor,
    non_target_token_kl: torch.Tensor,
    prompt_mask: torch.Tensor,
    target_mask: torch.Tensor,
    non_target_kl_mask: torch.Tensor,
    save_token_details: bool,
) -> Dict[str, Any]:
    prompt_values = token_kl[prompt_mask]
    target_values = token_kl[target_mask]
    non_target_values = non_target_token_kl[non_target_kl_mask]
    prompt_mean, prompt_sum, prompt_count = summarize_values(prompt_values)
    target_mean, target_sum, target_count = summarize_values(target_values)
    non_target_mean, non_target_sum, non_target_count = summarize_values(non_target_values)

    entry = make_empty_entry(item)
    entry.update(
        {
            "prompt_kl": prompt_mean,
            "prompt_kl_sum": prompt_sum,
            "prompt_kl_count": prompt_count,
            "target_kl": target_mean,
            "target_kl_sum": target_sum,
            "target_kl_count": target_count,
            "non_target_kl": non_target_mean,
            "non_target_kl_sum": non_target_sum,
            "non_target_kl_count": non_target_count,
        }
    )

    if save_token_details:
        label_ids = input_ids[1:]
        prompt_label_ids = label_ids[prompt_mask].detach().cpu().tolist()
        target_label_ids = label_ids[target_mask].detach().cpu().tolist()
        non_target_excluded_label_ids = label_ids[non_target_kl_mask].detach().cpu().tolist()
        entry.update(
            {
                "prompt_tokens": tokenizer.convert_ids_to_tokens(prompt_label_ids),
                "prompt_token_kls": [float(value) for value in prompt_values.detach().cpu().tolist()],
                "prompt_logit_positions": torch.nonzero(prompt_mask, as_tuple=False).squeeze(-1).cpu().tolist(),
                "target_tokens": tokenizer.convert_ids_to_tokens(target_label_ids),
                "target_token_kls": [float(value) for value in target_values.detach().cpu().tolist()],
                "target_logit_positions": torch.nonzero(target_mask, as_tuple=False).squeeze(-1).cpu().tolist(),
                "non_target_excluded_tokens": tokenizer.convert_ids_to_tokens(non_target_excluded_label_ids),
                "non_target_token_kls": [float(value) for value in non_target_values.detach().cpu().tolist()],
                "non_target_logit_positions": torch.nonzero(non_target_kl_mask, as_tuple=False).squeeze(-1).cpu().tolist(),
            }
        )

    return entry


def prepare_batch_tensors(
    tokenizer: AutoTokenizer,
    hparams: SimpleNamespace,
    items: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    formatted_prompts = [
        format_teacher_forcing_prompt(
            tokenizer,
            hparams,
            str(item["prompt"]),
            locality=item["kind"] == "locality",
        )
        for item in items
    ]
    targets = [str(item["target"]) for item in items]
    prompt_targets = [f"{prompt} {target}" for prompt, target in zip(formatted_prompts, targets)]

    max_prompt_len = max(len(tokenizer.encode(text)) for text in prompt_targets) + 1
    max_length = max(int(getattr(hparams, "max_length", 40)), max_prompt_len)

    before_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompt_target_tok = tokenizer(
            prompt_targets,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_tok = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = before_padding_side

    input_ids = prompt_target_tok["input_ids"]
    attention_mask = prompt_target_tok["attention_mask"].bool()
    batch_size, seq_len = input_ids.shape
    if seq_len < 2:
        return None

    pad_token_id = tokenizer.pad_token_id
    num_prompt_toks = [
        int((row != pad_token_id).sum().item())
        for row in prompt_tok["input_ids"]
    ]
    num_pad_toks = [
        int((row == pad_token_id).sum().item())
        for row in input_ids
    ]

    prompt_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool)
    target_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool)
    for row_idx in range(batch_size):
        seq_start = num_pad_toks[row_idx]
        seq_end = seq_start + int(attention_mask[row_idx].sum().item())
        prompt_end = min(seq_start + num_prompt_toks[row_idx], seq_end)

        prompt_start = seq_start
        prompt_stop = max(prompt_start, prompt_end - 1)
        if prompt_start < prompt_stop:
            prompt_mask[row_idx, prompt_start:prompt_stop] = True

        target_start = max(prompt_start, prompt_end - 1)
        target_stop = max(target_start, seq_end - 1)
        if target_start < target_stop:
            target_mask[row_idx, target_start:target_stop] = True

    selected_mask = prompt_mask | target_mask
    return {
        "prompt_target_tok": prompt_target_tok,
        "input_ids": input_ids,
        "attention_mask": prompt_target_tok["attention_mask"],
        "prompt_mask": prompt_mask,
        "target_mask": target_mask,
        "selected_mask": selected_mask,
    }


def compute_batch_kl(
    base_model: AutoModelForCausalLM,
    edited_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hparams: SimpleNamespace,
    items: Sequence[Dict[str, Any]],
    direction: str,
    save_token_details: bool,
) -> List[Dict[str, Any]]:
    batch_tensors = prepare_batch_tensors(tokenizer, hparams, items)
    if batch_tensors is None:
        return [make_empty_entry(item) for item in items]

    prompt_target_tok = batch_tensors["prompt_target_tok"]
    input_ids = batch_tensors["input_ids"]
    prompt_mask = batch_tensors["prompt_mask"]
    target_mask = batch_tensors["target_mask"]
    selected_mask = batch_tensors["selected_mask"]
    batch_size, seq_len = input_ids.shape
    token_kl = torch.full((batch_size, seq_len - 1), float("nan"), dtype=torch.float32)
    non_target_token_kl = torch.full((batch_size, seq_len - 1), float("nan"), dtype=torch.float32)
    non_target_kl_mask = torch.zeros_like(target_mask)
    for row_idx, item in enumerate(items):
        if item["kind"] in {"rewrite", "rephrase"}:
            non_target_kl_mask[row_idx] = target_mask[row_idx]

    if selected_mask.any():
        base_batch = {
            key: value.to(model_input_device(base_model))
            for key, value in prompt_target_tok.items()
        }
        edited_batch = {
            key: value.to(model_input_device(edited_model))
            for key, value in prompt_target_tok.items()
        }

        with torch.inference_mode():
            base_logits = forward_logits(base_model, base_batch)[:, :-1, :]
            edited_logits = forward_logits(edited_model, edited_batch)[:, :-1, :]

        base_selected = base_logits[selected_mask.to(base_logits.device)]
        edited_selected = edited_logits[selected_mask.to(edited_logits.device)]
        token_kl[selected_mask] = position_kl(base_selected, edited_selected, direction).detach().cpu()

        if non_target_kl_mask.any():
            label_ids = input_ids[:, 1:]
            base_non_target_selected = base_logits[non_target_kl_mask.to(base_logits.device)]
            edited_non_target_selected = edited_logits[non_target_kl_mask.to(edited_logits.device)]
            target_token_ids = label_ids[non_target_kl_mask]
            non_target_token_kl[non_target_kl_mask] = non_target_vocab_kl(
                base_non_target_selected,
                edited_non_target_selected,
                target_token_ids,
                direction,
            ).detach().cpu()

    return [
        make_entry(
            item=item,
            tokenizer=tokenizer,
            input_ids=input_ids[row_idx],
            token_kl=token_kl[row_idx],
            non_target_token_kl=non_target_token_kl[row_idx],
            prompt_mask=prompt_mask[row_idx],
            target_mask=target_mask[row_idx],
            non_target_kl_mask=non_target_kl_mask[row_idx],
            save_token_details=save_token_details,
        )
        for row_idx, item in enumerate(items)
    ]


def new_totals() -> Dict[str, Any]:
    return {
        "item_count": 0,
        "prompt_kl_sum": 0.0,
        "prompt_token_count": 0,
        "target_kl_sum": 0.0,
        "target_token_count": 0,
        "non_target_kl_sum": 0.0,
        "non_target_token_count": 0,
    }


def update_totals(totals: Dict[str, Any], entry: Dict[str, Any]) -> None:
    totals["item_count"] += 1
    totals["prompt_kl_sum"] += float(entry["prompt_kl_sum"])
    totals["prompt_token_count"] += int(entry["prompt_kl_count"])
    totals["target_kl_sum"] += float(entry["target_kl_sum"])
    totals["target_token_count"] += int(entry["target_kl_count"])
    totals["non_target_kl_sum"] += float(entry["non_target_kl_sum"])
    totals["non_target_token_count"] += int(entry["non_target_kl_count"])


def finalize_totals(totals: Dict[str, Any]) -> Dict[str, Any]:
    prompt_count = int(totals["prompt_token_count"])
    target_count = int(totals["target_token_count"])
    non_target_count = int(totals["non_target_token_count"])
    prompt_sum = float(totals["prompt_kl_sum"])
    target_sum = float(totals["target_kl_sum"])
    non_target_sum = float(totals["non_target_kl_sum"])
    return {
        "item_count": int(totals["item_count"]),
        "prompt_avg_kl": prompt_sum / prompt_count if prompt_count else None,
        "prompt_kl_sum": prompt_sum,
        "prompt_token_count": prompt_count,
        "target_avg_kl": target_sum / target_count if target_count else None,
        "target_kl_sum": target_sum,
        "target_token_count": target_count,
        "non_target_avg_kl": non_target_sum / non_target_count if non_target_count else None,
        "non_target_kl_sum": non_target_sum,
        "non_target_token_count": non_target_count,
    }


def aggregate_entries(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    totals = new_totals()
    materialized = list(entries)
    for entry in materialized:
        update_totals(totals, entry)
    aggregate = finalize_totals(totals)
    aggregate["items"] = materialized
    return aggregate


def compute_all_kl(
    base_model: AutoModelForCausalLM,
    edited_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hparams: SimpleNamespace,
    kl_items: List[Dict[str, Any]],
    batch_size: int,
    direction: str,
    save_token_details: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    rewrite_totals = new_totals()
    rephrase_totals = new_totals()
    locality_totals = new_totals()
    locality_totals_by_key: Dict[str, Dict[str, Any]] = {}

    for chunk_index, chunk in enumerate(iter_chunks(kl_items, batch_size), start=1):
        print_time(f"KL batch {chunk_index}: {len(chunk)} item(s)")
        batch_entries = compute_batch_kl(
            base_model=base_model,
            edited_model=edited_model,
            tokenizer=tokenizer,
            hparams=hparams,
            items=chunk,
            direction=direction,
            save_token_details=save_token_details,
        )
        for entry in batch_entries:
            entries.append(entry)
            if entry["kind"] == "rewrite":
                update_totals(rewrite_totals, entry)
            elif entry["kind"] == "rephrase":
                update_totals(rephrase_totals, entry)
            elif entry["kind"] == "locality":
                update_totals(locality_totals, entry)
                update_totals(locality_totals_by_key.setdefault(entry["metric_key"], new_totals()), entry)

    summary = {
        "direction": direction,
        "rewrite": finalize_totals(rewrite_totals),
        "rephrase": finalize_totals(rephrase_totals),
        "locality": finalize_totals(locality_totals),
        "locality_by_key": {
            key: finalize_totals(totals)
            for key, totals in locality_totals_by_key.items()
        },
    }
    return entries, summary


def build_case_results(
    requests: List[Dict[str, Any]],
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        case_index = int(entry["case_index"])
        if entry["kind"] == "rewrite":
            grouped.setdefault(case_index, {})["rewrite"] = entry
        elif entry["kind"] == "rephrase":
            grouped.setdefault(case_index, {})["rephrase"] = entry
        elif entry["kind"] == "locality":
            grouped.setdefault(case_index, {}).setdefault("locality", {}).setdefault(entry["metric_key"], []).append(entry)

    results: List[Dict[str, Any]] = []
    for case_index, request in enumerate(requests):
        row: Dict[str, Any] = {
            "case_index": case_index,
            "case_id": request.get("case_id"),
        }
        group = grouped.get(case_index, {})
        if "rewrite" in group:
            entry = group["rewrite"]
            row.update(
                {
                    "rewrite_prompt": entry["prompt"],
                    "rewrite_target": entry["target"],
                    "rewrite_prompt_kl": entry["prompt_kl"],
                    "rewrite_prompt_kl_count": entry["prompt_kl_count"],
                    "rewrite_target_kl": entry["target_kl"],
                    "rewrite_target_kl_count": entry["target_kl_count"],
                    "rewrite_non_target_kl": entry["non_target_kl"],
                    "rewrite_non_target_kl_count": entry["non_target_kl_count"],
                }
            )
        if "rephrase" in group:
            entry = group["rephrase"]
            row.update(
                {
                    "rephrase_prompt": entry["prompt"],
                    "rephrase_target": entry["target"],
                    "rephrase_prompt_kl": entry["prompt_kl"],
                    "rephrase_prompt_kl_count": entry["prompt_kl_count"],
                    "rephrase_target_kl": entry["target_kl"],
                    "rephrase_target_kl_count": entry["target_kl_count"],
                    "rephrase_non_target_kl": entry["non_target_kl"],
                    "rephrase_non_target_kl_count": entry["non_target_kl_count"],
                }
            )
        if "locality" in group:
            locality: Dict[str, Any] = {}
            for locality_key, locality_entries in group["locality"].items():
                locality[locality_key] = aggregate_entries(locality_entries)
            row["locality"] = locality
            if "neighborhood" in locality:
                row["locality_prompt_kl"] = locality["neighborhood"]["prompt_avg_kl"]
                row["locality_prompt_kl_count"] = locality["neighborhood"]["prompt_token_count"]
                row["locality_target_kl"] = locality["neighborhood"]["target_avg_kl"]
                row["locality_target_kl_count"] = locality["neighborhood"]["target_token_count"]
                row["locality_non_target_kl"] = locality["neighborhood"]["non_target_avg_kl"]
                row["locality_non_target_kl_count"] = locality["neighborhood"]["non_target_token_count"]
        if len(row) > 2:
            results.append(row)
    return results


def run_kl_evaluation() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument(
        "--requests_path",
        type=str,
        default=None,
        help="Optional saved editing requests.json. When set, evaluate exactly these requests instead of sampling data_path.",
    )
    parser.add_argument("--model_path", required=True, type=str, help="Path to edited LLM")
    parser.add_argument("--base_model_path", type=str, default=None, help="Original/base LLM path")
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--kl_direction",
        choices=["base_to_edit", "edit_to_base"],
        default="base_to_edit",
        help="base_to_edit means KL(original || edited).",
    )
    parser.add_argument(
        "--save_token_details",
        action="store_true",
        help="Save per-token KL values and token strings. This can make output JSON large.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    requests = load_requests(args)
    kl_items = build_kl_items(requests)
    print(f"KL items: {len(kl_items)} (rewrite/rephrase/locality prompt-target pairs)")

    hparams = load_eval_hparams(args)
    base_model_path = resolve_base_model_path(args, hparams)
    if not base_model_path and kl_items:
        raise ValueError(
            "Could not resolve the original/base model. Pass --base_model_path explicitly."
        )

    print_time("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    ensure_tokenizer_padding(tokenizer)

    if not kl_items:
        kl_summary = {
            "direction": args.kl_direction,
            "rewrite": finalize_totals(new_totals()),
            "rephrase": finalize_totals(new_totals()),
            "locality": finalize_totals(new_totals()),
            "locality_by_key": {},
        }
        entries: List[Dict[str, Any]] = []
    else:
        print_time(f"Loading original/base model: {base_model_path}")
        base_model = load_model(base_model_path, args.trust_remote_code)
        maybe_resize_embeddings(base_model, tokenizer)

        print_time(f"Loading edited model: {args.model_path}")
        edited_model = load_model(args.model_path, args.trust_remote_code)
        maybe_resize_embeddings(edited_model, tokenizer)

        print_time(f"Computing KL with both models loaded ({args.kl_direction})")
        entries, kl_summary = compute_all_kl(
            base_model=base_model,
            edited_model=edited_model,
            tokenizer=tokenizer,
            hparams=hparams,
            kl_items=kl_items,
            batch_size=args.batch_size,
            direction=args.kl_direction,
            save_token_details=args.save_token_details,
        )

    results = build_case_results(requests, entries)
    rewrite_summary = kl_summary["rewrite"]
    rephrase_summary = kl_summary["rephrase"]
    locality_summary = kl_summary["locality"]
    summary = {
        "metric_type": "easyedit_token_kl",
        "model_path": args.model_path,
        "base_model_path": base_model_path,
        "data_path": args.data_path,
        "requests_path": args.requests_path,
        "sample_source": "requests_path" if args.requests_path else "data_path",
        "seed": args.seed,
        "num_samples_requested": len(requests) if args.requests_path else args.num_samples,
        "num_samples_used": len(requests),
        "kl_direction": args.kl_direction,
        "kl_definition": {
            "base_to_edit": "KL(original || edited)",
            "edit_to_base": "KL(edited || original)",
        }[args.kl_direction],
        "prompt_scope": "prompt next-token KL positions up to the token before the final prompt token; excludes the final prompt position that predicts the first target token",
        "target_scope": "EasyEdit teacher-forced target positions from prompt + ' ' + target",
        "non_target_scope": "rewrite/rephrase target prediction positions; KL over vocab distributions renormalized after excluding that position's gold target token",
        "rewrite_prompt_kl": rewrite_summary["prompt_avg_kl"],
        "rewrite_prompt_token_count": rewrite_summary["prompt_token_count"],
        "rewrite_target_kl": rewrite_summary["target_avg_kl"],
        "rewrite_target_token_count": rewrite_summary["target_token_count"],
        "rewrite_non_target_kl": rewrite_summary["non_target_avg_kl"],
        "rewrite_non_target_token_count": rewrite_summary["non_target_token_count"],
        "rephrase_prompt_kl": rephrase_summary["prompt_avg_kl"],
        "rephrase_prompt_token_count": rephrase_summary["prompt_token_count"],
        "rephrase_target_kl": rephrase_summary["target_avg_kl"],
        "rephrase_target_token_count": rephrase_summary["target_token_count"],
        "rephrase_non_target_kl": rephrase_summary["non_target_avg_kl"],
        "rephrase_non_target_token_count": rephrase_summary["non_target_token_count"],
        "locality_prompt_kl": locality_summary["prompt_avg_kl"],
        "locality_prompt_token_count": locality_summary["prompt_token_count"],
        "locality_target_kl": locality_summary["target_avg_kl"],
        "locality_target_token_count": locality_summary["target_token_count"],
        "locality_non_target_kl": locality_summary["non_target_avg_kl"],
        "locality_non_target_token_count": locality_summary["non_target_token_count"],
        "kl_summary": kl_summary,
    }

    print("\n" + "=" * 60)
    print(f"KL Results for: {args.model_path}")
    print(f"Base Model    : {base_model_path}")
    print(f"KL Direction  : {summary['kl_definition']}")
    if summary["rewrite_prompt_kl"] is not None:
        print(
            f"Rewrite Prompt KL: {summary['rewrite_prompt_kl']:.6f} "
            f"(tokens={summary['rewrite_prompt_token_count']})"
        )
    if summary["rewrite_target_kl"] is not None:
        print(
            f"Rewrite Target KL: {summary['rewrite_target_kl']:.6f} "
            f"(tokens={summary['rewrite_target_token_count']})"
        )
    if summary["rewrite_non_target_kl"] is not None:
        print(
            f"Rewrite Non Target KL: {summary['rewrite_non_target_kl']:.6f} "
            f"(tokens={summary['rewrite_non_target_token_count']})"
        )
    if summary["rephrase_prompt_kl"] is not None:
        print(
            f"Rephrase Prompt KL: {summary['rephrase_prompt_kl']:.6f} "
            f"(tokens={summary['rephrase_prompt_token_count']})"
        )
    if summary["rephrase_target_kl"] is not None:
        print(
            f"Rephrase Target KL: {summary['rephrase_target_kl']:.6f} "
            f"(tokens={summary['rephrase_target_token_count']})"
        )
    if summary["rephrase_non_target_kl"] is not None:
        print(
            f"Rephrase Non Target KL: {summary['rephrase_non_target_kl']:.6f} "
            f"(tokens={summary['rephrase_non_target_token_count']})"
        )
    if summary["locality_prompt_kl"] is not None:
        print(
            f"Locality Prompt KL: {summary['locality_prompt_kl']:.6f} "
            f"(tokens={summary['locality_prompt_token_count']})"
        )
    if summary["locality_target_kl"] is not None:
        print(
            f"Locality Target KL: {summary['locality_target_kl']:.6f} "
            f"(tokens={summary['locality_target_token_count']})"
        )
    if summary["locality_non_target_kl"] is not None:
        print(
            f"Locality Non Target KL: {summary['locality_non_target_kl']:.6f} "
            f"(tokens={summary['locality_non_target_token_count']})"
        )
    print("=" * 60)

    if args.save_path:
        payload = {
            "summary": summary,
            "results": results,
            "entries": entries,
        }
        save_path = Path(args.save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved KL results to {save_path}")

    print_time("KL evaluation finished")


if __name__ == "__main__":
    run_kl_evaluation()
