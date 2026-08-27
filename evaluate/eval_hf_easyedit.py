import argparse
from copy import deepcopy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from EasyEdit.easyeditor.evaluate.evaluate import (  # noqa: E402
    compute_locality_quality,
    compute_portability_quality,
    compute_rewrite_or_rephrase_quality,
)


def print_time(process_name: str) -> None:
    now = datetime.now()
    formatted_time = now.strftime("%m-%d %H:%M:%S")
    print(f"[{formatted_time}] {process_name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_or_none(x: Optional[Any]) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, dict):
        x = x.get("str")
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


def first_non_none(*values: Any) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def maybe_format_prompt(prompt: Optional[str], subject: Optional[str]) -> Optional[str]:
    prompt = strip_or_none(prompt)
    subject = strip_or_none(subject)
    if prompt is None:
        return None
    if subject and "{}" in prompt:
        try:
            return prompt.format(subject)
        except Exception:
            return prompt.replace("{}", subject)
    return prompt


def get_target_text(record: Dict[str, Any]) -> Optional[str]:
    req = record.get("requested_rewrite") if isinstance(record.get("requested_rewrite"), dict) else None
    target = first_non_none(
        record.get("target_new"),
        record.get("alt"),
        req.get("target_new") if req else None,
    )
    return strip_or_none(target)


def get_ground_truth_text(record: Dict[str, Any]) -> Optional[str]:
    req = record.get("requested_rewrite") if isinstance(record.get("requested_rewrite"), dict) else None
    answers = record.get("answers")
    first_answer = answers[0] if isinstance(answers, list) and answers else None
    ground_truth = first_non_none(
        record.get("ground_truth"),
        first_answer,
        record.get("pred"),
        req.get("ground_truth") if req else None,
    )
    return strip_or_none(ground_truth)


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def normalize_metric_pairs(
    prompts: Any,
    ground_truths: Any,
    subject: Optional[str],
) -> List[Tuple[str, str]]:
    prompt_list = [maybe_format_prompt(prompt, subject) for prompt in ensure_list(prompts)]
    ground_truth_list = [strip_or_none(ground_truth) for ground_truth in ensure_list(ground_truths)]

    if len(prompt_list) == 1 and len(ground_truth_list) > 1:
        prompt_list = prompt_list * len(ground_truth_list)
    elif len(ground_truth_list) == 1 and len(prompt_list) > 1:
        ground_truth_list = ground_truth_list * len(prompt_list)
    elif len(prompt_list) != len(ground_truth_list):
        return []

    pairs: List[Tuple[str, str]] = []
    for prompt, ground_truth in zip(prompt_list, ground_truth_list):
        if prompt is None or ground_truth is None:
            continue
        pairs.append((prompt, ground_truth))
    return pairs


def build_metric_group_from_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    prompts = [prompt for prompt, _ in pairs]
    ground_truths = [ground_truth for _, ground_truth in pairs]
    if len(prompts) == 1:
        return {
            "prompt": prompts[0],
            "ground_truth": ground_truths[0],
        }
    return {
        "prompt": prompts,
        "ground_truth": ground_truths,
    }


def extract_metric_group(raw_group: Any, subject: Optional[str]) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_group, dict):
        return metrics

    for metric_key, payload in raw_group.items():
        if not isinstance(payload, dict):
            continue
        pairs = normalize_metric_pairs(
            payload.get("prompt"),
            payload.get("ground_truth"),
            subject,
        )
        if not pairs:
            continue
        metrics[metric_key] = build_metric_group_from_pairs(pairs)
    return metrics


def get_nested_metric_group(record: Dict[str, Any], group_key: str, subject: Optional[str]) -> Dict[str, Dict[str, Any]]:
    req = record.get("requested_rewrite") if isinstance(record.get("requested_rewrite"), dict) else None
    raw_group = first_non_none(
        record.get(group_key),
        req.get(group_key) if req else None,
    )
    return extract_metric_group(raw_group, subject)


def get_first_list_item(records: Any) -> Optional[Any]:
    if isinstance(records, list) and records:
        return records[0]
    return None


def get_first_prompt_entry(records: Any) -> Optional[Any]:
    first_item = get_first_list_item(records)
    if isinstance(first_item, dict):
        return first_item.get("prompt")
    return first_item


def parse_record(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    req = record.get("requested_rewrite") if isinstance(record.get("requested_rewrite"), dict) else None

    subject = strip_or_none(first_non_none(record.get("subject"), req.get("subject") if req else None))
    prompt = maybe_format_prompt(
        first_non_none(
            record.get("prompt"),
            record.get("src"),
            req.get("prompt") if req else None,
        ),
        subject,
    )
    locality = get_nested_metric_group(record, "locality", subject)
    portability = get_nested_metric_group(record, "portability", subject)
    rephrase_prompt = maybe_format_prompt(
        first_non_none(
            record.get("rephrase_prompt"),
            record.get("rephrase"),
            get_first_prompt_entry(record.get("paraphrase_prompts")),
        ),
        subject,
    )
    target = get_target_text(record)
    ground_truth = get_ground_truth_text(record)

    if prompt is None or target is None:
        return None

    if not locality:
        neighborhood_prompts = record.get("neighborhood_prompts")
        if isinstance(neighborhood_prompts, list) and neighborhood_prompts:
            prompt_values = []
            ground_truth_values = []
            for entry in neighborhood_prompts:
                if isinstance(entry, dict):
                    prompt_values.append(entry.get("prompt"))
                    ground_truth_values.append(first_non_none(entry.get("ground_truth"), entry.get("target")))
                else:
                    prompt_values.append(entry)
                    ground_truth_values.append(
                        first_non_none(
                            record.get("locality_ground_truth"),
                            record.get("loc_ans"),
                        )
                    )
            pairs = normalize_metric_pairs(prompt_values, ground_truth_values, subject)
            if pairs:
                locality["neighborhood"] = build_metric_group_from_pairs(pairs)
        else:
            locality_pairs = normalize_metric_pairs(
                first_non_none(record.get("locality_prompt"), record.get("loc")),
                first_non_none(record.get("locality_ground_truth"), record.get("loc_ans")),
                subject,
            )
            if locality_pairs:
                locality["neighborhood"] = build_metric_group_from_pairs(locality_pairs)

    return {
        "case_id": first_non_none(record.get("case_id"), record.get("id"), idx),
        "subject": subject,
        "prompt": prompt,
        "rephrase_prompt": rephrase_prompt,
        "locality": locality,
        "portability": portability,
        "ground_truth": ground_truth,
        "target_new": target,
    }


def build_easyedit_request(item: Dict[str, Any]) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "case_id": item["case_id"],
        "prompt": item["prompt"],
        "target_new": item["target_new"],
        "ground_truth": item.get("ground_truth") or "<|endoftext|>",
        "portability": item.get("portability", {}),
        "locality": item.get("locality", {}),
    }
    if item.get("subject") is not None:
        request["subject"] = item["subject"]
    if item.get("rephrase_prompt") is not None:
        request["rephrase_prompt"] = item["rephrase_prompt"]
    return request


def sanitize_cache_component(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    compacted = "".join(cleaned).strip("._")
    return compacted or "unknown"


def build_request_signature(request: Dict[str, Any]) -> str:
    return json.dumps(request, ensure_ascii=False, sort_keys=True)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_hparams_path(args: argparse.Namespace) -> Optional[Path]:
    if args.hparams_path:
        return Path(args.hparams_path).expanduser().resolve()

    metadata_paths = [
        Path(args.model_path) / "run_config.json",
        Path(args.model_path) / "training_manifest.json",
    ]
    config_keys = ("hparams_dir", "hparams_path", "pl_config_path", "ft_config_path")

    for metadata_path in metadata_paths:
        payload = load_json_if_exists(metadata_path)
        if not payload:
            continue
        for key in config_keys:
            config_path = payload.get(key)
            if config_path:
                return Path(config_path).expanduser().resolve()

    return None


def load_eval_hparams(args: argparse.Namespace) -> SimpleNamespace:
    hparams_path = resolve_hparams_path(args)
    config: Dict[str, Any] = {}

    if hparams_path and hparams_path.exists():
        with open(hparams_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    config.setdefault("alg_name", "FT")
    config.setdefault("max_length", 40)
    config.setdefault("model_name", None)
    config.setdefault("use_chat_template", False)
    config["hparams_path"] = str(hparams_path) if hparams_path else None
    config["device"] = 0
    return SimpleNamespace(**config)


def is_usable_base_model_ref(value: str) -> bool:
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return False

    path = Path(value)
    if any(part.startswith("checkpoint_") for part in path.parts):
        return False

    return True


def get_nested_string(payload: Dict[str, Any], key_path: Tuple[str, ...]) -> Optional[str]:
    current: Any = payload
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


def resolve_base_model_from_manifests(model_path: str) -> Optional[str]:
    model_dir = Path(model_path).expanduser()
    run_dir = model_dir.parent if model_dir.name.startswith("checkpoint_") else model_dir

    manifest_paths = [
        run_dir / "training_manifest.json",
        run_dir / "run_manifest.json",
        model_dir / "training_manifest.json",
        model_dir / "run_manifest.json",
        model_dir / "checkpoint_manifest.json",
    ]
    key_paths = [
        ("base_model_path",),
        ("base_model_name",),
        ("base_model",),
        ("original_base_model_path",),
        ("original_base_model_name",),
        ("original_model_name",),
        ("pretrained_model_name_or_path",),
        ("initial_model_name",),
        ("initial_model_path",),
        ("source_model_name",),
        ("source_model_path",),
        ("model_name",),
        ("effective_hparams", "model_name"),
        ("hparams", "model_name"),
    ]

    seen: set[Path] = set()
    for manifest_path in manifest_paths:
        try:
            resolved = manifest_path.resolve()
        except OSError:
            resolved = manifest_path
        if resolved in seen or not manifest_path.exists():
            continue
        seen.add(resolved)

        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue

        for key_path in key_paths:
            candidate = get_nested_string(payload, key_path)
            if candidate and is_usable_base_model_ref(candidate):
                return candidate

    return None


def resolve_base_model_path(args: argparse.Namespace, hparams: SimpleNamespace) -> Optional[str]:
    if args.base_model_path:
        return args.base_model_path

    manifest_model = resolve_base_model_from_manifests(args.model_path)
    if manifest_model:
        return manifest_model

    if getattr(hparams, "model_name", None) and is_usable_base_model_ref(str(hparams.model_name)):
        return str(hparams.model_name)

    config_path = Path(args.model_path) / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        base_name = config.get("_name_or_path") or config.get("name_or_path")
        if base_name and is_usable_base_model_ref(str(base_name)):
            return str(base_name)

    inferred = infer_base_model_from_dirname(str(Path(args.model_path)))
    if inferred:
        return inferred

    return None


def infer_base_model_from_dirname(dirname: str) -> Optional[str]:
    candidates = [
        ("Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
        ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
        ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
        ("Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1"),
    ]
    for needle, resolved in candidates:
        if needle in dirname:
            return resolved
    return None


def resolve_eval_device(model: AutoModelForCausalLM) -> Any:
    device = next(model.parameters()).device
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return "cpu"


def load_model(model_path: str, trust_remote_code: bool) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def iter_chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        size = len(items) if len(items) > 0 else 1
    for start in range(0, len(items), size):
        yield items[start : start + size]


def metric_list(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def compute_rewrite_scores(
    model: AutoModelForCausalLM,
    model_name: str,
    hparams: SimpleNamespace,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: Any,
    batch_size: int,
    test_rephrase: bool = False,
) -> List[float]:
    key = "rephrase_acc" if test_rephrase else "rewrite_acc"
    scores: List[float] = []

    for chunk in iter_chunks(list(zip(prompts, targets)), batch_size):
        chunk_prompts = [prompt for prompt, _ in chunk]
        chunk_targets = [target for _, target in chunk]
        ret = compute_rewrite_or_rephrase_quality(
            model,
            model_name,
            hparams,
            tokenizer,
            chunk_prompts,
            chunk_targets,
            device=device,
            test_rephrase=test_rephrase,
        )
        scores.extend(metric_list(ret[key]))

    return scores


def compute_locality_outputs(
    model: AutoModelForCausalLM,
    model_name: str,
    hparams: SimpleNamespace,
    tokenizer: AutoTokenizer,
    locality_items: List[Tuple[int, str, str, str]],
    device: Any,
    batch_size: int,
) -> Dict[int, Dict[str, List[Any]]]:
    outputs_by_index: Dict[int, Dict[str, List[Any]]] = {}
    grouped_items: Dict[str, List[Tuple[int, str, str]]] = {}

    for idx, locality_key, prompt, ground_truth in locality_items:
        grouped_items.setdefault(locality_key, []).append((idx, prompt, ground_truth))

    for locality_key, grouped_locality_items in grouped_items.items():
        for chunk in iter_chunks(grouped_locality_items, batch_size):
            indices = [idx for idx, _, _ in chunk]
            prompts = [prompt for _, prompt, _ in chunk]
            ground_truths = [ground_truth for _, _, ground_truth in chunk]
            ret = compute_locality_quality(
                model,
                model_name,
                hparams,
                tokenizer,
                locality_key,
                prompts,
                ground_truths,
                device=device,
            )
            chunk_outputs = ret[f"{locality_key}_output"]
            for idx, output in zip(indices, chunk_outputs):
                outputs_by_index.setdefault(idx, {}).setdefault(locality_key, []).append(output)

    return outputs_by_index


def compute_portability_scores(
    model: AutoModelForCausalLM,
    model_name: str,
    hparams: SimpleNamespace,
    tokenizer: AutoTokenizer,
    portability_items: List[Tuple[int, str, str, str]],
    device: Any,
    batch_size: int,
) -> Dict[int, Dict[str, List[float]]]:
    scores_by_index: Dict[int, Dict[str, List[float]]] = {}
    grouped_items: Dict[str, List[Tuple[int, str, str]]] = {}

    for idx, portability_key, prompt, ground_truth in portability_items:
        grouped_items.setdefault(portability_key, []).append((idx, prompt, ground_truth))

    for portability_key, grouped_portability_items in grouped_items.items():
        for chunk in iter_chunks(grouped_portability_items, batch_size):
            indices = [idx for idx, _, _ in chunk]
            prompts = [prompt for _, prompt, _ in chunk]
            ground_truths = [ground_truth for _, _, ground_truth in chunk]
            ret = compute_portability_quality(
                model,
                model_name,
                hparams,
                tokenizer,
                portability_key,
                prompts,
                ground_truths,
                device=device,
            )
            chunk_scores = metric_list(ret[f"{portability_key}_acc"])
            for idx, score in zip(indices, chunk_scores):
                scores_by_index.setdefault(idx, {}).setdefault(portability_key, []).append(float(score))

    return scores_by_index


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def metric_scalar(value: Any) -> float:
    if isinstance(value, list):
        flattened = [float(v) for v in value]
        return safe_mean(flattened)
    return float(value)


def aggregate_easyedit_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for stage in ("pre", "post"):
        stage_metrics: Dict[str, Any] = {}

        for key in ("rewrite_acc", "rephrase_acc", "rewrite_ppl", "ood_acc"):
            values = [
                metric_scalar(metric[stage][key])
                for metric in all_metrics
                if stage in metric and key in metric[stage]
            ]
            if values:
                stage_metrics[key] = safe_mean(values)

        for group_key in ("locality", "portability"):
            group_values: Dict[str, List[float]] = {}
            for metric in all_metrics:
                if stage not in metric:
                    continue
                group = metric[stage].get(group_key)
                if not isinstance(group, dict):
                    continue
                for metric_key, metric_value in group.items():
                    if not metric_key.endswith("acc"):
                        continue
                    group_values.setdefault(metric_key, []).append(metric_scalar(metric_value))
            if group_values:
                stage_metrics[group_key] = {
                    metric_key: safe_mean(values)
                    for metric_key, values in group_values.items()
                }

        if stage_metrics:
            summary[stage] = stage_metrics

    return summary


def collect_group_items(
    requests: List[Dict[str, Any]],
    group_key: str,
) -> List[Tuple[int, str, str, str]]:
    items: List[Tuple[int, str, str, str]] = []

    for idx, request in enumerate(requests):
        group = request.get(group_key)
        if not isinstance(group, dict):
            continue
        for metric_key, payload in group.items():
            if not isinstance(payload, dict):
                continue
            pairs = normalize_metric_pairs(
                payload.get("prompt"),
                payload.get("ground_truth"),
                request.get("subject"),
            )
            for prompt, ground_truth in pairs:
                items.append((idx, metric_key, prompt, ground_truth))

    return items


def resolve_pre_cache_path(
    args: argparse.Namespace,
    base_model_path: Optional[str],
) -> Optional[Path]:
    if args.pre_cache_path:
        return Path(args.pre_cache_path).expanduser().resolve()

    if not args.pre_cache_dir:
        return None

    cache_dir = Path(args.pre_cache_dir).expanduser().resolve()
    base_label_source = base_model_path or Path(args.model_path).name
    base_label = sanitize_cache_component(Path(str(base_label_source)).name)
    if args.requests_path:
        requests_path = Path(args.requests_path).expanduser().resolve()
        data_label = sanitize_cache_component(f"{requests_path.parent.name}__{requests_path.stem}")
        sample_label = "trained_requests"
    else:
        data_label = sanitize_cache_component(Path(args.data_path).stem)
        sample_label = f"seed{args.seed}_n{args.num_samples if args.num_samples > 0 else 'all'}"
    return cache_dir / f"{base_label}__{data_label}__{sample_label}.json"


def normalize_loaded_pre_metrics(raw_metrics: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, list):
        raise ValueError("Pre-metrics cache must contain a JSON list or a payload with all_metrics list.")

    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_metrics):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid pre-metrics entry at index {idx}: expected object, got {type(entry).__name__}")
        pre_metrics = entry.get("pre", entry)
        if not isinstance(pre_metrics, dict):
            raise ValueError(f"Invalid pre-metrics entry at index {idx}: missing pre metric dict")
        normalized.append({"pre": deepcopy(pre_metrics)})
    return normalized


def load_pre_metrics_cache(path: Path) -> Tuple[List[Dict[str, Any]], Optional[List[str]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata: Dict[str, Any] = payload if isinstance(payload, dict) else {}
    raw_metrics = payload.get("all_metrics", payload) if isinstance(payload, dict) else payload
    request_signatures = metadata.get("request_signatures") if isinstance(metadata, dict) else None

    if request_signatures is None and isinstance(raw_metrics, list):
        inferred_signatures: List[str] = []
        can_infer = True
        for entry in raw_metrics:
            if not isinstance(entry, dict) or "requested_rewrite" not in entry:
                can_infer = False
                break
            inferred_signatures.append(build_request_signature(entry["requested_rewrite"]))
        if can_infer:
            request_signatures = inferred_signatures

    return normalize_loaded_pre_metrics(raw_metrics), request_signatures, metadata


def validate_loaded_pre_metrics(
    pre_metrics: List[Dict[str, Any]],
    request_signatures: Optional[List[str]],
    requests: List[Dict[str, Any]],
    pre_cache_path: Path,
) -> None:
    if len(pre_metrics) != len(requests):
        raise ValueError(
            f"Pre-metrics cache size mismatch: {pre_cache_path} has {len(pre_metrics)} entries, "
            f"but current evaluation expects {len(requests)} requests."
        )

    if not request_signatures:
        return

    current_signatures = [build_request_signature(request) for request in requests]
    if len(request_signatures) != len(current_signatures):
        raise ValueError(
            f"Pre-metrics signature count mismatch: {pre_cache_path} has {len(request_signatures)} signatures, "
            f"but current evaluation expects {len(current_signatures)} requests."
        )

    for idx, (loaded_signature, current_signature) in enumerate(zip(request_signatures, current_signatures)):
        if loaded_signature != current_signature:
            raise ValueError(
                f"Pre-metrics cache request mismatch at index {idx} for {pre_cache_path}. "
                "Use the same dataset sampling config or regenerate the pre cache."
            )


def build_request_eval_items(
    requests: List[Dict[str, Any]],
) -> Tuple[
    List[str],
    List[str],
    List[Tuple[int, str, str]],
    List[Tuple[int, str, str, str]],
    List[Tuple[int, str, str, str]],
]:
    prompts = [request["prompt"] for request in requests]
    targets = [request["target_new"] for request in requests]
    rephrase_items = [
        (idx, request["rephrase_prompt"], request["target_new"])
        for idx, request in enumerate(requests)
        if request.get("rephrase_prompt")
    ]
    locality_items = collect_group_items(requests, "locality")
    portability_items = collect_group_items(requests, "portability")
    return prompts, targets, rephrase_items, locality_items, portability_items


def save_pre_metrics_cache(
    path: Path,
    pre_metrics: List[Dict[str, Any]],
    requests: List[Dict[str, Any]],
    args: argparse.Namespace,
    base_model_path: Optional[str],
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_type": "easyedit_pre_metrics",
        "base_model_path": base_model_path,
        "model_name": model_name,
        "data_path": args.data_path,
        "requests_path": args.requests_path,
        "sample_source": "requests_path" if args.requests_path else "data_path",
        "seed": args.seed,
        "num_samples_requested": len(requests) if args.requests_path else args.num_samples,
        "num_samples_used": len(requests),
        "request_signatures": [build_request_signature(request) for request in requests],
        "all_metrics": pre_metrics,
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def build_pre_metrics(
    requests: List[Dict[str, Any]],
    rewrite_scores: List[float],
    rephrase_scores_by_index: Dict[int, float],
    locality_outputs: Dict[int, Dict[str, List[Any]]],
    portability_scores_by_index: Dict[int, Dict[str, List[float]]],
) -> List[Dict[str, Any]]:
    all_metrics: List[Dict[str, Any]] = []

    for idx, request in enumerate(requests):
        pre_metrics: Dict[str, Any] = {
            "rewrite_acc": [float(rewrite_scores[idx])],
            "locality": {},
            "portability": {},
        }

        if idx in rephrase_scores_by_index:
            pre_metrics["rephrase_acc"] = [float(rephrase_scores_by_index[idx])]

        if idx in locality_outputs:
            for locality_key, outputs in locality_outputs[idx].items():
                pre_metrics["locality"][f"{locality_key}_output"] = deepcopy(outputs)

        if idx in portability_scores_by_index:
            for portability_key, scores in portability_scores_by_index[idx].items():
                pre_metrics["portability"][f"{portability_key}_acc"] = [float(score) for score in scores]

        all_metrics.append({"pre": pre_metrics})

    return all_metrics


def compute_pre_metrics_for_requests(
    model: AutoModelForCausalLM,
    model_name: str,
    hparams: SimpleNamespace,
    tokenizer: AutoTokenizer,
    requests: List[Dict[str, Any]],
    device: Any,
    batch_size: int,
) -> List[Dict[str, Any]]:
    prompts, targets, rephrase_items, locality_items, portability_items = build_request_eval_items(requests)

    print_time(f"Teacher-forcing pre rewrite evaluation ({len(prompts)})")
    pre_rewrite_scores = compute_rewrite_scores(
        model=model,
        model_name=model_name,
        hparams=hparams,
        tokenizer=tokenizer,
        prompts=prompts,
        targets=targets,
        device=device,
        batch_size=batch_size,
        test_rephrase=False,
    )

    pre_rephrase_scores_by_index: Dict[int, float] = {}
    if rephrase_items:
        print_time(f"Teacher-forcing pre rephrase evaluation ({len(rephrase_items)})")
        pre_rephrase_scores = compute_rewrite_scores(
            model=model,
            model_name=model_name,
            hparams=hparams,
            tokenizer=tokenizer,
            prompts=[prompt for _, prompt, _ in rephrase_items],
            targets=[target for _, _, target in rephrase_items],
            device=device,
            batch_size=batch_size,
            test_rephrase=True,
        )
        pre_rephrase_scores_by_index = {
            idx: score for (idx, _, _), score in zip(rephrase_items, pre_rephrase_scores)
        }

    pre_locality_outputs: Dict[int, Dict[str, List[Any]]] = {}
    if locality_items:
        print_time(f"Teacher-forcing pre locality evaluation ({len(locality_items)})")
        pre_locality_outputs = compute_locality_outputs(
            model=model,
            model_name=model_name,
            hparams=hparams,
            tokenizer=tokenizer,
            locality_items=locality_items,
            device=device,
            batch_size=batch_size,
        )

    pre_portability_scores_by_index: Dict[int, Dict[str, List[float]]] = {}
    if portability_items:
        print_time(f"Teacher-forcing pre portability evaluation ({len(portability_items)})")
        pre_portability_scores_by_index = compute_portability_scores(
            model=model,
            model_name=model_name,
            hparams=hparams,
            tokenizer=tokenizer,
            portability_items=portability_items,
            device=device,
            batch_size=batch_size,
        )

    return build_pre_metrics(
        requests=requests,
        rewrite_scores=pre_rewrite_scores,
        rephrase_scores_by_index=pre_rephrase_scores_by_index,
        locality_outputs=pre_locality_outputs,
        portability_scores_by_index=pre_portability_scores_by_index,
    )


def run_evaluation() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument(
        "--requests_path",
        type=str,
        default=None,
        help="Optional saved editing requests.json. When set, evaluate exactly these requests instead of sampling data_path.",
    )
    parser.add_argument("--model_path", required=True, type=str, help="Path to edited LLM")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--pre_cache_path", type=str, default=None)
    parser.add_argument("--pre_cache_dir", type=str, default=None)
    parser.add_argument(
        "--pre_cache_prefix_path",
        type=str,
        default=None,
        help="Optional pre-metrics cache for a prefix of the current requests. Missing suffix pre metrics will be computed and saved as a full cache.",
    )
    parser.add_argument("--force_recompute_pre", action="store_true")
    parser.add_argument("--save_pre_only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    source_path = args.requests_path or args.data_path
    with open(source_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    if not isinstance(source_data, list):
        raise ValueError(f"Expected {source_path} to contain a JSON list")

    parsed: List[Dict[str, Any]] = []
    skipped = 0
    for i, record in enumerate(source_data):
        item = parse_record(record, i)
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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This EasyEdit-style teacher-forcing evaluator currently requires CUDA, "
            "because EasyEdit's evaluation helpers move tensors to cuda:<device> internally."
        )

    print(f"Loaded {len(data)} usable samples from {source_path}")
    print(f"Sample source: {'requests_path' if args.requests_path else 'data_path'}")
    if not args.requests_path:
        print(f"Sampling seed: {args.seed}")
    print(f"Skipped invalid records: {skipped}")

    requests = [build_easyedit_request(item) for item in data]
    prompts, targets, rephrase_items, locality_items, portability_items = build_request_eval_items(requests)

    hparams = load_eval_hparams(args)
    base_model_path = resolve_base_model_path(args, hparams)
    model_name = hparams.model_name or args.model_path
    pre_cache_path = resolve_pre_cache_path(args, base_model_path)

    print_time("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    pre_metrics: List[Dict[str, Any]] = []
    pre_cache_metadata: Dict[str, Any] = {}
    pre_loaded_from_cache = False
    pre_prefix_loaded_from_cache = False
    pre_prefix_count = 0

    if pre_cache_path and pre_cache_path.exists() and not args.force_recompute_pre:
        print_time(f"Loading pre-metrics cache: {pre_cache_path}")
        loaded_pre_metrics, request_signatures, pre_cache_metadata = load_pre_metrics_cache(pre_cache_path)
        validate_loaded_pre_metrics(loaded_pre_metrics, request_signatures, requests, pre_cache_path)
        pre_metrics = loaded_pre_metrics
        pre_loaded_from_cache = True
    elif args.pre_cache_prefix_path and not args.force_recompute_pre:
        pre_cache_prefix_path = Path(args.pre_cache_prefix_path).expanduser().resolve()
        print_time(f"Loading prefix pre-metrics cache: {pre_cache_prefix_path}")
        loaded_pre_metrics, request_signatures, pre_cache_metadata = load_pre_metrics_cache(pre_cache_prefix_path)
        if len(loaded_pre_metrics) > len(requests):
            raise ValueError(
                f"Prefix pre-metrics cache is longer than the current request list: "
                f"{pre_cache_prefix_path} has {len(loaded_pre_metrics)} entries, "
                f"but current evaluation expects {len(requests)} requests."
            )
        validate_loaded_pre_metrics(
            loaded_pre_metrics,
            request_signatures,
            requests[: len(loaded_pre_metrics)],
            pre_cache_prefix_path,
        )
        pre_metrics = loaded_pre_metrics
        pre_prefix_count = len(pre_metrics)
        pre_prefix_loaded_from_cache = True
        if pre_prefix_count == len(requests):
            pre_loaded_from_cache = True

    if pre_metrics and len(pre_metrics) == len(requests):
        pass
    elif base_model_path:
        print_time(f"Loading base model for EasyEdit-style pre metrics: {base_model_path}")
        base_model = load_model(base_model_path, args.trust_remote_code)
        base_model.resize_token_embeddings(len(tokenizer))

        base_device = resolve_eval_device(base_model)
        hparams.device = base_device

        suffix_start = len(pre_metrics)
        if suffix_start > 0:
            print_time(
                f"Prefix pre cache covers {suffix_start} request(s); "
                f"computing suffix pre metrics ({len(requests) - suffix_start})"
            )
        suffix_pre_metrics = compute_pre_metrics_for_requests(
            model=base_model,
            model_name=model_name,
            hparams=hparams,
            tokenizer=tokenizer,
            requests=requests[suffix_start:],
            device=base_device,
            batch_size=args.batch_size,
        )
        pre_metrics = pre_metrics + suffix_pre_metrics

        if pre_cache_path:
            print_time(f"Saving pre-metrics cache: {pre_cache_path}")
            save_pre_metrics_cache(
                path=pre_cache_path,
                pre_metrics=pre_metrics,
                requests=requests,
                args=args,
                base_model_path=base_model_path,
                model_name=model_name,
            )

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_pre_only:
        if not pre_metrics:
            raise ValueError(
                "Could not build pre metrics. Pass --base_model_path explicitly, or run with a model directory "
                "that lets the script resolve the base model."
            )

        pre_summary = aggregate_easyedit_metrics(pre_metrics)
        print("\n" + "=" * 60)
        print("Saved/Loaded EasyEdit-style pre metrics")
        if pre_cache_path:
            print(f"Pre Cache Path: {pre_cache_path}")
        if "pre" in pre_summary and "rewrite_acc" in pre_summary["pre"]:
            print(f"Pre Rewrite Acc: {pre_summary['pre']['rewrite_acc']:.4f}")
        if "pre" in pre_summary and "rephrase_acc" in pre_summary["pre"]:
            print(f"Pre Rephrase Acc: {pre_summary['pre']['rephrase_acc']:.4f}")
        print("=" * 60)

        if args.save_path:
            save_pre_metrics_cache(
                path=Path(args.save_path).expanduser().resolve(),
                pre_metrics=pre_metrics,
                requests=requests,
                args=args,
                base_model_path=base_model_path,
                model_name=model_name,
            )
            print(f"Saved pre-only payload to {args.save_path}")

        print_time("Evaluation finished")
        return

    if locality_items and not pre_metrics:
        raise ValueError(
            "Locality evaluation now expects pre metrics from either a cache file or a resolvable base model. "
            "Pass --pre_cache_path/--pre_cache_dir, or keep --base_model_path available."
        )

    print_time("Loading edited model")
    edited_model = load_model(args.model_path, args.trust_remote_code)
    edited_device = resolve_eval_device(edited_model)
    hparams.device = edited_device

    print_time(f"Teacher-forcing rewrite evaluation ({len(prompts)})")
    rewrite_scores = compute_rewrite_scores(
        model=edited_model,
        model_name=model_name,
        hparams=hparams,
        tokenizer=tokenizer,
        prompts=prompts,
        targets=targets,
        device=edited_device,
        batch_size=args.batch_size,
        test_rephrase=False,
    )

    rephrase_scores_by_index: Dict[int, float] = {}
    if rephrase_items:
        print_time(f"Teacher-forcing rephrase evaluation ({len(rephrase_items)})")
        rephrase_scores = compute_rewrite_scores(
            model=edited_model,
            model_name=model_name,
            hparams=hparams,
            tokenizer=tokenizer,
            prompts=[prompt for _, prompt, _ in rephrase_items],
            targets=[target for _, _, target in rephrase_items],
            device=edited_device,
            batch_size=args.batch_size,
            test_rephrase=True,
        )
        rephrase_scores_by_index = {
            idx: score for (idx, _, _), score in zip(rephrase_items, rephrase_scores)
        }

    post_locality_outputs: Dict[int, Dict[str, List[Any]]] = {}
    if locality_items:
        print_time(f"Teacher-forcing locality evaluation ({len(locality_items)})")
        post_locality_outputs = compute_locality_outputs(
            model=edited_model,
            model_name=model_name,
            hparams=hparams,
            tokenizer=tokenizer,
            locality_items=locality_items,
            device=edited_device,
            batch_size=args.batch_size,
        )

    post_portability_scores_by_index: Dict[int, Dict[str, List[float]]] = {}
    if portability_items:
        print_time(f"Teacher-forcing portability evaluation ({len(portability_items)})")
        post_portability_scores_by_index = compute_portability_scores(
            model=edited_model,
            model_name=model_name,
            hparams=hparams,
            tokenizer=tokenizer,
            portability_items=portability_items,
            device=edited_device,
            batch_size=args.batch_size,
        )

    rewrite_acc_list: List[float] = []
    rephrase_acc_list: List[float] = []
    locality_acc_list: List[float] = []
    pre_rewrite_acc_list: List[float] = []
    pre_rephrase_acc_list: List[float] = []
    results: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, Any]] = []

    for idx, request in enumerate(requests):
        rewrite_acc = float(rewrite_scores[idx])
        rewrite_acc_list.append(rewrite_acc)

        row: Dict[str, Any] = {
            "case_id": request["case_id"],
            "prompt": request["prompt"],
            "target_new": request["target_new"],
            "rewrite_acc": rewrite_acc,
        }
        metric_row: Dict[str, Any] = deepcopy(pre_metrics[idx]) if idx < len(pre_metrics) else {}
        metric_row.update(
            {
                "case_id": request["case_id"],
                "requested_rewrite": request,
                "post": {
                    "rewrite_acc": [rewrite_acc],
                    "locality": {},
                    "portability": {},
                },
            }
        )

        pre_section = metric_row.get("pre", {})
        if "rewrite_acc" in pre_section:
            pre_rewrite_acc = metric_scalar(pre_section["rewrite_acc"])
            pre_rewrite_acc_list.append(pre_rewrite_acc)
            row["pre_rewrite_acc"] = pre_rewrite_acc

        if idx in rephrase_scores_by_index:
            rephrase_acc = float(rephrase_scores_by_index[idx])
            rephrase_acc_list.append(rephrase_acc)
            row.update(
                {
                    "rephrase_prompt": request["rephrase_prompt"],
                    "rephrase_acc": rephrase_acc,
                }
            )
            metric_row["post"]["rephrase_acc"] = [rephrase_acc]

        if "rephrase_acc" in pre_section:
            pre_rephrase_acc = metric_scalar(pre_section["rephrase_acc"])
            pre_rephrase_acc_list.append(pre_rephrase_acc)
            row["pre_rephrase_acc"] = pre_rephrase_acc

        if idx in post_locality_outputs:
            pre_locality_section = pre_section.get("locality", {})
            for locality_key, post_outputs in post_locality_outputs[idx].items():
                pre_outputs = pre_locality_section.get(f"{locality_key}_output")
                if pre_outputs is None:
                    raise ValueError(
                        f"Missing pre locality output for case_id={request['case_id']} locality={locality_key}. "
                        "Regenerate the pre cache with the same dataset and sampling config."
                    )
                locality_scores = [
                    float(np.mean(np.equal(post_output, pre_output)))
                    for post_output, pre_output in zip(post_outputs, pre_outputs)
                ]
                if not locality_scores:
                    continue
                metric_row["post"]["locality"][f"{locality_key}_acc"] = locality_scores
                if locality_key == "neighborhood":
                    locality_acc = safe_mean(locality_scores)
                    locality_acc_list.append(locality_acc)
                    locality_payload = request["locality"][locality_key]
                    row.update(
                        {
                            "locality_prompt": locality_payload["prompt"],
                            "locality_ground_truth": locality_payload["ground_truth"],
                            "locality_acc": locality_acc,
                        }
                    )
                else:
                    row[f"{locality_key}_acc"] = safe_mean(locality_scores)

            if "pre" in metric_row and "locality" in metric_row["pre"]:
                metric_row["pre"].pop("locality")

        if idx in post_portability_scores_by_index:
            for portability_key, portability_scores in post_portability_scores_by_index[idx].items():
                metric_row["post"]["portability"][f"{portability_key}_acc"] = portability_scores
                row[f"portability_{portability_key}_acc"] = safe_mean(portability_scores)

        results.append(row)
        all_metrics.append(metric_row)

        if args.verbose:
            print(f"Example {idx + 1}:")
            if "pre_rewrite_acc" in row:
                print(f"Pre Rewrite Acc: {row['pre_rewrite_acc']:.4f}")
            print(f"Rewrite Acc: {rewrite_acc:.4f}")
            if "rephrase_acc" in row:
                print(f"Rephrase Acc: {row['rephrase_acc']:.4f}")
            if "locality_acc" in row:
                print(f"Locality Acc: {row['locality_acc']:.4f}")
            print("-" * 50)

    easyedit_summary = aggregate_easyedit_metrics(all_metrics)
    summary = {
        "metric_type": "easyedit_teacher_forcing",
        "model_path": args.model_path,
        "base_model_path": base_model_path,
        "pre_cache_path": str(pre_cache_path) if pre_cache_path else None,
        "pre_loaded_from_cache": pre_loaded_from_cache,
        "pre_cache_prefix_path": args.pre_cache_prefix_path,
        "pre_prefix_loaded_from_cache": pre_prefix_loaded_from_cache,
        "pre_prefix_count": pre_prefix_count,
        "hparams_path": getattr(hparams, "hparams_path", None),
        "data_path": args.data_path,
        "requests_path": args.requests_path,
        "sample_source": "requests_path" if args.requests_path else "data_path",
        "seed": args.seed,
        "num_samples_requested": len(data) if args.requests_path else args.num_samples,
        "num_samples_used": len(data),
        "pre_rewrite_acc": safe_mean(pre_rewrite_acc_list),
        "pre_rephrase_acc": safe_mean(pre_rephrase_acc_list),
        "rewrite_acc": safe_mean(rewrite_acc_list),
        "rephrase_acc": safe_mean(rephrase_acc_list),
        "locality_acc": safe_mean(locality_acc_list),
        "pre_rewrite_count": len(pre_rewrite_acc_list),
        "pre_rephrase_count": len(pre_rephrase_acc_list),
        "rewrite_count": len(rewrite_acc_list),
        "rephrase_count": len(rephrase_acc_list),
        "locality_count": len(locality_acc_list),
    }

    print("\n" + "=" * 60)
    print(f"Evaluation Results for: {args.model_path}")
    print(f"Metric Type   : {summary['metric_type']}")
    if summary["pre_cache_path"]:
        cache_state = "loaded" if summary["pre_loaded_from_cache"] else "computed"
        print(f"Pre Cache     : {summary['pre_cache_path']} ({cache_state})")
    if summary["pre_rewrite_count"] > 0:
        print(f"Pre Rewrite Acc: {summary['pre_rewrite_acc']:.4f} (n={summary['pre_rewrite_count']})")
    if summary["pre_rephrase_count"] > 0:
        print(f"Pre Rephrase Acc: {summary['pre_rephrase_acc']:.4f} (n={summary['pre_rephrase_count']})")
    print(f"Rewrite Acc   : {summary['rewrite_acc']:.4f} (n={summary['rewrite_count']})")
    print(f"Rephrase Acc  : {summary['rephrase_acc']:.4f} (n={summary['rephrase_count']})")
    if summary["locality_count"] > 0:
        print(f"Locality Acc  : {summary['locality_acc']:.4f} (n={summary['locality_count']})")
    if easyedit_summary:
        print(f"EasyEdit Summary: {json.dumps(easyedit_summary, ensure_ascii=False)}")
    print("=" * 60)

    if args.save_path:
        payload = {
            "summary": summary,
            "easyedit_summary": easyedit_summary,
            "results": results,
            "all_metrics": all_metrics,
        }
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {args.save_path}")

    print_time("Evaluation finished")


if __name__ == "__main__":
    run_evaluation()
