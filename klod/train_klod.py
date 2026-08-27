import argparse
import json
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EasyEdit.easyeditor.models.klod.klod_hparams import KLODHyperParams


SCRIPT_DIR = PROJECT_ROOT
DEFAULT_TARGET_ALPHA = 0.85

# Cache for MEMIT/AlphaEdit-style generated context templates.
CONTEXT_TEMPLATES_CACHE = None


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class WeightSwapContext:
    """Temporarily swap editable weights with a reference snapshot."""

    def __init__(self, weights_to_update: Dict[str, torch.nn.Parameter], ref_snapshot: Dict[str, torch.Tensor]):
        self.weights_to_update = weights_to_update
        self.ref_snapshot = ref_snapshot
        self.current_snapshot: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        with torch.no_grad():
            for name, param in self.weights_to_update.items():
                self.current_snapshot[name] = param.detach().clone()
                ref = self.ref_snapshot[name].to(device=param.device, dtype=param.dtype)
                param.copy_(ref)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with torch.no_grad():
            for name, param in self.weights_to_update.items():
                cur = self.current_snapshot[name].to(device=param.device, dtype=param.dtype)
                param.copy_(cur)
        return False


def clone_weight_snapshot(weights_to_update: Dict[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in weights_to_update.items()}


def chunks(arr, n):
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def print_time(process_name):
    now = datetime.now()
    print(f"{process_name}: {now.strftime('%m-%d %H:%M:%S')}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device_index(device_arg: Optional[str], config_device) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script currently requires a CUDA device.")

    if device_arg is None:
        return int(config_device)
    if device_arg == "auto":
        return torch.cuda.current_device()

    try:
        device_idx = int(device_arg)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --device value: {device_arg}. Use an integer GPU id or 'auto'."
        ) from exc

    if device_idx < 0 or device_idx >= torch.cuda.device_count():
        raise ValueError(
            f"Requested cuda:{device_idx}, but only {torch.cuda.device_count()} visible CUDA device(s) are available."
        )
    return device_idx


def get_model_load_dtype(config) -> Optional[torch.dtype]:
    if not getattr(config, "bf16", False):
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    print("[load] bf16=True in config, but this GPU does not report bf16 support. Falling back to fp16.")
    return torch.float16


def get_primary_model_device_index(model: AutoModelForCausalLM, fallback_device: int) -> int:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for _, device in hf_device_map.items():
            if isinstance(device, int):
                return device
            if isinstance(device, str) and device.startswith("cuda:"):
                return int(device.split(":")[1])

    model_device = getattr(model, "device", None)
    if model_device is not None:
        model_device = str(model_device)
        if model_device.startswith("cuda:"):
            return int(model_device.split(":")[1])
    return fallback_device


def format_path_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def get_data_stem(data_path: str) -> str:
    return os.path.splitext(os.path.basename(data_path))[0]


def get_arg_or_config(arg_value, config, attr: str, default=None):
    if arg_value is not None:
        return arg_value
    return getattr(config, attr, default)


def normalize_bool_int(value, *, name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return 1
        if normalized in {"0", "false", "no", "n", "off"}:
            return 0
    try:
        normalized_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be 0/1 or boolean-like, got {value!r}") from exc
    if normalized_int not in {0, 1}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return normalized_int


def resolve_existing_path(path: str, *base_dirs: str) -> str:
    if path is None:
        return path
    if os.path.isabs(path):
        return path

    candidates = [os.path.abspath(path)]
    candidates.extend(os.path.abspath(os.path.join(base_dir, path)) for base_dir in base_dirs if base_dir)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def resolve_output_path(path: str, base_dir: str) -> str:
    if path is None or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def apply_hparam_overrides(
    config,
    target_alpha: Optional[float],
    early_stop_loss: Optional[float],
) -> None:
    if target_alpha is not None:
        config.target_alpha = target_alpha
    if early_stop_loss is not None:
        config.early_stop_loss = float(early_stop_loss)


def get_model_name(config) -> str:
    if hasattr(config, "model_name_or_path"):
        return config.model_name_or_path
    if hasattr(config, "model_name"):
        return config.model_name
    raise AttributeError("Config must define model_name_or_path or model_name.")


def get_rewrite_module_template(config) -> str:
    if hasattr(config, "rewrite_module"):
        return config.rewrite_module
    if hasattr(config, "rewrite_module_tmp"):
        return config.rewrite_module_tmp
    raise AttributeError("Config must define rewrite_module or rewrite_module_tmp.")


def get_layers_to_edit(config) -> List[int]:
    if hasattr(config, "layers") and config.layers is not None:
        return list(config.layers)
    if hasattr(config, "edit_layers") and config.edit_layers is not None:
        return list(config.edit_layers)
    if hasattr(config, "layer"):
        return [int(config.layer)]
    raise AttributeError("Config must define one of layers/edit_layers/layer.")


def normalize_target_text(
    target_text: str,
    tok: Optional[AutoTokenizer] = None,
    append_eos: bool = False,
) -> str:
    if target_text != " " and len(target_text) > 0 and target_text[0] != " ":
        target_text = " " + target_text
    eos_token = resolve_tokenizer_eos_token(tok) if append_eos else None
    if eos_token is not None and len(target_text) > 0 and not target_text.endswith(eos_token):
        target_text += eos_token
    return target_text


def resolve_tokenizer_eos_token(tok: Optional[AutoTokenizer]) -> Optional[str]:
    if tok is None:
        return None

    eos_token = getattr(tok, "eos_token", None)
    if eos_token:
        return str(eos_token)

    eos_token_id = getattr(tok, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if eos_token_id is not None and hasattr(tok, "decode"):
        return str(tok.decode([int(eos_token_id)], skip_special_tokens=False))

    return None


def count_targets_needing_eos(requests: List[Dict], tok: AutoTokenizer) -> int:
    eos_token = resolve_tokenizer_eos_token(tok)
    if eos_token is None:
        return 0
    return sum(
        1
        for request in requests
        if str(request.get("target_new", "")) and not str(request.get("target_new", "")).endswith(eos_token)
    )


def normalize_training_requests(requests: List[Dict], tok: AutoTokenizer) -> List[Dict]:
    out = deepcopy(requests)
    for request in out:
        request["target_new"] = normalize_target_text(
            request["target_new"],
            tok=tok,
            append_eos=True,
        )
    return out


def parse_context_aug_length_params(spec: str) -> List[tuple]:
    """Parse strings like '10:5' into (prefix length, number of prefixes)."""
    if spec is None or str(spec).strip() == "":
        return [(10, 5)]

    params = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --context_aug_length_params item: {item}. "
                "Expected format like '10:5' or '10:5,15:10'."
            )
        length_str, n_gen_str = item.split(":", 1)
        length = int(length_str)
        n_gen = int(n_gen_str)
        if length <= 0 or n_gen <= 0:
            raise ValueError(
                f"Invalid --context_aug_length_params item: {item}. "
                "Both length and n_gen must be positive."
            )
        params.append((length, n_gen))

    if not params:
        return [(10, 5)]
    return params


def get_context_templates(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    length_params: List[tuple],
) -> List[str]:
    """
    Build MEMIT/AlphaEdit-style context templates and flatten them into prompt templates.
    """
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        try:
            from util.generate import generate_fast
        except ImportError as first_exc:
            try:
                from EasyEdit.easyeditor.util.generate import generate_fast
            except ImportError as second_exc:
                raise ImportError(
                    "Could not import generate_fast. Make sure the MEMIT/AlphaEdit/EasyEdit "
                    "root is on PYTHONPATH, or pass --easyedit_path correctly before running."
                ) from second_exc

        seed_prompts = ["The", "Therefore", "Because", "I", "You"]
        generated_prefixes = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for length, n_gen in length_params:
                generation_prompts = [
                    seed_prompts[i % len(seed_prompts)]
                    for i in range(int(n_gen))
                ]
                generated_prefixes.extend(
                    generate_fast(
                        model,
                        tok,
                        generation_prompts,
                        n_gen_per_prompt=1,
                        max_out_len=int(length),
                    )
                )
        if was_training:
            model.train()

        cleaned_templates = []
        for prefix in generated_prefixes:
            prefix = prefix.replace("{", "").replace("}", "").strip()
            if prefix:
                cleaned_templates.append(prefix + ". {}")

        CONTEXT_TEMPLATES_CACHE = ["{}"] + cleaned_templates
        print(f"Cached context templates ({len(CONTEXT_TEMPLATES_CACHE)}): {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def render_context_template(prompt: str, template: str) -> str:
    """Render a prompt into a context template."""
    if "{}" in template:
        return template.format(prompt)
    return f"{template.rstrip()} {prompt}"


def augment_requests_with_context_templates(
    requests: List[Dict],
    context_templates: List[str],
    max_templates: Optional[int] = None,
) -> List[Dict]:
    """
    Expand each request using context templates.

    The first template is normally '{}', so the original prompt is retained.
    `max_templates` counts the original template as well.
    """
    if max_templates is not None and max_templates > 0:
        context_templates = context_templates[:max_templates]

    augmented = []
    for request in requests:
        seen_prompts = set()
        for template_idx, template in enumerate(context_templates):
            augmented_prompt = render_context_template(request["prompt"], template)
            if augmented_prompt in seen_prompts:
                continue
            seen_prompts.add(augmented_prompt)

            aug_request = deepcopy(request)
            aug_request["prompt"] = augmented_prompt
            aug_request["original_prompt"] = request["prompt"]
            aug_request["context_template"] = template
            aug_request["aug_type"] = "original" if template.strip() == "{}" else f"context_template_{template_idx}"
            augmented.append(aug_request)

    return augmented


def save_context_aug_artifacts(
    save_dir: str,
    train_requests_augmented: List[Dict],
    context_templates: List[str],
    *,
    context_aug_stage: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    aug_requests_path = os.path.join(save_dir, f"training_requests_context_aug_{context_aug_stage}.json")
    with open(aug_requests_path, "w", encoding="utf-8") as f:
        json.dump(train_requests_augmented, f, ensure_ascii=False, indent=2)

    templates_path = os.path.join(save_dir, "context_templates.json")
    with open(templates_path, "w", encoding="utf-8") as f:
        json.dump(context_templates, f, ensure_ascii=False, indent=2)

    print(f"Saved context-augmented training requests to {aug_requests_path}")
    print(f"Saved context templates to {templates_path}")


def normalize_context_aug_stage(stage: str) -> str:
    stage = (stage or "klod").lower()
    aliases = {
        "ft": "locft",
        "both": "all",
    }
    stage = aliases.get(stage, stage)
    if stage not in {"klod", "locft", "all"}:
        raise ValueError(f"context_aug_stage must be one of ['klod', 'locft', 'all'], got {stage}")
    return stage


def build_training_request(record: Dict, idx: int) -> Optional[Dict]:
    prompt = record.get("prompt", record.get("src"))
    target = record.get("target_new", record.get("alt"))
    subject = record.get("subject", "")

    if not prompt or not target:
        return None

    request = {
        "prompt": prompt,
        "target_new": target,
        "subject": subject,
        "source_index": idx,
    }

    case_id = record.get("case_id", record.get("id"))
    if case_id is not None:
        request["case_id"] = case_id

    rephrase_prompt = record.get("rephrase_prompt", record.get("rephrase"))
    if rephrase_prompt:
        request["rephrase_prompt"] = rephrase_prompt

    locality_prompt = record.get("locality_prompt", record.get("loc"))
    locality_ground_truth = record.get("locality_ground_truth", record.get("loc_ans"))
    if locality_prompt:
        request["locality_prompt"] = locality_prompt
    if locality_ground_truth:
        request["locality_ground_truth"] = locality_ground_truth

    return request


def save_training_artifacts(
    save_dir: str,
    requests: List[Dict],
    *,
    data_path: str,
    num_requests_before_sampling: int,
    sample_size_requested: Optional[int],
    seed: int,
    num_steps: int,
    rewrite_kl_lambda: float,
    non_target_kl_lambda: float,
    early_stop_loss: Optional[float],
    target_alpha: Optional[float],
    use_context_aug: bool = False,
    context_aug_stage: str = "klod",
    context_aug_length_params: str = "10:5",
    context_aug_max_templates: Optional[int] = None,
    append_eos_to_target: bool = True,
    eos_token: Optional[str] = None,
    eos_appended_count: int = 0,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    requests_path = os.path.join(save_dir, "training_requests.json")
    with open(requests_path, "w", encoding="utf-8") as f:
        json.dump(requests, f, ensure_ascii=False, indent=2)

    manifest = {
        "objective": "KLOD",
        "data_path": data_path,
        "data_stem": get_data_stem(data_path),
        "num_requests_before_sampling": num_requests_before_sampling,
        "num_requests_used": len(requests),
        "sample_size_requested": sample_size_requested,
        "seed": seed,
        "num_steps": num_steps,
        "rewrite_kl_lambda": rewrite_kl_lambda,
        "non_target_kl_lambda": non_target_kl_lambda,
        "early_stop_loss": early_stop_loss,
        "target_alpha": target_alpha,
        "edit_loss": "one_sided_logit_odds_hinge",
        "use_context_aug": use_context_aug,
        "context_aug_stage": context_aug_stage,
        "context_aug_length_params": context_aug_length_params,
        "context_aug_max_templates": context_aug_max_templates,
        "append_eos_to_target": append_eos_to_target,
        "eos_token": eos_token,
        "eos_appended_count": eos_appended_count,
        "training_requests_path": requests_path,
    }

    manifest_path = os.path.join(save_dir, "training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved training requests to {requests_path}")
    print(f"Saved training manifest to {manifest_path}")


# -----------------------------------------------------------------------------
# KLOD objective:
# one-sided target-vs-rest logit odds hinge + rewrite-prefix KL + non-target KL
# -----------------------------------------------------------------------------


def build_full_batch_inputs(
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    prompt_inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
    full_texts = [p + t for p, t in zip(prompts, targets)]
    full_inputs = tok(full_texts, return_tensors="pt", padding=True).to(device)

    prompt_token_counts = prompt_inputs["attention_mask"].sum(dim=1)
    full_token_counts = full_inputs["attention_mask"].sum(dim=1)
    full_seq_len = full_inputs["input_ids"].size(1)
    full_pad_counts = full_seq_len - full_token_counts
    prompt_end_positions = full_pad_counts + prompt_token_counts

    position_ids = torch.arange(full_seq_len, device=device).unsqueeze(0)
    label_mask = position_ids >= prompt_end_positions.unsqueeze(1)
    label_mask = label_mask & full_inputs["attention_mask"].bool()

    if bool((label_mask.sum(dim=1) == 0).any().item()):
        raise RuntimeError("A batch contains samples with zero target tokens after tokenization.")

    return full_inputs, label_mask


def compute_rewrite_reference_logits(
    model: AutoModelForCausalLM,
    inputs: Dict[str, torch.Tensor],
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Dict[str, torch.Tensor],
) -> torch.Tensor:
    with WeightSwapContext(weights_to_update, ref_snapshot):
        with torch.no_grad():
            ref_logits = model(**inputs).logits.float()
    return ref_logits


def compute_non_target_kl_loss(
    shift_logits: torch.Tensor,
    ref_shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    target_mask: torch.Tensor,
    zero_loss: torch.Tensor,
) -> torch.Tensor:
    if not bool(target_mask.any().item()):
        return zero_loss

    target_position_logits = shift_logits[target_mask]
    ref_target_position_logits = ref_shift_logits[target_mask].to(
        device=target_position_logits.device,
        dtype=target_position_logits.dtype,
    )
    target_position_ids = shift_labels[target_mask].unsqueeze(-1)
    neg_inf = torch.finfo(target_position_logits.dtype).min

    cur_non_target_logits = target_position_logits.clone()
    ref_non_target_logits = ref_target_position_logits.clone()
    cur_non_target_logits.scatter_(dim=-1, index=target_position_ids, value=neg_inf)
    ref_non_target_logits.scatter_(dim=-1, index=target_position_ids, value=neg_inf)

    cur_non_target_log_probs = F.log_softmax(cur_non_target_logits, dim=-1)
    ref_non_target_log_probs = F.log_softmax(ref_non_target_logits, dim=-1)
    return F.kl_div(
        cur_non_target_log_probs,
        ref_non_target_log_probs,
        log_target=True,
        reduction="batchmean",
    )

def compute_klod_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
    *,
    target_alpha: float,
    rewrite_kl_lambda: float,
    non_target_kl_lambda: float,
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    full_inputs, label_mask = build_full_batch_inputs(tok, prompts, targets, device)
    target_mask = label_mask[:, 1:]

    ref_logits = None
    needs_reference_logits = rewrite_kl_lambda > 0.0 or non_target_kl_lambda > 0.0
    if needs_reference_logits:
        if ref_snapshot is None:
            raise ValueError(
                "ref_snapshot must be provided when rewrite_kl_lambda > 0 "
                "or non_target_kl_lambda > 0."
            )
        ref_logits = compute_rewrite_reference_logits(
            model=model,
            inputs=full_inputs,
            weights_to_update=weights_to_update,
            ref_snapshot=ref_snapshot,
        )

    logits = model(**full_inputs).logits  # [B, T, V]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = full_inputs["input_ids"][:, 1:].contiguous()
    shift_log_p_theta = F.log_softmax(shift_logits, dim=-1)

    alpha = float(target_alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"target_alpha must be in [0,1], got {alpha}")
    alpha_eps = 1e-6
    alpha = min(max(alpha, alpha_eps), 1.0 - alpha_eps)
    beta = torch.logit(torch.tensor(alpha, device=device, dtype=shift_logits.dtype))

    target_logits = shift_logits.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    all_logsumexp = torch.logsumexp(shift_logits, dim=-1)
    target_mass_ratio = torch.exp(target_logits - all_logsumexp).clamp(max=1.0 - 1e-7)
    rest_logsumexp = all_logsumexp + torch.log1p(-target_mass_ratio)
    logit_odds = target_logits - rest_logsumexp

    target_probs = torch.sigmoid(logit_odds.detach())
    num_target_positions = int(target_mask.sum().item())
    zero_loss = shift_logits.sum() * 0.0
    avg_target_prob = target_probs[target_mask].mean().detach()

    target_logit_odds = logit_odds[target_mask]
    target_logit_deficit = torch.relu(beta - target_logit_odds)
    logit_edit_loss = target_logit_deficit.mean()

    prefix_kl_loss = zero_loss
    if rewrite_kl_lambda > 0.0:
        prefix_mask = (
            full_inputs["attention_mask"][:, :-1].bool()
            & full_inputs["attention_mask"][:, 1:].bool()
            & ~target_mask
        )

        if bool(prefix_mask.any().item()):
            cur_log_probs = shift_log_p_theta
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            kl_per_pos = F.kl_div(
                cur_log_probs,
                ref_log_probs,
                log_target=True,
                reduction="none",
            ).sum(dim=-1)
            prefix_kl_loss = (kl_per_pos * prefix_mask).sum() / prefix_mask.sum()
            # prefix_kl_loss = (kl_per_pos * prefix_mask.float()).sum() / prefix_mask.float().sum()

    non_target_kl_loss = zero_loss
    if non_target_kl_lambda > 0.0:
        non_target_kl_loss = compute_non_target_kl_loss(
            shift_logits=shift_logits,
            ref_shift_logits=ref_logits[:, :-1, :],
            shift_labels=shift_labels,
            target_mask=target_mask,
            zero_loss=zero_loss,
        )

    total_loss = (
        logit_edit_loss
        + float(rewrite_kl_lambda) * prefix_kl_loss
        + float(non_target_kl_lambda) * non_target_kl_loss
    )

    avg_logit_deficit = target_logit_deficit.mean().detach()
    avg_rest_logsumexp = rest_logsumexp[target_mask].mean().detach()
    num_positive_deficit_positions = int((target_logit_deficit.detach() > 0.0).sum().item())

    return {
        "loss": total_loss,
        "logit_edit_loss": logit_edit_loss.detach(),
        "prefix_kl_loss": prefix_kl_loss.detach(),
        "non_target_kl_loss": non_target_kl_loss.detach(),
        "avg_target_prob": avg_target_prob,
        "target_logit_odds_beta": beta.detach(),
        "avg_logit_deficit": avg_logit_deficit,
        "num_positive_deficit_positions": num_positive_deficit_positions,
        "rest_logsumexp": avg_rest_logsumexp,
        "num_target_positions": num_target_positions,
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def execute_training(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    config,
    *,
    rewrite_kl_lambda: float,
    non_target_kl_lambda: float,
    use_context_aug: bool = False,
    context_aug_stage: str = "klod",
    context_aug_length_params: Optional[List[tuple]] = None,
    context_aug_max_templates: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> AutoModelForCausalLM:
    device = torch.device(f"cuda:{config.device}")

    if tok.padding_side != "left":
        tok.padding_side = "left"

    base_train_requests = normalize_training_requests(requests, tok)
    train_requests = base_train_requests

    if use_context_aug:
        context_aug_stage = normalize_context_aug_stage(context_aug_stage)
        if context_aug_length_params is None:
            context_aug_length_params = [(10, 5)]

        context_templates = []
        selected_context_templates = []
        if context_aug_stage in {"klod", "all"}:
            context_templates = get_context_templates(model, tok, context_aug_length_params)
            selected_context_templates = (
                context_templates[:context_aug_max_templates]
                if context_aug_max_templates
                else context_templates
            )
            train_requests = augment_requests_with_context_templates(
                base_train_requests,
                context_templates,
                max_templates=context_aug_max_templates,
            )

            if save_dir:
                save_context_aug_artifacts(
                    save_dir,
                    train_requests,
                    selected_context_templates,
                    context_aug_stage=context_aug_stage,
                )
        else:
            print(
                f"[TRAIN] Context augmentation requested for stage={context_aug_stage}, "
                "but this script only has a KLOD training stage. Using base requests."
            )

        print(
            f"[TRAIN] Context augmentation enabled: stage={context_aug_stage}, "
            f"base_requests={len(base_train_requests)}, "
            f"klod_train_requests={len(train_requests)}, "
            f"templates_used={len(selected_context_templates)}"
        )

    preview_requests = base_train_requests if train_requests is not base_train_requests else train_requests
    for i, request in enumerate(preview_requests[:3]):
        print(f"[TRAIN] Request: [{request['prompt']}] -> [{request['target_new']}]")
    if len(preview_requests) > 3:
        print(f"[TRAIN] ... and {len(preview_requests)-3} more requests.")
    if train_requests is not base_train_requests:
        for request in train_requests[:3]:
            print(f"[TRAIN][KLOD-AUG] Request: [{request['prompt']}] -> [{request['target_new']}]")
        if len(train_requests) > 3:
            print(f"[TRAIN][KLOD-AUG] ... and {len(train_requests)-3} more KLOD requests.")

    layers_to_edit = get_layers_to_edit(config)
    rewrite_module_template = get_rewrite_module_template(config)

    weights_to_update = {}
    for n, p in model.named_parameters():
        for layer in layers_to_edit:
            if f"{rewrite_module_template.format(layer)}.weight" == n:
                weights_to_update[n] = p

    if not weights_to_update:
        raise ValueError(
            f"No weights found matching template={rewrite_module_template} layers={layers_to_edit}"
        )

    print(f"[TRAIN] Updating {len(weights_to_update)} params: {list(weights_to_update.keys())}")

    opt = torch.optim.Adam(
        weights_to_update.values(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in weights_to_update.values():
        p.requires_grad = True

    loss_meter = AverageMeter()
    early_stop_loss = float(getattr(config, "early_stop_loss", 55e-3) or 0.0)
    target_alpha = float(getattr(config, "target_alpha", DEFAULT_TARGET_ALPHA))
    ref_snapshot = (
        clone_weight_snapshot(weights_to_update)
        if rewrite_kl_lambda > 0.0 or non_target_kl_lambda > 0.0
        else None
    )

    for epoch in range(config.num_steps):
        print(f"\n=== Epoch {epoch} ===")
        loss_meter.reset()
        batch_count = 0
        optimizer_step_count = 0

        model.train()
        random.shuffle(train_requests)

        print(f"[TRAIN] Phase: KLOD objective (epoch {epoch})")

        for batch_reqs in chunks(train_requests, config.batch_size):
            txt_batch = [r["prompt"] for r in batch_reqs]
            tgt_batch = [r["target_new"] for r in batch_reqs]

            opt.zero_grad()
            batch_label = batch_count + 1

            try:
                out = compute_klod_loss(
                    model=model,
                    tok=tok,
                    prompts=txt_batch,
                    targets=tgt_batch,
                    device=device,
                    target_alpha=target_alpha,
                    rewrite_kl_lambda=rewrite_kl_lambda,
                    non_target_kl_lambda=non_target_kl_lambda,
                    weights_to_update=weights_to_update,
                    ref_snapshot=ref_snapshot,
                )

                batch_count += 1

                final_loss = out["loss"]
                print(
                    f"[KLOD] batch {batch_label} | "
                    f"loss={final_loss.item():.4f} "
                    f"logit_edit={float(out['logit_edit_loss']):.4f} "
                    f"prefix_kl={float(out['prefix_kl_loss']):.4f} "
                    f"non_target_kl={float(out['non_target_kl_loss']):.4f} "
                    f"avg_target_prob={float(out['avg_target_prob']):.6f} "
                    f"target_logit_odds_beta={float(out['target_logit_odds_beta']):.6f} "
                    f"rest_logsumexp={float(out['rest_logsumexp']):.4f} "
                    f"avg_logit_deficit={float(out['avg_logit_deficit']):.6f} "
                    f"deficit_positions={out['num_positive_deficit_positions']}/{out['num_target_positions']} "
                    f"target_positions={out['num_target_positions']}"
                )

            except Exception as exc:
                print(f"[TRAIN][warn] Skipping batch {batch_label} due to loss construction failure: {exc}")
                continue

            if not bool(torch.isfinite(final_loss).item()):
                print(f"[TRAIN][warn] Skipping batch {batch_label}: non-finite loss ({final_loss.item()}).")
                continue

            loss_meter.update(final_loss.item(), n=len(txt_batch))

            final_loss.backward()

            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
                for p in weights_to_update.values()
            ):
                print(f"[TRAIN][warn] Skipping optimizer step for batch {batch_label}: non-finite gradient.")
                opt.zero_grad(set_to_none=True)
                continue

            opt.step()
            optimizer_step_count += 1

        if batch_count == 0 or loss_meter.count == 0:
            print(f"Epoch {epoch} avg loss: n/a")
            print(f"Epoch {epoch} processed batches: {batch_count}")
            print(f"Epoch {epoch} optimizer steps: {optimizer_step_count}")
            continue

        print(f"Epoch {epoch} avg loss: {loss_meter.avg:.4f}")
        print(f"Epoch {epoch} processed batches: {batch_count}")
        print(f"Epoch {epoch} optimizer steps: {optimizer_step_count}")

        if early_stop_loss > 0 and loss_meter.avg < early_stop_loss:
            print(f"Converged under early_stop_loss={early_stop_loss:.4f}. Stopping early.")
            break

    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "KLOD: one-sided target-vs-rest logit odds hinge "
            "+ rewrite-prefix KL + optional non-target KL"
        )
    )
    parser.add_argument(
        "hparams_path",
        nargs="?",
        help="Path to KLOD yaml config. Equivalent to --klod_config_path.",
    )
    parser.add_argument(
        "--klod_config_path",
        "--hparams",
        "--config_path",
        dest="klod_config_path",
        type=str,
        default=None,
        help="Path to KLOD yaml config",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Override data path in config")
    parser.add_argument("--save_model_dir", type=str, default=None, help="Override save dir in config")
    parser.add_argument("--easyedit_path", type=str, default=None, help="Path to EasyEdit root")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or 'auto'; overrides config device")
    parser.add_argument("--rewrite_kl_lambda", type=float, default=None, help="Weight for rewrite-prompt prefix next-token KL during KLOD")
    parser.add_argument("--non_target_kl_lambda", type=float, default=None, help="Weight for target-position non-target distribution KL against the initial model")
    parser.add_argument("--num_steps", "--epochs", type=int, default=None, help="Override training epoch count in config.num_steps")
    parser.add_argument("--sample_size", type=int, default=None, help="Randomly sample this many requests; <= 0 uses all data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling and training order")
    parser.add_argument("--early_stop_loss", type=float, default=None, help="Override early-stop loss threshold; <= 0 disables early stopping")
    parser.add_argument("--target_alpha", type=float, default=None, help="Desired target-token probability for the target-vs-rest logit odds objective")
    parser.add_argument("--use_context_aug", type=int, choices=[0, 1], default=None, help="Use MEMIT/AlphaEdit-style generated context-template prompt augmentation")
    parser.add_argument("--context_aug_stage", type=str, default=None, help="Apply context augmentation to klod or all; locft is accepted for compatibility but has no effect here")
    parser.add_argument("--context_aug_length_params", type=str, default=None, help="Generation spec for context templates, e.g. '10:5' or '10:5,15:10'")
    parser.add_argument("--context_aug_max_templates", type=int, default=None, help="Maximum templates per request, counting the original '{}' template; <=0 means no limit")
    parser.add_argument(
        "--save_tag",
        type=str,
        default=None,
        help="Default method subdirectory under outputs/Models when save_model_dir is not provided",
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset label used in default save directory naming")
    args = parser.parse_args()

    config_path = args.klod_config_path or args.hparams_path
    if config_path is None:
        parser.error("--klod_config_path/--hparams or positional hparams_path is required")
    config_path = resolve_existing_path(config_path, SCRIPT_DIR)

    print(f"Loading KLOD config from {config_path}")
    config = KLODHyperParams.from_hparams(config_path)
    if hasattr(config, "alg_name"):
        config.alg_name = "KLOD"

    config_dir = os.path.dirname(os.path.abspath(config_path))
    args.easyedit_path = get_arg_or_config(args.easyedit_path, config, "easyedit_path", None)
    if args.easyedit_path:
        args.easyedit_path = resolve_existing_path(args.easyedit_path, SCRIPT_DIR, config_dir)
        sys.path.insert(0, args.easyedit_path)

    args.seed = int(get_arg_or_config(args.seed, config, "seed", 42))
    set_seed(args.seed)

    config.data_path = get_arg_or_config(args.data_path, config, "data_path", None)
    if config.data_path is None:
        parser.error("data_path must be provided in the hparams file or with --data_path")
    config.data_path = resolve_existing_path(config.data_path, SCRIPT_DIR, config_dir)
    if not os.path.exists(config.data_path):
        parser.error(f"data_path not found: {config.data_path}")

    args.rewrite_kl_lambda = get_arg_or_config(
        args.rewrite_kl_lambda,
        config,
        "rewrite_kl_lambda",
        getattr(config, "kl_lambda", 0.0),
    )
    if args.rewrite_kl_lambda is None:
        args.rewrite_kl_lambda = getattr(config, "kl_lambda", 0.0)
    args.rewrite_kl_lambda = float(args.rewrite_kl_lambda)
    args.non_target_kl_lambda = float(
        get_arg_or_config(args.non_target_kl_lambda, config, "non_target_kl_lambda", 0.0) or 0.0
    )
    args.sample_size = get_arg_or_config(args.sample_size, config, "sample_size", None)
    if args.sample_size is not None:
        args.sample_size = int(args.sample_size)
    args.use_context_aug = normalize_bool_int(
        get_arg_or_config(args.use_context_aug, config, "use_context_aug", 0),
        name="use_context_aug",
    )
    args.context_aug_stage = normalize_context_aug_stage(
        get_arg_or_config(args.context_aug_stage, config, "context_aug_stage", "klod")
    )
    args.context_aug_length_params = str(
        get_arg_or_config(args.context_aug_length_params, config, "context_aug_length_params", "10:5")
    )
    args.context_aug_max_templates = get_arg_or_config(
        args.context_aug_max_templates,
        config,
        "context_aug_max_templates",
        None,
    )
    if args.context_aug_max_templates is not None:
        args.context_aug_max_templates = int(args.context_aug_max_templates)
        if args.context_aug_max_templates <= 0:
            args.context_aug_max_templates = None
    args.save_model_dir = get_arg_or_config(args.save_model_dir, config, "save_model_dir", None)
    if args.save_model_dir:
        args.save_model_dir = resolve_output_path(args.save_model_dir, SCRIPT_DIR)
    args.save_tag = str(get_arg_or_config(args.save_tag, config, "save_tag", "KLOD") or "KLOD")
    args.dataset_name = get_arg_or_config(args.dataset_name, config, "dataset_name", None)
    context_aug_length_params = parse_context_aug_length_params(args.context_aug_length_params)

    if args.num_steps is not None:
        if args.num_steps <= 0:
            parser.error("--num_steps/--epochs must be a positive integer")
        config.num_steps = args.num_steps
    apply_hparam_overrides(
        config,
        args.target_alpha,
        args.early_stop_loss,
    )
    if getattr(config, "target_alpha", None) is None:
        config.target_alpha = DEFAULT_TARGET_ALPHA

    use_model_parallel = bool(getattr(config, "model_parallel", False))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script currently requires a CUDA device.")
    if use_model_parallel:
        if args.device not in (None, "auto"):
            print(
                f"[load] Ignoring --device={args.device} because config has model_parallel=True. "
                "Using device_map='auto' instead."
            )
        resolved_device = int(config.device)
        print("Config has model_parallel=True, so the model will be loaded with device_map='auto'.")
    else:
        resolved_device = resolve_device_index(args.device, config.device)
        config.device = resolved_device
        print(f"Using cuda:{resolved_device}")

    print(f"Loading data from {config.data_path}")
    with open(config.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    requests = []
    for idx, d in enumerate(data):
        request = build_training_request(d, idx)
        if request is not None:
            requests.append(request)
        else:
            print(f"Skipping invalid data entry: {list(d.keys())}")

    num_requests_before_sampling = len(requests)
    print(f"Loaded {num_requests_before_sampling} valid editing requests before sampling.")

    if args.sample_size is not None and args.sample_size > 0:
        if args.sample_size < len(requests):
            print(
                f"Randomly sampling {args.sample_size} requests from {len(requests)} "
                f"total with seed={args.seed}."
            )
            requests = random.sample(requests, args.sample_size)
        else:
            print(
                f"Requested sample_size={args.sample_size}, but only {len(requests)} "
                "valid requests are available. Using all requests."
            )

    print(f"Using {len(requests)} editing requests for training.")
    print(
        "[config] "
        f"num_steps={getattr(config, 'num_steps', None)} "
        f"rewrite_kl_lambda={args.rewrite_kl_lambda} "
        f"non_target_kl_lambda={args.non_target_kl_lambda} "
        f"early_stop_loss={getattr(config, 'early_stop_loss', None)} "
        f"target_alpha={getattr(config, 'target_alpha', None)} "
        f"edit_loss=one_sided_logit_odds_hinge "
        f"use_context_aug={bool(args.use_context_aug)} "
        f"context_aug_stage={args.context_aug_stage} "
        f"context_aug_length_params={args.context_aug_length_params} "
        f"context_aug_max_templates={args.context_aug_max_templates}"
    )

    model_name = get_model_name(config)
    model_dtype = get_model_load_dtype(config)
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {"low_cpu_mem_usage": True}
    if model_dtype is not None:
        load_kwargs["torch_dtype"] = model_dtype

    if use_model_parallel:
        print(
            "Loading model with device_map='auto'"
            + (f" and dtype={model_dtype}" if model_dtype is not None else "")
            + "..."
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **load_kwargs)
        resolved_device = get_primary_model_device_index(model, fallback_device=resolved_device)
        config.device = resolved_device
        print(f"Primary execution device resolved to cuda:{resolved_device}")
    else:
        compute_device = torch.device(f"cuda:{resolved_device}")
        try:
            print(
                f"Loading model directly on cuda:{resolved_device}"
                + (f" with dtype={model_dtype}" if model_dtype is not None else "")
                + "..."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": resolved_device},
                **load_kwargs,
            )
        except Exception as exc:
            print(f"[load] Direct GPU loading failed: {exc}")
            print("[load] Falling back to standard load followed by model.to(cuda).")
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model.to(compute_device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    eos_token = resolve_tokenizer_eos_token(tokenizer)
    eos_appended_count = count_targets_needing_eos(requests, tokenizer)
    requests = normalize_training_requests(requests, tokenizer)
    print(
        "[data] append_eos_to_target=True "
        f"eos_token={eos_token!r} appended={eos_appended_count}"
    )

    if args.save_model_dir:
        config.save_model_dir = args.save_model_dir
    else:
        data_stem = get_data_stem(config.data_path)
        dataset_label = args.dataset_name or data_stem
        dataset_name = dataset_label if dataset_label == data_stem else f"{dataset_label}_{data_stem}"
        model_slug = model_name.split("/")[-1]
        target_alpha_tag = format_path_value(getattr(config, "target_alpha", None))
        early_stop_loss_tag = format_path_value(getattr(config, "early_stop_loss", None))
        context_aug_tag = ""
        if bool(args.use_context_aug):
            context_aug_tag = (
                f"_ctxaug1"
                f"_ctxstage{args.context_aug_stage}"
                f"_ctxtmpl{format_path_value(args.context_aug_max_templates or 'all')}"
            )
        run_name = (
            f"{model_slug}_{dataset_name}"
            f"_rewritekl{format_path_value(args.rewrite_kl_lambda)}"
            f"_ntkl{format_path_value(args.non_target_kl_lambda)}"
            f"_early_stop_loss{early_stop_loss_tag}"
            f"_target_alpha{target_alpha_tag}"
            f"{context_aug_tag}"
        )
        config.save_model_dir = os.path.join(SCRIPT_DIR, "outputs", "Models", args.save_tag, run_name)

    save_training_artifacts(
        save_dir=config.save_model_dir,
        requests=requests,
        data_path=config.data_path,
        num_requests_before_sampling=num_requests_before_sampling,
        sample_size_requested=args.sample_size,
        seed=args.seed,
        num_steps=config.num_steps,
        rewrite_kl_lambda=args.rewrite_kl_lambda,
        non_target_kl_lambda=args.non_target_kl_lambda,
        early_stop_loss=getattr(config, "early_stop_loss", None),
        target_alpha=getattr(config, "target_alpha", None),
        use_context_aug=bool(args.use_context_aug),
        context_aug_stage=args.context_aug_stage,
        context_aug_length_params=args.context_aug_length_params,
        context_aug_max_templates=args.context_aug_max_templates,
        append_eos_to_target=True,
        eos_token=eos_token,
        eos_appended_count=eos_appended_count,
    )

    stage_label = "KLOD one-sided logit-odds hinge + rewrite-prefix KL + non-target KL"
    print_time(f"Begin {stage_label}")
    model = execute_training(
        model=model,
        tok=tokenizer,
        requests=requests,
        config=config,
        rewrite_kl_lambda=args.rewrite_kl_lambda,
        non_target_kl_lambda=args.non_target_kl_lambda,
        use_context_aug=bool(args.use_context_aug),
        context_aug_stage=args.context_aug_stage,
        context_aug_length_params=context_aug_length_params,
        context_aug_max_templates=args.context_aug_max_templates,
        save_dir=config.save_model_dir,
    )
    print_time(f"End {stage_label}")

    save_path = config.save_model_dir
    print(f"Saving model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")
