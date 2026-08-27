import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT_DIR = PROJECT_ROOT
LOCFT_BF_MODEL_DIR = os.path.join(SCRIPT_DIR, "EasyEdit", "easyeditor", "models", "locft-bf")
if LOCFT_BF_MODEL_DIR not in sys.path:
    sys.path.insert(0, LOCFT_BF_MODEL_DIR)

from EasyEdit.easyeditor.models.locft_bf.locft_bf_hparams import LocFTBFHyperParams


DEFAULT_OVERTONE_LAMBDA = 0.1
DEFAULT_OVERTONE_EPSILON = 0.01
DEFAULT_OVERTONE_NSIGMA = 0.5


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


def compute_reference_logits(
    model: AutoModelForCausalLM,
    model_inputs: Dict[str, torch.Tensor],
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Dict[str, torch.Tensor],
) -> torch.Tensor:
    with WeightSwapContext(weights_to_update, ref_snapshot):
        with torch.no_grad():
            ref_logits = model(**model_inputs).logits.float()
    return ref_logits


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


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)


def start_timer() -> float:
    synchronize_cuda()
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    synchronize_cuda()
    return time.perf_counter() - start_time


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes:02d}m {secs:06.3f}s"
    if minutes:
        return f"{minutes}m {secs:06.3f}s"
    return f"{secs:.3f}s"


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
    break_loss: Optional[float],
    break_prob: Optional[float],
    norm_factor: Optional[float],
    target_alpha: Optional[float],
) -> None:
    if break_loss is not None and hasattr(config, "break_loss"):
        config.break_loss = break_loss
    if break_prob is not None and hasattr(config, "break_prob"):
        config.break_prob = break_prob
    if norm_factor is not None and hasattr(config, "clamp_norm_factor"):
        config.clamp_norm_factor = norm_factor
    if target_alpha is not None and hasattr(config, "target_alpha"):
        config.target_alpha = target_alpha


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


def normalize_ft_requests(requests: List[Dict], tok: AutoTokenizer) -> List[Dict]:
    out = deepcopy(requests)
    for request in out:
        request["target_new"] = normalize_target_text(
            request["target_new"],
            tok=tok,
            append_eos=True,
        )
    return out


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
    objective: str,
    overtone_lambda: float,
    overtone_epsilon: float,
    overtone_nsigma: float,
    target_kl_lambda: float,
    target_kl_direction: str,
    break_loss: Optional[float],
    break_prob: Optional[float],
    norm_factor: Optional[float],
    target_alpha: Optional[float],
    append_eos_to_target: bool = True,
    eos_token: Optional[str] = None,
    eos_appended_count: int = 0,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    requests_path = os.path.join(save_dir, "training_requests.json")
    with open(requests_path, "w", encoding="utf-8") as f:
        json.dump(requests, f, ensure_ascii=False, indent=2)

    manifest = {
        "data_path": data_path,
        "data_stem": get_data_stem(data_path),
        "num_requests_before_sampling": num_requests_before_sampling,
        "num_requests_used": len(requests),
        "sample_size_requested": sample_size_requested,
        "seed": seed,
        "objective": objective,
        "overtone_lambda": overtone_lambda,
        "overtone_epsilon": overtone_epsilon,
        "overtone_nsigma": overtone_nsigma,
        "target_kl_lambda": target_kl_lambda,
        "target_kl_direction": target_kl_direction,
        "break_loss": break_loss,
        "break_prob": break_prob,
        "norm_factor": norm_factor,
        "target_alpha": target_alpha,
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


def save_timing_summary(save_dir: str, timing_summary: Dict) -> None:
    timing_path = os.path.join(save_dir, "timing_summary.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved timing summary to {timing_path}")


# -----------------------------------------------------------------------------
# Sequence helpers
# -----------------------------------------------------------------------------


def build_full_batch_inputs(
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    full_texts = [p + t for p, t in zip(prompts, targets)]
    return tok(full_texts, return_tensors="pt", padding=True).to(device)


def compute_prompt_target_lengths(
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
    full_inputs = build_full_batch_inputs(tok, prompts, targets, device)

    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
    full_lens = full_inputs["attention_mask"].sum(dim=1)
    target_lens = full_lens - prompt_lens
    if bool((target_lens <= 0).any().item()):
        raise RuntimeError("A batch contains samples with zero target tokens.")
    return prompt_lens, full_lens, target_lens


def build_shift_label_mask(
    seq_len: int,
    prompt_lens: torch.Tensor,
    full_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a [B, seq_len-1] mask over shift_labels / shift_logits positions.

    With left padding, if the non-pad span starts at `seq_len - full_len[b]`, then the
    first target token is located at input_ids position:
        start_target = seq_len - full_len[b] + prompt_len[b]
    and corresponds to shift index `start_target - 1`.
    """
    batch_size = prompt_lens.shape[0]
    mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool, device=prompt_lens.device)
    for b in range(batch_size):
        full_len = int(full_lens[b].item())
        prompt_len = int(prompt_lens[b].item())
        start_target = seq_len - full_len + prompt_len
        start_shift = max(start_target - 1, 0)
        end_shift = seq_len - 1
        if start_shift < end_shift:
            mask[b, start_shift:end_shift] = True
    return mask


def compute_target_reference_kl_loss(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    direction: str,
) -> torch.Tensor:
    """
    KL over next-token distributions at target-token prediction positions.

    `target_mask` is aligned to shifted logits, i.e. [B, T-1].
    direction='current_to_ref' matches the ROME/AlphaEdit-style argument order:
        KL(p_current || p_reference)
    direction='ref_to_current' matches the FT-M prefix-KL helper:
        KL(p_reference || p_current)
    """
    zero_loss = current_logits.sum() * 0.0
    if not bool(target_mask.any().item()):
        return zero_loss

    current_log_probs = F.log_softmax(current_logits[:, :-1, :][target_mask].float(), dim=-1)
    reference_log_probs = F.log_softmax(reference_logits[:, :-1, :][target_mask].float(), dim=-1)

    if direction == "current_to_ref":
        return F.kl_div(
            reference_log_probs,
            current_log_probs,
            log_target=True,
            reduction="batchmean",
        )
    elif direction == "ref_to_current":
        return F.kl_div(
            current_log_probs,
            reference_log_probs,
            log_target=True,
            reduction="batchmean",
        )
    else:
        raise ValueError(
            f"target_kl_direction must be 'current_to_ref' or 'ref_to_current', got {direction}"
        )


# -----------------------------------------------------------------------------
# OVERTONE objective
# -----------------------------------------------------------------------------


def top_nsigma_filtered_distribution(
    target_logits: torch.Tensor,
    nsigma: float,
) -> torch.Tensor:
    """
    target_logits: [N, V]
    returns filtered probabilities [N, V]
    """
    if nsigma < 0:
        raise ValueError(f"overtone_nsigma must be >= 0, got {nsigma}")

    row_max = target_logits.max(dim=-1, keepdim=True).values
    row_std = target_logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    threshold = row_max - float(nsigma) * row_std
    filtered_logits = target_logits.masked_fill(target_logits <= threshold, float("-inf"))

    # Always keep argmax token to avoid all -inf rows.
    argmax_idx = target_logits.argmax(dim=-1, keepdim=True)
    filtered_logits.scatter_(1, argmax_idx, target_logits.gather(1, argmax_idx))
    return F.softmax(filtered_logits, dim=-1)


def compute_overtone_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
    *,
    overtone_lambda: float,
    overtone_epsilon: float,
    overtone_nsigma: float,
    target_kl_lambda: float,
    target_kl_direction: str,
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not (0.0 <= overtone_lambda <= 1.0):
        raise ValueError(f"overtone_lambda must be in [0,1], got {overtone_lambda}")
    if overtone_epsilon < 0.0:
        raise ValueError(f"overtone_epsilon must be >= 0, got {overtone_epsilon}")

    full_inputs = build_full_batch_inputs(tok, prompts, targets, device)
    prompt_lens, full_lens, target_lens = compute_prompt_target_lengths(tok, prompts, targets, device)
    label_mask = build_shift_label_mask(full_inputs["input_ids"].shape[1], prompt_lens, full_lens)

    ref_logits = None
    if target_kl_lambda > 0.0:
        if ref_snapshot is None:
            raise ValueError("ref_snapshot must be provided when target_kl_lambda > 0.")
        ref_logits = compute_reference_logits(
            model=model,
            model_inputs=full_inputs,
            weights_to_update=weights_to_update,
            ref_snapshot=ref_snapshot,
        )

    logits = model(**full_inputs).logits.float()  # [B, T, V]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full_inputs["input_ids"][:, 1:].contiguous()

    target_kl_loss = logits.sum() * 0.0
    if target_kl_lambda > 0.0:
        target_kl_loss = compute_target_reference_kl_loss(
            current_logits=logits,
            reference_logits=ref_logits,
            target_mask=label_mask,
            direction=target_kl_direction,
        )

    flat_logits = shift_logits[label_mask]  # [N, V]
    flat_labels = shift_labels[label_mask]  # [N]
    if flat_logits.numel() == 0:
        zero_loss = logits.sum() * 0.0
        return {
            "loss": zero_loss,
            "base_loss": zero_loss.detach(),
            "target_kl_loss": zero_loss.detach(),
            "avg_token_kl": zero_loss.detach(),
            "avg_target_prob": zero_loss.detach(),
            "avg_filtered_target_prob": zero_loss.detach(),
            "num_target_tokens": 0,
            "num_onehot_fallback_tokens": 0,
            "num_clipped_tokens": 0,
        }

    log_p_theta = F.log_softmax(flat_logits, dim=-1)
    p_theta = log_p_theta.exp()
    row_idx = torch.arange(flat_logits.shape[0], device=device)
    target_probs = p_theta[row_idx, flat_labels]

    pi_flt = top_nsigma_filtered_distribution(flat_logits.detach(), overtone_nsigma)
    one_hot = F.one_hot(flat_labels, num_classes=flat_logits.shape[-1]).to(dtype=p_theta.dtype)
    pi_tar_candidate = float(overtone_lambda) * one_hot + (1.0 - float(overtone_lambda)) * pi_flt
    candidate_argmax = pi_tar_candidate.argmax(dim=-1)
    fallback_mask = candidate_argmax != flat_labels
    pi_tar = torch.where(fallback_mask.unsqueeze(-1), one_hot, pi_tar_candidate)

    token_kl = F.kl_div(
        log_p_theta,
        pi_tar.clamp_min(1e-12),
        reduction="none",
        log_target=False,
    ).sum(dim=-1)

    clipped_mask = token_kl <= float(overtone_epsilon)
    clipped_token_loss = torch.clamp_min(token_kl, float(overtone_epsilon))
    base_loss = clipped_token_loss.mean()
    loss = base_loss + float(target_kl_lambda) * target_kl_loss

    avg_filtered_target_prob = pi_flt[row_idx, flat_labels].mean().detach()
    avg_token_kl = token_kl.mean().detach()

    return {
        "loss": loss,
        "base_loss": base_loss.detach(),
        "target_kl_loss": target_kl_loss.detach(),
        "avg_token_kl": avg_token_kl,
        "avg_target_prob": target_probs.mean().detach(),
        "avg_filtered_target_prob": avg_filtered_target_prob,
        "num_target_tokens": int(flat_labels.shape[0]),
        "num_onehot_fallback_tokens": int(fallback_mask.sum().item()),
        "num_clipped_tokens": int(clipped_mask.sum().item()),
        "avg_target_len": target_lens.float().mean().detach(),
    }


# -----------------------------------------------------------------------------
# Vanilla CE objective
# -----------------------------------------------------------------------------


def compute_standard_ft_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
    *,
    target_kl_lambda: float,
    target_kl_direction: str,
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    full_inputs = build_full_batch_inputs(tok, prompts, targets, device)
    prompt_lens, full_lens, _ = compute_prompt_target_lengths(tok, prompts, targets, device)
    label_mask = build_shift_label_mask(full_inputs["input_ids"].shape[1], prompt_lens, full_lens)

    ref_logits = None
    if target_kl_lambda > 0.0:
        if ref_snapshot is None:
            raise ValueError("ref_snapshot must be provided when target_kl_lambda > 0.")
        ref_logits = compute_reference_logits(
            model=model,
            model_inputs=full_inputs,
            weights_to_update=weights_to_update,
            ref_snapshot=ref_snapshot,
        )

    logits = model(**full_inputs).logits.float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full_inputs["input_ids"][:, 1:].contiguous()

    target_kl_loss = logits.sum() * 0.0
    if target_kl_lambda > 0.0:
        target_kl_loss = compute_target_reference_kl_loss(
            current_logits=logits,
            reference_logits=ref_logits,
            target_mask=label_mask,
            direction=target_kl_direction,
        )

    loss_fct = CrossEntropyLoss(reduction="none")
    flat_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.shape)

    target_token_counts = label_mask.sum(dim=1)
    if bool((target_token_counts == 0).any().item()):
        raise RuntimeError("A batch contains samples with zero target tokens under the current label mask.")

    masked_loss = (flat_loss * label_mask.float()).sum(dim=1) / target_token_counts.float()
    ce_loss = masked_loss.mean()
    return {
        "loss": ce_loss + float(target_kl_lambda) * target_kl_loss,
        "avg_ce": ce_loss.detach(),
        "target_kl_loss": target_kl_loss.detach(),
        "num_target_tokens": int(label_mask.sum().item()),
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
    objective: str,
    overtone_lambda: float,
    overtone_epsilon: float,
    overtone_nsigma: float,
    target_kl_lambda: float,
    target_kl_direction: str,
) -> AutoModelForCausalLM:
    device = torch.device(f"cuda:{config.device}")

    if tok.padding_side != "left":
        tok.padding_side = "left"

    train_requests = normalize_ft_requests(requests, tok)
    for request in train_requests[:3]:
        print(f"[TRAIN] Request: [{request['prompt']}] -> [{request['target_new']}]")
    if len(train_requests) > 3:
        print(f"[TRAIN] ... and {len(train_requests) - 3} more requests.")

    layers_to_edit = get_layers_to_edit(config)
    rewrite_module_template = get_rewrite_module_template(config)

    weights_to_update = {}
    for name, param in model.named_parameters():
        for layer in layers_to_edit:
            if f"{rewrite_module_template.format(layer)}.weight" == name:
                weights_to_update[name] = param

    if not weights_to_update:
        raise ValueError(
            f"No weights found matching template={rewrite_module_template} layers={layers_to_edit}"
        )

    print(f"[TRAIN] Updating {len(weights_to_update)} params: {list(weights_to_update.keys())}")
    ref_snapshot = clone_weight_snapshot(weights_to_update) if target_kl_lambda > 0.0 else None

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
    early_stop_loss = float(getattr(config, "early_stop_loss", 1e-2) or 0.0)
    low_loss_skip_threshold = float(getattr(config, "low_loss_skip_threshold", 1e-2) or 0.0)

    for epoch in range(config.num_steps):
        print(f"\n=== Epoch {epoch} ===")
        loss_meter.reset()
        batch_count = 0
        optimizer_step_count = 0

        model.train()
        random.shuffle(train_requests)
        print(f"[TRAIN] Phase: {objective}")

        for batch_label, batch_reqs in enumerate(chunks(train_requests, config.batch_size), start=1):
            txt_batch = [r["prompt"] for r in batch_reqs]
            tgt_batch = [r["target_new"] for r in batch_reqs]

            opt.zero_grad()

            try:
                if objective == "overtone":
                    out = compute_overtone_loss(
                        model=model,
                        tok=tok,
                        prompts=txt_batch,
                        targets=tgt_batch,
                        device=device,
                        overtone_lambda=overtone_lambda,
                        overtone_epsilon=overtone_epsilon,
                        overtone_nsigma=overtone_nsigma,
                        target_kl_lambda=target_kl_lambda,
                        target_kl_direction=target_kl_direction,
                        weights_to_update=weights_to_update,
                        ref_snapshot=ref_snapshot,
                    )
                    final_loss = out["loss"]
                    print(
                        f"[OVERTONE] batch {batch_label} | "
                        f"loss={final_loss.item():.4f} "
                        f"base={float(out['base_loss']):.4f} "
                        f"target_kl={float(out['target_kl_loss']):.4f} "
                        f"avg_token_kl={float(out['avg_token_kl']):.4f} "
                        f"avg_target_prob={float(out['avg_target_prob']):.6f} "
                        f"avg_filtered_target_prob={float(out['avg_filtered_target_prob']):.6f} "
                        f"avg_target_len={float(out['avg_target_len']):.2f} "
                        f"target_tokens={out['num_target_tokens']} "
                        f"onehot_fallback={out['num_onehot_fallback_tokens']} "
                        f"clipped={out['num_clipped_tokens']}"
                    )
                elif objective == "ce":
                    out = compute_standard_ft_loss(
                        model=model,
                        tok=tok,
                        prompts=txt_batch,
                        targets=tgt_batch,
                        device=device,
                        target_kl_lambda=target_kl_lambda,
                        target_kl_direction=target_kl_direction,
                        weights_to_update=weights_to_update,
                        ref_snapshot=ref_snapshot,
                    )
                    final_loss = out["loss"]
                    print(
                        f"[CE] batch {batch_label} | "
                        f"loss={final_loss.item():.4f} "
                        f"ce={float(out['avg_ce']):.4f} "
                        f"target_kl={float(out['target_kl_loss']):.4f} "
                        f"target_tokens={out['num_target_tokens']}"
                    )
                else:
                    raise ValueError(f"Unsupported objective: {objective}")
            except Exception as exc:
                print(f"[TRAIN][warn] Skipping batch {batch_label} due to loss construction failure: {exc}")
                continue

            if not bool(torch.isfinite(final_loss).item()):
                print(f"[TRAIN][warn] Skipping optimizer step for batch {batch_label}: non-finite loss ({final_loss.item()}).")
                continue

            loss_value = float(final_loss.item())
            if low_loss_skip_threshold > 0 and loss_value < low_loss_skip_threshold:
                print(
                    f"[TRAIN] batch {batch_label} | optimizer step skipped "
                    f"(total_loss < {low_loss_skip_threshold:g})"
                )
                continue

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
            batch_count += 1
            loss_meter.update(loss_value, n=len(txt_batch))

        if batch_count == 0:
            print(f"Epoch {epoch} avg loss: n/a")
            print(f"Epoch {epoch} processed batches: 0")
            print(f"Epoch {epoch} optimizer steps: 0")
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
        description="OVERTONE-style full-target editing objective on top of the provided EasyEdit FT backbone."
    )
    parser.add_argument(
        "hparams_path",
        nargs="?",
        help="Path to FT/LocFT yaml config. Equivalent to --ft_config_path.",
    )
    parser.add_argument(
        "--ft_config_path",
        "--hparams",
        "--config_path",
        dest="ft_config_path",
        type=str,
        default=None,
        help="Path to FT/LocFT yaml config",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Override data path in config")
    parser.add_argument("--save_model_dir", type=str, default=None, help="Override save dir in config")
    parser.add_argument("--easyedit_path", type=str, default=None, help="Path to EasyEdit root")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or 'auto'; overrides config device")
    parser.add_argument("--sample_size", type=int, default=None, help="Randomly sample this many requests; <= 0 uses all data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling and training order")
    parser.add_argument("--objective", type=str, choices=["overtone", "ce"], default=None, help="Training objective")
    parser.add_argument("--overtone_lambda", type=float, default=None, help="Mixture weight λ for OVERTONE target distribution")
    parser.add_argument("--overtone_epsilon", type=float, default=None, help="Clipping threshold ε for token-level KL")
    parser.add_argument("--overtone_nsigma", type=float, default=None, help="Top-nσ filtering threshold")
    parser.add_argument("--target_kl_lambda", type=float, default=None, help="Weight for target-token KL against the initial editable-weight snapshot")
    parser.add_argument(
        "--target_kl_direction",
        type=str,
        choices=["current_to_ref", "ref_to_current"],
        default=None,
        help="Direction for target-token KL: current_to_ref matches ROME/AlphaEdit; ref_to_current matches FT-M prefix KL.",
    )
    parser.add_argument("--break_loss", type=float, default=None, help="Kept only for config bookkeeping")
    parser.add_argument("--break_prob", type=float, default=None, help="Kept only for config bookkeeping")
    parser.add_argument("--norm_factor", type=float, default=None, help="Kept only for config bookkeeping")
    parser.add_argument("--target_alpha", type=float, default=None, help="Kept only for config bookkeeping")
    parser.add_argument(
        "--save_tag",
        type=str,
        default=None,
        help="Optional method subdirectory override under outputs/Models when save_model_dir is not provided",
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset label used in default save directory naming")
    args = parser.parse_args()
    run_started_at = datetime.now()
    run_start = start_timer()

    config_path = args.ft_config_path or args.hparams_path
    if config_path is None:
        parser.error("--ft_config_path/--hparams or positional hparams_path is required")
    config_path = resolve_existing_path(config_path, SCRIPT_DIR)

    if args.easyedit_path:
        sys.path.insert(0, args.easyedit_path)

    print(f"Loading FT config from {config_path}")
    config = LocFTBFHyperParams.from_hparams(config_path)
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

    args.objective = get_arg_or_config(args.objective, config, "objective", "overtone")
    if args.objective not in {"overtone", "ce"}:
        parser.error("--objective must be one of: overtone, ce")
    args.overtone_lambda = float(
        get_arg_or_config(args.overtone_lambda, config, "overtone_lambda", DEFAULT_OVERTONE_LAMBDA)
    )
    args.overtone_epsilon = float(
        get_arg_or_config(args.overtone_epsilon, config, "overtone_epsilon", DEFAULT_OVERTONE_EPSILON)
    )
    args.overtone_nsigma = float(
        get_arg_or_config(args.overtone_nsigma, config, "overtone_nsigma", DEFAULT_OVERTONE_NSIGMA)
    )
    args.target_kl_lambda = float(
        get_arg_or_config(args.target_kl_lambda, config, "target_kl_lambda", 0.0) or 0.0
    )
    args.target_kl_direction = get_arg_or_config(
        args.target_kl_direction,
        config,
        "target_kl_direction",
        "current_to_ref",
    )
    if args.target_kl_direction not in {"current_to_ref", "ref_to_current"}:
        parser.error("--target_kl_direction must be one of: current_to_ref, ref_to_current")
    if args.target_kl_lambda < 0.0:
        parser.error("--target_kl_lambda must be >= 0")
    args.sample_size = get_arg_or_config(args.sample_size, config, "sample_size", None)
    if args.sample_size is not None:
        args.sample_size = int(args.sample_size)
    args.save_model_dir = get_arg_or_config(args.save_model_dir, config, "save_model_dir", None)
    if args.save_model_dir:
        args.save_model_dir = resolve_output_path(args.save_model_dir, SCRIPT_DIR)
    args.save_tag = str(get_arg_or_config(args.save_tag, config, "save_tag", "baseline") or "baseline")
    args.dataset_name = get_arg_or_config(args.dataset_name, config, "dataset_name", None)

    apply_hparam_overrides(config, args.break_loss, args.break_prob, args.norm_factor, args.target_alpha)

    use_model_parallel = bool(getattr(config, "model_parallel", False))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script currently requires a CUDA device.")
    if use_model_parallel:
        if args.device not in (None, "auto"):
            print(
                f"[load] Ignoring --device={args.device} because ft_config has model_parallel=True. "
                "Using device_map='auto' instead."
            )
        resolved_device = int(config.device)
        print("FT config has model_parallel=True, so the model will be loaded with device_map='auto'.")
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
                f"Randomly sampling {args.sample_size} requests from {len(requests)} total with seed={args.seed}."
            )
            requests = random.sample(requests, args.sample_size)
        else:
            print(
                f"Requested sample_size={args.sample_size}, but only {len(requests)} valid requests are available. Using all requests."
            )

    print(f"Using {len(requests)} editing requests for training.")
    print(
        "[config] "
        f"objective={args.objective} "
        f"overtone_lambda={args.overtone_lambda} "
        f"overtone_epsilon={args.overtone_epsilon} "
        f"overtone_nsigma={args.overtone_nsigma} "
        f"target_kl_lambda={args.target_kl_lambda} "
        f"target_kl_direction={args.target_kl_direction} "
        f"break_loss={getattr(config, 'break_loss', None)} "
        f"break_prob={getattr(config, 'break_prob', None)} "
        f"target_alpha={getattr(config, 'target_alpha', None)} "
        f"norm_factor={getattr(config, 'clamp_norm_factor', None)}"
    )

    model_name = get_model_name(config)
    model_dtype = get_model_load_dtype(config)
    model_load_start = start_timer()
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
    model_load_elapsed = elapsed_since(model_load_start)

    eos_token = resolve_tokenizer_eos_token(tokenizer)
    eos_appended_count = count_targets_needing_eos(requests, tokenizer)
    requests = normalize_ft_requests(requests, tokenizer)
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
        norm_factor_tag = format_path_value(getattr(config, "clamp_norm_factor", None))
        target_alpha_tag = format_path_value(getattr(config, "target_alpha", None))
        target_kl_tag = (
            f"_targetkl{format_path_value(args.target_kl_lambda)}_{args.target_kl_direction}"
            if args.target_kl_lambda > 0.0
            else ""
        )
        run_name = (
            f"{model_slug}_{dataset_name}"
            f"_obj{args.objective}"
            f"_lambda{format_path_value(args.overtone_lambda)}"
            f"_eps{format_path_value(args.overtone_epsilon)}"
            f"_nsigma{format_path_value(args.overtone_nsigma)}"
            f"{target_kl_tag}"
            f"_norm_factor{norm_factor_tag}"
            f"_target_alpha{target_alpha_tag}"
        )
        objective_output_dir = {
            "ce": "LocFT-BF",
            "overtone": "OVERTONE",
        }[args.objective]
        save_tag = (args.save_tag or "").strip()
        method_output_dir = (
            save_tag if save_tag and save_tag.lower() != "baseline" else objective_output_dir
        )
        config.save_model_dir = os.path.join(
            SCRIPT_DIR,
            "outputs",
            "Models",
            method_output_dir,
            run_name,
        )

    save_training_artifacts(
        save_dir=config.save_model_dir,
        requests=requests,
        data_path=config.data_path,
        num_requests_before_sampling=num_requests_before_sampling,
        sample_size_requested=args.sample_size,
        seed=args.seed,
        objective=args.objective,
        overtone_lambda=args.overtone_lambda,
        overtone_epsilon=args.overtone_epsilon,
        overtone_nsigma=args.overtone_nsigma,
        target_kl_lambda=args.target_kl_lambda,
        target_kl_direction=args.target_kl_direction,
        break_loss=getattr(config, "break_loss", None),
        break_prob=getattr(config, "break_prob", None),
        norm_factor=getattr(config, "clamp_norm_factor", None),
        target_alpha=getattr(config, "target_alpha", None),
        append_eos_to_target=True,
        eos_token=eos_token,
        eos_appended_count=eos_appended_count,
    )

    stage_label = f"Training objective={args.objective}"
    training_start = start_timer()
    print_time(f"Begin {stage_label}")
    model = execute_training(
        model=model,
        tok=tokenizer,
        requests=requests,
        config=config,
        objective=args.objective,
        overtone_lambda=args.overtone_lambda,
        overtone_epsilon=args.overtone_epsilon,
        overtone_nsigma=args.overtone_nsigma,
        target_kl_lambda=args.target_kl_lambda,
        target_kl_direction=args.target_kl_direction,
    )
    print_time(f"End {stage_label}")
    training_elapsed = elapsed_since(training_start)

    save_path = config.save_model_dir
    print(f"Saving model to {save_path}")
    save_start = start_timer()
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    save_elapsed = elapsed_since(save_start)
    total_elapsed = elapsed_since(run_start)
    run_finished_at = datetime.now()
    timing_summary = {
        "started_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": run_finished_at.isoformat(timespec="seconds"),
        "model_load_elapsed_sec": model_load_elapsed,
        "model_load_elapsed": format_elapsed(model_load_elapsed),
        "training_elapsed_sec": training_elapsed,
        "training_elapsed": format_elapsed(training_elapsed),
        "save_elapsed_sec": save_elapsed,
        "save_elapsed": format_elapsed(save_elapsed),
        "total_elapsed_sec": total_elapsed,
        "total_elapsed": format_elapsed(total_elapsed),
    }
    save_timing_summary(save_path, timing_summary)
    print("Done!")
