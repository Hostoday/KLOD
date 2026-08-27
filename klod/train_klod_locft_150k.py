import argparse
import json
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT_DIR = PROJECT_ROOT
from EasyEdit.easyeditor.models.locft_bf.locft_bf_hparams import LocFTBFHyperParams
from EasyEdit.easyeditor.models.klod.klod_hparams import KLODHyperParams


DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "zsre", "zsre_mend_train.json")
DEFAULT_SAMPLE_SIZE = 150_000
DEFAULT_SAVE_COUNTS = "3k,5k,10k,20k,50k,100k,150k"
DEFAULT_TARGET_ALPHA = 1.0
DEFAULT_SAVE_TAG = "KLOD_LocFT_rewrite_prefix_kl"

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


def restore_weight_snapshot(
    weights_to_update: Dict[str, torch.nn.Parameter],
    snapshot: Dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, param in weights_to_update.items():
            ref = snapshot[name].to(device=param.device, dtype=param.dtype)
            param.copy_(ref)


def chunks(arr, n):
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def checkpoint_aware_chunks(arr, batch_size: int, save_counts: List[int]):
    checkpoint_counts = sorted(count for count in set(save_counts) if 0 < count <= len(arr))
    checkpoint_idx = 0
    start = 0

    while start < len(arr):
        while checkpoint_idx < len(checkpoint_counts) and checkpoint_counts[checkpoint_idx] <= start:
            checkpoint_idx += 1

        end = min(start + batch_size, len(arr))
        if checkpoint_idx < len(checkpoint_counts):
            next_checkpoint = checkpoint_counts[checkpoint_idx]
            if start < next_checkpoint < end:
                end = next_checkpoint

        yield arr[start:end]
        start = end


def parse_count_token(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    if not text:
        raise ValueError("Empty checkpoint count token.")

    multiplier = 1
    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]

    count_float = float(text) * multiplier
    count = int(count_float)
    if count_float != count or count <= 0:
        raise ValueError(f"Invalid checkpoint count: {value}")
    return count


def parse_save_counts(value: Optional[str]) -> List[int]:
    if value is None:
        return []
    if value.strip().lower() in {"", "0", "none", "off", "false"}:
        return []

    counts = []
    for chunk in value.replace(";", ",").split(","):
        for token in chunk.split():
            counts.append(parse_count_token(token))
    return sorted(set(counts))


def filter_save_counts(save_counts: List[int], total_requests: int) -> List[int]:
    filtered = [count for count in save_counts if count <= total_requests]
    skipped = [count for count in save_counts if count > total_requests]
    if skipped:
        print(
            "[config][warn] Ignoring save_counts beyond sampled request count "
            f"({total_requests}): {skipped}"
        )
    return filtered


def filter_cumulative_save_counts(
    save_counts: List[int],
    new_request_count: int,
    initial_count: int,
) -> List[int]:
    if initial_count < 0:
        raise ValueError(f"--initial_count must be >= 0, got {initial_count}")

    max_cumulative_count = initial_count + new_request_count
    filtered = [
        count
        for count in save_counts
        if initial_count < count <= max_cumulative_count
    ]
    skipped_already_done = [count for count in save_counts if count <= initial_count]
    skipped_beyond = [count for count in save_counts if count > max_cumulative_count]

    if skipped_already_done:
        print(
            "[config][warn] Ignoring save_counts already covered by the loaded model "
            f"(initial_count={initial_count}): {skipped_already_done}"
        )
    if skipped_beyond:
        print(
            "[config][warn] Ignoring save_counts beyond loaded model + sampled new requests "
            f"({initial_count}+{new_request_count}={max_cumulative_count}): {skipped_beyond}"
        )
    return filtered


def format_count_tag(count: int) -> str:
    if count % 1_000_000 == 0:
        return f"{count // 1_000_000}m"
    if count % 1_000 == 0:
        return f"{count // 1_000}k"
    return str(count)


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


def load_yaml_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def get_default_save_model_slug(load_model_name: str, fallback_model_name: str) -> str:
    slug_source = fallback_model_name if os.path.exists(load_model_name) else load_model_name
    return str(slug_source).rstrip("/").split("/")[-1]


def get_matching_resume_run_dir(load_model_name: str, run_name: str) -> Optional[str]:
    if not os.path.exists(load_model_name):
        return None

    model_path = os.path.abspath(load_model_name)
    candidates = [model_path]
    if os.path.basename(model_path).startswith("checkpoint_"):
        candidates.insert(0, os.path.dirname(model_path))

    for candidate in candidates:
        if os.path.basename(candidate.rstrip(os.sep)) == run_name:
            return candidate
    return None


def set_model_name(config, model_name: str) -> None:
    if hasattr(config, "model_name_or_path"):
        config.model_name_or_path = model_name
    elif hasattr(config, "model_name"):
        config.model_name = model_name
    else:
        config.model_name = model_name


def expand_path_args(paths: Optional[List[str]]) -> List[str]:
    expanded: List[str] = []
    for value in paths or []:
        for path in str(value).replace(",", " ").split():
            if path:
                expanded.append(path)
    return expanded


def identity_value(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def request_identity_keys(request: Dict) -> set:
    keys = set()
    case_id = request.get("case_id")
    if case_id is not None:
        keys.add(("case_id", identity_value(case_id)))

    source_index = request.get("source_index")
    if source_index is not None:
        keys.add(("source_index", identity_value(source_index)))

    prompt = request.get("prompt", request.get("src"))
    target = request.get("target_new", request.get("alt"))
    subject = request.get("subject", "")
    if prompt is not None and target is not None:
        keys.add(
            (
                "content",
                identity_value(prompt),
                identity_value(target),
                identity_value(subject),
            )
        )
    return keys


def load_json_record_list(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            records = payload["data"]
        elif isinstance(payload.get("requests"), list):
            records = payload["requests"]
        else:
            raise ValueError(
                f"Unsupported exclusion JSON structure in {path}; expected a list, "
                "or a dict with a list-valued 'data'/'requests' field."
            )
    else:
        raise ValueError(f"Unsupported exclusion JSON structure in {path}")

    return [record for record in records if isinstance(record, dict)]


def load_exclusion_keys_and_records(paths: List[str]) -> tuple[set, List[Dict]]:
    exclusion_keys = set()
    exclusion_records: List[Dict] = []
    for path in paths:
        records = load_json_record_list(path)
        exclusion_records.extend(records)
        for record in records:
            exclusion_keys.update(request_identity_keys(record))
        print(
            f"[data] Loaded {len(records)} exclusion record(s) from {path}; "
            f"identity_keys_total={len(exclusion_keys)}"
        )
    return exclusion_keys, exclusion_records


def exclude_previously_used_requests(
    requests: List[Dict],
    exclude_paths: List[str],
) -> tuple[List[Dict], int, int, List[Dict]]:
    if not exclude_paths:
        return requests, 0, 0, []

    exclusion_keys, exclusion_records = load_exclusion_keys_and_records(exclude_paths)
    kept = []
    excluded_count = 0
    for request in requests:
        if request_identity_keys(request) & exclusion_keys:
            excluded_count += 1
        else:
            kept.append(request)
    return kept, excluded_count, len(exclusion_keys), exclusion_records


def apply_hparam_overrides(
    config,
    break_loss: Optional[float],
    break_prob: Optional[float],
    norm_factor: Optional[float],
    target_alpha: Optional[float],
    early_stop_loss: Optional[float],
) -> None:
    if break_loss is not None and hasattr(config, "break_loss"):
        config.break_loss = break_loss
    if break_prob is not None and hasattr(config, "break_prob"):
        config.break_prob = break_prob
    if norm_factor is not None and hasattr(config, "clamp_norm_factor"):
        config.clamp_norm_factor = norm_factor
    if target_alpha is not None:
        config.target_alpha = target_alpha
    if early_stop_loss is not None:
        config.early_stop_loss = early_stop_loss


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


def get_invalid_request_reasons(record: Dict) -> List[str]:
    reasons = []
    if not record.get("prompt", record.get("src")):
        reasons.append("missing_or_empty_prompt/src")
    if not record.get("target_new", record.get("alt")):
        reasons.append("missing_or_empty_target_new/alt")
    return reasons or ["unknown"]


def save_training_artifacts(
    save_dir: str,
    requests: List[Dict],
    *,
    prior_requests: Optional[List[Dict]] = None,
    data_path: str,
    num_requests_before_sampling: int,
    num_requests_after_exclusion: int,
    exclude_requests_paths: List[str],
    num_requests_excluded: int,
    sample_size_requested: Optional[int],
    seed: int,
    initial_count: int,
    model_name: str,
    use_locft: bool,
    warmup_epochs: int,
    rewrite_kl_lambda: float,
    non_target_kl_lambda: float,
    shuffle_requests: bool,
    break_loss: Optional[float],
    break_prob: Optional[float],
    norm_factor: Optional[float],
    target_alpha: Optional[float],
    early_stop_loss: Optional[float],
    save_counts: Optional[List[int]] = None,
    append_eos_to_target: bool = True,
    eos_token: Optional[str] = None,
    eos_appended_count: int = 0,
    prior_eos_appended_count: int = 0,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    prior_requests = prior_requests or []
    cumulative_requests = prior_requests + requests
    requests_path = os.path.join(save_dir, "training_requests.json")
    with open(requests_path, "w", encoding="utf-8") as f:
        json.dump(cumulative_requests, f, ensure_ascii=False, indent=2)

    new_requests_path = None
    if prior_requests:
        new_requests_path = os.path.join(save_dir, "new_training_requests.json")
        with open(new_requests_path, "w", encoding="utf-8") as f:
            json.dump(requests, f, ensure_ascii=False, indent=2)

    has_klod_phase = int(warmup_epochs) > 0
    if has_klod_phase and use_locft:
        objective = "KLOD+LocFT"
    elif has_klod_phase:
        objective = "KLOD"
    elif use_locft:
        objective = "LocFT"
    else:
        objective = "none"

    manifest = {
        "objective": objective,
        "data_path": data_path,
        "data_stem": get_data_stem(data_path),
        "num_requests_before_sampling": num_requests_before_sampling,
        "num_requests_after_exclusion": num_requests_after_exclusion,
        "exclude_requests_paths": exclude_requests_paths,
        "num_requests_excluded": num_requests_excluded,
        "num_prior_requests": len(prior_requests),
        "num_new_requests_used": len(requests),
        "num_requests_used": len(cumulative_requests),
        "sample_size_requested": sample_size_requested,
        "seed": seed,
        "initial_count": initial_count,
        "model_name": model_name,
        "use_locft": use_locft,
        # Backward-compatible aliases for older analysis scripts.
        "use_ft": use_locft,
        "warmup_epochs": warmup_epochs,
        "rewrite_kl_lambda": rewrite_kl_lambda,
        "non_target_kl_lambda": non_target_kl_lambda,
        "edit_loss": "one_sided_logit_odds_hinge" if has_klod_phase else None,
        "shuffle_requests": shuffle_requests,
        "break_loss": break_loss,
        "break_prob": break_prob,
        "norm_factor": norm_factor,
        "target_alpha": target_alpha,
        "early_stop_loss": early_stop_loss,
        "save_counts": save_counts or [],
        "checkpoint_training_mode": "staged_noreplay" if save_counts else "single_full_run",
        "append_eos_to_target": append_eos_to_target,
        "eos_token": eos_token,
        "eos_appended_count": eos_appended_count,
        "prior_eos_appended_count": prior_eos_appended_count,
        "training_requests_path": requests_path,
        "new_training_requests_path": new_requests_path,
    }

    manifest_path = os.path.join(save_dir, "training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved training requests to {requests_path}")
    print(f"Saved training manifest to {manifest_path}")


def save_checkpoint_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_model_dir: str,
    count: int,
    *,
    epoch: int,
    batch_count: int,
    processed_requests: int,
    training_mode: str,
    num_epochs: int,
    extra_manifest: Optional[Dict] = None,
) -> str:
    checkpoint_dir = os.path.join(save_model_dir, f"checkpoint_{format_count_tag(count)}")
    print(f"[checkpoint] saving model after {count} requests to {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    manifest = {
        "checkpoint_count": count,
        "checkpoint_tag": format_count_tag(count),
        "epoch": epoch,
        "batch_count": batch_count,
        "processed_requests": processed_requests,
        "training_mode": training_mode,
        "num_epochs": num_epochs,
        "save_model_dir": save_model_dir,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return checkpoint_dir


# -----------------------------------------------------------------------------
# KLOD stage: one-sided target-vs-rest logit odds hinge + rewrite-prefix KL
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
# LocFT stage
# -----------------------------------------------------------------------------


def compute_locft_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
) -> torch.Tensor:
    full_inputs, label_mask = build_full_batch_inputs(tok, prompts, targets, device)

    logits = model(**full_inputs).logits
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = full_inputs["input_ids"][..., 1:].contiguous()
    target_token_counts = label_mask[:, 1:].sum(1)
    if bool((target_token_counts == 0).any().item()):
        raise RuntimeError("A batch contains samples with zero target tokens under the current label mask.")

    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(full_inputs["input_ids"].shape[0], -1)
    masked_loss = (loss * label_mask[:, 1:]).sum(1) / target_token_counts
    return masked_loss.mean()


def empty_phase_stats() -> Dict[str, int]:
    return {
        "epochs_completed": 0,
        "last_batch_count": 0,
        "last_optimizer_step_count": 0,
    }


def combine_phase_stats(
    klod_stats: Dict[str, int],
    locft_stats: Dict[str, int],
) -> Dict[str, int]:
    last_stats = locft_stats if locft_stats["epochs_completed"] > 0 else klod_stats
    return {
        "epochs_completed": klod_stats["epochs_completed"] + locft_stats["epochs_completed"],
        "last_batch_count": last_stats["last_batch_count"],
        "last_optimizer_step_count": last_stats["last_optimizer_step_count"],
        "klod_epochs_completed": klod_stats["epochs_completed"],
        "klod_last_batch_count": klod_stats["last_batch_count"],
        "klod_last_optimizer_step_count": klod_stats["last_optimizer_step_count"],
        "locft_epochs_completed": locft_stats["epochs_completed"],
        "locft_last_batch_count": locft_stats["last_batch_count"],
        "locft_last_optimizer_step_count": locft_stats["last_optimizer_step_count"],
    }


def run_klod_epochs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    train_requests: List[Dict],
    config,
    *,
    device: torch.device,
    opt: torch.optim.Optimizer,
    num_epochs: int,
    rewrite_kl_lambda: float,
    non_target_kl_lambda: float,
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Optional[Dict[str, torch.Tensor]],
    early_stop_loss: float,
    target_alpha: float,
    shuffle_requests: bool,
    stage_name: str,
) -> Dict[str, int]:
    epochs_completed = 0
    last_batch_count = 0
    last_optimizer_step_count = 0

    for epoch in range(num_epochs):
        print(f"\n=== {stage_name} | KLOD epoch {epoch} ===")
        loss_meter = AverageMeter()
        batch_count = 0
        optimizer_step_count = 0

        model.train()
        epoch_requests = list(train_requests)
        if shuffle_requests:
            random.shuffle(epoch_requests)

        print(
            "[TRAIN] Phase: KLOD "
            f"(one-sided logit odds hinge + rewrite-prefix KL, epoch {epoch})"
        )

        for batch_reqs in chunks(epoch_requests, config.batch_size):
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
                print(f"[KLOD][warn] Skipping batch {batch_label} due to loss construction failure: {exc}")
                continue

            if not bool(torch.isfinite(final_loss).item()):
                print(f"[KLOD][warn] Skipping batch {batch_label}: non-finite loss ({final_loss.item()}).")
                continue

            loss_meter.update(final_loss.item(), n=len(txt_batch))

            final_loss.backward()

            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
                for p in weights_to_update.values()
            ):
                print(f"[KLOD][warn] Skipping optimizer step for batch {batch_label}: non-finite gradient.")
                opt.zero_grad(set_to_none=True)
                continue

            opt.step()
            optimizer_step_count += 1

        epochs_completed += 1
        last_batch_count = batch_count
        last_optimizer_step_count = optimizer_step_count

        if batch_count == 0:
            print(f"KLOD epoch {epoch} avg loss: n/a")
            print(f"KLOD epoch {epoch} processed batches: 0")
            print(f"KLOD epoch {epoch} optimizer steps: 0")
            continue

        print(f"KLOD epoch {epoch} avg loss: {loss_meter.avg:.4f}")
        print(f"KLOD epoch {epoch} processed batches: {batch_count}")
        print(f"KLOD epoch {epoch} optimizer steps: {optimizer_step_count}")

        if early_stop_loss > 0 and loss_meter.avg < early_stop_loss:
            print(
                f"Converged in KLOD under early_stop_loss={early_stop_loss:.4f}. "
                "Stopping KLOD phase early."
            )
            break

    return {
        "epochs_completed": epochs_completed,
        "last_batch_count": last_batch_count,
        "last_optimizer_step_count": last_optimizer_step_count,
    }


def run_locft_epochs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    train_requests: List[Dict],
    config,
    *,
    device: torch.device,
    opt: torch.optim.Optimizer,
    num_epochs: int,
    weights_to_update: Dict[str, torch.nn.Parameter],
    early_stop_loss: float,
    shuffle_requests: bool,
    stage_name: str,
) -> Dict[str, int]:
    epochs_completed = 0
    last_batch_count = 0
    last_optimizer_step_count = 0

    for epoch in range(num_epochs):
        print(f"\n=== {stage_name} | LocFT epoch {epoch} ===")
        loss_meter = AverageMeter()
        batch_count = 0
        optimizer_step_count = 0

        model.train()
        epoch_requests = list(train_requests)
        if shuffle_requests:
            random.shuffle(epoch_requests)
        print(f"[TRAIN] Phase: LocFT (target-token cross entropy, epoch {epoch})")

        for batch_reqs in chunks(epoch_requests, config.batch_size):
            txt_batch = [r["prompt"] for r in batch_reqs]
            tgt_batch = [r["target_new"] for r in batch_reqs]

            opt.zero_grad()
            skip_optimizer_step = False
            batch_label = batch_count + 1

            try:
                final_loss = compute_locft_loss(
                    model=model,
                    tok=tok,
                    prompts=txt_batch,
                    targets=tgt_batch,
                    device=device,
                )

                batch_count += 1
                print(f"[LocFT] batch {batch_label} | ce_loss={final_loss.item():.4f}")

                if float(final_loss.item()) < 1e-2:
                    skip_optimizer_step = True
                    print(f"[LocFT] batch {batch_label} | optimizer step skipped (ce_loss < 1e-2)")

            except Exception as exc:
                print(f"[LocFT][warn] Skipping batch {batch_label} due to loss construction failure: {exc}")
                continue

            if not bool(torch.isfinite(final_loss).item()):
                print(f"[LocFT][warn] Skipping batch {batch_label}: non-finite loss ({final_loss.item()}).")
                continue

            loss_meter.update(final_loss.item(), n=len(txt_batch))

            if skip_optimizer_step:
                continue

            final_loss.backward()

            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
                for p in weights_to_update.values()
            ):
                print(f"[LocFT][warn] Skipping optimizer step for batch {batch_label}: non-finite gradient.")
                opt.zero_grad(set_to_none=True)
                continue

            opt.step()
            optimizer_step_count += 1

        epochs_completed += 1
        last_batch_count = batch_count
        last_optimizer_step_count = optimizer_step_count

        if batch_count == 0:
            print(f"LocFT epoch {epoch} avg loss: n/a")
            print(f"LocFT epoch {epoch} processed batches: 0")
            print(f"LocFT epoch {epoch} optimizer steps: 0")
            continue

        print(f"LocFT epoch {epoch} avg loss: {loss_meter.avg:.4f}")
        print(f"LocFT epoch {epoch} processed batches: {batch_count}")
        print(f"LocFT epoch {epoch} optimizer steps: {optimizer_step_count}")

        if early_stop_loss > 0 and loss_meter.avg < early_stop_loss:
            print(
                f"Converged in LocFT under early_stop_loss={early_stop_loss:.4f}. "
                "Stopping LocFT phase early."
            )
            break

    return {
        "epochs_completed": epochs_completed,
        "last_batch_count": last_batch_count,
        "last_optimizer_step_count": last_optimizer_step_count,
    }


def run_training_phases(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    train_requests: List[Dict],
    config,
    *,
    device: torch.device,
    use_locft: bool,
    warmup_epochs: int,
    rewrite_kl_lambda: float,
    non_target_kl_lambda: float,
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Optional[Dict[str, torch.Tensor]],
    early_stop_loss: float,
    target_alpha: float,
    shuffle_requests: bool,
    stage_name: str,
) -> Dict[str, int]:
    opt = torch.optim.Adam(
        weights_to_update.values(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    klod_epoch_count = min(max(int(warmup_epochs), 0), int(config.num_steps))
    locft_epoch_count = max(int(config.num_steps) - klod_epoch_count, 0) if use_locft else 0

    print(
        "[TRAIN] Phase schedule: "
        f"KLOD epochs={klod_epoch_count}, LocFT epochs={locft_epoch_count}"
    )

    klod_stats = empty_phase_stats()
    if klod_epoch_count > 0:
        klod_stats = run_klod_epochs(
            model=model,
            tok=tok,
            train_requests=train_requests,
            config=config,
            device=device,
            opt=opt,
            num_epochs=klod_epoch_count,
            rewrite_kl_lambda=rewrite_kl_lambda,
            non_target_kl_lambda=non_target_kl_lambda,
            weights_to_update=weights_to_update,
            ref_snapshot=ref_snapshot,
            early_stop_loss=early_stop_loss,
            target_alpha=target_alpha,
            shuffle_requests=shuffle_requests,
            stage_name=stage_name,
        )

    locft_stats = empty_phase_stats()
    if locft_epoch_count > 0:
        locft_stats = run_locft_epochs(
            model=model,
            tok=tok,
            train_requests=train_requests,
            config=config,
            device=device,
            opt=opt,
            num_epochs=locft_epoch_count,
            weights_to_update=weights_to_update,
            early_stop_loss=early_stop_loss,
            shuffle_requests=shuffle_requests,
            stage_name=stage_name,
        )

    return combine_phase_stats(klod_stats, locft_stats)



def execute_training(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    config,
    *,
    use_locft: bool,
    warmup_epochs: int,
    rewrite_kl_lambda: float,
    non_target_kl_lambda: float,
    shuffle_requests: bool,
    initial_count: int = 0,
    save_counts: Optional[List[int]] = None,
    save_model_dir: Optional[str] = None,
    checkpoint_records: Optional[List[Dict]] = None,
) -> AutoModelForCausalLM:
    device = torch.device(f"cuda:{config.device}")

    if tok.padding_side != "left":
        tok.padding_side = "left"

    train_requests = normalize_training_requests(requests, tok)
    for i, request in enumerate(train_requests[:3]):
        print(f"[TRAIN] Request: [{request['prompt']}] -> [{request['target_new']}]")
    if len(train_requests) > 3:
        print(f"[TRAIN] ... and {len(train_requests)-3} more requests.")

    if int(warmup_epochs) <= 0 and not use_locft:
        print("[TRAIN] No KLOD or LocFT epochs requested.")
        return model

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

    for p in model.parameters():
        p.requires_grad = False
    for p in weights_to_update.values():
        p.requires_grad = True

    early_stop_loss = float(getattr(config, "early_stop_loss", 1e-2) or 0.0)
    target_alpha = float(getattr(config, "target_alpha", DEFAULT_TARGET_ALPHA) or DEFAULT_TARGET_ALPHA)
    checkpoint_counts = sorted(set(save_counts or []))
    if initial_count < 0:
        raise ValueError(f"initial_count must be >= 0, got {initial_count}")
    if checkpoint_counts and not save_model_dir:
        raise ValueError("save_model_dir must be provided when save_counts is not empty.")
    if checkpoint_records is None:
        checkpoint_records = []

    if checkpoint_counts:
        print(
            "[TRAIN] Staged no-replay mode: continue training the same model and "
            "train only the new slice for each checkpoint "
            "(checkpoint_3k=requests[0:3000], checkpoint_5k=requests[3000:5000], etc.)."
        )
        if rewrite_kl_lambda > 0.0 or non_target_kl_lambda > 0.0:
            print(
                "[TRAIN] KL ref snapshot will be refreshed from the "
                "current model at the start of each no-replay stage."
            )
        previous_total_count = initial_count
        for stage_index, checkpoint_count in enumerate(checkpoint_counts, start=1):
            stage_start = previous_total_count - initial_count
            stage_end = checkpoint_count - initial_count
            stage_requests = train_requests[stage_start:stage_end]
            stage_name = (
                f"noreplay stage {stage_index}/{len(checkpoint_counts)} "
                f"new_requests[{stage_start}:{stage_end}] "
                f"cumulative[{previous_total_count}:{checkpoint_count}] "
                f"({len(stage_requests)} new requests)"
            )
            stage_ref_snapshot = (
                clone_weight_snapshot(weights_to_update)
                if rewrite_kl_lambda > 0.0 or non_target_kl_lambda > 0.0
                else None
            )
            stats = run_training_phases(
                model=model,
                tok=tok,
                train_requests=stage_requests,
                config=config,
                device=device,
                use_locft=use_locft,
                warmup_epochs=warmup_epochs,
                rewrite_kl_lambda=rewrite_kl_lambda,
                non_target_kl_lambda=non_target_kl_lambda,
                weights_to_update=weights_to_update,
                ref_snapshot=stage_ref_snapshot,
                early_stop_loss=early_stop_loss,
                target_alpha=target_alpha,
                shuffle_requests=shuffle_requests,
                stage_name=stage_name,
            )
            checkpoint_dir = save_checkpoint_model(
                model,
                tok,
                save_model_dir,
                checkpoint_count,
                epoch=stats["epochs_completed"],
                batch_count=stats["last_batch_count"],
                processed_requests=checkpoint_count,
                training_mode="staged_noreplay",
                num_epochs=config.num_steps,
                extra_manifest={
                    "initial_count": initial_count,
                    "segment_start": stage_start,
                    "segment_end": stage_end,
                    "stage_processed_requests": len(stage_requests),
                    "cumulative_start": previous_total_count,
                    "cumulative_end": checkpoint_count,
                    "edit_loss": "one_sided_logit_odds_hinge" if int(warmup_epochs) > 0 else None,
                    "rewrite_kl_ref_scope": "stage_start_model",
                    "klod_epochs_completed": stats["klod_epochs_completed"],
                    "locft_epochs_completed": stats["locft_epochs_completed"],
                },
            )
            checkpoint_records.append(
                {
                    "count": checkpoint_count,
                    "path": checkpoint_dir,
                    "training_mode": "staged_noreplay",
                    "num_epochs": config.num_steps,
                    "initial_count": initial_count,
                    "segment_start": stage_start,
                    "segment_end": stage_end,
                    "stage_processed_requests": len(stage_requests),
                    "cumulative_start": previous_total_count,
                    "cumulative_end": checkpoint_count,
                    "edit_loss": "one_sided_logit_odds_hinge" if int(warmup_epochs) > 0 else None,
                    "rewrite_kl_ref_scope": "stage_start_model",
                    "klod_epochs_completed": stats["klod_epochs_completed"],
                    "locft_epochs_completed": stats["locft_epochs_completed"],
                }
            )
            previous_total_count = checkpoint_count
    else:
        ref_snapshot = (
            clone_weight_snapshot(weights_to_update)
            if rewrite_kl_lambda > 0.0 or non_target_kl_lambda > 0.0
            else None
        )
        run_training_phases(
            model=model,
            tok=tok,
            train_requests=train_requests,
            config=config,
            device=device,
            use_locft=use_locft,
            warmup_epochs=warmup_epochs,
            rewrite_kl_lambda=rewrite_kl_lambda,
            non_target_kl_lambda=non_target_kl_lambda,
            weights_to_update=weights_to_update,
            ref_snapshot=ref_snapshot,
            early_stop_loss=early_stop_loss,
            target_alpha=target_alpha,
            shuffle_requests=shuffle_requests,
            stage_name=f"all requests ({len(train_requests)} requests)",
        )

    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Staged KLOD/LocFT-BF runner; choose objective phases explicitly with --use_locft and --klod_epochs"
    )
    parser.add_argument("hparams_path", nargs="?", help="Path to KLOD or LocFT-BF yaml config")
    parser.add_argument(
        "--klod_config_path",
        "--ft_config_path",
        "--hparams",
        "--config_path",
        dest="klod_config_path",
        type=str,
        default=None,
        help="Path to KLOD or LocFT-BF yaml config",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Override data path in config")
    parser.add_argument("--save_model_dir", type=str, default=None, help="Override save dir in config")
    parser.add_argument("--easyedit_path", type=str, default=None, help="Path to EasyEdit root")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or 'auto'; overrides config device")
    parser.add_argument("--model_name", type=str, default=None, help="Override config model_name/model_name_or_path; can be a local checkpoint directory.")
    parser.add_argument(
        "--exclude_requests_path",
        action="append",
        default=None,
        help="JSON request file(s) to exclude before sampling, e.g. a previous training_requests.json. May be passed multiple times or comma-separated.",
    )
    parser.add_argument(
        "--initial_count",
        type=int,
        default=0,
        help="Number of requests already represented by the loaded model. Save counts are treated as cumulative totals.",
    )
    parser.add_argument(
        "--use_locft",
        "--use_ft",
        dest="use_locft",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to run the LocFT phase. --use_ft is kept as a backward-compatible alias.",
    )
    parser.add_argument(
        "--klod_epochs",
        "--warmup_epochs",
        dest="warmup_epochs",
        type=int,
        default=1,
        help="How many initial epochs use the KLOD objective",
    )
    parser.add_argument("--rewrite_kl_lambda", type=float, default=None, help="Weight for rewrite-prompt prefix next-token KL during KLOD")
    parser.add_argument("--non_target_kl_lambda", type=float, default=None, help="Weight for target-position non-target distribution KL against the stage-start model")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Randomly sample this many requests; <= 0 uses all data. Defaults to 150k for this runner.",
    )
    parser.add_argument(
        "--save_counts",
        type=str,
        default=DEFAULT_SAVE_COUNTS,
        help="Comma/space separated cumulative request counts for staged no-replay saves, e.g. '3k,5k,10k,20k,50k,100k,150k'. Use 'none' to disable.",
    )
    parser.add_argument(
        "--shuffle_requests",
        type=int,
        choices=[0, 1],
        default=1,
        help="Shuffle training requests at each epoch. Set to 0 to preserve request-file order.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling and training order")
    parser.add_argument("--break_loss", type=float, default=None, help="Kept only for config bookkeeping")
    parser.add_argument("--break_prob", type=float, default=None, help="Kept only for config bookkeeping")
    parser.add_argument("--norm_factor", type=float, default=None, help="Kept only for config bookkeeping")
    parser.add_argument("--target_alpha", type=float, default=None, help="Desired target-token probability for the KLOD objective")
    parser.add_argument("--delta", type=float, default=None, help="Deprecated compatibility option; ignored by the KLOD objective")
    parser.add_argument(
        "--early_stop_loss",
        type=float,
        default=None,
        help="Stop KLOD or LocFT early when that phase's epoch average loss falls below this value. Defaults to config value or 1e-2.",
    )
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
        parser.error("--ft_config_path/--hparams or positional hparams_path is required")
    config_path = resolve_existing_path(config_path, SCRIPT_DIR)

    config_payload = load_yaml_config(config_path)
    config_keys = set(config_payload.keys())
    alg_name = config_payload.get("alg_name")
    if alg_name == "KLOD":
        print(f"Loading KLOD config for KLOD/LocFT runner from {config_path}")
        config = KLODHyperParams.from_hparams(config_path)
        if hasattr(config, "alg_name"):
            config.alg_name = "KLOD"
    elif alg_name == "LocFT-BF":
        print(f"Loading LocFT-BF config for KLOD/LocFT runner from {config_path}")
        config = LocFTBFHyperParams.from_hparams(config_path)
    elif alg_name == "KLEdit":
        locft_keys = {"objective", "overtone_lambda", "overtone_epsilon", "overtone_nsigma", "target_kl_lambda", "target_kl_direction"}
        if config_keys & locft_keys:
            print(f"Loading legacy KLEdit config as LocFT-BF for KLOD/LocFT runner from {config_path}")
            config = LocFTBFHyperParams.from_hparams(config_path)
        else:
            print(f"Loading legacy KLEdit config as KLOD for KLOD/LocFT runner from {config_path}")
            config = KLODHyperParams.from_hparams(config_path)
            if hasattr(config, "alg_name"):
                config.alg_name = "KLOD"
    else:
        parser.error(f"Unsupported alg_name for this runner: {alg_name!r}. Expected KLOD or LocFT-BF.")
    config_dir = os.path.dirname(os.path.abspath(config_path))

    args.easyedit_path = get_arg_or_config(args.easyedit_path, config, "easyedit_path", None)
    if args.easyedit_path:
        args.easyedit_path = resolve_existing_path(args.easyedit_path, SCRIPT_DIR, config_dir)
        sys.path.insert(0, args.easyedit_path)

    args.seed = int(get_arg_or_config(args.seed, config, "seed", 42))
    set_seed(args.seed)

    original_model_name = get_model_name(config)
    config.data_path = get_arg_or_config(args.data_path, config, "data_path", DEFAULT_DATA_PATH)
    if config.data_path is None:
        parser.error("data_path must be provided in the hparams file or with --data_path")
    config.data_path = resolve_existing_path(config.data_path, SCRIPT_DIR, config_dir)
    if not os.path.exists(config.data_path):
        parser.error(f"data_path not found: {config.data_path}")

    if args.model_name:
        set_model_name(config, args.model_name)
    apply_hparam_overrides(
        config,
        args.break_loss,
        args.break_prob,
        args.norm_factor,
        args.target_alpha,
        args.early_stop_loss,
    )
    if args.target_alpha is None and "target_alpha" not in config_keys:
        config.target_alpha = DEFAULT_TARGET_ALPHA
    elif getattr(config, "target_alpha", None) is None:
        config.target_alpha = DEFAULT_TARGET_ALPHA
    if args.delta is not None:
        print("[config][warn] --delta is ignored because KLOD uses a hinge loss.")

    args.rewrite_kl_lambda = get_arg_or_config(
        args.rewrite_kl_lambda,
        config,
        "rewrite_kl_lambda",
        getattr(config, "kl_lambda", 0.0),
    )
    if args.rewrite_kl_lambda is None:
        args.rewrite_kl_lambda = getattr(config, "kl_lambda", 0.0)
    args.rewrite_kl_lambda = float(args.rewrite_kl_lambda or 0.0)
    args.non_target_kl_lambda = float(
        get_arg_or_config(args.non_target_kl_lambda, config, "non_target_kl_lambda", 0.0) or 0.0
    )
    args.sample_size = get_arg_or_config(args.sample_size, config, "sample_size", DEFAULT_SAMPLE_SIZE)
    if args.sample_size is None:
        args.sample_size = DEFAULT_SAMPLE_SIZE
    args.sample_size = int(args.sample_size)
    args.save_model_dir = get_arg_or_config(args.save_model_dir, config, "save_model_dir", None)
    if args.save_model_dir:
        args.save_model_dir = resolve_output_path(args.save_model_dir, SCRIPT_DIR)
    args.save_tag = str(args.save_tag or DEFAULT_SAVE_TAG)
    args.dataset_name = get_arg_or_config(args.dataset_name, config, "dataset_name", None)

    use_model_parallel = bool(getattr(config, "model_parallel", False))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script currently requires a CUDA device.")
    if use_model_parallel:
        if args.device not in (None, "auto"):
            print(
                f"[load] Ignoring --device={args.device} because KLOD/LocFT config has model_parallel=True. "
                "Using device_map='auto' instead."
            )
        resolved_device = int(config.device)
        print("KLOD/LocFT config has model_parallel=True, so the model will be loaded with device_map='auto'.")
    else:
        resolved_device = resolve_device_index(args.device, config.device)
        config.device = resolved_device
        print(f"Using cuda:{resolved_device}")

    print(f"Loading data from {config.data_path}")
    with open(config.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    requests = []
    invalid_count = 0
    invalid_reasons = Counter()
    invalid_examples = []
    for idx, d in enumerate(data):
        request = build_training_request(d, idx)
        if request is not None:
            requests.append(request)
        else:
            invalid_count += 1
            reasons = get_invalid_request_reasons(d)
            invalid_reasons.update(reasons)
            if len(invalid_examples) < 5:
                invalid_examples.append((idx, reasons, list(d.keys())))

    num_requests_before_sampling = len(requests)
    print(f"Loaded {num_requests_before_sampling} valid editing requests before sampling.")
    if invalid_count:
        print(
            f"Skipped {invalid_count} invalid data entries before sampling "
            f"(reasons={dict(invalid_reasons)})."
        )
        print(f"Invalid examples (index, reasons, keys): {invalid_examples}")

    exclude_requests_paths = expand_path_args(args.exclude_requests_path)
    requests, excluded_count, exclusion_key_count, prior_requests = exclude_previously_used_requests(
        requests,
        exclude_requests_paths,
    )
    num_requests_after_exclusion = len(requests)
    if exclude_requests_paths:
        print(
            f"[data] Excluded {excluded_count} previously used request(s) "
            f"using {exclusion_key_count} identity key(s). "
            f"Remaining before sampling: {num_requests_after_exclusion}."
        )

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

    save_counts = filter_cumulative_save_counts(
        parse_save_counts(args.save_counts),
        len(requests),
        args.initial_count,
    )
    print(f"Using {len(requests)} editing requests for training.")
    print(f"[config] initial_count={args.initial_count}")
    print(f"[config] save_counts={save_counts if save_counts else 'none'}")
    print(f"[config] shuffle_requests={bool(args.shuffle_requests)}")
    if save_counts:
        print(
            "[config] checkpoint_training_mode=staged_noreplay "
            "(save_counts are cumulative; each stage trains only the new slice after initial_count.)"
        )
    print(
        "[config] klod_epochs="
        f"{args.warmup_epochs} rewrite_kl_lambda={args.rewrite_kl_lambda} "
        f"non_target_kl_lambda={args.non_target_kl_lambda} "
        f"edit_loss=one_sided_logit_odds_hinge "
        f"break_loss={getattr(config, 'break_loss', None)} "
        f"break_prob={getattr(config, 'break_prob', None)} "
        f"target_alpha={getattr(config, 'target_alpha', None)} "
        f"norm_factor={getattr(config, 'clamp_norm_factor', None)} "
        f"early_stop_loss={getattr(config, 'early_stop_loss', 1e-2)}"
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
    prior_eos_appended_count = count_targets_needing_eos(prior_requests, tokenizer)
    requests = normalize_training_requests(requests, tokenizer)
    prior_requests = normalize_training_requests(prior_requests, tokenizer) if prior_requests else prior_requests
    print(
        "[data] append_eos_to_target=True "
        f"eos_token={eos_token!r} appended={eos_appended_count} "
        f"prior_appended={prior_eos_appended_count}"
    )

    if args.save_model_dir:
        config.save_model_dir = args.save_model_dir
    else:
        data_stem = get_data_stem(config.data_path)
        dataset_label = args.dataset_name or data_stem
        dataset_name = dataset_label if dataset_label == data_stem else f"{dataset_label}_{data_stem}"
        model_slug = get_default_save_model_slug(model_name, original_model_name)
        break_loss_tag = format_path_value(getattr(config, "break_loss", None))
        target_alpha_tag = format_path_value(getattr(config, "target_alpha", None))
        run_name = (
            f"{model_slug}_{dataset_name}"
            f"_use_locft{args.use_locft}"
            f"_klod_epochs{args.warmup_epochs}"
            f"_rewritekl{format_path_value(args.rewrite_kl_lambda)}"
            f"_ntkl{format_path_value(args.non_target_kl_lambda)}"
            f"_target_alpha{target_alpha_tag}"
            f"_break_loss{break_loss_tag}"
            f"KLOD-HINGE_FULL-TARGET"
        )
        resume_run_dir = get_matching_resume_run_dir(model_name, run_name)
        if resume_run_dir is not None:
            config.save_model_dir = resume_run_dir
            print(
                "[save] Reusing existing run directory because loaded model path "
                f"matches run_name: {config.save_model_dir}"
            )
        else:
            config.save_model_dir = os.path.join(SCRIPT_DIR, "outputs", "Models", args.save_tag, run_name)

    save_training_artifacts(
        save_dir=config.save_model_dir,
        requests=requests,
        prior_requests=prior_requests,
        data_path=config.data_path,
        num_requests_before_sampling=num_requests_before_sampling,
        num_requests_after_exclusion=num_requests_after_exclusion,
        exclude_requests_paths=exclude_requests_paths,
        num_requests_excluded=excluded_count,
        sample_size_requested=args.sample_size,
        seed=args.seed,
        initial_count=args.initial_count,
        model_name=model_name,
        use_locft=bool(args.use_locft),
        warmup_epochs=args.warmup_epochs,
        rewrite_kl_lambda=args.rewrite_kl_lambda,
        non_target_kl_lambda=args.non_target_kl_lambda,
        shuffle_requests=bool(args.shuffle_requests),
        break_loss=getattr(config, "break_loss", None),
        break_prob=getattr(config, "break_prob", None),
        norm_factor=getattr(config, "clamp_norm_factor", None),
        target_alpha=getattr(config, "target_alpha", None),
        early_stop_loss=getattr(config, "early_stop_loss", 1e-2),
        save_counts=save_counts,
        append_eos_to_target=True,
        eos_token=eos_token,
        eos_appended_count=eos_appended_count,
        prior_eos_appended_count=prior_eos_appended_count,
    )

    if args.warmup_epochs > 0 and args.use_locft:
        stage_label = "KLOD + LocFT"
    elif args.warmup_epochs > 0:
        stage_label = "KLOD"
    elif args.use_locft:
        stage_label = "LocFT"
    else:
        stage_label = "No training"
    print_time(f"Begin {stage_label}")
    saved_checkpoints: List[Dict] = []
    model = execute_training(
        model=model,
        tok=tokenizer,
        requests=requests,
        config=config,
        use_locft=bool(args.use_locft),
        warmup_epochs=args.warmup_epochs,
        rewrite_kl_lambda=args.rewrite_kl_lambda,
        non_target_kl_lambda=args.non_target_kl_lambda,
        shuffle_requests=bool(args.shuffle_requests),
        initial_count=args.initial_count,
        save_counts=save_counts,
        save_model_dir=config.save_model_dir,
        checkpoint_records=saved_checkpoints,
    )
    print_time(f"End {stage_label}")

    if saved_checkpoints:
        checkpoints_path = os.path.join(config.save_model_dir, "saved_checkpoints.json")
        with open(checkpoints_path, "w", encoding="utf-8") as f:
            json.dump(saved_checkpoints, f, ensure_ascii=False, indent=2)
        print(f"Saved checkpoint list to {checkpoints_path}")

    save_path = config.save_model_dir
    print(f"Saving model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")
