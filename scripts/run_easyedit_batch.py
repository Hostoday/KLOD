#!/usr/bin/env python3

import argparse
import ast
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml


BaseEditor = None
BatchEditor = None
compute_edit_quality = None
HPARAMS_REGISTRY: Dict[str, Any] = {}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ALG_ALIASES = {
    "alphaedit": "AlphaEdit",
    "ft": "FT",
    "memit": "MEMIT",
    "rome": "ROME",
    "ultraedit": "ULTRAEDIT",
}

RUNNER_EXTRA_METHODS = {"ROME"}


def add_easyedit_to_syspath(easyedit_path: Optional[str]) -> None:
    if easyedit_path is None:
        candidate_root = PROJECT_ROOT
    else:
        candidate_root = Path(easyedit_path).resolve()

    if (candidate_root / "EasyEdit" / "easyeditor").is_dir():
        sys.path.insert(0, str(candidate_root))
        return

    if candidate_root.name == "EasyEdit" and (candidate_root / "easyeditor").is_dir():
        sys.path.insert(0, str(candidate_root.parent))
        return

    raise FileNotFoundError(
        f"Could not locate EasyEdit package from --easyedit_path={candidate_root}. "
        "Pass either the KLOD root or the EasyEdit directory itself."
    )


def init_easyedit_imports() -> None:
    global BaseEditor, BatchEditor, HPARAMS_REGISTRY, compute_edit_quality

    from EasyEdit.easyeditor import (
        BaseEditor as _BaseEditor,
        AlphaEditHyperParams,
        FTHyperParams,
        MEMITHyperParams,
        ROMEHyperParams,
        UltraEditHyperParams,
    )
    from EasyEdit.easyeditor.editors.batch_editor import BatchEditor as _BatchEditor
    from EasyEdit.easyeditor.evaluate import compute_edit_quality as _compute_edit_quality

    BaseEditor = _BaseEditor
    BatchEditor = _BatchEditor
    compute_edit_quality = _compute_edit_quality

    HPARAMS_REGISTRY = {
        "AlphaEdit": AlphaEditHyperParams,
        "FT": FTHyperParams,
        "MEMIT": MEMITHyperParams,
        "ROME": ROMEHyperParams,
        "ULTRAEDIT": UltraEditHyperParams,
    }


def is_runner_supported_method(alg_name: str) -> bool:
    return BatchEditor.is_batchable_method(alg_name) or alg_name in RUNNER_EXTRA_METHODS


def fix_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except Exception:
        pass


def canonicalize_alg_name(name: str) -> str:
    if name in HPARAMS_REGISTRY:
        return name
    lowered = name.lower()
    if lowered in ALG_ALIASES:
        return ALG_ALIASES[lowered]
    raise ValueError(
        f"Unsupported editing method: {name}. "
        f"Supported methods: {', '.join(sorted(HPARAMS_REGISTRY))}"
    )


def infer_alg_name_from_yaml(hparams_path: str) -> str:
    with open(hparams_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    alg_name = config.get("alg_name") or config.get("alg")
    if alg_name is None:
        raise ValueError(f"Could not infer editing method from {hparams_path}")
    return canonicalize_alg_name(str(alg_name))


def resolve_device_arg(device_arg: Optional[str], default_device: int) -> int:
    if device_arg is None:
        return int(default_device)
    if device_arg == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, so --device auto cannot be resolved.")
        return torch.cuda.current_device()
    return int(device_arg)


def maybe_parse_literal_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list literal, but got: {value}")
    return parsed


def apply_hparam_overrides(hparams: Any, args: argparse.Namespace) -> None:
    if hasattr(hparams, "model_parallel") and hparams.model_parallel and args.device is not None:
        print(
            f"[config] Ignoring --device={args.device} because this hparams file uses model_parallel=True."
        )
    elif hasattr(hparams, "device"):
        hparams.device = resolve_device_arg(args.device, getattr(hparams, "device", 0))

    if args.batch_size is not None:
        hparams.batch_size = int(args.batch_size)

    if args.model_name is not None:
        hparams.model_name = args.model_name

    if args.layers is not None and hasattr(hparams, "layers"):
        hparams.layers = args.layers

    if args.edit_layers is not None and hasattr(hparams, "edit_layers"):
        hparams.edit_layers = args.edit_layers

    if args.break_loss is not None and hasattr(hparams, "break_loss"):
        hparams.break_loss = args.break_loss

    if args.break_prob is not None and hasattr(hparams, "break_prob"):
        hparams.break_prob = args.break_prob

    if args.mse_lambda is not None and hasattr(hparams, "mse_lambda"):
        hparams.mse_lambda = args.mse_lambda

    if args.objective_optimization is not None and hasattr(hparams, "objective_optimization"):
        hparams.objective_optimization = args.objective_optimization

    if args.ft_loss_objective is not None and hasattr(hparams, "ft_loss_objective"):
        hparams.ft_loss_objective = args.ft_loss_objective

    if args.target_alpha is not None and hasattr(hparams, "target_alpha"):
        hparams.target_alpha = args.target_alpha

    if args.rewrite_kl_lambda is not None and hasattr(hparams, "rewrite_kl_lambda"):
        hparams.rewrite_kl_lambda = args.rewrite_kl_lambda

    if args.skip_prob is not None and hasattr(hparams, "skip_prob"):
        hparams.skip_prob = args.skip_prob

    if args.pl_skip_above_target is not None and hasattr(hparams, "pl_skip_above_target"):
        hparams.pl_skip_above_target = bool(args.pl_skip_above_target)

    if args.append_eos_to_target is not None and hasattr(hparams, "append_eos_to_target"):
        hparams.append_eos_to_target = bool(args.append_eos_to_target)

    if args.context_num is not None and hasattr(hparams, "context_template_length_params"):
        if args.context_num == 0:
            hparams.context_template_length_params = "None"
        else:
            half = int(args.context_num // 2)
            hparams.context_template_length_params = [[5, half], [10, half]]

    if args.p_loc is not None and hasattr(hparams, "P_loc"):
        hparams.P_loc = args.p_loc


def normalize_hparam_paths(hparams: Any, project_root: Path) -> None:
    for attr in ("stats_dir", "P_loc", "save_path", "load_path"):
        if not hasattr(hparams, attr):
            continue
        raw_value = getattr(hparams, attr)
        if raw_value is None:
            continue
        path = Path(str(raw_value))
        if path.is_absolute():
            continue
        setattr(hparams, attr, str((project_root / path).resolve()))


def first_present(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_ground_truth(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else "<|endoftext|>"
    if value is None:
        return "<|endoftext|>"
    return str(value)


def normalize_target_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("str", "text", "value"):
            if key in value and value[key] is not None:
                return str(value[key])
        raise ValueError(f"Unsupported target_new dict format: {value}")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def resolve_tokenizer_eos_token(tokenizer: Any) -> str:
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        return str(eos_token)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if eos_token_id is not None and hasattr(tokenizer, "decode"):
        return str(tokenizer.decode([int(eos_token_id)], skip_special_tokens=False))

    raise ValueError(
        "--append_eos_to_target=1 was requested, but the tokenizer has no eos_token or eos_token_id."
    )


def append_eos_to_request_targets(
    requests: List[Dict[str, Any]],
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], str, int]:
    eos_token = resolve_tokenizer_eos_token(tokenizer)
    appended_count = 0

    for request in requests:
        target_new = str(request["target_new"])
        if not target_new.endswith(eos_token):
            request["target_new"] = target_new + eos_token
            appended_count += 1

    return requests, eos_token, appended_count


def normalize_eval_group(raw_group: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_group, dict):
        return {}

    if "prompt" in raw_group and ("ground_truth" in raw_group or "answer" in raw_group):
        ground_truth = first_present(raw_group, ("ground_truth", "answer"))
        return {
            "default": {
                "prompt": raw_group["prompt"],
                "ground_truth": ground_truth,
            }
        }

    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in raw_group.items():
        if not isinstance(value, dict):
            continue
        prompt = first_present(value, ("prompt", "src"))
        ground_truth = first_present(value, ("ground_truth", "answer", "answers"))
        if prompt is None or ground_truth is None:
            continue
        normalized[key] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
        }
    return normalized


def build_request(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    prompt = first_present(record, ("prompt", "src"))
    target_new = first_present(record, ("target_new", "alt"))

    if prompt is None or target_new is None:
        return None

    request: Dict[str, Any] = {
        "case_id": first_present(record, ("case_id", "id"), idx),
        "prompt": str(prompt),
        "target_new": normalize_target_text(target_new),
        "ground_truth": normalize_ground_truth(
            first_present(record, ("ground_truth", "pred", "answers"))
        ),
        "portability": {},
        "locality": {},
    }

    subject = first_present(record, ("subject",))
    if subject is not None:
        request["subject"] = str(subject)
        request.setdefault("loc_prompt", str(subject))

    loc_prompt = first_present(record, ("loc_prompt",))
    if loc_prompt is not None:
        request["loc_prompt"] = str(loc_prompt)

    rephrase_prompt = first_present(record, ("rephrase_prompt", "rephrase"))
    if rephrase_prompt is not None:
        request["rephrase_prompt"] = str(rephrase_prompt)

    locality_prompt = first_present(record, ("locality_prompt", "loc"))
    locality_ground_truth = first_present(record, ("locality_ground_truth", "loc_ans"))
    if locality_prompt is not None and locality_ground_truth is not None:
        request["locality"]["neighborhood"] = {
            "prompt": locality_prompt,
            "ground_truth": locality_ground_truth,
        }

    request["locality"].update(normalize_eval_group(record.get("locality")))
    request["portability"].update(normalize_eval_group(record.get("portability")))

    return request


def load_requests(data_path: str) -> Tuple[List[Dict[str, Any]], int]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError(f"Unsupported JSON structure in {data_path}")
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {data_path}")

    requests: List[Dict[str, Any]] = []
    for idx, record in enumerate(data):
        request = build_request(record, idx)
        if request is not None:
            requests.append(request)

    return requests, len(data)


def maybe_sample_requests(
    requests: List[Dict[str, Any]],
    sample_size: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(requests):
        return requests

    rng = random.Random(seed)
    sampled = rng.sample(requests, sample_size)
    sampled.sort(key=lambda x: int(x["case_id"]) if str(x["case_id"]).isdigit() else str(x["case_id"]))
    return sampled


def apply_method_runtime_defaults(alg_name: str, hparams: Any) -> None:
    if alg_name == "ROME":
        if getattr(hparams, "batch_size", 1) != 1:
            print(
                "[config] ROME does not support batch editing; "
                f"forcing batch_size=1 from {hparams.batch_size}."
            )
        hparams.batch_size = 1

    if alg_name == "WISE":
        # WISE's adapter code expects this attribute and this runner performs
        # cumulative/sequential edits by construction.
        hparams.sequential_edit = True


def prepare_method_specific_requests(
    alg_name: str,
    requests: List[Dict[str, Any]],
) -> None:
    if alg_name != "WISE":
        return

    for request in requests:
        request.setdefault("loc_prompt", request.get("subject", request["prompt"]))


def validate_method_specific_constraints(
    alg_name: str,
    hparams: Any,
    requests: List[Dict[str, Any]],
) -> None:
    if alg_name == "ROME":
        missing_subject = [req["case_id"] for req in requests if "subject" not in req]
        if missing_subject:
            raise ValueError(
                "ROME requires a subject for every request. "
                f"Missing subject for case_ids={missing_subject[:10]}"
                + ("..." if len(missing_subject) > 10 else "")
            )
        return

    if alg_name != "ULTRAEDIT":
        return

    if not hasattr(hparams, "batch_size_once"):
        raise ValueError("ULTRAEDIT hparams must define batch_size_once.")
    if hparams.batch_size_once <= 0:
        raise ValueError(f"ULTRAEDIT batch_size_once must be > 0, got {hparams.batch_size_once}")
    if hparams.batch_size % hparams.batch_size_once != 0:
        raise ValueError(
            "ULTRAEDIT requires batch_size to be divisible by batch_size_once, "
            f"got batch_size={hparams.batch_size}, batch_size_once={hparams.batch_size_once}"
        )
    if len(requests) < hparams.batch_size:
        raise ValueError(
            "ULTRAEDIT requires at least one full batch. "
            f"num_requests={len(requests)}, batch_size={hparams.batch_size}"
        )
    if len(requests) % hparams.batch_size != 0:
        raise ValueError(
            "ULTRAEDIT original implementation expects every chunk to have exactly batch_size requests. "
            f"Current num_requests={len(requests)} is not divisible by batch_size={hparams.batch_size}. "
            "Please set --sample_size to a multiple of batch_size or use a dataset size that is already divisible."
        )


def chunk_list(items: List[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def collapse_locality_outputs(
    pre_eval: Optional[Dict[str, Any]],
    post_eval: Optional[Dict[str, Any]],
    request: Dict[str, Any],
    evaluation_type: Optional[str],
) -> None:
    if not pre_eval or not post_eval:
        return
    if "locality" not in post_eval or not post_eval["locality"]:
        return

    for locality_key in request.get("locality", {}).keys():
        output_key = f"{locality_key}_output"
        if output_key not in post_eval["locality"] or output_key not in pre_eval.get("locality", {}):
            continue

        post_output = post_eval["locality"][output_key]
        pre_output = pre_eval["locality"][output_key]

        if evaluation_type == "LLM-judge":
            acc = [float(post_output == pre_output)]
        else:
            if not isinstance(post_output, list):
                post_output = [post_output]
            if not isinstance(pre_output, list):
                pre_output = [pre_output]
            acc = [float(np.mean(np.equal(a, b))) for a, b in zip(post_output, pre_output)]

        post_eval["locality"][f"{locality_key}_acc"] = acc
        post_eval["locality"].pop(output_key, None)

    if "locality" in pre_eval:
        pre_eval.pop("locality", None)


def flatten_numeric_metrics(obj: Any, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_numeric_metrics(value, next_prefix))
        return flat

    if isinstance(obj, (bool, np.bool_)):
        flat[prefix] = float(obj)
        return flat

    if isinstance(obj, (int, float, np.integer, np.floating)):
        flat[prefix] = float(obj)
        return flat

    if isinstance(obj, list) and obj and all(
        isinstance(x, (bool, int, float, np.integer, np.floating, np.bool_)) for x in obj
    ):
        flat[prefix] = float(np.mean(obj))
        return flat

    return flat


def summarize_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_requests": len(metrics),
        "num_batches": len({m["batch_index"] for m in metrics}) if metrics else 0,
    }

    if metrics:
        summary["mean_edit_time_sec"] = float(np.mean([m["edit_time_sec"] for m in metrics]))

    for stage in ("pre", "post"):
        collected: Dict[str, List[float]] = {}
        for metric in metrics:
            if stage not in metric or metric[stage] is None:
                continue
            for key, value in flatten_numeric_metrics(metric[stage]).items():
                collected.setdefault(key, []).append(value)
        if collected:
            summary[stage] = {
                key: float(np.mean(values))
                for key, values in sorted(collected.items())
            }
    return summary


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def save_model_and_tokenizer(
    model: Any,
    tokenizer: Any,
    save_dir: str,
    method_artifact: Optional[Any] = None,
    method_artifact_path: Optional[str] = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model

    if method_artifact is not None:
        if method_artifact_path is not None and hasattr(method_artifact, "save"):
            method_artifact.save(method_artifact_path)
            print(f"[save] method_artifact={method_artifact_path}")
        if hasattr(method_artifact, "model"):
            model_to_save = method_artifact.model

    if not hasattr(model_to_save, "save_pretrained"):
        raise TypeError(
            f"Edited model of type {type(model_to_save)} does not support save_pretrained()."
        )
    model_to_save.save_pretrained(save_dir)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)


def determine_output_dir(
    args: argparse.Namespace,
    alg_name: str,
    model_name: str,
    batch_size: int,
) -> str:
    if args.output_dir is not None:
        return os.path.abspath(args.output_dir)

    return determine_default_model_dir(args, alg_name, model_name, batch_size)


def build_default_run_name(args: argparse.Namespace, model_name: str, batch_size: int) -> str:
    if args.run_name is not None:
        return args.run_name

    model_slug = str(model_name).rstrip("/").split("/")[-1]
    data_stem = Path(args.data_path).stem
    sample_tag = args.sample_size if args.sample_size and args.sample_size > 0 else "all"
    return f"{model_slug}_{data_stem}_bs{batch_size}_n{sample_tag}_seed{args.seed}"


def determine_default_model_dir(
    args: argparse.Namespace,
    alg_name: str,
    model_name: str,
    batch_size: int,
) -> str:
    run_name = build_default_run_name(args, model_name, batch_size)
    return str(PROJECT_ROOT / "outputs" / "Models" / alg_name / run_name)


def determine_save_model_dir(
    args: argparse.Namespace,
    alg_name: str,
    model_name: str,
    batch_size: int,
) -> str:
    if args.save_model_dir is not None:
        return os.path.abspath(args.save_model_dir)
    return determine_default_model_dir(args, alg_name, model_name, batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-wise sequential editing runner for EasyEdit methods such as MEMIT, AlphaEdit, ROME, and WISE."
    )
    parser.add_argument("--editing_method", type=str, default=None, help="Method name, e.g. MEMIT, AlphaEdit, ROME, or WISE.")
    parser.add_argument("--hparams_path", type=str, required=True, help="Path to an EasyEdit hparams yaml.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to editing data json.")
    parser.add_argument("--easyedit_path", type=str, default=None, help="KLOD root or EasyEdit root.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for metrics and manifests. Defaults to outputs/Models/<method>/<run_name>.",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name when --output_dir is omitted.")
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default=None,
        help="Directory to save the edited model. Defaults to outputs/Models/<method>/<run_name>.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override hparams.batch_size.")
    parser.add_argument("--sample_size", type=int, default=None, help="Randomly sample this many requests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id or 'auto'.")
    parser.add_argument("--layers", type=maybe_parse_literal_list, default=None, help="Override hparams.layers.")
    parser.add_argument("--edit_layers", type=maybe_parse_literal_list, default=None, help="Override hparams.edit_layers.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model_name override in hparams.")
    parser.add_argument("--break_loss", type=float, default=None)
    parser.add_argument("--break_prob", type=float, default=None)
    parser.add_argument("--mse_lambda", type=float, default=None)
    parser.add_argument("--context_num", type=int, default=None)
    parser.add_argument("--objective_optimization", type=str, default=None)
    parser.add_argument(
        "--ft_loss_objective",
        type=str,
        default=None,
        help="FT loss objective override: standard_ft or kmtd_prefix_kl.",
    )
    parser.add_argument("--target_alpha", type=float, default=None, help="KMTD target alpha override.")
    parser.add_argument(
        "--rewrite_kl_lambda",
        type=float,
        default=None,
        help="Weight for rewrite-prompt prefix KL when ft_loss_objective=kmtd_prefix_kl.",
    )
    parser.add_argument("--skip_prob", type=float, default=None, help="KMTD active-token probability threshold.")
    parser.add_argument(
        "--pl_skip_above_target",
        type=int,
        choices=[0, 1],
        default=None,
        help="Skip KMTD target positions whose current target probability is already >= skip_prob.",
    )
    parser.add_argument(
        "--append_eos_to_target",
        type=int,
        choices=[0, 1],
        default=1,
        help="Append tokenizer EOS token to each request target_new before editing. Set 0 to disable.",
    )
    parser.add_argument("--p_loc", type=str, default=None, help="Optional AlphaEdit projection path override.")
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Run in-process pre/post evaluation. Default is to skip evaluation and evaluate later from the saved model.",
    )
    parser.add_argument("--skip_eval", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test_generation", action="store_true", help="Enable generation-based fluency eval.")
    parser.add_argument("--eval_metric", type=str, default="exact match", help="Evaluation metric for EasyEdit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    add_easyedit_to_syspath(args.easyedit_path)
    init_easyedit_imports()
    fix_seed(args.seed)

    alg_name = (
        canonicalize_alg_name(args.editing_method)
        if args.editing_method is not None
        else infer_alg_name_from_yaml(args.hparams_path)
    )
    hparams_cls = HPARAMS_REGISTRY[alg_name]
    hparams = hparams_cls.from_hparams(args.hparams_path)

    if not is_runner_supported_method(alg_name):
        raise ValueError(f"{alg_name} is not supported by this runner.")

    apply_hparam_overrides(hparams, args)
    apply_method_runtime_defaults(alg_name, hparams)
    normalize_hparam_paths(hparams, project_root)

    if not hasattr(hparams, "batch_size"):
        raise ValueError(f"{alg_name} hparams does not define batch_size.")
    if hparams.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {hparams.batch_size}")

    requests, raw_count = load_requests(args.data_path)
    if not requests:
        raise ValueError(f"No valid edit requests found in {args.data_path}")

    requests = maybe_sample_requests(requests, args.sample_size, args.seed)
    prepare_method_specific_requests(alg_name, requests)
    validate_method_specific_constraints(alg_name, hparams, requests)
    hparams_model_name = getattr(hparams, "model_name", "unknown_model")
    output_dir = determine_output_dir(
        args,
        alg_name,
        hparams_model_name,
        hparams.batch_size,
    )
    save_model_dir = determine_save_model_dir(
        args,
        alg_name,
        hparams_model_name,
        hparams.batch_size,
    )
    setattr(hparams, "save_model_dir", save_model_dir)
    if alg_name == "WISE" and not getattr(hparams, "save_path", None):
        hparams.save_path = os.path.join(save_model_dir, "wise.pt")
    os.makedirs(output_dir, exist_ok=True)
    run_eval = bool(args.do_eval and not args.skip_eval)
    editor = BaseEditor.from_hparams(hparams)
    append_eos_enabled = bool(args.append_eos_to_target)
    eos_token = None
    eos_appended_count = 0
    if append_eos_enabled:
        requests, eos_token, eos_appended_count = append_eos_to_request_targets(requests, editor.tok)

    print(f"[config] method={alg_name}")
    print(f"[config] model={hparams.model_name}")
    print(f"[config] data_path={args.data_path}")
    print(f"[config] requests={len(requests)} / raw_records={raw_count}")
    print(f"[config] batch_size={hparams.batch_size}")
    print(
        "[config] append_eos_to_target="
        f"{append_eos_enabled}"
        + (f" eos_token={eos_token!r} appended={eos_appended_count}" if append_eos_enabled else "")
    )
    if hasattr(hparams, "ft_loss_objective"):
        print(f"[config] ft_loss_objective={hparams.ft_loss_objective}")
    print(f"[config] output_dir={output_dir}")
    print(f"[config] save_model_dir={save_model_dir}")
    print(f"[config] do_eval={run_eval}")

    evaluation_type = getattr(editor.hparams, "evaluation_type", None)

    all_metrics: List[Dict[str, Any]] = []
    started_at = time.time()
    configured_batch_size = int(hparams.batch_size)
    total_batches = (len(requests) + configured_batch_size - 1) // configured_batch_size
    method_artifact = None

    for batch_index, batch_requests in enumerate(chunk_list(requests, configured_batch_size), start=1):
        case_ids = [req["case_id"] for req in batch_requests]
        print(
            f"[batch {batch_index}/{total_batches}] editing {len(batch_requests)} request(s) "
            f"case_ids={case_ids[:3]}{'...' if len(case_ids) > 3 else ''}"
        )

        pre_batch = [None] * len(batch_requests)
        if run_eval:
            for i, request in enumerate(batch_requests):
                pre_batch[i] = compute_edit_quality(
                    editor.model,
                    editor.model_name,
                    editor.hparams,
                    editor.tok,
                    request,
                    editor.hparams.device,
                    eval_metric=args.eval_metric,
                    test_generation=args.test_generation,
                )

        edit_start = time.perf_counter()
        if alg_name == "WISE":
            editor.hparams.batch_size = len(batch_requests)
            if hasattr(editor.model, "config"):
                editor.model.config.batch_size = len(batch_requests)
        edited_model, _ = editor.apply_algo(
            editor.model,
            editor.tok,
            batch_requests,
            editor.hparams,
            copy=False,
            return_orig_weights=False,
            keep_original_weight=False,
        )
        edit_time = time.perf_counter() - edit_start
        edited_model_for_eval = edited_model
        if alg_name == "WISE" and hasattr(edited_model, "model"):
            method_artifact = edited_model
            editor.model = edited_model.model
        else:
            editor.model = edited_model

        for request_index, request in enumerate(batch_requests):
            post_eval = None
            if run_eval:
                post_eval = compute_edit_quality(
                    edited_model_for_eval,
                    editor.model_name,
                    editor.hparams,
                    editor.tok,
                    request,
                    editor.hparams.device,
                    eval_metric=args.eval_metric,
                    test_generation=args.test_generation,
                )
                collapse_locality_outputs(pre_batch[request_index], post_eval, request, evaluation_type)

            all_metrics.append(
                {
                    "case_id": request["case_id"],
                    "batch_index": batch_index,
                    "index_in_batch": request_index,
                    "requested_rewrite": request,
                    "edit_time_sec": edit_time,
                    "pre": pre_batch[request_index],
                    "post": post_eval,
                }
            )

        print(
            f"[batch {batch_index}/{total_batches}] done in {edit_time:.2f}s "
            f"(cumulative requests={len(all_metrics)})"
        )

    hparams.batch_size = configured_batch_size
    if alg_name == "WISE" and hasattr(editor.model, "config"):
        editor.model.config.batch_size = configured_batch_size

    total_elapsed = time.time() - started_at
    summary = summarize_metrics(all_metrics)
    summary["total_elapsed_sec"] = total_elapsed

    metrics_path = os.path.join(output_dir, "metrics.json")
    requests_path = os.path.join(output_dir, "requests.json")
    summary_path = os.path.join(output_dir, "summary.json")
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    method_artifact_path = getattr(hparams, "save_path", None) if alg_name == "WISE" else None

    manifest = {
        "editing_method": alg_name,
        "hparams_path": os.path.abspath(args.hparams_path),
        "data_path": os.path.abspath(args.data_path),
        "raw_record_count": raw_count,
        "num_requests_used": len(requests),
        "batch_size": hparams.batch_size,
        "seed": args.seed,
        "append_eos_to_target": append_eos_enabled,
        "eos_token": eos_token,
        "eos_appended_count": eos_appended_count,
        "do_eval": run_eval,
        "test_generation": args.test_generation,
        "eval_metric": args.eval_metric,
        "output_dir": output_dir,
        "save_model_dir": save_model_dir,
        "method_artifact_path": method_artifact_path,
        "total_elapsed_sec": total_elapsed,
        "effective_hparams": vars(hparams),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=json_default)
    with open(requests_path, "w", encoding="utf-8") as f:
        json.dump(requests, f, ensure_ascii=False, indent=2, default=json_default)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=json_default)

    print(f"[save] writing edited model to {save_model_dir}")
    save_model_and_tokenizer(
        editor.model,
        editor.tok,
        save_model_dir,
        method_artifact=method_artifact,
        method_artifact_path=method_artifact_path,
    )

    print(f"[done] metrics: {metrics_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] saved_model: {save_model_dir}")


if __name__ == "__main__":
    main()
