#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

CALLER_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -gt 0 ]]; then
  NORMALIZED_ARGS=()
  for arg in "$@"; do
    if [[ "$arg" == /* ]]; then
      NORMALIZED_ARGS+=("$arg")
    else
      NORMALIZED_ARGS+=("$CALLER_DIR/$arg")
    fi
  done
  set -- "${NORMALIZED_ARGS[@]}"
fi
cd "$SCRIPT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [[ -x "python" ]]; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python executable not found. Set PYTHON_BIN explicitly."
    exit 1
  fi
fi

EVAL_SCRIPT="${EVAL_SCRIPT:-$SCRIPT_DIR/evaluate/eval_hf_easyedit.py}"
METHOD="${METHOD:-KLOD}"
case "$METHOD" in
  KLOD|klod)
    METHOD_DIR="KLOD_150k"
    ;;
  LocFT-BF|locft-bf|locft_bf|locft)
    METHOD_DIR="LocFT-BF_150k"
    ;;
  *)
    echo "Unknown METHOD='$METHOD'. Use METHOD=KLOD or METHOD=LocFT-BF."
    exit 1
    ;;
esac
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/outputs/Models/$METHOD_DIR}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$SCRIPT_DIR/outputs/evaluation}"
RESULT_ROOT="${RESULT_ROOT:-$EVALUATION_ROOT/eval_results_easyedit_150k/$METHOD_DIR}"
REQUEST_SLICE_ROOT="${REQUEST_SLICE_ROOT:-$EVALUATION_ROOT/request_slices_150k/$METHOD_DIR}"
PRE_CACHE_DIR="${PRE_CACHE_DIR:-$RESULT_ROOT/_pre_cache}"

CHECKPOINT_COUNTS="${CHECKPOINT_COUNTS:-3k,5k,10k,20k,50k,100k,150k}"
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
VERBOSE="${VERBOSE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING_EVALS="${SKIP_EXISTING_EVALS:-0}"
FORCE_RECOMPUTE_PRE="${FORCE_RECOMPUTE_PRE:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-auto}"
USE_PREFIX_PRE_CACHE="${USE_PREFIX_PRE_CACHE:-1}"
PREFIX_PRE_CACHE_COUNT="${PREFIX_PRE_CACHE_COUNT:-150k}"

EVAL_GPUS="${EVAL_GPUS:-$CUDA_VISIBLE_DEVICES}"
PARALLEL_EVALS="${PARALLEL_EVALS:-auto}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"
GPU_IDLE_UTIL="${GPU_IDLE_UTIL:-10}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-60}"
JOB_POLL_INTERVAL="${JOB_POLL_INTERVAL:-5}"

declare -A DATA_PATHS=(
  ["counterfact-edit"]="$SCRIPT_DIR/data/counterfact/counterfact_3k.json"
  ["zsre-mend-eval"]="$SCRIPT_DIR/data/zsre/zsre_3k.json"
)

normalize_path_part() {
  printf '%s' "$1" | sed -E 's#^/##; s#[^[:alnum:]._-]+#_#g; s#_+#_#g; s#^_+##; s#_+$##'
}

parse_count() {
  local raw="$1"
  local number

  raw="${raw#checkpoint_}"
  case "$raw" in
    *[kK])
      number="${raw%[kK]}"
      [[ "$number" =~ ^[0-9]+$ ]] || return 1
      echo $((number * 1000))
      ;;
    *)
      [[ "$raw" =~ ^[0-9]+$ ]] || return 1
      echo "$raw"
      ;;
  esac
}

count_label() {
  local count="$1"
  if [[ "$count" =~ ^[0-9]+$ && $((count % 1000)) -eq 0 ]]; then
    echo "$((count / 1000))k"
  else
    echo "$count"
  fi
}

resolve_dataset_name() {
  local text
  text="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"

  if [[ "$text" == *counterfact* ]]; then
    echo "counterfact-edit"
  elif [[ "$text" == *zsre* ]]; then
    echo "zsre-mend-eval"
  else
    return 1
  fi
}

resolve_base_model_path_for_run() {
  local run_dir="$1"
  local model_dir="$2"

  if [[ -n "$BASE_MODEL_PATH" && "$BASE_MODEL_PATH" != "auto" ]]; then
    echo "$BASE_MODEL_PATH"
    return 0
  fi
  if [[ -z "$BASE_MODEL_PATH" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - "$run_dir" "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
model_dir = Path(sys.argv[2])

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


def read_key_path(payload, key_path):
    cur = payload
    for key in key_path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, str) and cur.strip() else None


def usable_model_ref(value):
    value = value.strip()
    lower = value.lower()
    if lower in {"none", "null"}:
        return False

    parts = Path(value).parts
    if any(part.startswith("checkpoint_") for part in parts):
        return False
    return True


for manifest_path in manifest_paths:
    if not manifest_path.exists():
        continue
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        continue
    for key_path in key_paths:
        value = read_key_path(payload, key_path)
        if value and usable_model_ref(value):
            print(value)
            raise SystemExit(0)

raise SystemExit(0)
PY
}

eval_record_matches_base_model() {
  local path="$1"
  local expected_base="$2"

  [[ -s "$path" ]] || return 1
  if [[ -z "$expected_base" ]]; then
    has_existing_eval_record "$path"
    return $?
  fi

  "$PYTHON_BIN" - "$path" "$expected_base" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
actual_base = summary.get("base_model_path")
if actual_base != sys.argv[2]:
    raise SystemExit(1)
PY
}

find_prefix_pre_cache_path() {
  local dataset_name="$1"
  local rel_run_dir="$2"
  local base_model_path="$3"
  local prefix_label="$4"
  local base_label
  local filename
  local candidate
  local rel_run_basename

  base_label="$(normalize_path_part "${base_model_path##*/}")"
  filename="${base_label}__checkpoint_${prefix_label}__training_requests_prefix_${prefix_label}__trained_requests.json"
  rel_run_basename="$(basename "$rel_run_dir")"

  for candidate in \
    "$PRE_CACHE_DIR/$dataset_name/$rel_run_dir/$filename" \
    "$PRE_CACHE_DIR/$dataset_name/$rel_run_basename/$filename" \
    "$PRE_CACHE_DIR/$dataset_name/$filename"
  do
    if [[ -s "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

has_existing_eval_record() {
  local path="$1"

  [[ -s "$path" ]] || return 1

  "$PYTHON_BIN" - "$path" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    json.load(f)
PY
}

gpu_is_idle() {
  local gpu="$1"
  local mem
  local util
  local pids

  if [[ "$WAIT_FOR_FREE_GPUS" != "1" || "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    if [[ "${NVIDIA_SMI_WARNED:-0}" != "1" ]]; then
      echo "[gpu] nvidia-smi not found; skipping external idle checks."
      NVIDIA_SMI_WARNED=1
    fi
    return 0
  fi

  if ! read -r mem util < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{print $1, $2}'
  ); then
    echo "[gpu] Could not query GPU $gpu; treating it as busy."
    return 1
  fi

  pids="$(
    nvidia-smi -i "$gpu" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
  )"

  mem="${mem//[^0-9]/}"
  util="${util//[^0-9]/}"
  mem="${mem:-0}"
  util="${util:-0}"

  if [[ -n "$pids" || "$mem" -gt "$GPU_IDLE_MEM_MB" || "$util" -gt "$GPU_IDLE_UTIL" ]]; then
    echo "[gpu] GPU $gpu busy: mem=${mem}MiB util=${util}% pids=${pids:-none}"
    return 1
  fi

  return 0
}

make_request_slice() {
  local source_path="$1"
  local count="$2"
  local output_path="$3"

  "$PYTHON_BIN" - "$source_path" "$count" "$output_path" <<'PY'
import json
import os
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
count = int(sys.argv[2])
output_path = Path(sys.argv[3])

with source_path.open("r", encoding="utf-8") as f:
    requests = json.load(f)

if not isinstance(requests, list):
    raise ValueError(f"Expected JSON list: {source_path}")

used_count = min(count, len(requests))
if used_count < count:
    print(
        f"[slice][warn] requested first {count} records, but {source_path} has only {len(requests)}; "
        f"using {used_count}.",
        file=sys.stderr,
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
with tmp_path.open("w", encoding="utf-8") as f:
    json.dump(requests[:used_count], f, ensure_ascii=False, indent=2)
    f.write("\n")
os.replace(tmp_path, output_path)
print(used_count)
PY
}

resolve_requests_path() {
  local run_dir="$1"

  if [[ -f "$run_dir/training_requests.json" ]]; then
    echo "$run_dir/training_requests.json"
  elif [[ -f "$run_dir/requests.json" ]]; then
    echo "$run_dir/requests.json"
  else
    return 1
  fi
}

discover_entries_for_run() {
  local run_dir="$1"
  local cp_dir
  local cp_name
  local count
  local label
  local requests_path

  requests_path="$(resolve_requests_path "$run_dir")" || {
    echo "[skip] missing training_requests.json/requests.json: $run_dir" >&2
    return 0
  }

  while IFS= read -r -d '' cp_dir; do
    [[ -f "$cp_dir/config.json" ]] || continue
    cp_name="$(basename "$cp_dir")"
    count="$(parse_count "$cp_name")" || continue
    label="$(count_label "$count")"
    printf '%s\t%s\t%s\t%s\n' "$count" "$label" "$run_dir" "$cp_dir"
  done < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' -print0)

  if [[ "$INCLUDE_FINAL_MODEL" == "1" && -f "$run_dir/config.json" ]]; then
    count="$("$PYTHON_BIN" - "$requests_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

if not isinstance(payload, list):
    raise ValueError(f"Expected JSON list: {sys.argv[1]}")
print(len(payload))
PY
)"
    label="$(count_label "$count")"
    printf '%s\t%s\t%s\t%s\n' "$count" "$label" "$run_dir" "$run_dir"
  fi
}

count_is_requested() {
  local count="$1"
  local raw
  local requested_count

  if [[ "$CHECKPOINT_COUNTS" == "all" || "$CHECKPOINT_COUNTS" == "*" ]]; then
    return 0
  fi

  IFS=',' read -r -a RAW_COUNTS <<< "$CHECKPOINT_COUNTS"
  for raw in "${RAW_COUNTS[@]}"; do
    raw="${raw//[[:space:]]/}"
    [[ -n "$raw" ]] || continue
    requested_count="$(parse_count "$raw")" || {
      echo "Invalid CHECKPOINT_COUNTS entry: $raw" >&2
      return 2
    }
    if [[ "$requested_count" == "$count" ]]; then
      return 0
    fi
  done

  return 1
}

build_entries() {
  local arg
  local run_dir
  local count
  local label
  local entry_run_dir
  local model_dir

  if [[ "$#" -gt 0 ]]; then
    for arg in "$@"; do
      if [[ ! -d "$arg" ]]; then
        echo "Model/run directory not found: $arg" >&2
        return 1
      fi

      if [[ "$(basename "$arg")" == checkpoint_* ]]; then
        model_dir="$arg"
        run_dir="$(dirname "$model_dir")"
        if [[ -n "$MODEL_FILTER" && "$model_dir" != *"$MODEL_FILTER"* && "$run_dir" != *"$MODEL_FILTER"* ]]; then
          continue
        fi

        [[ -f "$model_dir/config.json" ]] || {
          echo "[skip] missing config.json: $model_dir" >&2
          continue
        }
        resolve_requests_path "$run_dir" >/dev/null || {
          echo "[skip] missing training_requests.json/requests.json: $run_dir" >&2
          continue
        }

        count="$(parse_count "$(basename "$model_dir")")" || continue
        label="$(count_label "$count")"
        if count_is_requested "$count"; then
          printf '%s\t%s\t%s\t%s\n' "$count" "$label" "$run_dir" "$model_dir"
        fi
        continue
      fi

      run_dir="$arg"
      if [[ -n "$MODEL_FILTER" && "$run_dir" != *"$MODEL_FILTER"* ]]; then
        continue
      fi

      while IFS=$'\t' read -r count label entry_run_dir model_dir; do
        [[ -n "${count:-}" ]] || continue
        if count_is_requested "$count"; then
          printf '%s\t%s\t%s\t%s\n' "$count" "$label" "$entry_run_dir" "$model_dir"
        fi
      done < <(discover_entries_for_run "$run_dir")
    done
    return 0
  fi

  while IFS= read -r run_dir; do
    [[ -n "$run_dir" ]] || continue
    if [[ -n "$MODEL_FILTER" && "$run_dir" != *"$MODEL_FILTER"* ]]; then
      continue
    fi

    while IFS=$'\t' read -r count label entry_run_dir model_dir; do
      [[ -n "${count:-}" ]] || continue
      if count_is_requested "$count"; then
        printf '%s\t%s\t%s\t%s\n' "$count" "$label" "$entry_run_dir" "$model_dir"
      fi
    done < <(discover_entries_for_run "$run_dir")
  done < <(find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d -print)
}

run_eval_entry() {
  local count="$1"
  local label="$2"
  local run_dir="$3"
  local model_dir="$4"

  local dataset_name
  local data_path
  local source_requests_path
  local rel_model_dir
  local rel_run_dir
  local slice_dir
  local sliced_requests_path
  local used_count
  local save_dir
  local save_path
  local log_path
  local pre_cache_run_dir
  local base_model_path
  local prefix_count
  local prefix_label
  local prefix_pre_cache_path
  local -a cmd

  dataset_name="$(resolve_dataset_name "$run_dir")" || {
    echo "Could not infer dataset from run directory: $run_dir"
    return 1
  }
  data_path="${DATA_PATHS[$dataset_name]}"
  if [[ ! -f "$data_path" ]]; then
    echo "Data file not found: $data_path"
    return 1
  fi

  source_requests_path="$(resolve_requests_path "$run_dir")" || {
    echo "Requests file not found under run directory: $run_dir"
    return 1
  }

  rel_model_dir="${model_dir#$MODEL_ROOT/}"
  if [[ "$rel_model_dir" == "$model_dir" ]]; then
    rel_model_dir="$(normalize_path_part "$model_dir")"
  fi

  rel_run_dir="${run_dir#$MODEL_ROOT/}"
  if [[ "$rel_run_dir" == "$run_dir" ]]; then
    rel_run_dir="$(normalize_path_part "$run_dir")"
  fi

  slice_dir="$REQUEST_SLICE_ROOT/$rel_run_dir/checkpoint_$label"
  sliced_requests_path="$slice_dir/training_requests_prefix_$label.json"

  save_dir="$RESULT_ROOT/$dataset_name/$rel_model_dir"
  save_path="$save_dir/eval.json"
  log_path="$save_dir/eval.log"
  pre_cache_run_dir="$PRE_CACHE_DIR/$dataset_name/$rel_run_dir"
  base_model_path="$(resolve_base_model_path_for_run "$run_dir" "$model_dir")"

  prefix_pre_cache_path=""
  if [[ "$USE_PREFIX_PRE_CACHE" == "1" && "$FORCE_RECOMPUTE_PRE" != "1" && -n "$base_model_path" ]]; then
    prefix_count="$(parse_count "$PREFIX_PRE_CACHE_COUNT")" || {
      echo "Invalid PREFIX_PRE_CACHE_COUNT entry: $PREFIX_PRE_CACHE_COUNT" >&2
      return 1
    }
    if [[ "$count" -gt "$prefix_count" ]]; then
      prefix_label="$(count_label "$prefix_count")"
      prefix_pre_cache_path="$(find_prefix_pre_cache_path "$dataset_name" "$rel_run_dir" "$base_model_path" "$prefix_label" || true)"
    fi
  fi

  if [[ "$SKIP_EXISTING_EVALS" == "1" ]] && has_existing_eval_record "$save_path"; then
    if eval_record_matches_base_model "$save_path" "$base_model_path"; then
      echo
      echo "Dataset : $dataset_name"
      echo "Model   : $model_dir"
      echo "Base    : ${base_model_path:-auto}"
      echo "Skip    : existing eval record found at $save_path"
      return 0
    fi
    echo
    echo "Dataset : $dataset_name"
    echo "Model   : $model_dir"
    echo "Base    : ${base_model_path:-auto}"
    echo "Rerun   : existing eval record uses a different/missing base_model_path"
  fi

  mkdir -p "$save_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    used_count="$count"
  else
    used_count="$(make_request_slice "$source_requests_path" "$count" "$sliced_requests_path")"
  fi

  cmd=(
    "$PYTHON_BIN" "$EVAL_SCRIPT"
    --data_path "$data_path"
    --requests_path "$sliced_requests_path"
    --model_path "$model_dir"
    --num_samples "$used_count"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --save_path "$save_path"
    --pre_cache_dir "$pre_cache_run_dir"
  )

  if [[ -n "$base_model_path" ]]; then
    cmd+=(--base_model_path "$base_model_path")
  fi
  if [[ -n "$prefix_pre_cache_path" ]]; then
    cmd+=(--pre_cache_prefix_path "$prefix_pre_cache_path")
  fi
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi
  if [[ "$VERBOSE" == "1" ]]; then
    cmd+=(--verbose)
  fi
  if [[ "$FORCE_RECOMPUTE_PRE" == "1" ]]; then
    cmd+=(--force_recompute_pre)
  fi

  echo
  echo "Dataset : $dataset_name"
  echo "Run     : $run_dir"
  echo "Model   : $model_dir"
  echo "Base    : ${base_model_path:-auto}"
  echo "Requests: $sliced_requests_path (first $used_count from $source_requests_path)"
  echo "PreCache: $pre_cache_run_dir"
  if [[ -n "$prefix_pre_cache_path" ]]; then
    echo "PrePref : $prefix_pre_cache_path"
  elif [[ "$USE_PREFIX_PRE_CACHE" == "1" && -n "${prefix_count:-}" && "$count" -gt "$prefix_count" && "$FORCE_RECOMPUTE_PRE" != "1" ]]; then
    echo "PrePref : not found for prefix checkpoint_$(count_label "$prefix_count"); full pre cache will be computed if needed"
  fi
  echo "Save to : $save_path"
  printf 'Command :'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "$log_path"
}

active_job_count() {
  local count=0
  local gpu

  for gpu in "${EVAL_GPU_LIST[@]}"; do
    if [[ -n "${ACTIVE_PIDS_BY_GPU[$gpu]:-}" ]]; then
      count=$((count + 1))
    fi
  done

  echo "$count"
}

reap_finished_jobs() {
  local gpu
  local pid
  local status_file
  local wait_status
  local status_exit_code
  local status_desc

  for gpu in "${EVAL_GPU_LIST[@]}"; do
    pid="${ACTIVE_PIDS_BY_GPU[$gpu]:-}"
    if [[ -z "$pid" ]]; then
      continue
    fi

    status_file="${ACTIVE_STATUS_BY_PID[$pid]}"
    if [[ ! -f "$status_file" ]]; then
      continue
    fi

    if wait "$pid"; then
      wait_status=0
    else
      wait_status=$?
    fi

    status_exit_code="$wait_status"
    status_desc="${ACTIVE_DESC_BY_PID[$pid]:-}"
    while IFS='=' read -r key value; do
      case "$key" in
        exit_code)
          status_exit_code="$value"
          ;;
      esac
    done < "$status_file"

    unset "ACTIVE_PIDS_BY_GPU[$gpu]"
    unset "ACTIVE_STATUS_BY_PID[$pid]"
    unset "ACTIVE_DESC_BY_PID[$pid]"

    if [[ "$status_exit_code" -ne 0 ]]; then
      failed_jobs=$((failed_jobs + 1))
      echo "[worker gpu=$gpu] Failed with exit code $status_exit_code: $status_desc"
    else
      echo "[worker gpu=$gpu] Finished: $status_desc"
    fi
  done
}

wait_for_free_gpu_slot() {
  local gpu

  FREE_GPU=""
  while true; do
    reap_finished_jobs

    if [[ "$(active_job_count)" -lt "$MAX_PARALLEL_EVALS" ]]; then
      for gpu in "${EVAL_GPU_LIST[@]}"; do
        if [[ -n "${ACTIVE_PIDS_BY_GPU[$gpu]:-}" ]]; then
          continue
        fi

        if gpu_is_idle "$gpu"; then
          FREE_GPU="$gpu"
          return 0
        fi
      done
    fi

    if [[ "$(active_job_count)" -ge "$MAX_PARALLEL_EVALS" ]]; then
      sleep "$JOB_POLL_INTERVAL"
    else
      echo "[gpu] No configured GPU is idle; sleeping ${GPU_WAIT_INTERVAL}s..."
      sleep "$GPU_WAIT_INTERVAL"
    fi
  done
}

wait_for_all_jobs() {
  while [[ "$(active_job_count)" -gt 0 ]]; do
    reap_finished_jobs
    if [[ "$(active_job_count)" -gt 0 ]]; then
      sleep "$JOB_POLL_INTERVAL"
    fi
  done
}

run_eval_worker() {
  set +e

  local gpu="$1"
  local status_file="$2"
  local count="$3"
  local label="$4"
  local run_dir="$5"
  local model_dir="$6"
  local exit_code=0

  export CUDA_VISIBLE_DEVICES="$gpu"
  trap 'exit_code=$?; if [[ ! -f "$status_file" ]]; then { echo "exit_code=$exit_code"; echo "gpu=$gpu"; echo "model_dir=$model_dir"; } > "$status_file"; fi' EXIT

  echo
  echo "[worker gpu=$gpu] ============================================================"
  echo "[worker gpu=$gpu] Evaluating $label checkpoint"
  echo "[worker gpu=$gpu] Run  : $run_dir"
  echo "[worker gpu=$gpu] Model: $model_dir"
  echo "[worker gpu=$gpu] ============================================================"

  run_eval_entry "$count" "$label" "$run_dir" "$model_dir"
  exit_code=$?

  {
    echo "exit_code=$exit_code"
    echo "gpu=$gpu"
    echo "model_dir=$model_dir"
  } > "$status_file"

  return "$exit_code"
}

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Evaluation script not found: $EVAL_SCRIPT"
  exit 1
fi

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "Model root not found: $MODEL_ROOT"
  exit 1
fi

EVAL_GPUS="${EVAL_GPUS//[[:space:]]/}"
IFS=',' read -r -a EVAL_GPU_LIST <<< "$EVAL_GPUS"
if [[ "${#EVAL_GPU_LIST[@]}" -eq 0 || -z "${EVAL_GPU_LIST[0]}" ]]; then
  EVAL_GPU_LIST=("0")
fi

if [[ "$PARALLEL_EVALS" == "auto" ]]; then
  MAX_PARALLEL_EVALS="${#EVAL_GPU_LIST[@]}"
else
  MAX_PARALLEL_EVALS="$PARALLEL_EVALS"
fi
if ! [[ "$MAX_PARALLEL_EVALS" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL_EVALS" -lt 1 ]]; then
  echo "PARALLEL_EVALS must be 'auto' or a positive integer: $PARALLEL_EVALS"
  exit 1
fi
if [[ "$MAX_PARALLEL_EVALS" -gt "${#EVAL_GPU_LIST[@]}" ]]; then
  MAX_PARALLEL_EVALS="${#EVAL_GPU_LIST[@]}"
fi

mapfile -t EVAL_ENTRIES < <(build_entries "$@" | sort -n -k1,1 -k4,4)

if [[ "${#EVAL_ENTRIES[@]}" -eq 0 ]]; then
  echo "No checkpoint model directories found under: $MODEL_ROOT"
  echo "Set MODEL_FILTER/CHECKPOINT_COUNTS or pass run/checkpoint directories explicitly if needed."
  exit 1
fi

echo "Found ${#EVAL_ENTRIES[@]} checkpoint model(s) to evaluate."
echo "Model root          : $MODEL_ROOT"
echo "Method              : $METHOD_DIR"
echo "Evaluation root     : $EVALUATION_ROOT"
echo "Results             : $RESULT_ROOT"
echo "Request slices      : $REQUEST_SLICE_ROOT"
echo "Checkpoint counts   : $CHECKPOINT_COUNTS"
echo "Include final model : $INCLUDE_FINAL_MODEL"
echo "Eval script         : $EVAL_SCRIPT"
echo "Skip existing evals : $SKIP_EXISTING_EVALS"
echo "Eval GPUs           : ${EVAL_GPU_LIST[*]}"
echo "Parallel evals      : $MAX_PARALLEL_EVALS"
echo "Wait for free GPUs  : $WAIT_FOR_FREE_GPUS"
if [[ "$WAIT_FOR_FREE_GPUS" == "1" ]]; then
  echo "GPU idle threshold  : mem<=${GPU_IDLE_MEM_MB}MiB util<=${GPU_IDLE_UTIL}% and no compute pids"
fi

failed_jobs=0
job_id=0

declare -A ACTIVE_PIDS_BY_GPU=()
declare -A ACTIVE_STATUS_BY_PID=()
declare -A ACTIVE_DESC_BY_PID=()

RUN_STATUS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kledit_eval_150k_status.XXXXXX")"
mkdir -p "$RUN_STATUS_DIR"
cleanup_run_status_dir() {
  local pid

  for pid in "${ACTIVE_PIDS_BY_GPU[@]}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done

  if [[ -n "${RUN_STATUS_DIR:-}" && "$RUN_STATUS_DIR" == "${TMPDIR:-/tmp}/kledit_eval_150k_status."* ]]; then
    rm -rf "$RUN_STATUS_DIR"
  fi
}
trap cleanup_run_status_dir EXIT

for entry in "${EVAL_ENTRIES[@]}"; do
  IFS=$'\t' read -r count label run_dir model_dir <<< "$entry"

  wait_for_free_gpu_slot
  job_id=$((job_id + 1))
  status_file="$RUN_STATUS_DIR/job_${job_id}.status"

  run_eval_worker "$FREE_GPU" "$status_file" "$count" "$label" "$run_dir" "$model_dir" &
  worker_pid=$!
  ACTIVE_PIDS_BY_GPU["$FREE_GPU"]="$worker_pid"
  ACTIVE_STATUS_BY_PID["$worker_pid"]="$status_file"
  ACTIVE_DESC_BY_PID["$worker_pid"]="$model_dir"
  echo "[scheduler] Started PID $worker_pid on GPU $FREE_GPU"
done

wait_for_all_jobs

echo
echo "Evaluated checkpoint jobs: ${#EVAL_ENTRIES[@]}"
echo "Failed jobs             : $failed_jobs"
if [[ "$failed_jobs" -ne 0 ]]; then
  echo "Some evaluations failed. Check the eval.log files above."
  exit 1
fi
echo "All evaluations finished."
