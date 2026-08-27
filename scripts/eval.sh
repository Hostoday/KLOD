#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python executable not found. Set PYTHON_BIN explicitly."
    exit 1
  fi
fi

EVAL_SCRIPT="${EVAL_SCRIPT:-$SCRIPT_DIR/evaluate/eval_hf_easyedit.py}"
MODEL_ROOT="${MODEL_ROOT:-${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/Models/}}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$SCRIPT_DIR/outputs/evaluation}"
RESULT_ROOT="${RESULT_ROOT:-$EVALUATION_ROOT/eval_results_easyedit}"
COMPLETED_ROOT="${COMPLETED_ROOT:-$EVALUATION_ROOT}"
PRE_CACHE_DIR="${PRE_CACHE_DIR:-$RESULT_ROOT/_pre_cache}"
NUM_SAMPLES="${NUM_SAMPLES:-3000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
VERBOSE="${VERBOSE:-1}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
METHOD_FILTER="${METHOD_FILTER:-KLOD}"
MATCH_DATASETS_FROM_NAME="${MATCH_DATASETS_FROM_NAME:-1}"
MOVE_COMPLETED_MODELS="${MOVE_COMPLETED_MODELS:-0}"
SKIP_EXISTING_EVALS="${SKIP_EXISTING_EVALS:-1}"
USE_MODEL_REQUESTS="${USE_MODEL_REQUESTS:-0}"
EVAL_GPUS="${EVAL_GPUS:-$CUDA_VISIBLE_DEVICES}"
PARALLEL_EVALS="${PARALLEL_EVALS:-auto}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"
GPU_IDLE_UTIL="${GPU_IDLE_UTIL:-10}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-60}"
JOB_POLL_INTERVAL="${JOB_POLL_INTERVAL:-5}"

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

declare -A DATA_PATHS=(
  ["counterfact-edit"]="$SCRIPT_DIR/data/counterfact/counterfact_3k.json"
  ["zsre-mend-eval"]="$SCRIPT_DIR/data/zsre/zsre_3k.json"
)

declare -A DATASET_ALIASES=(
  ["counterfact-edit"]="counterfact_3k counterfact"
  ["zsre-mend-eval"]="zsre_3k zsre zsre-mend-eval"
)

resolve_dataset_name() {
  case "$1" in
    counterfact|counterfact-edit)
      echo "counterfact-edit"
      ;;
    zsre|zsre-mend-eval)
      echo "zsre-mend-eval"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Evaluation script not found: $EVAL_SCRIPT"
  exit 1
fi

supports_eval_arg() {
  local arg="$1"
  grep -Eq "[\"']$arg[\"']" "$EVAL_SCRIPT"
}

EVAL_SUPPORTS_PRE_CACHE_DIR=0
EVAL_SUPPORTS_REQUESTS_PATH=0
if supports_eval_arg "--pre_cache_dir"; then
  EVAL_SUPPORTS_PRE_CACHE_DIR=1
fi
if supports_eval_arg "--requests_path"; then
  EVAL_SUPPORTS_REQUESTS_PATH=1
fi

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "Model root not found: $MODEL_ROOT"
  exit 1
fi

if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a RAW_DATASETS <<< "$DATASETS"
else
  RAW_DATASETS=("counterfact-edit" "zsre-mend-eval")
fi

REQUESTED_DATASET_NAMES=()
for raw_dataset in "${RAW_DATASETS[@]}"; do
  dataset_name="$(resolve_dataset_name "$raw_dataset")" || {
    echo "Unsupported dataset: $raw_dataset"
    echo "Choose from: counterfact-edit, zsre-mend-eval"
    exit 1
  }

  if [[ -z "${DATA_PATHS[$dataset_name]+x}" ]]; then
    echo "Dataset key is not configured in DATA_PATHS: $dataset_name"
    echo "Configured dataset keys: ${!DATA_PATHS[*]}"
    exit 1
  fi
  data_path="${DATA_PATHS[$dataset_name]}"
  if [[ ! -f "$data_path" ]]; then
    echo "Data file not found: $data_path"
    exit 1
  fi

  REQUESTED_DATASET_NAMES+=("$dataset_name")
done

normalize_match_text() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^[:alnum:]]+/ /g; s/^ +//; s/ +$//; s/ +/ /g'
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

select_datasets_for_model() {
  local model_dir="$1"
  local model_name_norm
  local dataset_name
  local alias
  local alias_norm
  local -a matches=()

  if [[ "$MATCH_DATASETS_FROM_NAME" != "1" ]]; then
    printf '%s\n' "${REQUESTED_DATASET_NAMES[@]}"
    return 0
  fi

  model_name_norm="$(normalize_match_text "$(basename "$model_dir")")"

  for dataset_name in "${REQUESTED_DATASET_NAMES[@]}"; do
    for alias in ${DATASET_ALIASES[$dataset_name]}; do
      alias_norm="$(normalize_match_text "$alias")"
      if [[ -n "$alias_norm" && " $model_name_norm " == *" $alias_norm "* ]]; then
        matches+=("$dataset_name")
        break
      fi
    done
  done

  if [[ "${#matches[@]}" -eq 0 ]]; then
    return 1
  fi

  printf '%s\n' "${matches[@]}" | sort -u
}

MODEL_DIRS=()
if [[ "$#" -gt 0 ]]; then
  for model_dir in "$@"; do
    if [[ ! -d "$model_dir" ]]; then
      echo "Model directory not found: $model_dir"
      exit 1
    fi

    if [[ ! -f "$model_dir/config.json" ]]; then
      echo "Skipping non-HF directory (missing config.json): $model_dir"
      continue
    fi

    model_dir="$(cd "$model_dir" && pwd -P)"
    method="$(basename "$(dirname "$model_dir")")"

    if [[ -n "$MODEL_FILTER" && "$model_dir" != *"$MODEL_FILTER"* ]]; then
      continue
    fi

    if [[ -n "$METHOD_FILTER" && "$method" != *"$METHOD_FILTER"* ]]; then
      continue
    fi

    MODEL_DIRS+=("$model_dir")
  done
else
  while IFS= read -r -d '' config_file; do
    model_dir="$(dirname "$config_file")"
    method="$(basename "$(dirname "$model_dir")")"

    if [[ "$model_dir" == "$RESULT_ROOT"* ]]; then
      continue
    fi

    if [[ -n "$MODEL_FILTER" && "$model_dir" != *"$MODEL_FILTER"* ]]; then
      continue
    fi

    if [[ -n "$METHOD_FILTER" && "$method" != *"$METHOD_FILTER"* ]]; then
      continue
    fi

    MODEL_DIRS+=("$model_dir")
  done < <(find "$MODEL_ROOT" -type f -name config.json -print0)
fi

if [[ "${#MODEL_DIRS[@]}" -eq 0 ]]; then
  echo "No model directories found under: $MODEL_ROOT"
  echo "Set MODEL_FILTER or pass model directories explicitly if needed."
  exit 1
fi

mapfile -t MODEL_DIRS < <(printf '%s\n' "${MODEL_DIRS[@]}" | sort -u)

mkdir -p "$RESULT_ROOT"

run_eval() {
  local dataset_name="$1"
  local model_dir="$2"

  if [[ -z "${DATA_PATHS[$dataset_name]+x}" ]]; then
    echo "Dataset key is not configured in DATA_PATHS: $dataset_name"
    echo "Configured dataset keys: ${!DATA_PATHS[*]}"
    return 1
  fi

  local data_path="${DATA_PATHS[$dataset_name]}"
  local rel_model_dir
  local save_dir
  local save_path
  local log_path
  local requests_path
  local use_model_requests=0
  local -a cmd

  rel_model_dir="${model_dir#$MODEL_ROOT/}"
  if [[ "$rel_model_dir" == "$model_dir" ]]; then
    rel_model_dir="$(basename "$model_dir")"
  fi

  save_dir="$RESULT_ROOT/$dataset_name/$rel_model_dir"
  save_path="$save_dir/eval.json"
  log_path="$save_dir/eval.log"

  if [[ "$SKIP_EXISTING_EVALS" == "1" ]] && has_existing_eval_record "$save_path"; then
    echo
    echo "Dataset : $dataset_name"
    echo "Model   : $model_dir"
    echo "Skip    : existing eval record found at $save_path"
    skipped_existing_evals=$((skipped_existing_evals + 1))
    last_eval_skipped=1
    return 0
  fi

  last_eval_skipped=0
  requests_path="$model_dir/requests.json"
  if [[ ! -f "$requests_path" && -f "$model_dir/training_requests.json" ]]; then
    requests_path="$model_dir/training_requests.json"
  fi

  if [[ "$USE_MODEL_REQUESTS" == "1" && -f "$requests_path" && "$EVAL_SUPPORTS_REQUESTS_PATH" == "1" ]]; then
    use_model_requests=1
  fi

  mkdir -p "$save_dir"

  cmd=(
    "$PYTHON_BIN" "$EVAL_SCRIPT"
    --data_path "$data_path"
    --model_path "$model_dir"
    --num_samples "$NUM_SAMPLES"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --save_path "$save_path"
  )

  if [[ "$EVAL_SUPPORTS_PRE_CACHE_DIR" == "1" ]]; then
    cmd+=(--pre_cache_dir "$PRE_CACHE_DIR/$dataset_name")
  fi

  if [[ "$use_model_requests" == "1" ]]; then
    cmd+=(--requests_path "$requests_path")
  fi

  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust_remote_code)
  fi

  if [[ "$VERBOSE" == "1" ]]; then
    cmd+=(--verbose)
  fi

  echo
  echo "Dataset : $dataset_name"
  echo "Model   : $model_dir"
  if [[ "$use_model_requests" == "1" ]]; then
    echo "Requests: $requests_path"
  elif [[ "$USE_MODEL_REQUESTS" == "1" && -f "$requests_path" ]]; then
    echo "Requests: sampled from $data_path (NUM_SAMPLES=$NUM_SAMPLES; --requests_path unsupported by $(basename "$EVAL_SCRIPT"))"
  else
    echo "Requests: sampled from $data_path (NUM_SAMPLES=$NUM_SAMPLES)"
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

move_completed_model() {
  local model_dir="$1"
  local rel_model_dir
  local destination_dir
  local destination_parent

  if [[ "$MOVE_COMPLETED_MODELS" != "1" ]]; then
    return 0
  fi

  if [[ "$COMPLETED_ROOT" == "$MODEL_ROOT" ]]; then
    echo "Skipping move because COMPLETED_ROOT matches MODEL_ROOT: $COMPLETED_ROOT"
    return 0
  fi

  rel_model_dir="${model_dir#$MODEL_ROOT/}"
  if [[ "$rel_model_dir" == "$model_dir" ]]; then
    rel_model_dir="$(basename "$model_dir")"
  fi

  destination_dir="$COMPLETED_ROOT/$rel_model_dir"
  destination_parent="$(dirname "$destination_dir")"

  if [[ "$destination_dir" == "$model_dir" ]]; then
    echo "Model already in completed root: $model_dir"
    return 0
  fi

  echo "Move to : $destination_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  mkdir -p "$destination_parent"

  if [[ -e "$destination_dir" ]]; then
    echo "Destination already exists, refusing to overwrite: $destination_dir"
    return 1
  fi

  mv "$model_dir" "$destination_dir"
}

run_model_eval_worker() {
  set +e

  local gpu="$1"
  local model_dir="$2"
  local status_file="$3"
  shift 3

  local dataset_name
  local dataset_status
  local exit_code=0
  local model_ran_eval=0
  local worker_skipped_existing=0

  export CUDA_VISIBLE_DEVICES="$gpu"
  trap 'exit_code=$?; if [[ ! -f "$status_file" ]]; then { echo "exit_code=$exit_code"; echo "model_ran_eval=$model_ran_eval"; echo "skipped_existing=$worker_skipped_existing"; echo "gpu=$gpu"; echo "model_dir=$model_dir"; } > "$status_file"; fi' EXIT

  echo
  echo "[worker gpu=$gpu] ============================================================"
  echo "[worker gpu=$gpu] Evaluating model: $model_dir"
  echo "[worker gpu=$gpu] Datasets        : $*"
  echo "[worker gpu=$gpu] ============================================================"

  for dataset_name in "$@"; do
    run_eval "$dataset_name" "$model_dir"
    dataset_status=$?
    if [[ "$dataset_status" -ne 0 ]]; then
      exit_code="$dataset_status"
      break
    fi

    if [[ "$last_eval_skipped" == "1" ]]; then
      worker_skipped_existing=$((worker_skipped_existing + 1))
    else
      model_ran_eval=1
    fi
  done

  if [[ "$exit_code" -eq 0 && "$model_ran_eval" == "1" ]]; then
    move_completed_model "$model_dir"
    exit_code=$?
  elif [[ "$exit_code" -eq 0 && "$MOVE_COMPLETED_MODELS" == "1" ]]; then
    echo "Skipping move because all requested eval records already exist."
  fi

  {
    echo "exit_code=$exit_code"
    echo "model_ran_eval=$model_ran_eval"
    echo "skipped_existing=$worker_skipped_existing"
    echo "gpu=$gpu"
    echo "model_dir=$model_dir"
  } > "$status_file"

  return "$exit_code"
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
  local status_skipped_existing
  local status_model_dir

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
    status_skipped_existing=0
    status_model_dir="${ACTIVE_DESC_BY_PID[$pid]:-}"
    while IFS='=' read -r key value; do
      case "$key" in
        exit_code)
          status_exit_code="$value"
          ;;
        skipped_existing)
          status_skipped_existing="$value"
          ;;
      esac
    done < "$status_file"
    skipped_existing_evals=$((skipped_existing_evals + status_skipped_existing))

    unset "ACTIVE_PIDS_BY_GPU[$gpu]"
    unset "ACTIVE_STATUS_BY_PID[$pid]"
    unset "ACTIVE_DESC_BY_PID[$pid]"

    if [[ "$status_exit_code" -ne 0 ]]; then
      failed_jobs=$((failed_jobs + 1))
      echo "[worker gpu=$gpu] Failed with exit code $status_exit_code: $status_model_dir"
    else
      echo "[worker gpu=$gpu] Finished: $status_model_dir"
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

echo "Found ${#MODEL_DIRS[@]} model(s) to evaluate."
echo "Requested datasets: ${REQUESTED_DATASET_NAMES[*]}"
echo "Match datasets from model name: $MATCH_DATASETS_FROM_NAME"
echo "Model root       : $MODEL_ROOT"
echo "Method filter    : ${METHOD_FILTER:-<none>}"
echo "Model filter     : ${MODEL_FILTER:-<none>}"
echo "Evaluation root  : $EVALUATION_ROOT"
echo "Results          : $RESULT_ROOT"
echo "Eval script: $EVAL_SCRIPT"
echo "Move completed models: $MOVE_COMPLETED_MODELS"
if [[ "$MOVE_COMPLETED_MODELS" == "1" ]]; then
  echo "Completed root      : $COMPLETED_ROOT"
fi
echo "Skip existing evals : $SKIP_EXISTING_EVALS"
echo "Eval GPUs           : ${EVAL_GPU_LIST[*]}"
echo "Parallel evals      : $MAX_PARALLEL_EVALS"
echo "Wait for free GPUs  : $WAIT_FOR_FREE_GPUS"
if [[ "$WAIT_FOR_FREE_GPUS" == "1" ]]; then
  echo "GPU idle threshold  : mem<=${GPU_IDLE_MEM_MB}MiB util<=${GPU_IDLE_UTIL}% and no compute pids"
fi

evaluated_models=0
skipped_models=0
skipped_existing_evals=0
failed_jobs=0
job_id=0

declare -A ACTIVE_PIDS_BY_GPU=()
declare -A ACTIVE_STATUS_BY_PID=()
declare -A ACTIVE_DESC_BY_PID=()

RUN_STATUS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kledit_eval_parallel_status.XXXXXX")"
mkdir -p "$RUN_STATUS_DIR"
cleanup_run_status_dir() {
  local pid

  for pid in "${ACTIVE_PIDS_BY_GPU[@]}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done

  if [[ -n "${RUN_STATUS_DIR:-}" && "$RUN_STATUS_DIR" == "${TMPDIR:-/tmp}/kledit_eval_parallel_status."* ]]; then
    rm -rf "$RUN_STATUS_DIR"
  fi
}
trap cleanup_run_status_dir EXIT

for model_dir in "${MODEL_DIRS[@]}"; do
  MODEL_DATASETS=()
  mapfile -t MODEL_DATASETS < <(select_datasets_for_model "$model_dir" || true)
  if [[ "${#MODEL_DATASETS[@]}" -eq 0 ]]; then
    echo
    echo "Skipping model: $model_dir"
    echo "Reason   : no requested dataset token found in directory name"
    skipped_models=$((skipped_models + 1))
    continue
  fi

  echo
  echo "============================================================"
  echo "Evaluating model: $model_dir"
  echo "Datasets        : ${MODEL_DATASETS[*]}"
  echo "============================================================"
  evaluated_models=$((evaluated_models + 1))

  wait_for_free_gpu_slot
  job_id=$((job_id + 1))
  status_file="$RUN_STATUS_DIR/job_${job_id}.status"

  run_model_eval_worker "$FREE_GPU" "$model_dir" "$status_file" "${MODEL_DATASETS[@]}" &
  worker_pid=$!
  ACTIVE_PIDS_BY_GPU["$FREE_GPU"]="$worker_pid"
  ACTIVE_STATUS_BY_PID["$worker_pid"]="$status_file"
  ACTIVE_DESC_BY_PID["$worker_pid"]="$model_dir"
  echo "[scheduler] Started PID $worker_pid on GPU $FREE_GPU"
done

wait_for_all_jobs

if [[ "$evaluated_models" -eq 0 ]]; then
  echo
  echo "No model directories matched the requested datasets in their names."
  echo "Set MATCH_DATASETS_FROM_NAME=0 to restore the previous behavior."
  exit 1
fi

echo
echo "Evaluated models: $evaluated_models"
echo "Skipped models  : $skipped_models"
echo "Skipped existing evals: $skipped_existing_evals"
echo "Failed jobs     : $failed_jobs"
if [[ "$failed_jobs" -ne 0 ]]; then
  echo "Some evaluations failed. Check the eval.log files above."
  exit 1
fi
echo "All evaluations finished."
