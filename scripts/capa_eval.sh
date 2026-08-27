#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2"
fi

BASELINE_ROOT="${BASELINE_ROOT:-$SCRIPT_DIR/outputs/Models/}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$SCRIPT_DIR/outputs/evaluation}"
LM_EVAL_ROOT="${LM_EVAL_ROOT:-$EVALUATION_ROOT/lm_eval}"
EVAL_GPUS="${EVAL_GPUS:-$CUDA_VISIBLE_DEVICES}"
PARALLEL_EVALS="${PARALLEL_EVALS:-auto}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"
GPU_IDLE_UTIL="${GPU_IDLE_UTIL:-10}"
GPU_WAIT_INTERVAL="${GPU_WAIT_INTERVAL:-30}"
JOB_POLL_INTERVAL="${JOB_POLL_INTERVAL:-5}"
TASKS="${TASKS:-mmlu,sst2,gsm8k,nq_open,wmt16-de-en}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_FILTER="${MODEL_FILTER:-}"
DATASET_FILTER="${DATASET_FILTER:-}"
METHOD_FILTER="${METHOD_FILTER:-KLOD}"
SKIP_EXISTING_LM_EVALS="${SKIP_EXISTING_LM_EVALS:-1}"
DRY_RUN="${DRY_RUN:-0}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-}"

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

if [[ "$#" -eq 0 && ! -d "$BASELINE_ROOT" ]]; then
  echo "Baseline root not found: $BASELINE_ROOT"
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]] && ! command -v "$LM_EVAL_BIN" >/dev/null 2>&1; then
  echo "lm_eval executable not found: $LM_EVAL_BIN"
  echo "Set LM_EVAL_BIN=/path/to/lm_eval if needed."
  exit 1
fi

sanitize_component() {
  printf '%s' "$1" \
    | sed -E 's/[[:space:]\/]+/_/g; s/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_//; s/_$//'
}

infer_dataset() {
  local name_lc
  name_lc="$(printf '%s' "$(basename "$1")" | tr '[:upper:]' '[:lower:]')"

  case "$name_lc" in
    *counterfact*)
      echo "counterfact"
      ;;
    *zsre*)
      echo "zsre"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

infer_model_name() {
  local name_lc
  name_lc="$(printf '%s' "$(basename "$1")" | tr '[:upper:]' '[:lower:]')"

  case "$name_lc" in
    *qwen2.5-7b*|*qwen2_5-7b*|*qwen*)
      echo "Qwen2.5-7B-Instruct"
      ;;
    *meta-llama-3-8b-instruct*|*llama3-8b*|*llama3_8b*|*llama3*)
      echo "Meta-Llama-3-8B-Instruct"
      ;;
    *)
      basename "$1"
      ;;
  esac
}

output_dir_for_model() {
  local model_dir="$1"
  local method
  local dataset
  local run_name

  method="$(basename "$(dirname "$model_dir")")"
  dataset="$(infer_dataset "$model_dir")"
  run_name="$(basename "$model_dir")"

  printf '%s/%s_%s_%s' \
    "$LM_EVAL_ROOT" \
    "$(sanitize_component "$method")" \
    "$(sanitize_component "$run_name")" \
    "$(sanitize_component "$dataset")"
}

has_existing_lm_eval_result() {
  local output_dir="$1"
  # lm_eval writes results under --output_path/<model-path-slug>/results_*.json.
  find "$output_dir" -maxdepth 2 -type f -name 'results_*.json' -size +0c 2>/dev/null | grep -q .
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

run_lm_eval_worker() {
  set +e

  local gpu="$1"
  local model_dir="$2"
  local status_file="$3"
  local method
  local model_name
  local dataset
  local output_dir
  local log_path
  local model_args
  local exit_code=0
  local skipped_existing=0
  local -a cmd

  method="$(basename "$(dirname "$model_dir")")"
  model_name="$(infer_model_name "$model_dir")"
  dataset="$(infer_dataset "$model_dir")"
  output_dir="$(output_dir_for_model "$model_dir")"
  log_path="$output_dir/lm_eval.log"
  model_args="pretrained=$model_dir,dtype=$DTYPE"
  if [[ -n "$MODEL_ARGS_EXTRA" ]]; then
    model_args="$model_args,$MODEL_ARGS_EXTRA"
  fi

  export CUDA_VISIBLE_DEVICES="$gpu"
  trap 'exit_code=$?; if [[ ! -f "$status_file" ]]; then { echo "exit_code=$exit_code"; echo "skipped_existing=$skipped_existing"; echo "gpu=$gpu"; echo "model_dir=$model_dir"; echo "output_dir=$output_dir"; } > "$status_file"; fi' EXIT

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$output_dir"
  fi

  echo
  echo "[lm-eval gpu=$gpu] ============================================================"
  echo "[lm-eval gpu=$gpu] Method : $method"
  echo "[lm-eval gpu=$gpu] Model  : $model_name"
  echo "[lm-eval gpu=$gpu] Dataset: $dataset"
  echo "[lm-eval gpu=$gpu] Path   : $model_dir"
  echo "[lm-eval gpu=$gpu] Output : $output_dir"
  echo "[lm-eval gpu=$gpu] ============================================================"

  if [[ "$SKIP_EXISTING_LM_EVALS" == "1" ]] && has_existing_lm_eval_result "$output_dir"; then
    echo "[lm-eval gpu=$gpu] Skip existing result in $output_dir"
    skipped_existing=1
    {
      echo "exit_code=0"
      echo "skipped_existing=$skipped_existing"
      echo "gpu=$gpu"
      echo "model_dir=$model_dir"
      echo "output_dir=$output_dir"
    } > "$status_file"
    return 0
  fi

  cmd=(
    "$LM_EVAL_BIN"
    --model hf
    --model_args "$model_args"
    --batch_size "$BATCH_SIZE"
    --tasks "$TASKS"
    --device cuda:0
    --output_path "$output_dir"
  )

  printf '[lm-eval gpu=%s] Command :' "$gpu"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    {
      echo "exit_code=0"
      echo "skipped_existing=0"
      echo "gpu=$gpu"
      echo "model_dir=$model_dir"
      echo "output_dir=$output_dir"
    } > "$status_file"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "$log_path"
  exit_code=$?

  {
    echo "exit_code=$exit_code"
    echo "skipped_existing=$skipped_existing"
    echo "gpu=$gpu"
    echo "model_dir=$model_dir"
    echo "output_dir=$output_dir"
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
  local status_output_dir

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
    status_output_dir=""
    while IFS='=' read -r key value; do
      case "$key" in
        exit_code)
          status_exit_code="$value"
          ;;
        skipped_existing)
          status_skipped_existing="$value"
          ;;
        output_dir)
          status_output_dir="$value"
          ;;
      esac
    done < "$status_file"

    skipped_existing_evals=$((skipped_existing_evals + status_skipped_existing))
    unset "ACTIVE_PIDS_BY_GPU[$gpu]"
    unset "ACTIVE_STATUS_BY_PID[$pid]"
    unset "ACTIVE_DESC_BY_PID[$pid]"

    if [[ "$status_exit_code" -ne 0 ]]; then
      failed_jobs=$((failed_jobs + 1))
      echo "[lm-eval gpu=$gpu] Failed with exit code $status_exit_code: $status_model_dir"
    else
      completed_jobs=$((completed_jobs + 1))
      echo "[lm-eval gpu=$gpu] Finished: ${status_output_dir:-$status_model_dir}"
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

discover_model_dirs() {
  while IFS= read -r -d '' config_path; do
    dirname "$config_path"
  done < <(find -L "$BASELINE_ROOT" -type f -name config.json -print0 2>/dev/null)
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
    dataset="$(infer_dataset "$model_dir")"

    if [[ -n "$MODEL_FILTER" && "$model_dir" != *"$MODEL_FILTER"* ]]; then
      continue
    fi
    if [[ -n "$METHOD_FILTER" && "$method" != *"$METHOD_FILTER"* ]]; then
      continue
    fi
    if [[ -n "$DATASET_FILTER" && "$dataset" != "$DATASET_FILTER" ]]; then
      continue
    fi

    MODEL_DIRS+=("$model_dir")
  done
else
  while IFS= read -r model_dir; do
    method="$(basename "$(dirname "$model_dir")")"
    dataset="$(infer_dataset "$model_dir")"

    if [[ -n "$MODEL_FILTER" && "$model_dir" != *"$MODEL_FILTER"* ]]; then
      continue
    fi
    if [[ -n "$METHOD_FILTER" && "$method" != *"$METHOD_FILTER"* ]]; then
      continue
    fi
    if [[ -n "$DATASET_FILTER" && "$dataset" != "$DATASET_FILTER" ]]; then
      continue
    fi

    MODEL_DIRS+=("$model_dir")
  done < <(discover_model_dirs | sort -u)
fi

if [[ "${#MODEL_DIRS[@]}" -eq 0 ]]; then
  echo "No baseline model directories found under: $BASELINE_ROOT"
  exit 1
fi

mkdir -p "$LM_EVAL_ROOT"

echo "Found baseline model(s): ${#MODEL_DIRS[@]}"
if [[ "$#" -gt 0 ]]; then
  echo "Model source        : explicit model directories"
else
  echo "Baseline root       : $BASELINE_ROOT"
fi
echo "LM eval root        : $LM_EVAL_ROOT"
echo "Tasks               : $TASKS"
echo "Batch size          : $BATCH_SIZE"
echo "Dtype               : $DTYPE"
echo "Eval GPUs           : ${EVAL_GPU_LIST[*]}"
echo "Parallel evals      : $MAX_PARALLEL_EVALS"
echo "Wait for free GPUs  : $WAIT_FOR_FREE_GPUS"
if [[ "$WAIT_FOR_FREE_GPUS" == "1" ]]; then
  echo "GPU idle threshold  : mem<=${GPU_IDLE_MEM_MB}MiB util<=${GPU_IDLE_UTIL}% and no compute pids"
fi
echo "Skip existing evals : $SKIP_EXISTING_LM_EVALS"
echo "Dry run             : $DRY_RUN"

scheduled_jobs=0
completed_jobs=0
failed_jobs=0
skipped_existing_evals=0
job_id=0

declare -A ACTIVE_PIDS_BY_GPU=()
declare -A ACTIVE_STATUS_BY_PID=()
declare -A ACTIVE_DESC_BY_PID=()

RUN_STATUS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kledit_lm_eval_status.XXXXXX")"
cleanup_run_status_dir() {
  local pid

  for pid in "${ACTIVE_PIDS_BY_GPU[@]}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done

  if [[ -n "${RUN_STATUS_DIR:-}" && "$RUN_STATUS_DIR" == "${TMPDIR:-/tmp}/kledit_lm_eval_status."* ]]; then
    rm -rf "$RUN_STATUS_DIR"
  fi
}
trap cleanup_run_status_dir EXIT

for model_dir in "${MODEL_DIRS[@]}"; do
  output_dir="$(output_dir_for_model "$model_dir")"
  if [[ "$SKIP_EXISTING_LM_EVALS" == "1" ]] && has_existing_lm_eval_result "$output_dir"; then
    echo
    echo "[scheduler] Skip existing result: $output_dir"
    skipped_existing_evals=$((skipped_existing_evals + 1))
    continue
  fi

  wait_for_free_gpu_slot
  job_id=$((job_id + 1))
  scheduled_jobs=$((scheduled_jobs + 1))
  status_file="$RUN_STATUS_DIR/job_${job_id}.status"

  run_lm_eval_worker "$FREE_GPU" "$model_dir" "$status_file" &
  worker_pid=$!
  ACTIVE_PIDS_BY_GPU["$FREE_GPU"]="$worker_pid"
  ACTIVE_STATUS_BY_PID["$worker_pid"]="$status_file"
  ACTIVE_DESC_BY_PID["$worker_pid"]="$model_dir"
  echo "[scheduler] Started PID $worker_pid on GPU $FREE_GPU"
done

wait_for_all_jobs

echo
echo "Scheduled jobs       : $scheduled_jobs"
echo "Completed jobs       : $completed_jobs"
echo "Skipped existing     : $skipped_existing_evals"
echo "Failed jobs          : $failed_jobs"

if [[ "$failed_jobs" -ne 0 ]]; then
  echo "Some lm_eval jobs failed. Check lm_eval.log files under $LM_EVAL_ROOT."
  exit 1
fi

echo "All lm_eval jobs finished."
