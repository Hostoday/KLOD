#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
REWRITE_KL_LAMBDA="${REWRITE_KL_LAMBDA:-1.2}"
NON_TARGET_KL_LAMBDA="${NON_TARGET_KL_LAMBDA:-0.6}"
TARGET_ALPHA="${TARGET_ALPHA:-0.85}"
EARLY_STOP_LOSS="${EARLY_STOP_LOSS:-0.1}"
EARLY_STOP_LOSS_TAG="${EARLY_STOP_LOSS_TAG:-${EARLY_STOP_LOSS}}"
SAVE_TAG="${SAVE_TAG:-KLOD}"

# NUM_SAMPLES=0 means use all usable records when evaluating from the raw data_path.
# When USE_MODEL_REQUESTS=1 and training_requests.json exists, eval_hf_easyedit_kl.py
# evaluates the saved training requests exactly.
NUM_SAMPLES="${NUM_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
KL_DIRECTION="${KL_DIRECTION:-base_to_edit}"
USE_MODEL_REQUESTS="${USE_MODEL_REQUESTS:-1}"
SKIP_MISSING_MODELS="${SKIP_MISSING_MODELS:-0}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/outputs/evaluation/Analysis/kl_analysis/KLOD}"

MODEL_CONFIGS=(
  "llama3-8b:Meta-Llama-3-8B-Instruct:meta-llama/Meta-Llama-3-8B-Instruct"
  "qwen2.5-7b:Qwen2.5-7B-Instruct:Qwen/Qwen2.5-7B-Instruct"
)

DATASET_CONFIGS=(
  "counterfact:counterfact_3k:${SCRIPT_DIR}/data/counterfact/counterfact_3k.json"
  "zsre:zsre_3k:${SCRIPT_DIR}/data/zsre/zsre_3k.json"
)

context_aug_tag=""
if [[ "${USE_CONTEXT_AUG:-0}" == "1" ]]; then
  context_aug_stage="${CONTEXT_AUG_STAGE:-klod}"
  context_aug_templates="${CONTEXT_AUG_MAX_TEMPLATES:-all}"
  context_aug_tag="_ctxaug1_ctxstage${context_aug_stage}_ctxtmpl${context_aug_templates}"
fi

default_model_path() {
  local model_label="$1"
  local model_slug="$2"
  local dataset_label="$3"
  local data_stem="$4"

  if [[ -n "${SAVE_MODEL_ROOT:-}" ]]; then
    printf '%s/%s/%s' "$SAVE_MODEL_ROOT" "$model_label" "$dataset_label"
    return 0
  fi

  if [[ -n "${SAVE_MODEL_DIR:-}" ]]; then
    printf '%s/%s/%s' "$SAVE_MODEL_DIR" "$model_label" "$dataset_label"
    return 0
  fi

  printf '%s/outputs/Models/%s/%s_%s_rewritekl%s_ntkl%s_early_stop_loss%s_target_alpha%s%s' \
    "$SCRIPT_DIR" \
    "$SAVE_TAG" \
    "$model_slug" \
    "$data_stem" \
    "$REWRITE_KL_LAMBDA" \
    "$NON_TARGET_KL_LAMBDA" \
    "$EARLY_STOP_LOSS_TAG" \
    "$TARGET_ALPHA" \
    "$context_aug_tag"
}

total_runs=$((${#MODEL_CONFIGS[@]} * ${#DATASET_CONFIGS[@]}))
run_index=0

for model_config in "${MODEL_CONFIGS[@]}"; do
  IFS=':' read -r model_label model_slug base_model_path <<< "$model_config"

  for dataset_config in "${DATASET_CONFIGS[@]}"; do
    IFS=':' read -r dataset_label data_stem data_path <<< "$dataset_config"
    run_index=$((run_index + 1))

    model_path="$(default_model_path "$model_label" "$model_slug" "$dataset_label" "$data_stem")"
    save_path="${RESULT_ROOT}/${model_label}_${dataset_label}_rewritekl${REWRITE_KL_LAMBDA}_ntkl${NON_TARGET_KL_LAMBDA}_earlystop${EARLY_STOP_LOSS_TAG}_${KL_DIRECTION}_kl.json"

    if [[ ! -f "$model_path/config.json" ]]; then
      message="[run_eval_kl] model not found or incomplete: $model_path"
      if [[ "$SKIP_MISSING_MODELS" == "1" ]]; then
        echo "$message; skipping"
        continue
      fi
      echo "$message"
      exit 1
    fi

    CMD=(
      "$PYTHON_BIN" "$SCRIPT_DIR/evaluate/eval_hf_easyedit_kl.py"
      --data_path "$data_path"
      --model_path "$model_path"
      --base_model_path "$base_model_path"
      --num_samples "$NUM_SAMPLES"
      --batch_size "$BATCH_SIZE"
      --kl_direction "$KL_DIRECTION"
      --save_path "$save_path"
      --trust_remote_code
    )

    requests_path="$model_path/training_requests.json"
    if [[ "$USE_MODEL_REQUESTS" == "1" && -f "$requests_path" ]]; then
      CMD+=(--requests_path "$requests_path")
    fi

    echo "[run_eval_kl] run ${run_index}/${total_runs}: model=${model_label} dataset=${dataset_label}"
    echo "[run_eval_kl] model_path=${model_path}"
    echo "[run_eval_kl] save_path=${save_path}"
    printf '[run_eval_kl] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
  done
done
