#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
REWRITE_KL_LAMBDA="${REWRITE_KL_LAMBDA:-1.2}"
NON_TARGET_KL_LAMBDA="${NON_TARGET_KL_LAMBDA:-0.6}"
EARLY_STOP_LOSS="${EARLY_STOP_LOSS:-0.1}"
TARGET_ALPHA="${TARGET_ALPHA:-0.85}"

MODEL_CONFIGS=(
  "llama3-8b:${SCRIPT_DIR}/hparams/KLOD/llama3-8b.yaml"
  "qwen2.5-7b:${SCRIPT_DIR}/hparams/KLOD/qwen2.5-7b.yaml"
)

DATASET_CONFIGS=(
  "counterfact:${SCRIPT_DIR}/data/counterfact/counterfact_3k.json"
  "zsre:${SCRIPT_DIR}/data/zsre/zsre_3k.json"
)

total_runs=$((${#MODEL_CONFIGS[@]} * ${#DATASET_CONFIGS[@]}))
run_index=0

for model_config in "${MODEL_CONFIGS[@]}"; do
  model_label="${model_config%%:*}"
  hparams_path="${model_config#*:}"

  for dataset_config in "${DATASET_CONFIGS[@]}"; do
    dataset_label="${dataset_config%%:*}"
    data_path="${dataset_config#*:}"
    run_index=$((run_index + 1))

    CMD=(
      "$PYTHON_BIN" "$SCRIPT_DIR/klod/train_klod.py"
      --klod_config_path "$hparams_path"
      --data_path "$data_path"
      --rewrite_kl_lambda "$REWRITE_KL_LAMBDA"
      --non_target_kl_lambda "$NON_TARGET_KL_LAMBDA"
      --early_stop_loss "$EARLY_STOP_LOSS"
      --target_alpha "$TARGET_ALPHA"
    )

    if [[ -n "${SAVE_MODEL_ROOT:-}" ]]; then
      CMD+=(--save_model_dir "$SAVE_MODEL_ROOT/$model_label/$dataset_label")
    elif [[ -n "${SAVE_MODEL_DIR:-}" ]]; then
      CMD+=(--save_model_dir "$SAVE_MODEL_DIR/$model_label/$dataset_label")
    fi

    if [[ -n "${SAVE_TAG:-}" ]]; then
      CMD+=(--save_tag "$SAVE_TAG")
    fi

    if [[ -n "${SAMPLE_SIZE:-}" ]]; then
      CMD+=(--sample_size "$SAMPLE_SIZE")
    fi

    if [[ -n "${SEED:-}" ]]; then
      CMD+=(--seed "$SEED")
    fi

    if [[ -n "${NUM_STEPS:-}" ]]; then
      CMD+=(--num_steps "$NUM_STEPS")
    fi

    if [[ -n "${DEVICE:-}" ]]; then
      CMD+=(--device "$DEVICE")
    fi

    if [[ -n "${USE_CONTEXT_AUG:-}" ]]; then
      CMD+=(--use_context_aug "$USE_CONTEXT_AUG")
    fi

    if [[ -n "${CONTEXT_AUG_STAGE:-}" ]]; then
      CMD+=(--context_aug_stage "$CONTEXT_AUG_STAGE")
    fi

    if [[ -n "${CONTEXT_AUG_LENGTH_PARAMS:-}" ]]; then
      CMD+=(--context_aug_length_params "$CONTEXT_AUG_LENGTH_PARAMS")
    fi

    if [[ -n "${CONTEXT_AUG_MAX_TEMPLATES:-}" ]]; then
      CMD+=(--context_aug_max_templates "$CONTEXT_AUG_MAX_TEMPLATES")
    fi

    echo "[run_klod] run ${run_index}/${total_runs}: model=${model_label} dataset=${dataset_label}"
    printf '[run_klod] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
  done
done
