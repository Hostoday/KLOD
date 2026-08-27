#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/hparams/LocFT-BF/llama3-8b.yaml}"
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data/zsre/zsre_mend_train.json}"
SAVE_TAG="${SAVE_TAG:-LocFT-BF_150k}"
DATASET_NAME="${DATASET_NAME:-zsre}"
SAMPLE_SIZE="${SAMPLE_SIZE:-150000}"
SAVE_COUNTS="${SAVE_COUNTS:-3k,5k,10k,20k,50k,100k,150k}"

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/klod/train_klod_locft_150k.py"
  --klod_config_path "$CONFIG_PATH"
  --data_path "$DATA_PATH"
  --dataset_name "$DATASET_NAME"
  --sample_size "$SAMPLE_SIZE"
  --save_counts "$SAVE_COUNTS"
  --save_tag "$SAVE_TAG"
  --use_locft 1
  --klod_epochs 0
  --rewrite_kl_lambda 0
  --non_target_kl_lambda 0
)

if [[ -n "${DEVICE:-}" ]]; then
  CMD+=(--device "$DEVICE")
fi
if [[ -n "${MODEL_NAME:-}" ]]; then
  CMD+=(--model_name "$MODEL_NAME")
fi
if [[ -n "${SAVE_MODEL_DIR:-}" ]]; then
  CMD+=(--save_model_dir "$SAVE_MODEL_DIR")
fi
if [[ -n "${SEED:-}" ]]; then
  CMD+=(--seed "$SEED")
fi

echo "Running pure LocFT-BF 150K training"
echo "  config: $CONFIG_PATH"
echo "  output: outputs/Models/$SAVE_TAG/"
"${CMD[@]}"
