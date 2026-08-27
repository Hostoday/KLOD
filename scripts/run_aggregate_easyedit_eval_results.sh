#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGGREGATE_PY="${SCRIPT_DIR}/evaluate/aggregate_easyedit_eval_results.py"

DEFAULT_TARGET_DIR="${SCRIPT_DIR}/outputs/evaluation/eval_results_easyedit"

TARGET_DIR="${1:-${DEFAULT_TARGET_DIR}}"
PATTERN="${2:-eval.json}"
DIGITS="${DIGITS:-2}"
SCALE="${SCALE:-percent}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-easyedit_paper_table}"

if [[ ! -f "${AGGREGATE_PY}" ]]; then
  echo "[error] aggregate script not found: ${AGGREGATE_PY}" >&2
  exit 1
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "[error] target directory not found: ${TARGET_DIR}" >&2
  exit 1
fi

echo "[info] target_dir    = ${TARGET_DIR}"
echo "[info] pattern       = ${PATTERN}"
echo "[info] scale         = ${SCALE}"
echo "[info] digits        = ${DIGITS}"
echo "[info] output_prefix = ${OUTPUT_PREFIX}"

python3 "${AGGREGATE_PY}" "${TARGET_DIR}" \
  --pattern "${PATTERN}" \
  --scale "${SCALE}" \
  --digits "${DIGITS}" \
  --output-prefix "${OUTPUT_PREFIX}"
