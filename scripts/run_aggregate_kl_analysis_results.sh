#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/outputs/evaluation/Analysis/kl_analysis}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-kl_analysis_paper_table}"
DIGITS="${DIGITS:-3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate/aggregate_kl_analysis_results.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  --digits "${DIGITS}"
