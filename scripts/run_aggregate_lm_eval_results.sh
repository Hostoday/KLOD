#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGGREGATE_PY="${SCRIPT_DIR}/evaluate/aggregate_lm_eval_results.py"

DEFAULT_TARGET_DIR="${SCRIPT_DIR}/outputs/evaluation/lm_eval"

TARGET_DIR="${1:-${DEFAULT_TARGET_DIR}}"
DIGITS="${DIGITS:-2}"
OVERALL_PREFIX="${OVERALL_PREFIX:-lm_eval_5task_avg_table}"
DETAIL_PREFIX="${DETAIL_PREFIX:-lm_eval_dataset_score_table}"
INCLUDE_INCOMPLETE_LATEX="${INCLUDE_INCOMPLETE_LATEX:-0}"
EXCLUDE_RUN_GLOB="${EXCLUDE_RUN_GLOB:-KLEdit_*_3kuse_kledit1_*FULL-TARGET*}"

if [[ ! -f "${AGGREGATE_PY}" ]]; then
  echo "[error] aggregate script not found: ${AGGREGATE_PY}" >&2
  exit 1
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "[error] target directory not found: ${TARGET_DIR}" >&2
  exit 1
fi

echo "[info] target_dir    = ${TARGET_DIR}"
echo "[info] digits        = ${DIGITS}"
echo "[info] overall_prefix = ${OVERALL_PREFIX}"
echo "[info] detail_prefix  = ${DETAIL_PREFIX}"
echo "[info] include_incomplete_latex = ${INCLUDE_INCOMPLETE_LATEX}"
echo "[info] exclude_run_glob = ${EXCLUDE_RUN_GLOB:-none}"

args=(
  "${TARGET_DIR}"
  --digits "${DIGITS}"
  --overall-prefix "${OVERALL_PREFIX}"
  --detail-prefix "${DETAIL_PREFIX}"
)

if [[ "${INCLUDE_INCOMPLETE_LATEX}" == "1" || "${INCLUDE_INCOMPLETE_LATEX}" == "true" ]]; then
  args+=(--include-incomplete-latex)
fi

if [[ -n "${EXCLUDE_RUN_GLOB}" ]]; then
  args+=(--exclude-run-glob "${EXCLUDE_RUN_GLOB}")
fi

python3 "${AGGREGATE_PY}" "${args[@]}"
