#!/usr/bin/env python3
"""Build paper-style summary tables from lm-eval result JSON/log files."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


EDIT_DATASET_DISPLAY = {
    "counterfact": "CounterFact",
    "zsre": "ZSRE",
}
MODEL_PREFIXES = (
    "Meta-Llama",
    "Qwen",
    "Llama",
    "Mistral",
    "Gemma",
    "gemma",
)

PRIMARY_EVALS: tuple[dict[str, str], ...] = (
    {
        "key": "gsm8k",
        "display": "GSM8K",
        "metric": "exact_match",
        "filter": "flexible-extract",
        "value_key": "exact_match,flexible-extract",
        "stderr_key": "exact_match_stderr,flexible-extract",
        "scale": "percent",
    },
    {
        "key": "mmlu",
        "display": "MMLU",
        "metric": "acc",
        "filter": "none",
        "value_key": "acc,none",
        "stderr_key": "acc_stderr,none",
        "scale": "percent",
    },
    {
        "key": "nq_open",
        "display": "NQ Open",
        "metric": "exact_match",
        "filter": "remove_whitespace",
        "value_key": "exact_match,remove_whitespace",
        "stderr_key": "exact_match_stderr,remove_whitespace",
        "scale": "percent",
    },
    {
        "key": "sst2",
        "display": "SST-2",
        "metric": "acc",
        "filter": "none",
        "value_key": "acc,none",
        "stderr_key": "acc_stderr,none",
        "scale": "percent",
    },
    {
        "key": "wmt16-de-en",
        "display": "WMT16 De-En",
        "metric": "bleu",
        "filter": "none",
        "value_key": "bleu,none",
        "stderr_key": "bleu_stderr,none",
        "scale": "raw",
    },
)
WMT_CHRF_EVAL = {
    "key": "wmt16-de-en",
    "display": "WMT16 De-En chrF",
    "metric": "chrf",
    "filter": "none",
    "value_key": "chrf,none",
    "stderr_key": "chrf_stderr,none",
    "scale": "raw",
}

MMLU_FEATURES: tuple[tuple[str, str], ...] = (
    ("mmlu_humanities", "MMLU-Humanities"),
    ("mmlu_other", "MMLU-Other"),
    ("mmlu_social_sciences", "MMLU-Social Sciences"),
    ("mmlu_stem", "MMLU-STEM"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect lm-eval outputs and write paper-style CSV/Markdown/LaTeX/JSON "
            "tables for five-task averages and per-dataset scores."
        )
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory containing lm_eval run directories.",
    )
    parser.add_argument(
        "--overall-prefix",
        default="lm_eval_5task_avg_table",
        help="Output prefix for the five-task average table.",
    )
    parser.add_argument(
        "--detail-prefix",
        default="lm_eval_dataset_score_table",
        help="Output prefix for the per-dataset/MMLU-feature table.",
    )
    parser.add_argument(
        "--summary-json-name",
        default="lm_eval_paper_tables.json",
        help="Filename for the JSON summary written under target_dir.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places for table scores.",
    )
    parser.add_argument(
        "--include-incomplete-latex",
        action="store_true",
        help="Include incomplete/missing runs in LaTeX tables with -- values.",
    )
    parser.add_argument(
        "--include-run-glob",
        action="append",
        default=[],
        help=(
            "Only include run directories whose names match this shell-style glob. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--exclude-run-glob",
        action="append",
        default=[],
        help=(
            "Exclude run directories whose names match this shell-style glob. "
            "Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON is not an object")
    return payload


def latest_result_json(run_dir: Path) -> Path | None:
    candidates = sorted(
        run_dir.glob("**/results_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def split_run_name(run_name: str) -> tuple[str, str, str]:
    if "_" not in run_name:
        return "", run_name, ""

    head, edit_dataset = run_name.rsplit("_", 1)

    parts = head.split("_")
    for index, part in enumerate(parts):
        if part.startswith(MODEL_PREFIXES):
            method = "_".join(parts[:index])
            model = "_".join(parts[index:])
            return method, model, edit_dataset

    if "_" not in head:
        return "", head, edit_dataset

    method, model = head.split("_", 1)
    return method, model, edit_dataset


def display_edit_dataset(value: str) -> str:
    return EDIT_DATASET_DISPLAY.get(value.lower(), value)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def scale_value(value: float | None, scale: str) -> float | None:
    if value is None:
        return None
    if scale == "percent":
        return value * 100.0
    return value


def format_score(value: float | None, digits: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def parse_log_table(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None

    results: dict[str, dict[str, Any]] = {}
    task_aliases = {
        "gsm8k": "gsm8k",
        "mmlu": "mmlu",
        "nq_open": "nq_open",
        "sst2": "sst2",
        "wmt16-de-en": "wmt16-de-en",
        " - humanities": "mmlu_humanities",
        " - other": "mmlu_other",
        " - social sciences": "mmlu_social_sciences",
        " - stem": "mmlu_stem",
    }
    current_task = ""

    with log_path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            if not line.startswith("|"):
                continue
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) < 9:
                continue
            raw_task, _, filter_name, _, metric, _, value, _, stderr = parts[:9]
            if raw_task and raw_task not in {"Tasks", "Groups"} and not set(raw_task) <= {"-"}:
                current_task = raw_task
            elif not raw_task:
                raw_task = current_task

            task_key = task_aliases.get(raw_task)
            if not task_key:
                continue

            try:
                numeric_value = float(value)
            except ValueError:
                continue
            try:
                numeric_stderr = float(stderr)
            except ValueError:
                numeric_stderr = None

            result = results.setdefault(task_key, {"alias": raw_task})
            result[f"{metric},{filter_name}"] = numeric_value
            if numeric_stderr is not None:
                result[f"{metric}_stderr,{filter_name}"] = numeric_stderr

    required = {item["key"] for item in PRIMARY_EVALS}
    if not required.issubset(results):
        return None
    return {"results": results, "groups": {key: results[key] for key, _ in MMLU_FEATURES if key in results}}


def load_run_payload(run_dir: Path) -> tuple[dict[str, Any] | None, str, str | None]:
    json_path = latest_result_json(run_dir)
    if json_path:
        return load_json(json_path), "json", str(json_path)

    log_path = run_dir / "lm_eval.log"
    payload = parse_log_table(log_path)
    if payload:
        return payload, "log", str(log_path)

    if log_path.exists():
        return None, "incomplete", str(log_path)
    return None, "missing", None


def run_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("edit_dataset", "")),
        str(row.get("model", "")),
        str(row.get("method", "")),
    )


def extract_primary_scores(payload: dict[str, Any] | None) -> dict[str, dict[str, float | None]]:
    scores: dict[str, dict[str, float | None]] = {}
    results = payload.get("results", {}) if payload else {}
    if not isinstance(results, dict):
        return scores

    for spec in PRIMARY_EVALS:
        scores[spec["key"]] = extract_metric_score(payload, spec)
    return scores


def extract_metric_score(payload: dict[str, Any] | None, spec: dict[str, str]) -> dict[str, float | None]:
    results = payload.get("results", {}) if payload else {}
    task_payload = results.get(spec["key"], {}) if isinstance(results, dict) else {}
    if not isinstance(task_payload, dict):
        raw_value = None
        raw_stderr = None
    else:
        raw_value = as_float(task_payload.get(spec["value_key"]))
        raw_stderr = as_float(task_payload.get(spec["stderr_key"]))
    return {
        "score": scale_value(raw_value, spec["scale"]),
        "stderr": scale_value(raw_stderr, spec["scale"]),
    }


def extract_mmlu_feature_scores(payload: dict[str, Any] | None) -> dict[str, dict[str, float | None]]:
    scores: dict[str, dict[str, float | None]] = {}
    groups = payload.get("groups", {}) if payload else {}
    results = payload.get("results", {}) if payload else {}

    for feature_key, _ in MMLU_FEATURES:
        feature_payload = {}
        if isinstance(groups, dict) and isinstance(groups.get(feature_key), dict):
            feature_payload = groups[feature_key]
        elif isinstance(results, dict) and isinstance(results.get(feature_key), dict):
            feature_payload = results[feature_key]
        scores[feature_key] = {
            "score": scale_value(as_float(feature_payload.get("acc,none")), "percent"),
            "stderr": scale_value(as_float(feature_payload.get("acc_stderr,none")), "percent"),
        }
    return scores


def extract_detail_rows(base_row: dict[str, Any], payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary_scores = extract_primary_scores(payload)

    for spec in PRIMARY_EVALS:
        score = primary_scores.get(spec["key"], {}).get("score")
        stderr = primary_scores.get(spec["key"], {}).get("stderr")
        rows.append(
            {
                **base_row,
                "eval_dataset": spec["display"],
                "eval_key": spec["key"],
                "metric": spec["metric"],
                "filter": spec["filter"],
                "score": score,
                "stderr": stderr,
            }
        )

    chrf_score = extract_metric_score(payload, WMT_CHRF_EVAL)
    rows.append(
        {
            **base_row,
            "eval_dataset": WMT_CHRF_EVAL["display"],
            "eval_key": "wmt16-de-en-chrf",
            "metric": WMT_CHRF_EVAL["metric"],
            "filter": WMT_CHRF_EVAL["filter"],
            "score": chrf_score.get("score"),
            "stderr": chrf_score.get("stderr"),
        }
    )

    feature_scores = extract_mmlu_feature_scores(payload)
    for feature_key, feature_display in MMLU_FEATURES:
        score = feature_scores.get(feature_key, {}).get("score")
        stderr = feature_scores.get(feature_key, {}).get("stderr")
        rows.append(
            {
                **base_row,
                "eval_dataset": feature_display,
                "eval_key": feature_key,
                "metric": "acc",
                "filter": "none",
                "score": score,
                "stderr": stderr,
            }
        )

    return rows


def run_name_is_selected(run_name: str, include_globs: list[str], exclude_globs: list[str]) -> bool:
    if include_globs and not any(fnmatch.fnmatchcase(run_name, pattern) for pattern in include_globs):
        return False
    if exclude_globs and any(fnmatch.fnmatchcase(run_name, pattern) for pattern in exclude_globs):
        return False
    return True


def build_rows(
    target_dir: Path,
    include_globs: list[str],
    exclude_globs: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for run_dir in sorted(path for path in target_dir.iterdir() if path.is_dir()):
        if not run_name_is_selected(run_dir.name, include_globs, exclude_globs):
            continue

        method, model, edit_dataset = split_run_name(run_dir.name)
        payload, source_type, source_path = load_run_payload(run_dir)
        base_row = {
            "edit_dataset": display_edit_dataset(edit_dataset),
            "edit_dataset_key": edit_dataset,
            "model": model,
            "method": method,
            "run_dir": str(run_dir),
            "source_type": source_type,
            "source_path": source_path or "",
            "status": "complete" if payload else source_type,
        }

        primary_scores = extract_primary_scores(payload)
        chrf_score = extract_metric_score(payload, WMT_CHRF_EVAL)
        feature_scores = extract_mmlu_feature_scores(payload)
        bleu_avg_values = [
            primary_scores.get(spec["key"], {}).get("score")
            for spec in PRIMARY_EVALS
            if primary_scores.get(spec["key"], {}).get("score") is not None
        ]
        chrf_avg_values = [
            primary_scores.get(spec["key"], {}).get("score")
            for spec in PRIMARY_EVALS
            if spec["key"] != "wmt16-de-en" and primary_scores.get(spec["key"], {}).get("score") is not None
        ]
        if chrf_score.get("score") is not None:
            chrf_avg_values.append(chrf_score["score"])

        overall_avg = sum(bleu_avg_values) / len(bleu_avg_values) if len(bleu_avg_values) == len(PRIMARY_EVALS) else None
        overall_avg_chrf = sum(chrf_avg_values) / len(chrf_avg_values) if len(chrf_avg_values) == len(PRIMARY_EVALS) else None

        row = {
            **base_row,
            "gsm8k": primary_scores.get("gsm8k", {}).get("score"),
            "mmlu": primary_scores.get("mmlu", {}).get("score"),
            "nq_open": primary_scores.get("nq_open", {}).get("score"),
            "sst2": primary_scores.get("sst2", {}).get("score"),
            "wmt16_de_en_bleu": primary_scores.get("wmt16-de-en", {}).get("score"),
            "wmt16_de_en_chrf": chrf_score.get("score"),
            "mmlu_humanities": feature_scores.get("mmlu_humanities", {}).get("score"),
            "mmlu_other": feature_scores.get("mmlu_other", {}).get("score"),
            "mmlu_social_sciences": feature_scores.get("mmlu_social_sciences", {}).get("score"),
            "mmlu_stem": feature_scores.get("mmlu_stem", {}).get("score"),
            "avg_5_tasks": overall_avg,
            "avg_5_tasks_chrf": overall_avg_chrf,
        }
        overall_rows.append(row)
        detail_rows.extend(extract_detail_rows(base_row, payload))

        if not payload:
            skipped.append(base_row)

    overall_rows.sort(key=run_sort_key)
    detail_rows.sort(key=lambda row: (*run_sort_key(row), str(row.get("eval_key", ""))))
    return overall_rows, detail_rows, skipped


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str], digits: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = format_score(value, digits)
                out[column] = value
            writer.writerow(out)


def write_markdown(path: Path, headers: list[str], body: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overall_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    digits: int,
    avg_key: str,
    avg_header: str,
) -> None:
    headers = [
        "Dataset",
        "Model",
        "Method",
        "GSM8K",
        "MMLU",
        "NQ Open",
        "SST-2",
        "WMT16 BLEU",
        "WMT16 chrF",
        avg_header,
        "Status",
    ]
    body = [
        [
            str(row["edit_dataset"]),
            str(row["model"]),
            str(row["method"]),
            format_score(row.get("gsm8k"), digits),
            format_score(row.get("mmlu"), digits),
            format_score(row.get("nq_open"), digits),
            format_score(row.get("sst2"), digits),
            format_score(row.get("wmt16_de_en_bleu"), digits),
            format_score(row.get("wmt16_de_en_chrf"), digits),
            format_score(row.get(avg_key), digits),
            str(row["status"]),
        ]
        for row in rows
    ]
    write_markdown(path, headers, body)


def write_detail_markdown(path: Path, rows: list[dict[str, Any]], digits: int) -> None:
    headers = ["Dataset", "Model", "Method", "Eval", "Metric", "Score", "Stderr", "Status"]
    body = [
        [
            str(row["edit_dataset"]),
            str(row["model"]),
            str(row["method"]),
            str(row["eval_dataset"]),
            str(row["metric"]),
            format_score(row.get("score"), digits),
            format_score(row.get("stderr"), digits),
            str(row["status"]),
        ]
        for row in rows
    ]
    write_markdown(path, headers, body)


def write_overall_latex(
    path: Path,
    rows: list[dict[str, Any]],
    digits: int,
    include_incomplete: bool,
    avg_key: str,
    avg_header: str,
    avg_metric_name: str,
    label: str,
) -> None:
    table_rows = rows if include_incomplete else [row for row in rows if row["status"] == "complete"]
    headers = ["Dataset", "Model", "Method", "GSM8K", "MMLU", "NQ", "SST-2", "BLEU", "chrF", avg_header]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in table_rows:
        values = [
            str(row["edit_dataset"]),
            str(row["model"]),
            str(row["method"]),
            format_score(row.get("gsm8k"), digits),
            format_score(row.get("mmlu"), digits),
            format_score(row.get("nq_open"), digits),
            format_score(row.get("sst2"), digits),
            format_score(row.get("wmt16_de_en_bleu"), digits),
            format_score(row.get("wmt16_de_en_chrf"), digits),
            format_score(row.get(avg_key), digits),
        ]
        lines.append(" & ".join(latex_escape(value) for value in values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{lm-eval downstream results. Scores are percentages for "
                r"GSM8K, MMLU, NQ Open, and SST-2; WMT16 reports BLEU and chrF. Avg. "
                rf"uses WMT16 {avg_metric_name} as the fifth score.}}"
            ),
            rf"\label{{{latex_escape(label)}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_detail_latex(
    path: Path,
    rows: list[dict[str, Any]],
    digits: int,
    include_incomplete: bool,
) -> None:
    table_rows = rows if include_incomplete else [row for row in rows if row["status"] == "complete"]
    headers = ["Dataset", "Model", "Method", "Eval", "Metric", "Score"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllllr}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in table_rows:
        values = [
            str(row["edit_dataset"]),
            str(row["model"]),
            str(row["method"]),
            str(row["eval_dataset"]),
            str(row["metric"]),
            format_score(row.get("score"), digits),
        ]
        lines.append(" & ".join(latex_escape(value) for value in values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Per-dataset lm-eval scores with MMLU category scores. "
                r"Scores are percentages except WMT16, which uses BLEU.}"
            ),
            r"\label{tab:lm-eval-dataset-scores}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_scores_markdown(path: Path, rows: list[dict[str, Any]], digits: int) -> None:
    headers = [
        "Dataset",
        "Model",
        "Method",
        "GSM8K",
        "MMLU",
        "Hum.",
        "Other",
        "Social",
        "STEM",
        "NQ Open",
        "SST-2",
        "WMT16 BLEU",
        "WMT16 chrF",
        "Status",
    ]
    body = [
        [
            str(row["edit_dataset"]),
            str(row["model"]),
            str(row["method"]),
            format_score(row.get("gsm8k"), digits),
            format_score(row.get("mmlu"), digits),
            format_score(row.get("mmlu_humanities"), digits),
            format_score(row.get("mmlu_other"), digits),
            format_score(row.get("mmlu_social_sciences"), digits),
            format_score(row.get("mmlu_stem"), digits),
            format_score(row.get("nq_open"), digits),
            format_score(row.get("sst2"), digits),
            format_score(row.get("wmt16_de_en_bleu"), digits),
            format_score(row.get("wmt16_de_en_chrf"), digits),
            str(row["status"]),
        ]
        for row in rows
    ]
    write_markdown(path, headers, body)


def write_dataset_scores_latex(
    path: Path,
    rows: list[dict[str, Any]],
    digits: int,
    include_incomplete: bool,
) -> None:
    table_rows = rows if include_incomplete else [row for row in rows if row["status"] == "complete"]
    headers = ["Dataset", "Model", "Method", "GSM8K", "MMLU", "Hum.", "Other", "Social", "STEM", "NQ", "SST-2", "BLEU", "chrF"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllrrrrrrrrrr}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in table_rows:
        values = [
            str(row["edit_dataset"]),
            str(row["model"]),
            str(row["method"]),
            format_score(row.get("gsm8k"), digits),
            format_score(row.get("mmlu"), digits),
            format_score(row.get("mmlu_humanities"), digits),
            format_score(row.get("mmlu_other"), digits),
            format_score(row.get("mmlu_social_sciences"), digits),
            format_score(row.get("mmlu_stem"), digits),
            format_score(row.get("nq_open"), digits),
            format_score(row.get("sst2"), digits),
            format_score(row.get("wmt16_de_en_bleu"), digits),
            format_score(row.get("wmt16_de_en_chrf"), digits),
        ]
        lines.append(" & ".join(latex_escape(value) for value in values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Per-dataset lm-eval scores with MMLU category scores. "
                r"Scores are percentages except WMT16 BLEU and chrF, which are reported as lm-eval outputs.}"
            ),
            r"\label{tab:lm-eval-dataset-scores}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_summary(
    path: Path,
    target_dir: Path,
    overall_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    digits: int,
) -> None:
    payload = {
        "target_dir": str(target_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "digits": digits,
        "primary_metrics": PRIMARY_EVALS,
        "mmlu_features": MMLU_FEATURES,
        "notes": [
            "GSM8K, MMLU, NQ Open, and SST-2 scores are multiplied by 100.",
            "WMT16 De-En BLEU and chrF are reported as lm-eval outputs.",
            "avg_5_tasks uses WMT16 BLEU as the fifth score.",
            "avg_5_tasks_chrf uses WMT16 chrF as the fifth score.",
        ],
        "wmt_chrf_metric": WMT_CHRF_EVAL,
        "skipped_or_incomplete": skipped,
        "overall_rows": overall_rows,
        "detail_rows": detail_rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.resolve()
    overall_rows, detail_rows, skipped = build_rows(
        target_dir,
        args.include_run_glob,
        args.exclude_run_glob,
    )

    overall_columns = [
        "edit_dataset",
        "model",
        "method",
        "gsm8k",
        "mmlu",
        "nq_open",
        "sst2",
        "wmt16_de_en_bleu",
        "wmt16_de_en_chrf",
        "avg_5_tasks",
        "avg_5_tasks_chrf",
        "status",
        "source_type",
        "source_path",
        "run_dir",
    ]
    chrf_overall_columns = [
        "edit_dataset",
        "model",
        "method",
        "gsm8k",
        "mmlu",
        "nq_open",
        "sst2",
        "wmt16_de_en_bleu",
        "wmt16_de_en_chrf",
        "avg_5_tasks_chrf",
        "status",
        "source_type",
        "source_path",
        "run_dir",
    ]
    dataset_score_columns = [
        "edit_dataset",
        "model",
        "method",
        "gsm8k",
        "mmlu",
        "mmlu_humanities",
        "mmlu_other",
        "mmlu_social_sciences",
        "mmlu_stem",
        "nq_open",
        "sst2",
        "wmt16_de_en_bleu",
        "wmt16_de_en_chrf",
        "status",
        "source_type",
        "source_path",
        "run_dir",
    ]
    long_detail_columns = [
        "edit_dataset",
        "model",
        "method",
        "eval_dataset",
        "metric",
        "filter",
        "score",
        "stderr",
        "status",
        "source_type",
        "source_path",
        "run_dir",
    ]

    overall_prefix = target_dir / args.overall_prefix
    chrf_overall_name = (
        args.overall_prefix.replace("_table", "_chrf_table")
        if "_table" in args.overall_prefix
        else f"{args.overall_prefix}_chrf"
    )
    chrf_overall_prefix = target_dir / chrf_overall_name
    detail_prefix = target_dir / args.detail_prefix
    write_csv(overall_prefix.with_suffix(".csv"), overall_rows, overall_columns, args.digits)
    write_overall_markdown(
        overall_prefix.with_suffix(".md"),
        overall_rows,
        args.digits,
        "avg_5_tasks",
        "Avg. (BLEU)",
    )
    write_overall_latex(
        overall_prefix.with_suffix(".tex"),
        overall_rows,
        args.digits,
        args.include_incomplete_latex,
        "avg_5_tasks",
        "Avg.",
        "BLEU",
        "tab:lm-eval-five-task-average",
    )
    write_csv(chrf_overall_prefix.with_suffix(".csv"), overall_rows, chrf_overall_columns, args.digits)
    write_overall_markdown(
        chrf_overall_prefix.with_suffix(".md"),
        overall_rows,
        args.digits,
        "avg_5_tasks_chrf",
        "Avg. (chrF)",
    )
    write_overall_latex(
        chrf_overall_prefix.with_suffix(".tex"),
        overall_rows,
        args.digits,
        args.include_incomplete_latex,
        "avg_5_tasks_chrf",
        "Avg.",
        "chrF",
        "tab:lm-eval-five-task-average-chrf",
    )

    write_csv(detail_prefix.with_suffix(".csv"), overall_rows, dataset_score_columns, args.digits)
    write_dataset_scores_markdown(detail_prefix.with_suffix(".md"), overall_rows, args.digits)
    write_dataset_scores_latex(
        detail_prefix.with_suffix(".tex"),
        overall_rows,
        args.digits,
        args.include_incomplete_latex,
    )
    write_csv(
        detail_prefix.with_name(f"{detail_prefix.name}_long").with_suffix(".csv"),
        detail_rows,
        long_detail_columns,
        args.digits,
    )

    summary_json_path = target_dir / args.summary_json_name
    write_json_summary(
        summary_json_path,
        target_dir,
        overall_rows,
        detail_rows,
        skipped,
        args.digits,
    )

    print(f"Wrote {overall_prefix.with_suffix('.csv')}")
    print(f"Wrote {overall_prefix.with_suffix('.md')}")
    print(f"Wrote {overall_prefix.with_suffix('.tex')}")
    print(f"Wrote {chrf_overall_prefix.with_suffix('.csv')}")
    print(f"Wrote {chrf_overall_prefix.with_suffix('.md')}")
    print(f"Wrote {chrf_overall_prefix.with_suffix('.tex')}")
    print(f"Wrote {detail_prefix.with_suffix('.csv')}")
    print(f"Wrote {detail_prefix.with_suffix('.md')}")
    print(f"Wrote {detail_prefix.with_suffix('.tex')}")
    print(f"Wrote {detail_prefix.with_name(f'{detail_prefix.name}_long').with_suffix('.csv')}")
    print(f"Wrote {summary_json_path}")
    if skipped:
        print(f"Skipped/incomplete runs: {len(skipped)}")


if __name__ == "__main__":
    main()
