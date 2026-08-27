#!/usr/bin/env python3
"""Build paper-style summary tables from EasyEdit eval.json files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("pre_rewrite_acc", "Pre-R"),
    ("pre_rephrase_acc", "Pre-G"),
    ("rewrite_acc", "Edit"),
    ("rephrase_acc", "Gen."),
    ("locality_acc", "Loc."),
    ("post_avg", "Avg."),
)
POST_METRICS = ("rewrite_acc", "rephrase_acc", "locality_acc")
DATASET_DISPLAY = {
    "counterfact": "CounterFact",
    "zsre": "ZSRE",
}
EASYEDIT_EVAL_DIR = "eval_results_easyedit"
METHOD_CONTAINER_DIRS = {"baseline", "models"}
NON_METHOD_OUTPUT_DIRS = {"evaluation", "lm_eval"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively collect EasyEdit eval.json files and write paper-style "
            "CSV/Markdown/LaTeX/JSON summary tables."
        )
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory to scan for EasyEdit eval.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="eval.json",
        help="Filename glob used to find EasyEdit result files recursively.",
    )
    parser.add_argument(
        "--output-prefix",
        default="easyedit_paper_table",
        help="Output prefix written inside target_dir.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places for table scores.",
    )
    parser.add_argument(
        "--scale",
        choices=("percent", "ratio"),
        default="percent",
        help="Write scores as 0-100 percentages or raw 0-1 ratios.",
    )
    parser.add_argument(
        "--caption",
        default="EasyEdit teacher-forcing evaluation results.",
        help="Caption used in the LaTeX table.",
    )
    parser.add_argument(
        "--label",
        default="tab:easyedit-results",
        help="Label used in the LaTeX table.",
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON is not an object")
    return payload


def dataset_from_summary(summary: dict[str, Any], eval_path: Path) -> str:
    data_path = summary.get("data_path")
    if isinstance(data_path, str) and data_path:
        stem = Path(data_path).stem
        return re.sub(r"_\d+k$", "", stem)

    # eval_results_easyedit/<dataset-split>/<method>/<run>/eval.json
    parts = eval_path.parts
    if EASYEDIT_EVAL_DIR in parts:
        index = parts.index(EASYEDIT_EVAL_DIR)
        if index + 1 < len(parts):
            dataset_dir = parts[index + 1]
            return re.sub(r"-(edit|mend-eval|eval)$", "", dataset_dir)

    dataset_dir = eval_path.parent.parent.name
    if not (
        dataset_dir.lower() in DATASET_DISPLAY
        or re.search(r"-(edit|mend-eval|eval)$", dataset_dir)
    ):
        dataset_dir = eval_path.parent.parent.parent.name
    return re.sub(r"-(edit|mend-eval|eval)$", "", dataset_dir)


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset.lower(), dataset)


def model_from_summary(summary: dict[str, Any], run_name: str, dataset: str) -> str:
    base_model_path = summary.get("base_model_path")
    if isinstance(base_model_path, str) and base_model_path:
        return Path(base_model_path).name

    marker = f"_{dataset}_"
    if marker in run_name:
        return run_name.split(marker, 1)[0]

    return run_name.split("_", 1)[0]


def method_from_eval_path(eval_path: Path) -> str | None:
    parts = eval_path.parts
    if EASYEDIT_EVAL_DIR not in parts:
        return None

    index = parts.index(EASYEDIT_EVAL_DIR)
    if index + 4 >= len(parts):
        return None

    return normalize_method_name(parts[index + 2])


def method_from_model_path(model_path: str) -> str | None:
    parts = Path(model_path).parts
    for index, part in enumerate(parts):
        if part != "outputs" or index + 1 >= len(parts):
            continue

        candidate = parts[index + 1]
        candidate_key = candidate.lower()
        if candidate_key in NON_METHOD_OUTPUT_DIRS:
            continue
        if candidate_key in METHOD_CONTAINER_DIRS and index + 2 < len(parts):
            return normalize_method_name(parts[index + 2])
        return normalize_method_name(candidate)

    return None


def method_from_summary(summary: dict[str, Any], run_name: str, eval_path: Path) -> str:
    method = method_from_eval_path(eval_path)
    if method:
        return method

    model_path = summary.get("model_path")
    if isinstance(model_path, str) and model_path:
        method = method_from_model_path(model_path)
        if method:
            return method

    lower_name = run_name.lower()
    if "kledit" in lower_name:
        return "KLEdit"
    if "memit" in lower_name:
        return "MEMIT"
    if "rome" in lower_name:
        return "ROME"
    if "mend" in lower_name:
        return "MEND"

    return "Unknown"


def normalize_method_name(method: str) -> str:
    if method.lower() == "kledit":
        return "KLEdit"
    return method


def config_from_run_name(run_name: str) -> str:
    match = re.search(r"_\d+kuse_(.+)$", run_name)
    if match:
        return match.group(1)
    return ""


def collect_rows(target_dir: Path, pattern: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for eval_path in sorted(target_dir.rglob(pattern)):
        if not eval_path.is_file():
            continue

        try:
            payload = load_json(eval_path)
        except Exception as exc:
            skipped.append({"path": str(eval_path), "reason": str(exc)})
            continue

        summary = payload.get("summary")
        if not isinstance(summary, dict):
            skipped.append({"path": str(eval_path), "reason": "missing summary object"})
            continue

        metric_type = summary.get("metric_type")
        if metric_type != "easyedit_teacher_forcing":
            skipped.append(
                {
                    "path": str(eval_path),
                    "reason": f"unsupported metric_type: {metric_type!r}",
                }
            )
            continue

        metrics = {
            key: float(summary[key])
            for key, _ in METRIC_COLUMNS
            if key != "post_avg" and is_number(summary.get(key))
        }
        post_values = [metrics[key] for key in POST_METRICS if key in metrics]
        if post_values:
            metrics["post_avg"] = sum(post_values) / len(post_values)

        missing_metrics = [
            key for key, _ in METRIC_COLUMNS if key != "post_avg" and key not in metrics
        ]
        if missing_metrics:
            skipped.append(
                {
                    "path": str(eval_path),
                    "reason": f"missing metrics: {', '.join(missing_metrics)}",
                }
            )
            continue

        run_name = eval_path.parent.name
        dataset = dataset_from_summary(summary, eval_path)
        row = {
            "dataset": dataset,
            "dataset_display": display_dataset(dataset),
            "model": model_from_summary(summary, run_name, dataset),
            "method": method_from_summary(summary, run_name, eval_path),
            "config": config_from_run_name(run_name),
            "n": summary.get("num_samples_used", ""),
            "metric_type": metric_type,
            "run_name": run_name,
            "eval_json": str(eval_path),
            "metrics": metrics,
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["dataset_display"], row["model"], row["method"], row["run_name"]))
    return rows, skipped


def scale_value(value: float, scale: str) -> float:
    return value * 100.0 if scale == "percent" else value


def format_score(value: float | None, scale: str, digits: int) -> str:
    if value is None:
        return ""
    return f"{scale_value(value, scale):.{digits}f}"


def table_headers() -> list[str]:
    return ["Dataset", "Model", "Method"] + [label for _, label in METRIC_COLUMNS] + ["N"]


def table_row(row: dict[str, Any], scale: str, digits: int) -> list[str]:
    metrics = row["metrics"]
    return [
        row["dataset_display"],
        row["model"],
        row["method"],
        *[
            format_score(metrics.get(key), scale, digits)
            for key, _ in METRIC_COLUMNS
        ],
        str(row["n"]),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], scale: str, digits: int) -> None:
    fieldnames = [
        "dataset",
        "model",
        "method",
        "pre_rewrite_acc",
        "pre_rephrase_acc",
        "rewrite_acc",
        "rephrase_acc",
        "locality_acc",
        "post_avg",
        "n",
        "metric_type",
        "config",
        "run_name",
        "eval_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "dataset": row["dataset_display"],
                    "model": row["model"],
                    "method": row["method"],
                    "pre_rewrite_acc": format_score(metrics.get("pre_rewrite_acc"), scale, digits),
                    "pre_rephrase_acc": format_score(metrics.get("pre_rephrase_acc"), scale, digits),
                    "rewrite_acc": format_score(metrics.get("rewrite_acc"), scale, digits),
                    "rephrase_acc": format_score(metrics.get("rephrase_acc"), scale, digits),
                    "locality_acc": format_score(metrics.get("locality_acc"), scale, digits),
                    "post_avg": format_score(metrics.get("post_avg"), scale, digits),
                    "n": row["n"],
                    "metric_type": row["metric_type"],
                    "config": row["config"],
                    "run_name": row["run_name"],
                    "eval_json": row["eval_json"],
                }
            )


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    widths = [
        max(len(str(value)) for value in [header] + [row[index] for row in body])
        for index, header in enumerate(headers)
    ]
    lines = []
    lines.append("| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |")
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    for row in body:
        lines.append("| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, rows: list[dict[str, Any]], scale: str, digits: int) -> None:
    headers = table_headers()
    body = [table_row(row, scale, digits) for row in rows]
    unit = "percent" if scale == "percent" else "ratio"
    text = [
        "# EasyEdit Evaluation Summary",
        "",
        f"Scores are reported as {unit}. `Avg.` is the mean of Edit, Gen., and Loc.",
        "",
        markdown_table(headers, body).rstrip(),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


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


def write_latex(
    path: Path,
    rows: list[dict[str, Any]],
    scale: str,
    digits: int,
    caption: str,
    label: str,
) -> None:
    headers = table_headers()
    body = [table_row(row, scale, digits) for row in rows]
    unit_note = "Scores are percentages." if scale == "percent" else "Scores are ratios."

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in body:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)} {unit_note} Avg. is the mean of Edit, Gen., and Loc.}}",
            rf"\label{{{latex_escape(label)}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    path: Path,
    rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    target_dir: Path,
    pattern: str,
    scale: str,
    digits: int,
) -> None:
    payload = {
        "target_dir": str(target_dir),
        "pattern": pattern,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scale": scale,
        "digits": digits,
        "result_count": len(rows),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "rows": rows,
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def main() -> int:
    args = parse_args()
    target_dir = args.target_dir.resolve()
    if not target_dir.is_dir():
        raise SystemExit(f"Target directory does not exist: {target_dir}")

    rows, skipped = collect_rows(target_dir, args.pattern)
    if not rows:
        raise SystemExit(f"No EasyEdit eval.json files matched under {target_dir}")

    output_prefix = target_dir / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows, args.scale, args.digits)
    write_markdown(output_prefix.with_suffix(".md"), rows, args.scale, args.digits)
    write_latex(
        output_prefix.with_suffix(".tex"),
        rows,
        args.scale,
        args.digits,
        args.caption,
        args.label,
    )
    write_json(
        output_prefix.with_suffix(".json"),
        rows,
        skipped,
        target_dir,
        args.pattern,
        args.scale,
        args.digits,
    )

    print(
        f"[done] Wrote {len(rows)} rows to "
        f"{output_prefix.with_suffix('.csv')}, "
        f"{output_prefix.with_suffix('.md')}, "
        f"{output_prefix.with_suffix('.tex')}, "
        f"{output_prefix.with_suffix('.json')}"
    )
    if skipped:
        print(f"[warn] Skipped {len(skipped)} files; see JSON output for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
