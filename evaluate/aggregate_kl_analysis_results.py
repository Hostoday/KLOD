import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_METHOD_ORDER = [
    "KLEdit",
    "AlphaEdit",
    "UltraEdit",
    "LocFT-BF",
]

METHOD_ALIASES = {
    "AlphEdit": "AlphaEdit",
    "AlphaEdit": "AlphaEdit",
    "KLEdit": "KLEdit",
    "UltraEdit": "UltraEdit",
    "LocFT-BF": "LocFT-BF",
    "LocFT": "LocFT-BF",
}


def parse_method_from_name(path: Path) -> str:
    stem = path.stem
    prefix = stem.split("_", 1)[0]
    return METHOD_ALIASES.get(prefix, prefix)


def parse_dataset_from_summary(path: Path, summary: Dict[str, Any]) -> str:
    data_path = summary.get("data_path")
    if data_path:
        return Path(str(data_path)).stem

    stem = path.stem
    match = re.search(r"(counterfact|zsre)(?:_([0-9]+k))?", stem, flags=re.IGNORECASE)
    if not match:
        return "unknown"
    dataset = match.group(1).lower()
    size = match.group(2)
    return f"{dataset}_{size}" if size else dataset


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def weighted_average(
    first_value: Optional[float],
    first_count: int,
    second_value: Optional[float],
    second_count: int,
) -> Optional[float]:
    return weighted_average_many(
        [
            (first_value, first_count),
            (second_value, second_count),
        ]
    )


def weighted_average_many(items: List[Tuple[Optional[float], int]]) -> Optional[float]:
    total_count = sum(count for _, count in items)
    if total_count == 0:
        return None
    total = 0.0
    for value, count in items:
        if value is not None:
            total += value * count
    return total / total_count


def read_row(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} does not contain a summary object")

    rewrite_prompt = to_float(summary.get("rewrite_prompt_kl"))
    rewrite_target = to_float(summary.get("rewrite_target_kl"))
    rephrase_prompt = to_float(summary.get("rephrase_prompt_kl"))
    rephrase_target = to_float(summary.get("rephrase_target_kl"))
    locality_prompt = to_float(summary.get("locality_prompt_kl"))
    locality_target = to_float(summary.get("locality_target_kl"))

    rewrite_prompt_count = int(summary.get("rewrite_prompt_token_count") or 0)
    rewrite_target_count = int(summary.get("rewrite_target_token_count") or 0)
    rephrase_prompt_count = int(summary.get("rephrase_prompt_token_count") or 0)
    rephrase_target_count = int(summary.get("rephrase_target_token_count") or 0)
    locality_prompt_count = int(summary.get("locality_prompt_token_count") or 0)
    locality_target_count = int(summary.get("locality_target_token_count") or 0)

    prompt_avg = weighted_average_many(
        [
            (rewrite_prompt, rewrite_prompt_count),
            (rephrase_prompt, rephrase_prompt_count),
            (locality_prompt, locality_prompt_count),
        ]
    )
    target_avg = weighted_average_many(
        [
            (rewrite_target, rewrite_target_count),
            (rephrase_target, rephrase_target_count),
            (locality_target, locality_target_count),
        ]
    )
    overall_avg = weighted_average(
        prompt_avg,
        rewrite_prompt_count + rephrase_prompt_count + locality_prompt_count,
        target_avg,
        rewrite_target_count + rephrase_target_count + locality_target_count,
    )

    return {
        "method": parse_method_from_name(path),
        "dataset": parse_dataset_from_summary(path, summary),
        "num_samples": int(summary.get("num_samples_used") or 0),
        "kl_direction": summary.get("kl_direction"),
        "kl_definition": summary.get("kl_definition"),
        "rewrite_prompt_kl": rewrite_prompt,
        "rewrite_prompt_token_count": rewrite_prompt_count,
        "rewrite_target_kl": rewrite_target,
        "rewrite_target_token_count": rewrite_target_count,
        "rephrase_prompt_kl": rephrase_prompt,
        "rephrase_prompt_token_count": rephrase_prompt_count,
        "rephrase_target_kl": rephrase_target,
        "rephrase_target_token_count": rephrase_target_count,
        "locality_prompt_kl": locality_prompt,
        "locality_prompt_token_count": locality_prompt_count,
        "locality_target_kl": locality_target,
        "locality_target_token_count": locality_target_count,
        "prompt_avg_kl": prompt_avg,
        "target_avg_kl": target_avg,
        "overall_avg_kl": overall_avg,
        "source_file": str(path),
        "model_path": summary.get("model_path"),
    }


def method_rank(method: str, method_order: List[str]) -> int:
    try:
        return method_order.index(method)
    except ValueError:
        return len(method_order)


def format_float(value: Optional[float], digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def markdown_table(rows: List[Dict[str, Any]], digits: int) -> str:
    headers = [
        "Method",
        "Rewrite Prompt KL",
        "Rewrite Target KL",
        "Rephrase Prompt KL",
        "Rephrase Target KL",
        "Locality Prompt KL",
        "Locality Target KL",
        "Prompt Avg.",
        "Target Avg.",
        "Overall Avg.",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        values = [
            row["method"],
            format_float(row["rewrite_prompt_kl"], digits),
            format_float(row["rewrite_target_kl"], digits),
            format_float(row["rephrase_prompt_kl"], digits),
            format_float(row["rephrase_target_kl"], digits),
            format_float(row["locality_prompt_kl"], digits),
            format_float(row["locality_target_kl"], digits),
            format_float(row["prompt_avg_kl"], digits),
            format_float(row["target_avg_kl"], digits),
            format_float(row["overall_avg_kl"], digits),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows: List[Dict[str, Any]], digits: int) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Method & Rewrite Prompt & Rewrite Target & Reph. Prompt & Reph. Target & Loc. Prompt & Loc. Target & Prompt Avg. & Target Avg. & Overall \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            row["method"],
            format_float(row["rewrite_prompt_kl"], digits),
            format_float(row["rewrite_target_kl"], digits),
            format_float(row["rephrase_prompt_kl"], digits),
            format_float(row["rephrase_target_kl"], digits),
            format_float(row["locality_prompt_kl"], digits),
            format_float(row["locality_target_kl"], digits),
            format_float(row["prompt_avg_kl"], digits),
            format_float(row["target_avg_kl"], digits),
            format_float(row["overall_avg_kl"], digits),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "dataset",
        "num_samples",
        "kl_direction",
        "kl_definition",
        "rewrite_prompt_kl",
        "rewrite_prompt_token_count",
        "rewrite_target_kl",
        "rewrite_target_token_count",
        "rephrase_prompt_kl",
        "rephrase_prompt_token_count",
        "rephrase_target_kl",
        "rephrase_target_token_count",
        "locality_prompt_kl",
        "locality_prompt_token_count",
        "locality_target_kl",
        "locality_target_token_count",
        "prompt_avg_kl",
        "target_avg_kl",
        "overall_avg_kl",
        "source_file",
        "model_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs/evaluation/Analysis/kl_analysis"),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="kl_analysis_paper_table")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument(
        "--method_order",
        type=str,
        default=",".join(DEFAULT_METHOD_ORDER),
        help="Comma-separated method order for table rows.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    method_order = [method.strip() for method in args.method_order.split(",") if method.strip()]

    rows = []
    skipped_paths = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            rows.append(read_row(path))
        except ValueError as exc:
            skipped_paths.append((path, str(exc)))

    if not rows:
        raise SystemExit(f"No KL analysis result JSON files found under {input_dir}")

    rows.sort(key=lambda row: (row["dataset"], method_rank(row["method"], method_order), row["method"]))

    csv_path = output_dir / f"{args.output_prefix}.csv"
    md_path = output_dir / f"{args.output_prefix}.md"
    tex_path = output_dir / f"{args.output_prefix}.tex"
    json_path = output_dir / f"{args.output_prefix}.json"

    write_csv(csv_path, rows)
    md_path.write_text(markdown_table(rows, args.digits), encoding="utf-8")
    tex_path.write_text(latex_table(rows, args.digits), encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {json_path}")
    if skipped_paths:
        print("Skipped non-result JSON files:")
        for path, reason in skipped_paths:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    run()
