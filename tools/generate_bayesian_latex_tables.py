#!/usr/bin/env python3
"""Generate publication-ready LaTeX tables from Bayesian comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


METRICS = ("ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID")
STAT_METRICS = ("CMD", "ADE", "MMADE", "FDE", "MMFDE", "FID")
DISPLAY_NAMES = {
    "card": "CARD",
    "star_hand": "STAR-Hand",
    "fihard": "FI-HARD",
    "comusion": "CoMusion",
}
MATH_NAMES = {
    "card": r"\mathrm{CARD}",
    "star_hand": r"\text{STAR-Hand}",
    "fihard": r"\text{FI-HARD}",
    "comusion": r"\mathrm{CoMusion}",
}
REPORT_STAT_RE = re.compile(
    r"(?P<model>[a-z_]+) \(MD=(?P<median>[0-9.]+)\+-(?P<half>[0-9.]+), "
    r"MAD=(?P<mad>[0-9.]+)\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def latex_label_name(model: str) -> str:
    return model.replace("_", "-")


def probability(value: float) -> str:
    return f"{value:.3f}"


def decision(row: dict[str, str]) -> str:
    options = (
        (float(row["p_left_better"]), DISPLAY_NAMES[row["left_model"]]),
        (float(row["p_rope"]), "ROPE"),
        (float(row["p_right_better"]), DISPLAY_NAMES[row["right_model"]]),
    )
    best_probability, best_name = max(options)
    return best_name if best_probability >= 1.0 - float(row["alpha"]) else "Inconclusive"


def report_stats(report_path: Path) -> dict[str, tuple[float, float, float]]:
    matches = REPORT_STAT_RE.finditer(report_path.read_text(encoding="utf-8"))
    result = {}
    for match in matches:
        result[match["model"]] = (
            float(match["median"]),
            float(match["mad"]),
            float(match["half"]),
        )
    if len(result) != 2:
        raise RuntimeError(f"Could not parse two populations from {report_path}")
    return result


def main() -> None:
    args = parse_args()
    summary_path = args.artifact_dir / "bayesian_pairwise_summary.csv"
    metrics_path = args.artifact_dir / "all_sample_metrics.csv"

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))

    pair_order = []
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in summary:
        pair = (row["left_model"], row["right_model"])
        if pair not in grouped:
            pair_order.append(pair)
            grouped[pair] = {}
        grouped[pair][row["metric"]] = row

    medians = {
        (model, metric): statistics.median(
            float(row[f"{model}__{metric}"]) for row in sample_rows
        )
        for model in DISPLAY_NAMES
        for metric in METRICS
    }

    out = []
    for left, right in pair_order:
        left_name = DISPLAY_NAMES[left]
        right_name = DISPLAY_NAMES[right]
        slug = f"{latex_label_name(left)}_{latex_label_name(right)}"
        out.extend(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                (
                    r"\caption{Bayesian counterpart of the Wilcoxon signed-rank "
                    f"comparison between {left_name} and {right_name} on the Assembly "
                    r"\textit{pick up} split. Probabilities are posterior probabilities "
                    f"for {left_name} better, practical equivalence, and {right_name} "
                    r"better, respectively. Lower is better for all reported metrics. "
                    r"APD is omitted because the evaluation script does not run the "
                    r"Bayesian test on that diversity metric.}"
                ),
                rf"\label{{tab:bayesian_{slug}}}",
                r"\begin{adjustbox}{max width=\columnwidth}",
                r"\begin{tabular}{lrrrrrrc}",
                r"\toprule",
                (
                    f"Metric & $n$ & Median {left_name} & Median {right_name} & "
                    f"$p({MATH_NAMES[left]})$ & $p(\\mathrm{{ROPE}})$ & "
                    f"$p({MATH_NAMES[right]})$ & Decision \\\\"
                ),
                r"\midrule",
            ]
        )
        for metric in METRICS:
            row = grouped[(left, right)][metric]
            out.append(
                f"{metric:<5} & {row['n_pairs']} & {medians[left, metric]:.4f} & "
                f"{medians[right, metric]:.4f} & "
                f"{probability(float(row['p_left_better']))} & "
                f"{probability(float(row['p_rope']))} & "
                f"{probability(float(row['p_right_better']))} & "
                f"{decision(row)} \\\\"
            )
        out.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{adjustbox}",
                r"\end{table}",
                "",
                r"\begin{table}[htb]",
                r"\centering",
                r"\small",
                (
                    r"\caption{Statistical summary of forecasting metrics comparing "
                    f"{left_name} and {right_name}. MED: median, MAD: median absolute "
                    r"deviation, CI: 95\% interval reported by the Bayesian counterpart "
                    r"of the Wilcoxon signed-rank summary.}"
                ),
                rf"\label{{tab:stat_{slug}}}",
                r"\begin{adjustbox}{max width=\columnwidth}",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"Metric & MED & MAD & CI (95\%) \\",
                r"\midrule",
            ]
        )
        for metric_index, metric in enumerate(STAT_METRICS):
            path = (
                args.artifact_dir
                / "reports"
                / f"{left}_vs_{right}__{metric.lower()}_report.txt"
            )
            stats = report_stats(path)
            for model in (left, right):
                median, mad, half = stats[model]
                out.append(
                    f"{metric} ({DISPLAY_NAMES[model]}) & {median:.3f} & {mad:.3f} & "
                    f"[{median - half:.3f},\\,{median + half:.3f}] \\\\"
                )
            if metric_index != len(STAT_METRICS) - 1:
                out.append("")
        out.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{adjustbox}",
                r"\end{table}",
                "",
            ]
        )

    args.output.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
