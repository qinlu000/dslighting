#!/usr/bin/env python3
"""Plot audited semantic task categories for the 2,000-task training mix."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.patches import Patch

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = PAPER_ROOT / "artifacts" / "annotations" / "taxonomy_annotations.jsonl"
DEFAULT_TAXONOMY = PAPER_ROOT / "config" / "taxonomy_v1.yaml"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "artifacts" / "figures"

STAGE_ORDER = ["data_understanding", "data_preparation", "modeling", "evaluation", "unknown"]
STAGE_LABELS = {
    "data_understanding": "Data understanding",
    "data_preparation": "Data preparation",
    "modeling": "Modeling",
    "evaluation": "Evaluation",
    "unknown": "Unresolved",
}
STAGE_COLORS = {
    "data_understanding": "#3B6FB6",
    "data_preparation": "#2E8B70",
    "modeling": "#D9822B",
    "evaluation": "#7B61A8",
    "unknown": "#98A2B3",
}
DISPLAY_NAMES = {
    "record_retrieval": "Record retrieval",
    "fact_verification": "Fact verification",
    "data_quality_assessment": "Data quality assessment",
    "data_preprocessing": "Data preprocessing",
    "feature_engineering": "Feature engineering",
    "counting": "Counting",
    "aggregation": "Aggregation",
    "arithmetic_calculation": "Arithmetic calculation",
    "descriptive_statistics": "Descriptive statistics",
    "distribution_analysis": "Distribution analysis",
    "comparison": "Comparison",
    "ranking": "Ranking / Top-K",
    "correlation_analysis": "Correlation analysis",
    "statistical_inference": "Statistical inference",
    "anomaly_detection": "Anomaly detection",
    "temporal_analysis": "Temporal analysis",
    "causal_analysis": "Causal analysis",
    "impact_analysis": "Impact analysis",
    "classification_modeling": "Classification modeling",
    "regression_modeling": "Regression modeling",
    "time_series_forecasting": "Time-series forecasting",
    "unresolved": "Unresolved",
}


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    task_ids = [str(row.get("task_id")) for row in rows]
    if len(rows) != 2000 or len(set(task_ids)) != 2000:
        raise ValueError("Expected exactly 2,000 unique taxonomy annotations")
    failures = [row["task_id"] for row in rows if row.get("status") != "ok"]
    if failures:
        raise ValueError(f"Cannot plot {len(failures)} failed annotations")
    return rows


def load_task_metadata(path: Path) -> dict[str, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in raw["primary_tasks"]}


def shade(base: str, index: int, count: int) -> str:
    rgb = mcolors.to_rgb(base)
    if count <= 1:
        return base
    position = index / (count - 1)
    factor = 0.62 + 0.55 * position
    if factor <= 1:
        color = tuple(channel * factor for channel in rgb)
    else:
        amount = factor - 1
        color = tuple(channel + (1 - channel) * amount for channel in rgb)
    return mcolors.to_hex(color)


def distribute_labels(items: list[dict], minimum: float, maximum: float, gap: float) -> None:
    items.sort(key=lambda item: item["desired_y"])
    positions: list[float] = []
    for item in items:
        positions.append(max(item["desired_y"], positions[-1] + gap if positions else minimum))
    if positions and positions[-1] > maximum:
        shift = positions[-1] - maximum
        positions = [position - shift for position in positions]
    if positions and positions[0] < minimum:
        shift = minimum - positions[0]
        positions = [position + shift for position in positions]
    for item, position in zip(items, positions):
        item["label_y"] = position


def plot(annotations: Path, taxonomy_path: Path, output_dir: Path) -> None:
    rows = load_rows(annotations)
    metadata = load_task_metadata(taxonomy_path)
    counts = Counter(row["final"]["primary_task"] for row in rows)
    unknown = set(counts) - set(metadata)
    if unknown:
        raise ValueError(f"Unknown primary task labels: {sorted(unknown)}")

    categories: list[dict] = []
    for stage in STAGE_ORDER:
        stage_items = [
            (task_id, count)
            for task_id, count in counts.items()
            if metadata[task_id]["stage"] == stage
        ]
        stage_items.sort(key=lambda item: (-item[1], item[0]))
        for index, (task_id, count) in enumerate(stage_items):
            categories.append(
                {
                    "task_id": task_id,
                    "stage": stage,
                    "count": count,
                    "color": shade(STAGE_COLORS[stage], index, len(stage_items)),
                }
            )

    total = len(rows)
    stage_counts = Counter()
    for item in categories:
        stage_counts[item["stage"]] += item["count"]
    present_stages = [stage for stage in STAGE_ORDER if stage_counts[stage]]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#172033",
        }
    )
    figure = plt.figure(figsize=(13.2, 10.0))
    chart = figure.add_axes((0.055, 0.08, 0.89, 0.79))
    start_angle = 90

    outer_values = [item["count"] for item in categories]
    outer_wedges, _ = chart.pie(
        outer_values,
        radius=1.0,
        startangle=start_angle,
        counterclock=False,
        colors=[item["color"] for item in categories],
        wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 1.25},
    )

    inner_values = [stage_counts[stage] for stage in present_stages]
    _, inner_labels = chart.pie(
        inner_values,
        radius=0.695,
        startangle=start_angle,
        counterclock=False,
        labels=[
            (
                f"{STAGE_LABELS[stage]}\n{stage_counts[stage] / total:.1%}"
                if stage_counts[stage] / total >= 0.04
                else ""
            )
            for stage in present_stages
        ],
        labeldistance=0.63,
        rotatelabels=False,
        colors=[STAGE_COLORS[stage] for stage in present_stages],
        wedgeprops={"width": 0.32, "edgecolor": "white", "linewidth": 1.8},
        textprops={"color": "white", "fontsize": 8.4, "fontweight": "bold"},
    )
    for label in inner_labels:
        label.set_horizontalalignment("center")

    chart.text(0, 0.035, f"{total:,}", ha="center", va="center", fontsize=25, fontweight="bold")
    chart.text(0, -0.095, "tasks", ha="center", va="center", fontsize=11, color="#667085")
    chart.set(aspect="equal")

    by_side: dict[int, list[dict]] = {-1: [], 1: []}
    for wedge, item in zip(outer_wedges, categories):
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
        x, y = math.cos(angle), math.sin(angle)
        side = 1 if x >= 0 else -1
        share = item["count"] / total
        by_side[side].append(
            {
                "anchor": (1.01 * x, 1.01 * y),
                "desired_y": 1.25 * y,
                "text": f"{DISPLAY_NAMES[item['task_id']]}  {item['count']} ({share:.1%})",
                "color": item["color"],
            }
        )
        if share >= 0.045:
            chart.text(
                0.855 * x,
                0.855 * y,
                f"{share:.1%}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=(
                    "white"
                    if mcolors.rgb_to_hsv(mcolors.to_rgb(item["color"]))[2] < 0.72
                    else "#172033"
                ),
            )

    for side, items in by_side.items():
        distribute_labels(items, minimum=-1.24, maximum=1.24, gap=0.115)
        for item in items:
            chart.annotate(
                item["text"],
                xy=item["anchor"],
                xytext=(1.39 * side, item["label_y"]),
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=9.0,
                color="#344054",
                arrowprops={
                    "arrowstyle": "-",
                    "color": item["color"],
                    "linewidth": 0.95,
                    "connectionstyle": "angle3,angleA=0,angleB=90",
                },
            )

    chart.set_xlim(-1.78, 1.78)
    chart.set_ylim(-1.43, 1.36)
    figure.suptitle(
        "Semantic Task Composition of the 2,000-Task Training Mixture",
        x=0.5,
        y=0.965,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Inner ring: analytical lifecycle stage · Outer ring: audited primary task objective",
        ha="center",
        fontsize=10.8,
        color="#667085",
    )
    figure.legend(
        handles=[Patch(facecolor=STAGE_COLORS[stage]) for stage in present_stages],
        labels=[
            f"{STAGE_LABELS[stage]} {stage_counts[stage] / total:.1%}" for stage in present_stages
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.907),
        ncol=len(present_stages),
        frameon=False,
        fontsize=8.6,
        handlelength=1.2,
        columnspacing=1.5,
    )
    figure.text(
        0.5,
        0.035,
        "Primary objectives are mutually exclusive; secondary objectives and operation tags are reported separately.",
        ha="center",
        fontsize=9.5,
        color="#667085",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_dir / "taxonomy_task_donut"
    figure.savefig(basename.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(basename.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(basename.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

    pd.DataFrame(
        [
            {
                "stage": item["stage"],
                "primary_task": item["task_id"],
                "name_zh": metadata[item["task_id"]]["name_zh"],
                "count": item["count"],
                "share": item["count"] / total,
            }
            for item in categories
        ]
    ).to_csv(output_dir / "taxonomy_primary_counts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    plot(
        args.annotations.expanduser().resolve(),
        args.taxonomy.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
