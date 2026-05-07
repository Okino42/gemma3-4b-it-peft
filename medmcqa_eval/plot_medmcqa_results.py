#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

MODEL_LABELS = {
    "gemma-3-4b-it": "Gemma-3-4B\nIT",
    "gemma-3-4b-med-dora-s1": "Med-DoRA\ns1",
    "gemma-3-4b-med-lora-newdata-r16": "Med-LoRA new\nr16",
    "gemma-3-4b-med-lora-newdata-r32": "Med-LoRA new\nr32",
    "gemma-3-4b-med-lora-s1": "Med-LoRA\ns1",
    "gemma-3-4b-med-lora-s2": "Med-LoRA\ns2",
    "lora-fulltoken-s1": "Full-token\nLoRA s1",
    "lora-fulltoken-s2": "Full-token\nLoRA s2",
    "dora-fulltoken-s1": "Full-token\nDoRA s1",
    "dora-fulltoken-s2": "Full-token\nDoRA s2",
}

GREEN_PALETTE = [
    "#1F4D3A",
    "#2E6A4C",
    "#4F8B63",
    "#74A978",
    "#9BC391",
    "#C4DDB0",
    "#E4F0D8",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-style summary figures from MedMCQA metrics."
    )
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-subjects", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_metrics(eval_config: str | Path) -> list[dict[str, Any]]:
    cfg = load_yaml(eval_config)
    output_root = Path(cfg["outputs"]["output_dir"])
    metrics_name = cfg["outputs"]["metrics_filename"]

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.glob(f"*/{metrics_name}")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["model_name"] = metrics.get("model_name") or metrics_path.parent.name
        metrics["metrics_path"] = str(metrics_path)
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError(f"No metrics files found under {output_root}")

    rows.sort(key=lambda row: (-row.get("accuracy", 0.0), row["model_name"]))
    return rows


def wrap_model_name(name: str, width: int = 18) -> str:
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]
    return textwrap.fill(name, width=width, break_long_words=False, break_on_hyphens=True)


def build_positions(n_groups: int, n_bars: int, group_width: float = 0.78) -> list[list[float]]:
    bar_width = group_width / max(n_bars, 1)
    positions: list[list[float]] = []
    for idx in range(n_bars):
        offset = -group_width / 2 + bar_width / 2 + idx * bar_width
        positions.append([group + offset for group in range(n_groups)])
    return positions


def make_legend_handles(labels: list[str], colors: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for label, color in zip(labels, colors):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=10,
                label=label,
            )
        )
    return handles


def style_axes(ax: plt.Axes, x_labels: list[str], ymax: float) -> None:
    ax.set_xlim(-0.6, len(x_labels) - 0.4)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(list(range(len(x_labels))))
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_xlabel("Model", fontsize=15)
    ax.tick_params(axis="y", labelsize=11, pad=10, length=6)
    ax.tick_params(axis="x", pad=10, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def annotate_overall(ax: plt.Axes, rows: list[dict[str, Any]], group_maxima: list[float], ymax: float) -> None:
    offset = max(0.018, ymax * 0.025)
    for idx, (row, group_max) in enumerate(zip(rows, group_maxima)):
        y = min(group_max + offset, ymax - 0.015)
        ax.text(
            idx,
            y,
            f"Overall:\n{row.get('accuracy', 0.0):.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    saved_paths: list[Path] = []
    for suffix in (".png", ".svg"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=dpi if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def plot_grouped_chart(
    rows: list[dict[str, Any]],
    categories: list[tuple[str, str]],
    values_by_model: list[list[float]],
    title: str,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    fig_width = max(17.0, len(rows) * 1.95)
    fig, ax = plt.subplots(figsize=(fig_width, 9.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = GREEN_PALETTE[: len(categories)]
    positions = build_positions(len(rows), len(categories))
    group_maxima = [max(model_values) if model_values else 0.0 for model_values in values_by_model]
    ymax = min(1.0, max(0.56, max(group_maxima) + 0.12))

    for cat_idx, (_, label) in enumerate(categories):
        category_values = [model_values[cat_idx] for model_values in values_by_model]
        ax.bar(
            positions[cat_idx],
            category_values,
            width=0.72 / max(len(categories), 1),
            color=colors[cat_idx],
            edgecolor=colors[cat_idx],
        )

    x_labels = [wrap_model_name(row["model_name"]) for row in rows]
    style_axes(ax, x_labels, ymax)
    annotate_overall(ax, rows, group_maxima, ymax)

    fig.text(0.06, 0.955, title, ha="left", va="top", fontsize=19, fontweight="bold")
    legend_handles = make_legend_handles([label for _, label in categories], colors)
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.885),
        ncol=min(3, len(categories)),
        frameon=False,
        fontsize=12,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.subplots_adjust(top=0.77, left=0.08, right=0.985, bottom=0.19)

    return save_figure(fig, output_dir, stem, dpi)


def plot_metric_chart(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> list[Path]:
    categories = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro F1"),
        ("f1_weighted", "Weighted F1"),
        ("precision_macro", "Macro Precision"),
        ("recall_macro", "Macro Recall"),
    ]
    values_by_model = [
        [float(row.get(metric_key, 0.0)) for metric_key, _ in categories]
        for row in rows
    ]
    return plot_grouped_chart(
        rows,
        categories,
        values_by_model,
        "MedMCQA scores by metric",
        output_dir,
        "medmcqa_scores_by_metric",
        dpi,
    )


def plot_choice_type_chart(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> list[Path]:
    categories = [
        ("single", "Single choice"),
        ("multi", "Multiple choice"),
    ]
    values_by_model = []
    for row in rows:
        choice_metrics = row.get("by_choice_type", {})
        values_by_model.append(
            [float(choice_metrics.get(choice_type, {}).get("accuracy", 0.0)) for choice_type, _ in categories]
        )
    return plot_grouped_chart(
        rows,
        categories,
        values_by_model,
        "MedMCQA scores by question type",
        output_dir,
        "medmcqa_scores_by_question_type",
        dpi,
    )


def select_top_subjects(rows: list[dict[str, Any]], top_k: int) -> list[tuple[str, str]]:
    support_by_subject: dict[str, int] = {}
    for row in rows:
        subject_metrics = row.get("by_subject_name", {})
        for subject, stats in subject_metrics.items():
            support = int(stats.get("total", 0))
            support_by_subject[subject] = max(support_by_subject.get(subject, 0), support)

    sorted_subjects = sorted(support_by_subject.items(), key=lambda item: (-item[1], item[0]))
    return [(subject, subject) for subject, _ in sorted_subjects[:top_k]]


def plot_subject_chart(
    rows: list[dict[str, Any]],
    output_dir: Path,
    top_subjects: int,
    dpi: int,
) -> list[Path]:
    categories = select_top_subjects(rows, top_subjects)
    values_by_model = []
    for row in rows:
        subject_metrics = row.get("by_subject_name", {})
        values_by_model.append(
            [float(subject_metrics.get(subject, {}).get("accuracy", 0.0)) for subject, _ in categories]
        )
    return plot_grouped_chart(
        rows,
        categories,
        values_by_model,
        f"MedMCQA scores by subject (top {len(categories)} by support)",
        output_dir,
        f"medmcqa_scores_by_subject_top{len(categories)}",
        dpi,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.eval_config)
    results_dir = Path(cfg["outputs"]["results_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics(args.eval_config)

    saved_paths: list[Path] = []
    saved_paths.extend(plot_metric_chart(rows, output_dir, args.dpi))
    saved_paths.extend(plot_choice_type_chart(rows, output_dir, args.dpi))
    saved_paths.extend(plot_subject_chart(rows, output_dir, args.top_subjects, args.dpi))

    print(f"Generated {len(saved_paths)} files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
