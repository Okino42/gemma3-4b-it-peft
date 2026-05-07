#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MedMCQA metrics across model output dirs.")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml(args.eval_config)
    output_root = Path(eval_cfg["outputs"]["output_dir"])
    results_dir = Path(eval_cfg["outputs"]["results_dir"])
    metrics_name = eval_cfg["outputs"]["metrics_filename"]

    rows = []
    for metrics_path in sorted(output_root.glob(f"*/{metrics_name}")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "model_name": metrics.get("model_name") or metrics_path.parent.name,
                "accuracy": metrics.get("accuracy", 0.0),
                "precision_macro": metrics.get("precision_macro", 0.0),
                "recall_macro": metrics.get("recall_macro", 0.0),
                "f1_macro": metrics.get("f1_macro", 0.0),
                "precision_weighted": metrics.get("precision_weighted", 0.0),
                "recall_weighted": metrics.get("recall_weighted", 0.0),
                "f1_weighted": metrics.get("f1_weighted", 0.0),
                "correct": metrics.get("correct", 0),
                "total": metrics.get("total", 0),
                "answered": metrics.get("answered", 0),
                "unparsed": metrics.get("unparsed", 0),
                "accuracy_answered_only": metrics.get("accuracy_answered_only", 0.0),
                "metrics_path": str(metrics_path),
            }
        )

    rows.sort(key=lambda r: r["model_name"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "summary.csv"
    output_json = Path(args.output_json) if args.output_json else results_dir / "summary.json"

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "correct",
                "total",
                "answered",
                "unparsed",
                "accuracy_answered_only",
                "metrics_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Found {len(rows)} metrics files.")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")
    for row in rows:
        print(
            f"{row['model_name']}: acc={row['accuracy']:.6f}, "
            f"macro_f1={row['f1_macro']:.6f}, "
            f"weighted_f1={row['f1_weighted']:.6f} "
            f"({row['correct']}/{row['total']}), unparsed={row['unparsed']}"
        )


if __name__ == "__main__":
    main()
