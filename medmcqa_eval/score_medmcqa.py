#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MedMCQA predictions for one model.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--metrics-path", default=None)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_num}: {exc}") from exc
    return rows


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    answered = sum(1 for r in records if r.get("pred_letter") is not None)
    correct = sum(1 for r in records if r.get("pred_letter") == r.get("gold_letter"))
    labels = ["A", "B", "C", "D"]
    per_class: dict[str, dict[str, Any]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for label in labels:
        tp = sum(1 for r in records if r.get("gold_letter") == label and r.get("pred_letter") == label)
        fp = sum(1 for r in records if r.get("gold_letter") != label and r.get("pred_letter") == label)
        fn = sum(1 for r in records if r.get("gold_letter") == label and r.get("pred_letter") != label)
        support = sum(1 for r in records if r.get("gold_letter") == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    metrics: dict[str, Any] = {
        "total": total,
        "answered": answered,
        "unparsed": total - answered,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_answered_only": correct / answered if answered else 0.0,
        "precision_macro": macro_precision / len(labels),
        "recall_macro": macro_recall / len(labels),
        "f1_macro": macro_f1 / len(labels),
        "precision_weighted": weighted_precision / total if total else 0.0,
        "recall_weighted": weighted_recall / total if total else 0.0,
        "f1_weighted": weighted_f1 / total if total else 0.0,
        "per_class": per_class,
    }

    confusion = {
        gold: {pred: 0 for pred in ["A", "B", "C", "D", "UNPARSED"]}
        for gold in ["A", "B", "C", "D"]
    }
    for r in records:
        gold = r.get("gold_letter")
        pred = r.get("pred_letter") or "UNPARSED"
        if gold in confusion and pred in confusion[gold]:
            confusion[gold][pred] += 1
    metrics["confusion_matrix"] = confusion

    for key in ["subject_name", "choice_type"]:
        grouped: dict[str, dict[str, Any]] = {}
        for r in records:
            group = r.get(key) or "unknown"
            item = grouped.setdefault(group, {"total": 0, "correct": 0})
            item["total"] += 1
            if r.get("pred_letter") == r.get("gold_letter"):
                item["correct"] += 1
        for item in grouped.values():
            item["accuracy"] = item["correct"] / item["total"] if item["total"] else 0.0
        metrics[f"by_{key}"] = grouped

    return metrics


def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml(args.eval_config)
    output_dir = Path(eval_cfg["outputs"]["output_dir"]) / args.model_name
    predictions_path = Path(args.predictions_path) if args.predictions_path else output_dir / eval_cfg["outputs"]["predictions_filename"]
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_dir / eval_cfg["outputs"]["metrics_filename"]

    records = read_jsonl(predictions_path)
    metrics = compute_metrics(records)
    metrics["model_name"] = args.model_name
    metrics["predictions_path"] = str(predictions_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"model_name: {args.model_name}")
    print(f"predictions: {predictions_path}")
    print(f"metrics: {metrics_path}")
    print(f"accuracy: {metrics['accuracy']:.6f} ({metrics['correct']}/{metrics['total']})")
    print(f"precision_macro: {metrics['precision_macro']:.6f}")
    print(f"recall_macro: {metrics['recall_macro']:.6f}")
    print(f"f1_macro: {metrics['f1_macro']:.6f}")
    print(f"precision_weighted: {metrics['precision_weighted']:.6f}")
    print(f"recall_weighted: {metrics['recall_weighted']:.6f}")
    print(f"f1_weighted: {metrics['f1_weighted']:.6f}")
    print(f"unparsed: {metrics['unparsed']}")


if __name__ == "__main__":
    main()
