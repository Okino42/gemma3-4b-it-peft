#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from eval_medmcqa import (
    COP_TO_LETTER,
    compute_metrics,
    configure_logging,
    find_model_config,
    load_tokenizer_and_model,
    load_yaml,
    model_input_device,
    predict_by_generation,
    predict_by_next_token_logprob,
    safe_text,
    tokenize_batch,
)
from prompt_templates import OPTION_LETTERS, build_medmcqa_messages


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose MedMCQA option-letter bias by re-evaluating with shuffled option order."
    )
    parser.add_argument("--model-name", required=True, help="Model name from configs/models.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for faster diagnosis.")
    parser.add_argument(
        "--choice-type",
        choices=["all", "single", "multi"],
        default="all",
        help="Optionally restrict diagnosis to one MedMCQA choice_type.",
    )
    parser.add_argument(
        "--num-shuffles",
        type=int,
        default=5,
        help="How many independent random option-order shuffles to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the summary JSON. Defaults to outputs/<model_name>/option_permutation_bias.json",
    )
    return parser.parse_args()


def normalize_choice_type(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def filter_dataframe(df: pd.DataFrame, fields: dict[str, Any], choice_type: str) -> pd.DataFrame:
    if choice_type == "all":
        return df
    key = fields["choice_type"]
    mask = df[key].map(normalize_choice_type) == choice_type
    return df[mask]


def evaluate_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    model,
    device,
    fields: dict[str, Any],
    prompt_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    desc: str,
) -> list[dict[str, Any]]:
    method = generation_cfg.get("method", "logprob")
    batch_size = int(generation_cfg["batch_size"])
    records: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(rows), batch_size), desc=desc):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_medmcqa_messages(row, fields, prompt_cfg),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        inputs = tokenize_batch(tokenizer, prompts, generation_cfg, method, device)

        if method == "logprob":
            batch_predictions = predict_by_next_token_logprob(model, tokenizer, inputs)
        elif method == "generate":
            batch_predictions = predict_by_generation(model, tokenizer, inputs, generation_cfg)
        else:
            raise ValueError(f"Unsupported generation.method={method!r}")

        for row, prediction in zip(batch, batch_predictions):
            gold_index = int(row[fields["label"]])
            gold_letter = COP_TO_LETTER[gold_index]
            pred_letter = prediction["pred_letter"]
            record = {
                "id": str(row[fields["id"]]),
                "gold_letter": gold_letter,
                "pred_letter": pred_letter,
                "prediction_text": prediction["prediction_text"],
                "correct": pred_letter == gold_letter,
                "choice_type": safe_text(row.get(fields["choice_type"])),
                "subject_name": safe_text(row.get(fields["subject"])),
            }
            if "choice_logprobs" in prediction:
                record["choice_logprobs"] = prediction["choice_logprobs"]
            records.append(record)

    return records


def permute_row(
    row: dict[str, Any],
    fields: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    option_fields = fields["options"]
    shuffled_old_letters = list(OPTION_LETTERS)
    rng.shuffle(shuffled_old_letters)

    new_to_old = {
        new_letter: old_letter for new_letter, old_letter in zip(OPTION_LETTERS, shuffled_old_letters)
    }
    old_to_new = {old_letter: new_letter for new_letter, old_letter in new_to_old.items()}

    new_row = dict(row)
    for new_letter, old_letter in new_to_old.items():
        new_row[option_fields[new_letter]] = row[option_fields[old_letter]]

    old_gold_letter = COP_TO_LETTER[int(row[fields["label"]])]
    new_gold_letter = old_to_new[old_gold_letter]
    new_row[fields["label"]] = OPTION_LETTERS.index(new_gold_letter)

    metadata = {
        "new_to_old": new_to_old,
        "old_to_new": old_to_new,
        "old_gold_letter": old_gold_letter,
        "new_gold_letter": new_gold_letter,
    }
    return new_row, metadata


def count_letters(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = {letter: 0 for letter in OPTION_LETTERS}
    for record in records:
        value = record.get(key)
        if value in counts:
            counts[value] += 1
    return counts


def counts_to_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {letter: 0.0 for letter in OPTION_LETTERS}
    return {letter: counts[letter] / total for letter in OPTION_LETTERS}


def average_fraction_maps(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {letter: 0.0 for letter in OPTION_LETTERS}
    return {
        letter: sum(item[letter] for item in items) / len(items)
        for letter in OPTION_LETTERS
    }


def build_shuffle_summary(
    shuffle_index: int,
    shuffle_seed: int,
    original_records: list[dict[str, Any]],
    permuted_records: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = compute_metrics(permuted_records)
    same_surface_letter = 0
    same_option_content = 0
    comparable = 0

    for original_record, permuted_record, item_meta in zip(original_records, permuted_records, metadata):
        original_pred = original_record.get("pred_letter")
        permuted_pred = permuted_record.get("pred_letter")
        if original_pred is None or permuted_pred is None:
            continue
        comparable += 1
        if original_pred == permuted_pred:
            same_surface_letter += 1
        expected_permuted_letter = item_meta["old_to_new"][original_pred]
        if permuted_pred == expected_permuted_letter:
            same_option_content += 1

    gold_counts = count_letters(permuted_records, "gold_letter")
    pred_counts = count_letters(permuted_records, "pred_letter")
    return {
        "shuffle_index": shuffle_index,
        "seed": shuffle_seed,
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "gold_letter_counts": gold_counts,
        "gold_letter_fractions": counts_to_fractions(gold_counts),
        "pred_letter_counts": pred_counts,
        "pred_letter_fractions": counts_to_fractions(pred_counts),
        "same_surface_letter_rate": same_surface_letter / comparable if comparable else 0.0,
        "same_option_content_rate": same_option_content / comparable if comparable else 0.0,
        "metrics": metrics,
    }


def default_output_path(eval_cfg: dict[str, Any], model_name: str) -> Path:
    output_root = Path(eval_cfg["outputs"]["output_dir"])
    return output_root / model_name / "option_permutation_bias.json"


def print_summary(summary: dict[str, Any]) -> None:
    original = summary["original"]
    aggregate = summary["aggregate"]
    LOGGER.info(
        "Original accuracy: %.4f (%d/%d)",
        original["accuracy"],
        original["correct"],
        original["total"],
    )
    LOGGER.info(
        "Shuffled accuracy mean: %.4f +/- %.4f",
        aggregate["shuffle_accuracy_mean"],
        aggregate["shuffle_accuracy_std"],
    )
    LOGGER.info("Mean accuracy delta vs original: %.4f", aggregate["shuffle_accuracy_delta_mean"])
    LOGGER.info("Original pred fractions: %s", json.dumps(original["pred_letter_fractions"], sort_keys=True))
    LOGGER.info(
        "Shuffled pred fractions mean: %s",
        json.dumps(aggregate["shuffle_pred_letter_fractions_mean"], sort_keys=True),
    )
    LOGGER.info("Mean same-surface-letter rate: %.4f", aggregate["same_surface_letter_rate_mean"])
    LOGGER.info("Mean same-option-content rate: %.4f", aggregate["same_option_content_rate_mean"])


def main() -> None:
    args = parse_args()
    configure_logging()

    models_cfg = load_yaml(args.models_config)
    eval_cfg = load_yaml(args.eval_config)
    model_cfg = find_model_config(models_cfg, args.model_name)
    fields = eval_cfg["fields"]
    prompt_cfg = eval_cfg["prompt"]
    loading_cfg = eval_cfg["model_loading"]
    generation_cfg = eval_cfg["generation"]

    data_path = Path(eval_cfg["dataset"]["parquet_path"])
    LOGGER.info("Loading dataset from %s", data_path)
    df = pd.read_parquet(data_path)
    df = filter_dataframe(df, fields, args.choice_type)
    if args.limit is not None:
        df = df.head(args.limit)
    rows = [row.to_dict() for _, row in df.iterrows()]
    LOGGER.info("Loaded %d rows after filtering choice_type=%s", len(rows), args.choice_type)

    tokenizer, model = load_tokenizer_and_model(model_cfg, loading_cfg)
    device = model_input_device(model)

    LOGGER.info("Running original evaluation pass")
    original_records = evaluate_rows(
        rows,
        tokenizer,
        model,
        device,
        fields,
        prompt_cfg,
        generation_cfg,
        desc=f"{args.model_name}-original",
    )
    original_metrics = compute_metrics(original_records)
    original_gold_counts = count_letters(original_records, "gold_letter")
    original_pred_counts = count_letters(original_records, "pred_letter")

    shuffle_summaries: list[dict[str, Any]] = []
    shuffle_accuracies: list[float] = []
    shuffle_pred_fraction_maps: list[dict[str, float]] = []
    shuffle_gold_fraction_maps: list[dict[str, float]] = []
    same_surface_rates: list[float] = []
    same_option_rates: list[float] = []

    for shuffle_index in range(args.num_shuffles):
        shuffle_seed = args.seed + shuffle_index
        rng = random.Random(shuffle_seed)
        permuted_rows: list[dict[str, Any]] = []
        permutation_metadata: list[dict[str, Any]] = []

        for row in rows:
            permuted_row, item_metadata = permute_row(row, fields, rng)
            permuted_rows.append(permuted_row)
            permutation_metadata.append(item_metadata)

        LOGGER.info("Running shuffled evaluation pass %d/%d", shuffle_index + 1, args.num_shuffles)
        permuted_records = evaluate_rows(
            permuted_rows,
            tokenizer,
            model,
            device,
            fields,
            prompt_cfg,
            generation_cfg,
            desc=f"{args.model_name}-shuffle-{shuffle_index + 1}",
        )
        shuffle_summary = build_shuffle_summary(
            shuffle_index,
            shuffle_seed,
            original_records,
            permuted_records,
            permutation_metadata,
        )
        shuffle_summaries.append(shuffle_summary)
        shuffle_accuracies.append(shuffle_summary["accuracy"])
        shuffle_pred_fraction_maps.append(shuffle_summary["pred_letter_fractions"])
        shuffle_gold_fraction_maps.append(shuffle_summary["gold_letter_fractions"])
        same_surface_rates.append(shuffle_summary["same_surface_letter_rate"])
        same_option_rates.append(shuffle_summary["same_option_content_rate"])

    shuffle_accuracy_mean = statistics.mean(shuffle_accuracies) if shuffle_accuracies else 0.0
    shuffle_accuracy_std = statistics.pstdev(shuffle_accuracies) if len(shuffle_accuracies) > 1 else 0.0

    summary = {
        "model_name": args.model_name,
        "method": generation_cfg.get("method", "logprob"),
        "choice_type": args.choice_type,
        "num_rows": len(rows),
        "num_shuffles": args.num_shuffles,
        "seed": args.seed,
        "original": {
            "accuracy": original_metrics["accuracy"],
            "correct": original_metrics["correct"],
            "total": original_metrics["total"],
            "gold_letter_counts": original_gold_counts,
            "gold_letter_fractions": counts_to_fractions(original_gold_counts),
            "pred_letter_counts": original_pred_counts,
            "pred_letter_fractions": counts_to_fractions(original_pred_counts),
            "metrics": original_metrics,
        },
        "aggregate": {
            "shuffle_accuracy_mean": shuffle_accuracy_mean,
            "shuffle_accuracy_std": shuffle_accuracy_std,
            "shuffle_accuracy_min": min(shuffle_accuracies) if shuffle_accuracies else 0.0,
            "shuffle_accuracy_max": max(shuffle_accuracies) if shuffle_accuracies else 0.0,
            "shuffle_accuracy_delta_mean": shuffle_accuracy_mean - original_metrics["accuracy"],
            "shuffle_gold_letter_fractions_mean": average_fraction_maps(shuffle_gold_fraction_maps),
            "shuffle_pred_letter_fractions_mean": average_fraction_maps(shuffle_pred_fraction_maps),
            "same_surface_letter_rate_mean": statistics.mean(same_surface_rates) if same_surface_rates else 0.0,
            "same_option_content_rate_mean": statistics.mean(same_option_rates) if same_option_rates else 0.0,
        },
        "shuffles": shuffle_summaries,
    }

    output_path = Path(args.output_path) if args.output_path else default_output_path(eval_cfg, args.model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved option permutation summary to %s", output_path)
    print_summary(summary)


if __name__ == "__main__":
    main()
