#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_templates import OPTION_LETTERS, build_medmcqa_messages


LOGGER = logging.getLogger(__name__)
COP_TO_LETTER = {idx: letter for idx, letter in enumerate(OPTION_LETTERS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one configured model on MedMCQA validation.")
    parser.add_argument("--model-name", required=True, help="Model name from configs/models.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in predictions.jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing prediction file.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def find_model_config(models_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    for item in models_cfg.get("models", []):
        if item.get("name") == model_name:
            return item
    known = [item.get("name") for item in models_cfg.get("models", [])]
    raise KeyError(f"Unknown model_name={model_name!r}. Known models: {known}")


def torch_dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def build_quantization_config(cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not cfg.get("use_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch_dtype_from_name(cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
    )


def load_tokenizer_and_model(model_cfg: dict[str, Any], loading_cfg: dict[str, Any]):
    model_path = model_cfg["model_path"]
    adapter_path = model_cfg.get("adapter_path")
    tokenizer_path = adapter_path or model_path

    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=bool(loading_cfg.get("local_files_only", True)),
        trust_remote_code=bool(loading_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = build_quantization_config(loading_cfg)
    dtype = None if quantization_config is not None else torch_dtype_from_name(loading_cfg.get("torch_dtype"))

    LOGGER.info("Loading base model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=bool(loading_cfg.get("local_files_only", True)),
        trust_remote_code=bool(loading_cfg.get("trust_remote_code", False)),
        attn_implementation=loading_cfg.get("attn_implementation", "sdpa"),
        device_map=loading_cfg.get("device_map", "auto"),
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )

    if adapter_path:
        LOGGER.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return tokenizer, model


def model_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def padding_side_for_method(method: str) -> str:
    if method == "logprob":
        return "right"
    if method == "generate":
        return "left"
    raise ValueError(f"Unsupported generation.method={method!r}; expected 'logprob' or 'generate'")


def tokenize_batch(
    tokenizer,
    prompts: list[str],
    generation_cfg: dict[str, Any],
    method: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side_for_method(method)
    try:
        return tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(generation_cfg["max_input_length"]),
        ).to(device)
    finally:
        tokenizer.padding_side = original_padding_side


def parse_answer_letter(text: str) -> str | None:
    cleaned = text.strip().upper()
    patterns = [
        r"^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;\s]?",
        r"\b(?:ANSWER|ANS|OPTION|CHOICE)\s*(?:IS|:)?\s*[\(\[]?\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return None


def answer_token_ids(tokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for letter in OPTION_LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Answer letter {letter!r} is not a single token: {encoded}")
        ids[letter] = encoded[0]
    return ids


def predict_by_next_token_logprob(model, tokenizer, inputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    token_ids = answer_token_ids(tokenizer)
    option_tensor = torch.tensor([token_ids[letter] for letter in OPTION_LETTERS], device=inputs["input_ids"].device)

    with torch.no_grad():
        outputs = model(**inputs)

    positions = torch.arange(inputs["attention_mask"].shape[1], device=inputs["input_ids"].device)
    lengths = (inputs["attention_mask"].to(dtype=positions.dtype) * positions.unsqueeze(0)).max(dim=1).values
    batch_idx = torch.arange(inputs["input_ids"].shape[0], device=inputs["input_ids"].device)
    next_logits = outputs.logits[batch_idx, lengths, :]
    option_logits = next_logits.index_select(dim=1, index=option_tensor)
    if not torch.isfinite(option_logits).all():
        bad_count = int((~torch.isfinite(option_logits)).sum().detach().cpu())
        raise RuntimeError(
            f"Non-finite logits for A/B/C/D choices: {bad_count}. "
            "Try disabling 4-bit loading or lowering precision-sensitive settings."
        )
    option_logprobs = torch.log_softmax(option_logits.float(), dim=1)
    pred_indices = option_logprobs.argmax(dim=1).tolist()

    predictions = []
    for row_idx, pred_idx in enumerate(pred_indices):
        scores = {
            letter: float(option_logprobs[row_idx, col_idx].detach().cpu())
            for col_idx, letter in enumerate(OPTION_LETTERS)
        }
        predictions.append(
            {
                "pred_letter": OPTION_LETTERS[pred_idx],
                "prediction_text": OPTION_LETTERS[pred_idx],
                "choice_logprobs": scores,
            }
        )
    return predictions


def predict_by_generation(model, tokenizer, inputs: dict[str, torch.Tensor], generation_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    generate_kwargs = {
        "max_new_tokens": int(generation_cfg["max_new_tokens"]),
        "do_sample": bool(generation_cfg.get("do_sample", False)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_cfg.get("temperature") is not None:
        generate_kwargs["temperature"] = float(generation_cfg["temperature"])
    if generation_cfg.get("top_p") is not None:
        generate_kwargs["top_p"] = float(generation_cfg["top_p"])

    with torch.no_grad():
        generated = model.generate(**inputs, **generate_kwargs)

    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [
        {
            "pred_letter": parse_answer_letter(response),
            "prediction_text": response.strip(),
        }
        for response in decoded
    ]


def safe_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj:
                ids.add(str(obj["id"]))
    return ids


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    answered = sum(1 for r in records if r.get("pred_letter") is not None)
    correct = sum(1 for r in records if r.get("correct") is True)
    labels = list(OPTION_LETTERS)
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

    label_count = len(labels)
    metrics: dict[str, Any] = {
        "total": total,
        "answered": answered,
        "unparsed": total - answered,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_answered_only": correct / answered if answered else 0.0,
        "precision_macro": macro_precision / label_count if label_count else 0.0,
        "recall_macro": macro_recall / label_count if label_count else 0.0,
        "f1_macro": macro_f1 / label_count if label_count else 0.0,
        "precision_weighted": weighted_precision / total if total else 0.0,
        "recall_weighted": weighted_recall / total if total else 0.0,
        "f1_weighted": weighted_f1 / total if total else 0.0,
        "per_class": per_class,
    }

    for key in ["subject_name", "choice_type"]:
        grouped: dict[str, dict[str, Any]] = {}
        for r in records:
            group = r.get(key) or "unknown"
            item = grouped.setdefault(group, {"total": 0, "correct": 0})
            item["total"] += 1
            if r.get("correct") is True:
                item["correct"] += 1
        for item in grouped.values():
            item["accuracy"] = item["correct"] / item["total"] if item["total"] else 0.0
        metrics[f"by_{key}"] = grouped

    return metrics


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    configure_logging()

    models_cfg = load_yaml(args.models_config)
    eval_cfg = load_yaml(args.eval_config)
    model_cfg = find_model_config(models_cfg, args.model_name)

    output_root = Path(eval_cfg["outputs"]["output_dir"])
    model_output_dir = output_root / args.model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_output_dir / eval_cfg["outputs"]["predictions_filename"]
    metrics_path = model_output_dir / eval_cfg["outputs"]["metrics_filename"]

    if predictions_path.exists() and args.overwrite:
        predictions_path.unlink()

    done_ids = existing_ids(predictions_path) if args.resume else set()
    write_mode = "a" if args.resume and predictions_path.exists() else "w"

    data_path = Path(eval_cfg["dataset"]["parquet_path"])
    LOGGER.info("Loading dataset from %s", data_path)
    df = pd.read_parquet(data_path)
    if args.limit is not None:
        df = df.head(args.limit)

    fields = eval_cfg["fields"]
    loading_cfg = eval_cfg["model_loading"]
    generation_cfg = eval_cfg["generation"]
    method = generation_cfg.get("method", "logprob")
    if method not in {"logprob", "generate"}:
        raise ValueError(f"Unsupported generation.method={method!r}; expected 'logprob' or 'generate'")

    tokenizer, model = load_tokenizer_and_model(model_cfg, loading_cfg)
    device = model_input_device(model)

    rows = [row.to_dict() for _, row in df.iterrows() if str(row[fields["id"]]) not in done_ids]
    batch_size = int(generation_cfg["batch_size"])
    LOGGER.info("Evaluating %d rows for model %s", len(rows), args.model_name)

    with predictions_path.open(write_mode, encoding="utf-8") as fout:
        for start in tqdm(range(0, len(rows), batch_size), desc=args.model_name):
            batch = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_medmcqa_messages(row, fields, eval_cfg["prompt"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            inputs = tokenize_batch(tokenizer, prompts, generation_cfg, method, device)

            if method == "logprob":
                batch_predictions = predict_by_next_token_logprob(model, tokenizer, inputs)
            else:
                batch_predictions = predict_by_generation(model, tokenizer, inputs, generation_cfg)

            for row, prediction in zip(batch, batch_predictions):
                gold_index = int(row[fields["label"]])
                gold_letter = COP_TO_LETTER[gold_index]
                pred_letter = prediction["pred_letter"]
                record = {
                    "id": str(row[fields["id"]]),
                    "model_name": args.model_name,
                    "question": safe_text(row[fields["question"]]),
                    "options": {
                        letter: safe_text(row[fields["options"][letter]])
                        for letter in OPTION_LETTERS
                    },
                    "gold_index": gold_index,
                    "gold_letter": gold_letter,
                    "pred_letter": pred_letter,
                    "prediction_text": prediction["prediction_text"],
                    "correct": pred_letter == gold_letter,
                    "choice_type": safe_text(row.get(fields["choice_type"])),
                    "subject_name": safe_text(row.get(fields["subject"])),
                    "topic_name": safe_text(row.get(fields["topic"])),
                }
                if "choice_logprobs" in prediction:
                    record["choice_logprobs"] = prediction["choice_logprobs"]
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    records = load_prediction_records(predictions_path)
    metrics = compute_metrics(records)
    metrics["model_name"] = args.model_name
    metrics["predictions_path"] = str(predictions_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved predictions to %s", predictions_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Accuracy: %.6f (%d/%d)", metrics["accuracy"], metrics["correct"], metrics["total"])


if __name__ == "__main__":
    main()
