#!/usr/bin/env python3
"""
QDoRA SFT training script aligned with finetune_data.py outputs.

Expected input JSONL format (one example per line):
{
  "id": "...",
  "source": "chatdoctor_cleaned",
  "task": "single_turn_qa",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {...}
}

Typical stage-1 usage:
python train_lora.py \
  --model_name_or_path /projects/checkpoint/gemma-3-4b-it \
  --train_file data/sft_mix_v2/train_stage1.jsonl \
  --validation_file data/sft_mix_v2/val.jsonl \
  --output_dir outputs/gemma3-med-lora-stage1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


LOGGER = logging.getLogger(__name__)


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    """Parse all command-line options for DoRA/QDoRA SFT training."""
    parser = argparse.ArgumentParser(description="Train Gemma with QA-format JSONL by DoRA/QDoRA")

    # Paths
    parser.add_argument("--model_path", type=str, default="/projects/checkpoint/gemma-3-4b-it")
    parser.add_argument("--train_file", type=str, default="/projects/checkpoint/datasets/train_stage2.jsonl")
    parser.add_argument("--validation_file", type=str, default="/projects/checkpoint/datasets/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="/projects/checkpoint/gemma-3-4b-med-dora-s2")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)

    # Data
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing. Off by default for dialogue SFT.")
    parser.add_argument("--shuffle_seed", type=int, default=42)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # LoRA / QLoRA
    parser.add_argument("--use_4bit", action="store_true", help="Enable QDoRA 4-bit loading.")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default=",".join(DEFAULT_TARGET_MODULES))

    # Loss behavior
    parser.add_argument(
        "--assistant_only_loss",
        action="store_true",
        help="Use assistant-only loss when tokenizer chat template supports assistant masks.",
    )
    parser.add_argument(
        "--force_disable_assistant_only_loss",
        action="store_true",
        help="Disable assistant-only loss even if supported.",
    )

    # Misc
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--save_safetensors", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--log_first_sample", action="store_true")

    args = parser.parse_args()

    if args.gradient_checkpointing and args.no_gradient_checkpointing:
        raise ValueError("Cannot set both --gradient_checkpointing and --no_gradient_checkpointing")
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16")

    return args


def configure_logging() -> None:
    """Configure a consistent log format for training and cluster logs."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def get_torch_dtype(name: str) -> torch.dtype:
    """Convert a quantization compute dtype name into a PyTorch dtype."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def supports_assistant_only_loss(tokenizer: Any) -> bool:
    """
    Check whether the tokenizer/chat template can return a non-empty assistant token mask.
    TRL assistant_only_loss relies on this capability.
    """
    probe_messages = [
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello!"},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        mask = encoded.get("assistant_masks")
        if mask is None:
            return False
        if isinstance(mask, torch.Tensor):
            return bool(mask.sum().item() > 0)
        return sum(mask) > 0
    except Exception as exc:
        LOGGER.warning("Assistant-only loss probe failed: %s", exc)
        return False


def load_jsonl_dataset(train_file: str, validation_file: str | None, seed: int):
    """Load messages-format JSONL data and shuffle available splits with a fixed seed."""
    data_files = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file

    ds = load_dataset("json", data_files=data_files)
    ds["train"] = ds["train"].shuffle(seed=seed)
    if "validation" in ds:
        ds["validation"] = ds["validation"].shuffle(seed=seed)
    return ds


def validate_messages(example: dict[str, Any]) -> bool:
    """Check that a sample has valid non-empty system/user/assistant messages."""
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    valid_roles = {"system", "user", "assistant"}
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in valid_roles:
            return False
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return False

    if not any(m["role"] == "assistant" for m in messages):
        return False

    return True


def summarize_dataset(ds) -> dict[str, Any]:
    """Summarize row counts and source/task distributions for each dataset split."""
    out: dict[str, Any] = {}
    for split in ds.keys():
        num_rows = len(ds[split])
        source_counts: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        for ex in ds[split]:
            source = ex.get("source", "unknown")
            task = ex.get("task", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            task_counts[task] = task_counts.get(task, 0) + 1
        out[split] = {
            "num_rows": num_rows,
            "source_counts": source_counts,
            "task_counts": task_counts,
        }
    return out


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Enable gradient checkpointing by default unless explicitly disabled.
    use_gc = args.gradient_checkpointing or not args.no_gradient_checkpointing

    LOGGER.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        # Gemma-style causal LMs often have no dedicated pad token; eos is a practical fallback.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # use_4bit selects QDoRA; otherwise the model is loaded with the requested fp/bf dtype.
    quantization_config = None
    torch_dtype = None
    if args.use_4bit:
        compute_dtype = get_torch_dtype(args.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        LOGGER.info(
            "Using 4-bit quantization: quant_type=%s compute_dtype=%s double_quant=%s",
            args.bnb_4bit_quant_type,
            args.bnb_4bit_compute_dtype,
            args.bnb_4bit_use_double_quant,
        )
    else:
        if args.bf16:
            torch_dtype = torch.bfloat16
        elif args.fp16:
            torch_dtype = torch.float16

    LOGGER.info("Loading model from %s", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    if hasattr(model.config, "use_cache"):
        # Disable KV cache during training to reduce memory use and support checkpointing.
        model.config.use_cache = False

    # DoRA is configured through LoRAConfig with use_dora=True.
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        use_dora=True,
    )
    if args.adapter_path:
        # Continue training an existing DoRA adapter instead of creating a new adapter.
        LOGGER.info("Loading existing DoRA adapter from %s", args.adapter_path)
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=True,
        )
        trainer_peft_config = None
    else:
        trainer_peft_config = peft_config

    LOGGER.info("Loading dataset from jsonl")
    ds = load_jsonl_dataset(args.train_file, args.validation_file, args.shuffle_seed)

    for split in list(ds.keys()):
        before = len(ds[split])
        # Drop invalid records before TRL applies the tokenizer chat template.
        ds[split] = ds[split].filter(validate_messages, num_proc=max(1, args.dataset_num_proc))
        after = len(ds[split])
        LOGGER.info("Split %s: kept %d / %d valid rows", split, after, before)

    dataset_stats = summarize_dataset(ds)
    LOGGER.info("Dataset summary:\n%s", json.dumps(dataset_stats, indent=2, ensure_ascii=False))
    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)

    if args.log_first_sample and len(ds["train"]) > 0:
        # Optional preview for checking role order and rendered chat-template text.
        LOGGER.info("First train sample:\n%s", json.dumps(ds["train"][0], ensure_ascii=False, indent=2))
        rendered = tokenizer.apply_chat_template(
            ds["train"][0]["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        LOGGER.info("Rendered chat template preview:\n%s", rendered[:3000])

    assistant_only_loss = False
    if not args.force_disable_assistant_only_loss and args.assistant_only_loss:
        # Assistant-only loss requires the tokenizer template to return assistant token masks.
        assistant_only_loss = supports_assistant_only_loss(tokenizer)
        if assistant_only_loss:
            LOGGER.info("assistant_only_loss is enabled.")
        else:
            LOGGER.warning(
                "assistant_only_loss requested, but tokenizer/chat template does not provide assistant masks. Falling back to full-token loss."
            )

    eval_strategy = "steps" if "validation" in ds else "no"
    load_best_model_at_end = eval_strategy != "no"

    # SFTConfig carries both Transformers training settings and TRL SFT data-processing options.
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_strategy != "no" else None,
        save_total_limit=args.save_total_limit,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=args.packing,
        dataset_num_proc=max(1, args.dataset_num_proc),
        dataset_kwargs={"skip_prepare_dataset": False},
        save_safetensors=args.save_safetensors,
        logging_first_step=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        eval_strategy=eval_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss" if load_best_model_at_end else None,
        greater_is_better=False if load_best_model_at_end else None,
        assistant_only_loss=assistant_only_loss,
    )

    callbacks = []
    if args.early_stopping_patience > 0 and "validation" in ds:
        # Early stopping needs a validation metric to monitor.
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        processing_class=tokenizer,
        peft_config=trainer_peft_config,
        callbacks=callbacks,
    )

    trainable, total = trainer.model.get_nb_trainable_parameters()
    LOGGER.info("Trainable parameters: %s / %s (%.4f%%)", trainable, total, 100.0 * trainable / total)

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        # Persist the effective configuration for reproducibility and later debugging.
        payload = {
            "cli_args": vars(args),
            "peft_config": peft_config.to_dict() if hasattr(peft_config, "to_dict") else str(peft_config),
            "training_args": asdict(training_args),
            "assistant_only_loss_effective": assistant_only_loss,
        }
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    LOGGER.info("Starting training")
    if args.adapter_path:
        # adapter_path already defines the starting weights, so train directly.
        train_result = trainer.train()
    else:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(ds["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if "validation" in ds:
        # Save final validation metrics after training completes.
        LOGGER.info("Running final evaluation")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(ds["validation"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    LOGGER.info("Done. Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
