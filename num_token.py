#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def load_jsonl(path):
    """Read a JSONL file line by line, skipping empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    """Measure token length statistics after rendering messages with the chat template."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/projects/checkpoint/gemma-3-4b-it")
    parser.add_argument("--data_file", type=str, default="/projects/checkpoint/datasets/train_stage2.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--sample_size", type=int, default=0, help="0 means use all")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    lengths = []
    n_total = 0
    n_truncated = 0

    for ex in load_jsonl(args.data_file):
        messages = ex["messages"]

        # Render with the same chat template used in training to avoid undercounting sequence length.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # add_special_tokens=False matches templates that already include special tokens.
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        seq_len = len(input_ids)
        lengths.append(seq_len)
        n_total += 1
        if seq_len > args.max_seq_length:
            n_truncated += 1

        if args.sample_size > 0 and n_total >= args.sample_size:
            break

    arr = np.array(lengths)

    print(f"examples: {len(arr)}")
    print(f"mean: {arr.mean():.2f}")
    print(f"median: {np.percentile(arr, 50):.2f}")
    print(f"p75: {np.percentile(arr, 75):.2f}")
    print(f"p90: {np.percentile(arr, 90):.2f}")
    print(f"p95: {np.percentile(arr, 95):.2f}")
    print(f"p99: {np.percentile(arr, 99):.2f}")
    print(f"max: {arr.max()}")
    print(f"> max_seq_length ({args.max_seq_length}): {n_truncated} / {n_total} = {n_truncated / n_total:.2%}")

    # Print simple buckets to quickly judge whether max_seq_length should be adjusted.
    bins = [0, 512, 1024, 1536, 2048, 3072, 4096, 8192]
    print("\nBucket counts:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        cnt = ((arr > lo) & (arr <= hi)).sum()
        print(f"({lo:>4}, {hi:>4}]: {cnt}")
    cnt = (arr > bins[-1]).sum()
    print(f"> {bins[-1]}: {cnt}")


if __name__ == "__main__":
    main()
