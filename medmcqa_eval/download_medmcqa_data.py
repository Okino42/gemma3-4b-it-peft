#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "openlifescienceai/medmcqa"
DEFAULT_FILENAME = "data/validation-00000-of-00001.parquet"
DEFAULT_OUTPUT_NAME = "validation-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the MedMCQA validation parquet file from Hugging Face."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    if output_path.exists() and not args.force:
        print(f"Already exists: {output_path}")
        return

    cached_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=args.filename,
    )

    shutil.copy2(cached_path, output_path)
    print(f"Downloaded: {args.repo_id}/{args.filename}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
