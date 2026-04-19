import json
import os
from collections import Counter
from datasets import load_dataset

GEN_PATH = "/projects/checkpoint/healthbench_gemma_eval/outputs/gemma-3-4b-it_healthbench_generations.jsonl"
OUT_PATH = "/projects/checkpoint/healthbench_gemma_eval/outputs/gemma-3-4b-it_healthbench_judge.jsonl"

DATASET_PATH = "/home/jie018/.cache/huggingface/hub/datasets--openai--healthbench/snapshots/40ee1968852fc57f625934251ac22be47077a8fb/2025-05-07-06-14-12_oss_eval.jsonl"
SPLIT_NAME = "train"


def read_generation_file(path: str) -> dict[str, str]:
    """
    Read generation JSONL and build a prompt_id-to-response lookup table.

    every line of JSON must have:
      - prompt_id
      - response
    """
    pred_by_id: dict[str, str] = {}
    duplicate_counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_num}: {e}") from e

            if "prompt_id" not in obj:
                raise KeyError(f"Missing 'prompt_id' in generation file at line {line_num}")
            if "response" not in obj:
                raise KeyError(f"Missing 'response' in generation file at line {line_num}")

            prompt_id = str(obj["prompt_id"])
            response = obj["response"]

            if not isinstance(response, str):
                response = str(response)

            if prompt_id in pred_by_id:
                duplicate_counter[prompt_id] += 1

            pred_by_id[prompt_id] = response

    if duplicate_counter:
        dup_n = len(duplicate_counter)
        print(f"Warning: found duplicate prompt_id entries in generation file: {dup_n}. Keeping the last one.")

    if not pred_by_id:
        raise ValueError(f"No predictions loaded from {path}")

    return pred_by_id


def main():
    """Merge model generations with HealthBench rubrics into the judge input file."""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print(f"Reading generations from: {GEN_PATH}")
    pred_by_id = read_generation_file(GEN_PATH)
    print(f"Loaded {len(pred_by_id)} generated responses.")

    print(f"Loading dataset")
    ds = load_dataset("json", data_files=DATASET_PATH, split=SPLIT_NAME)

    required_dataset_fields = {"prompt_id", "prompt", "rubrics", "example_tags"}
    dataset_fields = set(ds.column_names)
    missing = required_dataset_fields - dataset_fields
    if missing:
        raise KeyError(
            f"Dataset is missing required fields: {sorted(missing)}. "
            f"Available fields: {sorted(dataset_fields)}"
        )

    written = 0
    skipped_missing_prediction = 0
    seen_prediction_ids = set()

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for ex in ds:
            prompt_id = str(ex["prompt_id"])

            # Skip dataset rows without generations because the judge requires an answer.
            if prompt_id not in pred_by_id:
                skipped_missing_prediction += 1
                continue

            seen_prediction_ids.add(prompt_id)

            record = {
                "prompt_id": prompt_id,
                "prompt": ex["prompt"],
                # The judge expects the response as a messages list, so wrap it as an assistant message.
                "prompt_response": [
                    {
                        "role": "assistant",
                        "content": pred_by_id[prompt_id],
                    }
                ],
                "rubrics": ex["rubrics"],
                "example_tags": ex["example_tags"],
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    unused_predictions = sorted(set(pred_by_id.keys()) - seen_prediction_ids)

    print(f"Wrote {written} samples to: {OUT_PATH}")
    print(f"Skipped dataset samples without prediction: {skipped_missing_prediction}")
    print(f"Predictions not matched to dataset rows: {len(unused_predictions)}")

    if written == 0:
        raise RuntimeError(
            "No judge input rows were written. "
            "Check whether prompt_id in the generation file matches the dataset prompt_id."
        )

    if unused_predictions:
        print("First 10 unmatched prediction prompt_ids:")
        for pid in unused_predictions[:10]:
            print(f"  {pid}")


if __name__ == "__main__":
    main()
