import os
import json
from typing import Iterable, List, Dict, Any
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


MODEL_PATH = "/projects/checkpoint/gemma-3-4b-it"
OUT_PATH = "/projects/checkpoint/healthbench_gemma_eval/outputs/gemma-3-4b-it_healthbench_generations.jsonl"

MAX_NEW_TOKENS = 1024
BATCH_SIZE = 16
MAX_EXAMPLES = None
TENSOR_PARALLEL_SIZE = 1


os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """Split HealthBench examples into fixed-size batches for vLLM generation."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    """Generate HealthBench responses and save the prompt_id-to-response mapping."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    # vLLM loads the local model directory directly, which improves throughput for evaluation.
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=MAX_NEW_TOKENS,
        top_p=0.95,
        seed=42
    )

    local_jsonl = hf_hub_download(
        repo_id="openai/healthbench",
        repo_type="dataset",
        filename="2025-05-07-06-14-12_oss_eval.jsonl",
    )

    ds = load_dataset("json", data_files=local_jsonl, split="train")
    examples = list(ds)

    # Set MAX_EXAMPLES during debugging to generate only the first few examples.
    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]

    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for batch in tqdm(batched(examples, BATCH_SIZE), total=num_batches, desc="Generating"):
            prompts = []
            metas = []

            for ex in batch:
                messages = ex["prompt"]

                # HealthBench prompts are already in messages format, so only template rendering is needed.
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                prompts.append(prompt_text)
                metas.append(
                    {
                        "prompt_id": ex["prompt_id"],
                    }
                )

            outputs = llm.generate(prompts, sampling_params)

            # Keep only the fields needed by the judge-input assembly step.
            for meta, output in zip(metas, outputs):
                response = output.outputs[0].text.strip()

                record = {
                    "prompt_id": meta["prompt_id"],
                    "response": response,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
