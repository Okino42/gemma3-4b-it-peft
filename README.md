# Gemma 3 4B IT PEFT

Utilities for preparing medical SFT datasets, training Gemma 3 4B IT with LoRA/QLoRA or DoRA/QDoRA adapters, and running quick local inference or HealthBench generation workflows.

## Repository Contents

- `download_model.py` downloads the `google/gemma-3-4b-it` model snapshot from Hugging Face Hub.
- `data_knowledge_expansion.py` builds a stage-1 medical knowledge SFT mixture from MedMCQA, MedQuAD, MedMeadow, BioInstruct, and a filtered Dolly supplement.
- `finetune_data.py` builds dialogue-focused stage-1/stage-2 SFT data from ChatDoctor, MediQ AskDocs, MedMeadow, and MedQA-dialog.
- `train_lora.py` trains LoRA or QLoRA adapters with TRL `SFTTrainer`.
- `train_dora.py` trains DoRA or QDoRA adapters with TRL `SFTTrainer`.
- `num_token.py` reports token-length statistics for messages-format JSONL data.
- `draw_loss.py` plots training loss from a TensorBoard event file.
- `dialogue.py` runs a single-turn vLLM smoke test with the base model.
- `lora_dialogue.py` runs an interactive vLLM chat loop with a LoRA adapter.
- `generation_hb.py` generates HealthBench responses with vLLM.
- `judge_input.py` combines HealthBench generations with prompts and rubrics for judge evaluation.
- `text.py` trains a RoBERTa sentiment classifier with layer-wise learning-rate decay.

## Data Format

The LoRA and DoRA training scripts expect JSONL files where each line is one example:

```json
{
  "id": "example-id",
  "source": "chatdoctor_cleaned",
  "task": "single_turn_consult",
  "messages": [
    {"role": "system", "content": "System instruction"},
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Assistant answer"}
  ],
  "meta": {}
}
```

Valid message roles are `system`, `user`, and `assistant`. Each example must contain at least one assistant message.

## Setup

Install the core libraries used by the scripts:

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes tqdm numpy matplotlib tensorboard scikit-learn pandas vllm huggingface_hub
```

Some packages, especially `torch`, `bitsandbytes`, `flash-attn`, and `vllm`, are CUDA-version dependent. Install versions that match the target GPU environment.

For gated Hugging Face models or datasets, export a token instead of hard-coding it:

```bash
export HF_TOKEN=your_huggingface_token
```

## Download Model

```bash
python download_model.py
```

By default this downloads `google/gemma-3-4b-it` into:

```text
/projects/checkpoint/gemma-3-4b-it
```

Change `local_dir` in `download_model.py` if your checkpoint directory is different.

## Build SFT Data

Dialogue-focused mixture:

```bash
python finetune_data.py \
  --output_dir /projects/checkpoint/datasets/dialogue_mix
```

Knowledge-expansion mixture:

```bash
python data_knowledge_expansion.py \
  --output_dir /projects/checkpoint/datasets/knowledge_mix
```

Both scripts write JSONL train/validation/test files and a `stats.json` summary.

## Check Sequence Lengths

```bash
python num_token.py \
  --model_name_or_path /projects/checkpoint/gemma-3-4b-it \
  --data_file /projects/checkpoint/datasets/dialogue_mix/train_stage1.jsonl \
  --max_seq_length 2048
```

Use the reported percentiles and truncation count to decide whether to change `--max_seq_length`.

## Train LoRA or QLoRA

```bash
python train_lora.py \
  --model_path /projects/checkpoint/gemma-3-4b-it \
  --train_file /projects/checkpoint/datasets/dialogue_mix/train_stage1.jsonl \
  --validation_file /projects/checkpoint/datasets/dialogue_mix/val.jsonl \
  --output_dir /projects/checkpoint/gemma-3-4b-med-lora-stage1 \
  --bf16
```

Enable QLoRA 4-bit loading:

```bash
python train_lora.py \
  --model_path /projects/checkpoint/gemma-3-4b-it \
  --train_file /projects/checkpoint/datasets/dialogue_mix/train_stage1.jsonl \
  --validation_file /projects/checkpoint/datasets/dialogue_mix/val.jsonl \
  --output_dir /projects/checkpoint/gemma-3-4b-med-qlora-stage1 \
  --use_4bit \
  --bf16
```

Continue training from an existing adapter:

```bash
python train_lora.py \
  --model_path /projects/checkpoint/gemma-3-4b-it \
  --adapter_path /projects/checkpoint/gemma-3-4b-med-lora-stage1 \
  --train_file /projects/checkpoint/datasets/dialogue_mix/train_stage2.jsonl \
  --validation_file /projects/checkpoint/datasets/dialogue_mix/val.jsonl \
  --output_dir /projects/checkpoint/gemma-3-4b-med-lora-stage2 \
  --bf16
```

## Train DoRA or QDoRA

```bash
python train_dora.py \
  --model_path /projects/checkpoint/gemma-3-4b-it \
  --train_file /projects/checkpoint/datasets/dialogue_mix/train_stage2.jsonl \
  --validation_file /projects/checkpoint/datasets/dialogue_mix/val.jsonl \
  --output_dir /projects/checkpoint/gemma-3-4b-med-dora-stage2 \
  --bf16
```

Add `--use_4bit` for QDoRA.

## Inference

Base model smoke test:

```bash
python dialogue.py
```

Interactive LoRA chat:

```bash
python lora_dialogue.py
```

Update `BASE_MODEL_PATH` and `LORA_ADAPTER_PATH` in `lora_dialogue.py` before running.

## HealthBench Generation

Generate responses:

```bash
python generation_hb.py
```

Build judge input:

```bash
python judge_input.py
```

The output paths are currently configured as constants inside each script.

## Notes

- Large logs and generated files under `out/` are ignored by Git.
- Do not commit API tokens or Hugging Face tokens. Use `HF_TOKEN` or another environment variable.
- Most scripts contain absolute paths for the original training environment. Adjust those paths before running in a different environment.
