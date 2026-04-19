from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# Specify the base model and LoRA adapter separately; vLLM mounts the adapter at generation time.
BASE_MODEL_PATH = "/projects/checkpoint/gemma-3-4b-it"
LORA_ADAPTER_PATH = "/projects/checkpoint/gemma-3-4b-med-lora/gemma-3-4b-med-lora-newdata-r32"

# ====== tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

# ====== vLLM  ======
llm = LLM(
    model=BASE_MODEL_PATH,
    tokenizer=BASE_MODEL_PATH,
    trust_remote_code=False,
    enable_lora=True,
    max_lora_rank=32,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# ====== Sampling parameters ======
# This open sampling setup is intended for interactive QA; lower temperature for evaluation.
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.9,
    max_tokens=1024,
)

# ====== LoRA request object ======
# Argument 1: adapter name.
# Argument 2: globally unique integer ID.
# Argument 3: local adapter path.
lora_request = LoRARequest(
    "medical-lora",
    1,
    LORA_ADAPTER_PATH,
)

def build_prompt(messages):
    """
    Use the Hugging Face tokenizer chat template to render multi-turn messages
    into a prompt string that vLLM can consume directly.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

def chat():
    """Start a command-line multi-turn chat loop while preserving conversation history."""
    print("LoRA + vLLM 对话已启动。输入 exit / quit 退出。\n")

    messages = []

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Rebuild the prompt from full history each turn so the model sees prior context.
        prompt = build_prompt(messages)

        outputs = llm.generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        assistant_text = outputs[0].outputs[0].text.strip()
        print(f"Assistant: {assistant_text}\n")

        # Append the assistant response to history for the next turn.
        messages.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    chat()
