from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Run a single-turn local inference to smoke-test the model and vLLM environment.
model_path = "/projects/checkpoint/gemma-3-4b-it"

# The tokenizer renders messages into the prompt format expected by the model chat template.
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

# The vLLM engine loads the local model; max_model_len must cover prompt plus generated tokens.
llm = LLM(
    model=model_path,
    tokenizer=model_path,
    trust_remote_code=False,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

messages = [
    {"role": "user", "content": "could you explain to me what is abetalipoproteimemia?"},
]

# add_generation_prompt=True appends the assistant start marker to the prompt.
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Sampling parameters control output randomness and maximum generation length.
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.9,
    max_tokens=1024,
)

outputs = llm.generate([prompt], sampling_params)
response = outputs[0].outputs[0].text.strip()
print(response)
