from huggingface_hub import snapshot_download, hf_hub_download
import os

# Download the full Gemma model snapshot from Hugging Face Hub into a fixed local directory.
# snapshot_download fetches the whole repository, which is suitable for training or vLLM loading.
local_dir = "/projects/checkpoint/gemma-3-4b-it"
snapshot_download(
    repo_id="google/gemma-3-4b-it",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    token=os.environ.get("HF_TOKEN"),
)

print(f"The model has been downloaded to: {local_dir}")
