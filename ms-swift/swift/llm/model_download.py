import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"



from huggingface_hub import snapshot_download
model = "zai-org/GLM-4-9B-0414"
local_dir="/mnt/data/LLMS/GLM/glm4-9b-0414"


snapshot_download(
  repo_id=model,
  repo_type="model",
  local_dir=local_dir,
  local_dir_use_symlinks=False,
  max_workers=8,
)