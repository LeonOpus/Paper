import os
import sys
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(__file__))
import config

model_dir = os.path.join(config.MODEL_BASE, "DeepSeek-R1-Distill-Qwen-1.5B")

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    token=None  # 公开模型不需要 token
)
print(f"下载完成！模型保存在: {model_dir}")
