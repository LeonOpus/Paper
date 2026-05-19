"""
从 ModelScope 下载 BLIP 模型权重。
在训练/评估前运行一次即可。

下载的模型：
  - Salesforce/blip-image-captioning-base  （图像描述微调模型）
  - Salesforce/blip-itm-base-coco          （检索评估模型，已在 COCO 上微调）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# BLIP 在 ModelScope 上的模型 ID
MODELS = {
    "blip-image-captioning-base": "Salesforce/blip-image-captioning-base",
    "blip-itm-base-coco":         "Salesforce/blip-itm-base-coco",
}

CACHE_DIR = os.path.join(os.path.expanduser("~"), "Model", "ModelScope")
os.makedirs(CACHE_DIR, exist_ok=True)


def model_local_path(model_id: str) -> str:
    # ModelScope 将模型存储于 CACHE_DIR/ORG/MODEL-NAME/ 目录
    return os.path.join(CACHE_DIR, *model_id.split("/"))


def download_all():
    from modelscope import snapshot_download

    for name, model_id in MODELS.items():
        dest = model_local_path(model_id)
        if os.path.isdir(dest) and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(dest)):
            print(f"[skip] {name} already at {dest}")
            continue
        print(f"[download] {model_id} -> {CACHE_DIR}")
        try:
            path = snapshot_download(model_id, cache_dir=CACHE_DIR)
            print(f"[done] {name}: {path}")
        except Exception as e:
            print(f"[error] {name}: {e}")
            print(f"  Try manually: modelscope download --model {model_id} --local_dir {dest}")


if __name__ == "__main__":
    download_all()
