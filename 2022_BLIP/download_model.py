"""
Download BLIP model weights from ModelScope.
Run once before training/evaluation.

Models downloaded:
  - Salesforce/blip-image-captioning-base  (captioning fine-tune)
  - Salesforce/blip-itm-base-coco          (retrieval eval, already COCO-tuned)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ModelScope model IDs for BLIP
MODELS = {
    "blip-image-captioning-base": "Salesforce/blip-image-captioning-base",
    "blip-itm-base-coco":         "Salesforce/blip-itm-base-coco",
}

CACHE_DIR = os.path.join(os.path.expanduser("~"), "Model", "ModelScope")
os.makedirs(CACHE_DIR, exist_ok=True)


def model_local_path(model_id: str) -> str:
    # ModelScope stores as CACHE_DIR/ORG/MODEL-NAME/
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
