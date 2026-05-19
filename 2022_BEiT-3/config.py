import os

PAPER_ID   = "2022_BEiT-3"
DATA_DIR   = os.path.expanduser(f"~/DataSet/HuggingFace/{PAPER_ID}/data")
OUTPUT_DIR = os.path.expanduser(f"~/Output/{PAPER_ID}/output")
LOG_DIR    = os.path.expanduser(f"~/Output/{PAPER_ID}/logs")
MODEL_DIR  = os.path.expanduser("~/Model/ModelScope")

os.environ.setdefault("HF_ENDPOINT",       "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_CACHE",      os.path.expanduser("~/Model/HuggingFace/hub"))
os.environ.setdefault("MODELSCOPE_CACHE",  MODEL_DIR)

def resolve_data(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(DATA_DIR, rel)

def resolve_model(model_id: str) -> str:
    """优先使用本地 ModelScope 缓存，若不存在则回退到 HuggingFace ID。"""
    org, name = (model_id.split("/") + [""])[:2]
    local = os.path.join(MODEL_DIR, org, name)
    return local if os.path.isdir(local) else model_id
