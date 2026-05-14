import os

PAPER_ID   = "2017_Transformer"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.expanduser(f"~/DataSet/HuggingFace/{PAPER_ID}/data")
OUTPUT_DIR = os.path.expanduser(f"~/Output/{PAPER_ID}/output")
LOG_DIR    = os.path.expanduser(f"~/Output/{PAPER_ID}/logs")

os.environ.setdefault("HF_ENDPOINT",      "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/DataSet/HuggingFace"))

def resolve_data(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(DATA_DIR, rel)
