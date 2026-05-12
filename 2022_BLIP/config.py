import os


def _find_paper_root() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        name = os.path.basename(current)
        if len(name) > 5 and name[:4].isdigit() and name[4] == "_":
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.dirname(os.path.abspath(__file__))
        current = parent


PAPER_ROOT = _find_paper_root()
PAPER_NAME = os.path.basename(PAPER_ROOT)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PAPER_DATA = os.environ.get("BLIP_DATA_ROOT", os.path.join(os.path.expanduser("~"), "DataSet", "HuggingFace", PAPER_NAME))
HF_HOME = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), "Model", "HuggingFace"))
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", os.path.join(os.path.expanduser("~"), "DataSet", "HuggingFace"))
HF_MODULES_CACHE = os.path.join(HF_HOME, "modules")
TRANSFORMERS_CACHE = os.path.join(HF_HOME, "transformers")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_MODULES_CACHE"] = HF_MODULES_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODELSCOPE_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.join(os.path.expanduser("~"), "Model", "ModelScope"))
os.environ.setdefault("MODELSCOPE_CACHE", MODELSCOPE_CACHE)

OUTPUT_ROOT = os.environ.get("BLIP_OUTPUT_ROOT", os.path.join(os.path.expanduser("~"), "Output", PAPER_NAME))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "output")
LOG_BASE = os.path.join(OUTPUT_ROOT, "logs")

# COCO images are shared with VLMO project
COCO_IMAGE_ROOT = os.environ.get(
    "COCO_IMAGE_ROOT",
    os.path.join(os.path.expanduser("~"), "DataSet", "HuggingFace", "2022_VLMO", "data"),
)
DATA_DIR = os.path.join(PAPER_DATA, "data")


def resolve_model(model_id: str) -> str:
    """Return local ModelScope path for a model ID, falling back to the ID itself."""
    local = os.path.join(MODELSCOPE_CACHE, *model_id.split("/"))
    return local if os.path.isdir(local) else model_id


def resolve_data(rel_path: str) -> str:
    """Resolve a data path relative to DATA_DIR."""
    return os.path.join(DATA_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path

for path in [DATA_DIR, HF_HUB_CACHE, HF_DATASETS_CACHE, HF_MODULES_CACHE,
             TRANSFORMERS_CACHE, OUTPUT_DIR, LOG_BASE]:
    os.makedirs(path, exist_ok=True)
