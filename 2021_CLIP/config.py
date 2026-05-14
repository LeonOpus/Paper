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

PAPER_DATA = os.environ.get("PAPER_DATA", f"/home/leon/DataSet/HuggingFace/{PAPER_NAME}")
HF_HOME = os.environ.get("HF_HOME", "/home/leon/Model/HuggingFace")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", "/home/leon/DataSet/HuggingFace")
HF_MODULES_CACHE = os.path.join(HF_HOME, "modules")
TRANSFORMERS_CACHE = os.path.join(HF_HOME, "transformers")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_MODULES_CACHE"] = HF_MODULES_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", f"/home/leon/Output/{PAPER_NAME}")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(OUTPUT_ROOT, "output"))
OUTPUT_BASE = OUTPUT_DIR
LOG_BASE = os.path.join(OUTPUT_ROOT, "logs")

DATA_DIR = os.path.join(PAPER_DATA, "data")
CIFAR10_DIR = os.path.join(DATA_DIR, "cifar10")
CIFAR100_DIR = os.path.join(DATA_DIR, "cifar100")

for path in [
    DATA_DIR,
    HF_HUB_CACHE,
    HF_DATASETS_CACHE,
    HF_MODULES_CACHE,
    TRANSFORMERS_CACHE,
    OUTPUT_BASE,
    LOG_BASE,
    CIFAR10_DIR,
    CIFAR100_DIR,
]:
    os.makedirs(path, exist_ok=True)
