"""
Convert official COCO caption annotations to ALBEF/VLMO JSON format.

ALBEF format: [{"image": "coco/images/train2014/xxx.jpg", "caption": "..."}, ...]
One entry per caption (not per image), so ~5 captions per image.

Usage:
    python prepare_coco.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

DATA_DIR = Path(cfg.DATA_DIR)
ANN_DIR  = DATA_DIR / "coco" / "annotations"


def convert(ann_file: Path, split: str, image_prefix: str) -> list[dict]:
    with open(ann_file) as f:
        raw = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in raw["images"]}
    records = []
    for ann in raw["annotations"]:
        file_name = id2file[ann["image_id"]]
        records.append({
            "image": f"coco/images/{image_prefix}/{file_name}",
            "caption": ann["caption"].strip(),
        })

    out = DATA_DIR / f"coco_{split}.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records):,} records → {out}")
    return records


def main():
    zip_path = DATA_DIR / "annotations_trainval2014.zip"
    if not zip_path.exists():
        print(f"Not found: {zip_path}")
        print("Run download_data.sh first.")
        sys.exit(1)

    # Unzip into coco/annotations/
    import zipfile
    ann_dir = DATA_DIR / "coco" / "annotations"
    if not ann_dir.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR / "coco")
        print("Extraction done.")

    train_ann = ann_dir / "captions_train2014.json"
    val_ann   = ann_dir / "captions_val2014.json"

    train_records = convert(train_ann, "train", "train2014")

    # Split val → val (1000 images) + test (rest)
    with open(val_ann) as f:
        val_raw = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in val_raw["images"]}
    all_image_ids = sorted({ann["image_id"] for ann in val_raw["annotations"]})
    val_ids  = set(all_image_ids[:1000])
    test_ids = set(all_image_ids[1000:])

    val_records, test_records = [], []
    for ann in val_raw["annotations"]:
        rec = {"image": f"coco/images/val2014/{id2file[ann['image_id']]}",
               "caption": ann["caption"].strip()}
        if ann["image_id"] in val_ids:
            val_records.append(rec)
        else:
            test_records.append(rec)

    for name, records in [("val", val_records), ("test", test_records)]:
        out = DATA_DIR / f"coco_{name}.json"
        with open(out, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote {len(records):,} records → {out}")

    print("\nDone. Update configs/Retrieval_coco.yaml if needed.")


if __name__ == "__main__":
    main()
