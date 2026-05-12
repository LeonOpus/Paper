"""
Construct BLIP-format COCO Karpathy val/test splits from existing COCO annotations.

Karpathy split convention (by order in captions_val2014.json):
  val:     first 5,000 images
  test:    images 5,000-9,999
  restval: images 10,000+ (used by some papers for training; skipped here)

Train split is taken directly from the existing VLMO coco_train.json
(82,783 images from train2014, same as Karpathy train split).

Output format matches BLIP/ALBEF:
  [{"image": "coco/images/val2014/COCO_val2014_XXXXXX.jpg",
    "image_id": XXXXXX, "caption": "..."}, ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

VLMO_DATA = os.path.join(os.path.expanduser("~"), "DataSet", "HuggingFace", "2022_VLMO", "data")
OUT_DIR = os.path.join(os.path.expanduser("~"), "DataSet", "HuggingFace", "2022_BLIP", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def build_train():
    src = os.path.join(VLMO_DATA, "coco_train.json")
    dst = os.path.join(OUT_DIR, "coco_karpathy_train.json")
    if os.path.exists(dst):
        print(f"[skip] {dst}")
        return
    anns = json.load(open(src))
    # Add image_id field from filename
    out = []
    for item in anns:
        fname = os.path.basename(item["image"])          # COCO_train2014_000000XXXXXX.jpg
        iid = int(fname.replace(".jpg", "").split("_")[-1])
        out.append({"image": item["image"], "image_id": iid, "caption": item["caption"]})
    json.dump(out, open(dst, "w"))
    print(f"[done] train: {len(out)} entries -> {dst}")


def build_val_test():
    cap_file = os.path.join(VLMO_DATA, "coco", "annotations", "captions_val2014.json")
    raw = json.load(open(cap_file))

    # Map image_id -> file_name
    id_to_fname = {img["id"]: img["file_name"] for img in raw["images"]}

    # Order images by their position in the JSON (Karpathy convention)
    ordered_ids = [img["id"] for img in raw["images"]]

    val_ids  = set(ordered_ids[:5000])
    test_ids = set(ordered_ids[5000:10000])

    # Build image_id -> list of captions
    id_to_caps = {}
    for ann in raw["annotations"]:
        id_to_caps.setdefault(ann["image_id"], []).append(ann["caption"])

    def build_split(id_set, name):
        out = []
        for iid in ordered_ids:
            if iid not in id_set:
                continue
            fname = id_to_fname[iid]
            img_path = f"coco/images/val2014/{fname}"
            for cap in id_to_caps.get(iid, []):
                out.append({"image": img_path, "image_id": iid, "caption": cap})
        dst = os.path.join(OUT_DIR, f"coco_karpathy_{name}.json")
        if os.path.exists(dst):
            print(f"[skip] {dst}")
            return
        json.dump(out, open(dst, "w"))
        imgs = len(id_set)
        print(f"[done] {name}: {len(out)} entries ({imgs} images) -> {dst}")

    build_split(val_ids,  "val")
    build_split(test_ids, "test")


if __name__ == "__main__":
    print("Building Karpathy train split from VLMO coco_train.json ...")
    build_train()
    print("Building Karpathy val/test splits from captions_val2014.json ...")
    build_val_test()
    print("Done.")
