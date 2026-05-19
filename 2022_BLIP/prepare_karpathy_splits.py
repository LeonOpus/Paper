"""
从现有 COCO 标注文件构建 BLIP 格式的 COCO Karpathy val/test 划分。

Karpathy 划分规则（依据 captions_val2014.json 中的顺序）：
  val:     前 5,000 张图像
  test:    第 5,000 至 9,999 张图像
  restval: 第 10,000 张起（部分论文用于训练，此处跳过）

训练集直接来自已有的 VLMO coco_train.json
（train2014 中的 82,783 张图像，与 Karpathy 训练划分相同）。

输出格式与 BLIP/ALBEF 一致：
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
    # 从文件名中提取 image_id 字段
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

    # 建立 image_id -> file_name 的映射
    id_to_fname = {img["id"]: img["file_name"] for img in raw["images"]}

    # 按图像在 JSON 中的位置排序（Karpathy 规范）
    ordered_ids = [img["id"] for img in raw["images"]]

    val_ids  = set(ordered_ids[:5000])
    test_ids = set(ordered_ids[5000:10000])

    # 建立 image_id -> 标注列表 的映射
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
