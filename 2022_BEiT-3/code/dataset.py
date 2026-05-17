"""
Datasets for BEiT-3 evaluation on COCO retrieval and VQAv2.
"""
import json
import os
from PIL import Image
from torch.utils.data import Dataset


class CocoRetrievalDataset(Dataset):
    """Returns (image, caption, img_idx, cap_idx) for COCO Karpathy test split."""

    def __init__(self, ann_file: str, image_root: str, transform):
        anns = json.load(open(ann_file))
        self.transform = transform
        self.image_root = image_root

        # deduplicate images while preserving order
        self.images = []          # list of {"image_id", "image"}
        self.texts  = []          # list of caption strings
        self.img2txt = {}         # img_idx -> [cap_idx, ...]
        self.txt2img = {}         # cap_idx -> img_idx

        seen = {}
        for item in anns:
            iid = item["image_id"]
            if iid not in seen:
                seen[iid] = len(self.images)
                self.images.append({"image_id": iid, "image": item["image"]})
            img_idx = seen[iid]

            caps = item["caption"] if isinstance(item["caption"], list) else [item["caption"]]
            for cap in caps:
                cap_idx = len(self.texts)
                self.texts.append(cap)
                self.img2txt.setdefault(img_idx, []).append(cap_idx)
                self.txt2img[cap_idx] = img_idx

    def __len__(self):
        return len(self.images)

    def get_image(self, idx):
        path = os.path.join(self.image_root, self.images[idx]["image"])
        return self.transform(Image.open(path).convert("RGB"))

    def get_text(self, idx):
        return self.texts[idx]
