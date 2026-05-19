"""
BEiT-3 在 COCO 检索和 VQAv2 任务上的数据集定义。
"""
import json
import os
from PIL import Image
from torch.utils.data import Dataset


class CocoRetrievalDataset(Dataset):
    """返回 COCO Karpathy 测试集中的 (image, caption, img_idx, cap_idx) 四元组。"""

    def __init__(self, ann_file: str, image_root: str, transform):
        anns = json.load(open(ann_file))
        self.transform = transform
        self.image_root = image_root

        # 去重图像，同时保留原始顺序
        self.images = []          # 元素格式为 {"image_id", "image"} 的列表
        self.texts  = []          # 标注文本字符串列表
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
