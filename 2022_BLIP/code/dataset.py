import json
import os
from PIL import Image
from torch.utils.data import Dataset


class CocoCaptionDataset(Dataset):
    def __init__(self, ann_file, image_root, processor, split="train"):
        self.image_root = image_root
        self.processor = processor
        self.split = split

        anns = json.load(open(ann_file))
        self.anns = anns

        # 验证/测试集：按图像分组标注，用于评估
        if split != "train":
            seen = {}
            self.images = []
            self.image_ids = []
            self.references = {}
            for item in anns:
                img_path = item["image"]
                if img_path not in seen:
                    seen[img_path] = len(self.images)
                    self.images.append(img_path)
                    iid = item.get("image_id", img_path)
                    self.image_ids.append(iid)
                    self.references[iid] = []
                iid = item.get("image_id", img_path)
                cap = item["caption"] if isinstance(item["caption"], str) else item["caption"][0]
                self.references[iid].append(cap)

    def __len__(self):
        if self.split == "train":
            return len(self.anns)
        return len(self.images)

    def __getitem__(self, idx):
        if self.split == "train":
            ann = self.anns[idx]
            img_path = os.path.join(self.image_root, ann["image"])
            image = Image.open(img_path).convert("RGB")
            caption = ann["caption"]
            if isinstance(caption, list):
                caption = caption[0]
            inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=40)
            return {k: v.squeeze(0) for k, v in inputs.items()}
        else:
            img_path = os.path.join(self.image_root, self.images[idx])
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in inputs.items()}, self.image_ids[idx]


class CocoRetrievalDataset(Dataset):
    """
    为检索评估返回 (image, text) 对。
    构建独立的图像列表和文本列表，用于 R@K 计算。
    """
    def __init__(self, ann_file, image_root, processor):
        self.image_root = image_root
        self.processor = processor

        anns = json.load(open(ann_file))

        # 去重图像，保留全部文本
        img_to_idx = {}
        self.images = []
        self.texts = []
        self.txt2img = []   # 文本索引 i -> 图像索引
        self.img2txt = []   # 图像索引 i -> 文本索引列表

        for item in anns:
            img_path = item["image"]
            captions = item["caption"] if isinstance(item["caption"], list) else [item["caption"]]

            if img_path not in img_to_idx:
                img_to_idx[img_path] = len(self.images)
                self.images.append(img_path)
                self.img2txt.append([])

            img_idx = img_to_idx[img_path]
            for cap in captions:
                txt_idx = len(self.texts)
                self.texts.append(cap)
                self.txt2img.append(img_idx)
                self.img2txt[img_idx].append(txt_idx)

    def __len__(self):
        return len(self.images)

    def get_image(self, idx):
        img_path = os.path.join(self.image_root, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def get_text(self, idx):
        return self.processor(text=self.texts[idx], return_tensors="pt", padding="max_length",
                              truncation=True, max_length=35)

    def __getitem__(self, idx):
        return self.get_image(idx), idx
