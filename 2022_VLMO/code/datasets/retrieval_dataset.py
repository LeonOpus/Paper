import json
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_res: int, is_train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_res, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_res, image_res)),
        transforms.ToTensor(),
        normalize,
    ])


class RetrievalDataset(Dataset):
    """
    JSON annotation format (same as ALBEF):
    [{"image": "coco/images/train2014/xxx.jpg", "caption": "A dog ..."}, ...]
    """

    def __init__(self, ann_file: str, image_root: str, transform: transforms.Compose):
        self.image_root = Path(image_root)
        with open(ann_file) as f:
            self.ann = json.load(f)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, idx: int):
        item = self.ann[idx]
        image = Image.open(self.image_root / item["image"]).convert("RGB")
        return self.transform(image), item["caption"], idx
