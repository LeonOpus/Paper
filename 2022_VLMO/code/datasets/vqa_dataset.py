import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.retrieval_dataset import build_transform


class VQADataset(Dataset):
    """
    Annotation format:
    [{"image": "...", "question": "...", "answer": [...], "weights": [...]}]
    """

    def __init__(self, ann_files: list[str], image_root: str, answer_list: str,
                 transform, max_ques_words: int = 30):
        self.image_root = Path(image_root)
        self.ann: list[dict] = []
        for f in ann_files:
            with open(f) as fp:
                self.ann += json.load(fp)

        with open(answer_list) as f:
            self.answer_list: list[str] = json.load(f)
        self.ans2id = {a: i for i, a in enumerate(self.answer_list)}

        self.transform = transform
        self.max_ques_words = max_ques_words

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, idx: int):
        item = self.ann[idx]
        image = Image.open(self.image_root / item["image"]).convert("RGB")
        image = self.transform(image)

        question = item["question"]

        # soft labels
        target = torch.zeros(len(self.answer_list))
        for ans, w in zip(item.get("answer", []), item.get("weights", [])):
            if ans in self.ans2id:
                target[self.ans2id[ans]] = min(1.0, target[self.ans2id[ans]] + w)

        return image, question, target
