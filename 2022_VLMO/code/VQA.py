"""
VQA fine-tuning for VLMO (fusion-encoder mode, VL-FFN expert).

Usage:
    python VQA.py --config configs/VQA.yaml --output_dir <path>
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/home/leon/Model/HuggingFace")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from datasets.retrieval_dataset import build_transform
from datasets.vqa_dataset import VQADataset
from models.vlmo import VLMoForVQA, build_vlmo_base


@torch.no_grad()
def evaluation(model: VLMoForVQA, loader: DataLoader, tokenizer: BertTokenizer,
               answer_list: list[str], device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for image, question, target in loader:
        image  = image.to(device)
        target = target.to(device)
        enc = tokenizer(list(question), padding=True, truncation=True,
                        max_length=30, return_tensors="pt").to(device)
        out = model(image, enc.input_ids, enc.attention_mask)
        preds = out["logits"].argmax(dim=-1)
        gt    = target.argmax(dim=-1)
        correct += (preds == gt).sum().item()
        total   += image.size(0)
    return 100 * correct / total


def train_one_epoch(model, loader, optimizer, scheduler, tokenizer, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (image, question, target) in enumerate(loader):
        image  = image.to(device)
        target = target.to(device)
        enc = tokenizer(list(question), padding=True, truncation=True,
                        max_length=30, return_tensors="pt").to(device)
        out  = model(image, enc.input_ids, enc.attention_mask, labels=target)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if i % 50 == 0:
            print(f"[epoch {epoch} step {i}] loss={loss.item():.4f}")

    return total_loss / len(loader)


def main(args):
    import ruamel.yaml as yaml
    with open(args.config) as f:
        config = yaml.YAML().load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])

    train_transform = build_transform(config["image_res"], is_train=True)
    test_transform  = build_transform(config["image_res"], is_train=False)

    data_dir = Path(cfg.PAPER_DATA)

    train_dataset = VQADataset(
        ann_files=[str(data_dir / f) for f in config["train_file"]],
        image_root=str(data_dir / config.get("image_root", "")),
        answer_list=str(data_dir / config["answer_list"]),
        transform=train_transform,
    )
    test_dataset = VQADataset(
        ann_files=[str(data_dir / config["test_file"])],
        image_root=str(data_dir / config.get("image_root", "")),
        answer_list=str(data_dir / config["answer_list"]),
        transform=test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=config["batch_size_test"],
                              shuffle=False, num_workers=4, pin_memory=True)

    backbone = build_vlmo_base(img_size=config["image_res"])
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        backbone.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")

    num_answers = len(json.load(open(str(data_dir / config["answer_list"]))))
    model = VLMoForVQA(backbone, num_answers=num_answers).to(device)

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                  weight_decay=opt_cfg["weight_decay"])
    total_steps  = len(train_loader) * config["schedular"]["epochs"]
    warmup_steps = len(train_loader) * config["schedular"]["warmup_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt_cfg["lr"], total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    output_dir = Path(args.output_dir or cfg.OUTPUT_BASE) / "vqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(config["schedular"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     tokenizer, device, epoch)
        acc = evaluation(model, test_loader, tokenizer,
                         json.load(open(str(data_dir / config["answer_list"]))), device)
        print(f"Epoch {epoch} | loss={train_loss:.4f} | acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "acc": acc},
                       output_dir / "best_model.pth")

    print(f"Best VQA acc: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/VQA.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    main(args)
