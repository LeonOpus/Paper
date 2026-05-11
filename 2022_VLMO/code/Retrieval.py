"""
Image-text retrieval fine-tuning for VLMO.
Supports COCO and Flickr30K.

Usage:
    python Retrieval.py --config configs/Retrieval_coco.yaml --output_dir <path>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must set before any HuggingFace imports
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/home/leon/Model/HuggingFace")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from datasets.retrieval_dataset import RetrievalDataset, build_transform
from models.vlmo import VLMoForRetrieval, build_vlmo_base


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluation(model: VLMoForRetrieval, data_loader: DataLoader,
               tokenizer: BertTokenizer, device: torch.device) -> dict:
    model.eval()
    texts, img_ids = [], []
    for _, caption, idx in data_loader:
        texts.append(caption)
        img_ids.append(idx)

    # encode all texts
    all_text_feats = []
    for batch_texts in texts:
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=40, return_tensors="pt").to(device)
        feat = model.encode_text(enc.input_ids, enc.attention_mask)
        all_text_feats.append(feat)
    text_feats = torch.cat(all_text_feats)  # (N_text, D)

    # encode all images
    all_img_feats = []
    for image, _, _ in data_loader:
        feat = model.encode_image(image.to(device))
        all_img_feats.append(feat)
    img_feats = torch.cat(all_img_feats)  # (N_img, D)

    # compute similarity matrix
    sims = img_feats @ text_feats.T  # (N_img, N_text)

    return _compute_recall(sims.cpu().numpy())


def _compute_recall(sims) -> dict:
    N = sims.shape[0]
    ranks = []
    for i in range(N):
        row = sims[i]
        sorted_idx = row.argsort()[::-1]
        rank = (sorted_idx == i).nonzero()[0].item()
        ranks.append(rank)

    ranks = [r for r in ranks]
    r1  = 100 * sum(r < 1 for r in ranks) / N
    r5  = 100 * sum(r < 5 for r in ranks) / N
    r10 = 100 * sum(r < 10 for r in ranks) / N
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedR": float(sorted(ranks)[N // 2])}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, tokenizer, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (image, caption, _) in enumerate(loader):
        image = image.to(device)
        enc = tokenizer(list(caption), padding=True, truncation=True,
                        max_length=40, return_tensors="pt").to(device)
        out = model(image, enc.input_ids, enc.attention_mask)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
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

    code_root = Path(__file__).parent
    data_dir  = Path(cfg.PAPER_DATA)

    train_dataset = RetrievalDataset(
        ann_file=str(data_dir / config["train_file"][0]),
        image_root=str(data_dir / "data"),
        transform=train_transform,
    )
    val_dataset = RetrievalDataset(
        ann_file=str(data_dir / config["val_file"]),
        image_root=str(data_dir / "data"),
        transform=test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"],
                              shuffle=True, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size_test"],
                              shuffle=False, num_workers=2, pin_memory=False)

    backbone = build_vlmo_base(img_size=config["image_res"])
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        backbone.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")

    model = VLMoForRetrieval(backbone, embed_dim=config["embed_dim"],
                             temp=config["temp"]).to(device)

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                  weight_decay=opt_cfg["weight_decay"])
    total_steps = len(train_loader) * opt_cfg.get("epochs", config["schedular"]["epochs"])
    warmup_steps = len(train_loader) * config["schedular"]["warmup_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt_cfg["lr"], total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    output_dir = Path(args.output_dir or cfg.OUTPUT_BASE)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_r1 = 0.0
    for epoch in range(config["schedular"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     tokenizer, device, epoch)
        metrics = evaluation(model, val_loader, tokenizer, device)
        print(f"Epoch {epoch} | loss={train_loss:.4f} | {metrics}")

        if metrics["R@1"] > best_r1:
            best_r1 = metrics["R@1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics},
                       output_dir / "best_model.pth")

    print(f"Best R@1: {best_r1:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/Retrieval_coco.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    main(args)
