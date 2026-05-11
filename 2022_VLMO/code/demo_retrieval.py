"""
VLMO 10-minute reproduction demo.

Samples 4000 training pairs from COCO, trains 3 epochs,
then evaluates on the full val set and prints retrieval metrics.

Usage:
    python demo_retrieval.py
"""

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/home/leon/Model/HuggingFace")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from datasets.retrieval_dataset import RetrievalDataset, build_transform
from models.vlmo import VLMoForRetrieval, build_vlmo_base

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_SAMPLES = 4000
BATCH_TRAIN   = 64
BATCH_EVAL    = 64
EPOCHS        = 3
LR            = 1e-4   # higher lr for fast demo convergence
IMAGE_RES     = 224
EMBED_DIM     = 256
TEMP          = 0.07
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, loader, tokenizer, device):
    model.eval()
    all_img_feats, all_txt_feats = [], []

    for image, caption, _ in loader:
        img_feat = model.encode_image(image.to(device))
        all_img_feats.append(img_feat.cpu())

        enc = tokenizer(list(caption), padding=True, truncation=True,
                        max_length=40, return_tensors="pt").to(device)
        txt_feat = model.encode_text(enc.input_ids, enc.attention_mask)
        all_txt_feats.append(txt_feat.cpu())

    img_feats = torch.cat(all_img_feats)   # (N, D)
    txt_feats = torch.cat(all_txt_feats)   # (N, D)

    sims = img_feats @ txt_feats.T         # (N, N)
    N = sims.size(0)

    def recall(sims_matrix):
        ranks = []
        for i in range(N):
            row = sims_matrix[i].numpy()
            rank = (row.argsort()[::-1] == i).nonzero()[0].item()
            ranks.append(rank)
        r1  = 100 * sum(r < 1  for r in ranks) / N
        r5  = 100 * sum(r < 5  for r in ranks) / N
        r10 = 100 * sum(r < 10 for r in ranks) / N
        return r1, r5, r10

    i2t = recall(sims)
    t2i = recall(sims.T)
    return i2t, t2i


def main():
    data_dir = Path(cfg.PAPER_DATA)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer     = BertTokenizer.from_pretrained("bert-base-uncased")
    train_tf      = build_transform(IMAGE_RES, is_train=True)
    val_tf        = build_transform(IMAGE_RES, is_train=False)

    # ── Sample small training subset ──────────────────────────────────────────
    all_train = json.load(open(data_dir / "data/coco_train.json"))
    random.seed(42)
    subset = random.sample(all_train, TRAIN_SAMPLES)
    subset_path = "/tmp/coco_demo_train.json"
    json.dump(subset, open(subset_path, "w"))
    print(f"训练集: {TRAIN_SAMPLES} 条 (从 {len(all_train)} 条采样)")

    train_ds = RetrievalDataset(subset_path, str(data_dir / "data"), train_tf)
    val_ds   = RetrievalDataset(str(data_dir / "data/coco_val.json"),
                                str(data_dir / "data"), val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                              num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL,  shuffle=False,
                              num_workers=2, pin_memory=False)

    print(f"验证集: {len(val_ds)} 条  |  随机基线 R@1 ≈ {100/len(val_ds):.2f}%")

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone = build_vlmo_base(img_size=IMAGE_RES)
    model    = VLMoForRetrieval(backbone, embed_dim=EMBED_DIM, temp=TEMP).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    print(f"\n每轮 {len(train_loader)} 步，共 {EPOCHS} 轮\n{'─'*50}")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i, (image, caption, _) in enumerate(train_loader):
            image = image.to(device)
            enc   = tokenizer(list(caption), padding=True, truncation=True,
                              max_length=40, return_tensors="pt").to(device)
            out  = model(image, enc.input_ids, enc.attention_mask)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed  = (time.time() - t0) / 60

        i2t, t2i = evaluate(model, val_loader, tokenizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  [{elapsed:.1f}min]")
        print(f"  图→文  R@1={i2t[0]:.1f}%  R@5={i2t[1]:.1f}%  R@10={i2t[2]:.1f}%")
        print(f"  文→图  R@1={t2i[0]:.1f}%  R@5={t2i[1]:.1f}%  R@10={t2i[2]:.1f}%")
        model.train()

    total_min = (time.time() - t0) / 60
    print(f"\n{'─'*50}")
    print(f"完成，总耗时 {total_min:.1f} 分钟")


if __name__ == "__main__":
    main()
