"""
VLMO 驱动稳定性测试 —— 跑满 ~60 分钟，确认 GPU 无崩溃。

使用全量 COCO train（414k 条），batch=64，最多 1 epoch。
每 200 步打印 loss / GPU 温度 / 显存。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/home/leon/Model/HuggingFace")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from datasets.retrieval_dataset import RetrievalDataset, build_transform
from models.vlmo import VLMoForRetrieval, build_vlmo_base

BATCH      = 96
IMAGE_RES  = 224
EMBED_DIM  = 256
TEMP       = 0.07
LR         = 1e-5
LOG_EVERY  = 200   # 每 N 步打印一次
TIME_LIMIT = 60    # 分钟


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            text=True).strip()
        temp, mem, pwr = out.split(", ")
        return f"GPU {temp}°C  {int(mem)//1024}GB  {float(pwr):.0f}W"
    except Exception:
        return "gpu-stats N/A"


def main():
    data_dir = Path(cfg.PAPER_DATA)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_tf  = build_transform(IMAGE_RES, is_train=True)

    train_file = str(data_dir / "data/coco_train.json")
    train_ds   = RetrievalDataset(train_file, str(data_dir / "data"), train_tf)
    loader     = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                            num_workers=4, pin_memory=True)

    backbone = build_vlmo_base(img_size=IMAGE_RES)
    model    = VLMoForRetrieval(backbone, embed_dim=EMBED_DIM, temp=TEMP).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)

    total_steps = len(loader)
    print(f"全量训练集: {len(train_ds)} 条  |  {total_steps} 步/epoch  |  上限 {TIME_LIMIT} 分钟")
    print(f"开始时间: {time.strftime('%H:%M:%S')}")
    print(f"{'─'*60}")

    t0 = time.time()
    model.train()

    for step, (image, caption, _) in enumerate(loader, 1):
        elapsed = (time.time() - t0) / 60
        if elapsed >= TIME_LIMIT:
            print(f"\n⏱  已达 {TIME_LIMIT} 分钟上限，提前结束（{step}/{total_steps} 步）")
            break

        image = image.to(device)
        enc   = tokenizer(list(caption), padding=True, truncation=True,
                          max_length=40, return_tensors="pt").to(device)
        out   = model(image, enc.input_ids, enc.attention_mask)

        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == total_steps:
            print(f"[{elapsed:5.1f}min]  step {step:>5}/{total_steps}  "
                  f"loss={out['loss'].item():.4f}  {gpu_stats()}")

    total = (time.time() - t0) / 60
    print(f"{'─'*60}")
    print(f"✅ 稳定性测试完成，历时 {total:.1f} 分钟，无崩溃")


if __name__ == "__main__":
    main()
