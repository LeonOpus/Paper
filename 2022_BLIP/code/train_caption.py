"""
Fine-tune BLIP for image captioning on COCO Karpathy train split.
Usage: python train_caption.py --config ../configs/caption_coco.yaml
"""
import argparse
import os
import sys
import time
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

from dataset import CocoCaptionDataset


def train(args):
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = cfg.resolve_model(conf["model_name"])
    print(f"Loading model: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    if conf.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    image_root = cfg.resolve_data(conf.get("image_root", ""))
    train_dataset = CocoCaptionDataset(
        ann_file=cfg.resolve_data(conf["train_ann"]),
        image_root=image_root,
        processor=processor,
        split="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf.get("num_workers", 4),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(conf["lr"]),
        weight_decay=float(conf.get("weight_decay", 0.05)),
    )
    total_steps = len(train_loader) * conf["epochs"]
    warmup_steps = int(total_steps * conf.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=conf.get("fp16", True))

    out_rel = conf.get("output_dir", "")
    output_dir = os.path.join(cfg.OUTPUT_DIR, out_rel) if out_rel else cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"Train samples: {len(train_dataset)}, steps/epoch: {len(train_loader)}")

    global_step = 0
    for epoch in range(conf["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=conf.get("fp16", True)):
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss

            scaler.scale(loss).backward()

            if (step + 1) % conf.get("grad_accum", 1) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item()

            if step % 200 == 0:
                elapsed = time.time() - t0
                print(f"Epoch {epoch+1} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"time={elapsed:.0f}s")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} done. avg_loss={avg_loss:.4f}")

        save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}")
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args)
