"""
在 WMT14 英德翻译任务上训练 Transformer。
用法：python train.py --config ../configs/translation_wmt14.yaml
"""
import argparse
import os
import sys
import time
import math
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from model import Transformer
from dataset import build_loader


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx

    def forward(self, logits, target):
        # logits: (B*T, V)，target: (B*T,)
        # 避免完整展开 (B*T, V) 分布——分别计算 NLL 和均匀平滑项
        log_probs = torch.log_softmax(logits, dim=-1)
        nll    = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # 负对数似然 (B*T,)
        smooth = -log_probs.mean(dim=-1)                               # 均匀平滑项 (B*T,)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        mask = target == self.pad_idx
        return loss.masked_fill(mask, 0.0).sum() / (~mask).sum().clamp(min=1)


class WarmupScheduler:
    """论文公式：学习率 = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)"""
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup ** -1.5
        )
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr


def train(args):
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    sp_model = cfg.resolve_data("tokenizer.model")
    train_loader = build_loader(
        cfg.resolve_data("train.json"), sp_model,
        batch_size=conf.get("batch_size_sentences", 128),
        max_len=conf["max_len"], shuffle=True,
        num_workers=conf.get("num_workers", 4),
    )
    val_loader = build_loader(
        cfg.resolve_data("val.json"), sp_model,
        batch_size=conf.get("batch_size_sentences", 128),
        max_len=conf["max_len"], shuffle=False,
        num_workers=conf.get("num_workers", 4),
    )

    model = Transformer(
        vocab_size=conf["vocab_size"],
        d_model=conf["d_model"],
        n_heads=conf["n_heads"],
        d_ff=conf["d_ff"],
        n_layers=conf["n_layers"],
        dropout=conf["dropout"],
        max_len=conf["max_len"] + 2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.1f}M", flush=True)

    accum_steps = conf.get("gradient_accumulation_steps", 1)
    criterion = LabelSmoothingLoss(conf["vocab_size"], conf["label_smoothing"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, conf["d_model"], conf["warmup_steps"])

    out_rel = conf.get("output_dir", "")
    output_dir = os.path.join(cfg.OUTPUT_DIR, out_rel) if out_rel else cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    global_step = 0
    max_steps = conf.get("max_steps", None)

    optimizer.zero_grad()
    for epoch in range(1, conf["epochs"] + 1):
        model.train()
        epoch_loss, n_updates, t0 = 0.0, 0, time.time()

        for mini_step, (src, tgt_in, tgt_out) in enumerate(train_loader):
            if max_steps and global_step >= max_steps:
                break

            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            # 损失除以累积步数，使梯度等效于大 batch
            logits = model(src, tgt_in)                                  # 形状 (B, T, V)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            (loss / accum_steps).backward()

            # 每 accum_steps 个 mini-batch 做一次参数更新
            if (mini_step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                lr = scheduler.step()
                global_step += 1
                epoch_loss += loss.item()
                n_updates += 1

                if global_step % 500 == 0:
                    log(f"Epoch {epoch} global={global_step} "
                        f"loss={loss.item():.4f} lr={lr:.2e} time={time.time()-t0:.0f}s")

        if n_updates > 0:
            avg_loss = epoch_loss / n_updates
            log(f"Epoch {epoch} done. avg_loss={avg_loss:.4f} ppl={math.exp(avg_loss):.2f}")
        else:
            log(f"Epoch {epoch} done. (no updates this epoch)")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt_in, tgt_out in val_loader:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                logits = model(src, tgt_in)
                val_loss += criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1)).item()
        val_loss /= len(val_loader)
        log(f"Epoch {epoch} val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}")

        ckpt = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_step": scheduler.step_num,
        }, ckpt)
        log(f"Saved {ckpt}")

        if max_steps and global_step >= max_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    train(parser.parse_args())
