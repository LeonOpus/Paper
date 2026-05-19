"""快速完整性检查：构建模型并用随机输入执行一次前向传播。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg  # 设置 HF 环境变量

import torch
from models.vlmo import build_vlmo_base, VLMoForRetrieval, VLMoForVQA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

backbone = build_vlmo_base(img_size=224).to(device)
total = sum(p.numel() for p in backbone.parameters()) / 1e6
print(f"VLMo-Base params: {total:.1f}M")

B = 2
image    = torch.randn(B, 3, 224, 224, device=device)
input_ids = torch.randint(0, 30522, (B, 20), device=device)
attn_mask = torch.ones(B, 20, device=device)

# 仅视觉
v_out = backbone(image=image, mode="vision")
print(f"Vision output:   {v_out.shape}")

# 仅语言
l_out = backbone(input_ids=input_ids, attention_mask=attn_mask, mode="language")
print(f"Language output: {l_out.shape}")

# 视觉-语言联合
vl_v, vl_l = backbone(image=image, input_ids=input_ids,
                       attention_mask=attn_mask, mode="vl")
print(f"VL vision:       {vl_v.shape}")
print(f"VL language:     {vl_l.shape}")

# 检索头
ret_model = VLMoForRetrieval(backbone).to(device)
out = ret_model(image, input_ids, attn_mask)
print(f"Retrieval loss:  {out['loss'].item():.4f}")

# VQA 头
vqa_model = VLMoForVQA(backbone, num_answers=3129).to(device)
labels = torch.zeros(B, 3129, device=device)
out = vqa_model(image, input_ids, attn_mask, labels=labels)
print(f"VQA loss:        {out['loss'].item():.4f}")

print("\nAll checks passed.")
