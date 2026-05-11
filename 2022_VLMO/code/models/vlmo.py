"""
VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts
https://arxiv.org/abs/2111.02358

Key idea: replace standard FFN in transformer layers with Mixture-of-Modality-Experts (MoME).
Each layer has three expert FFNs:
  - V-FFN  : activated when processing vision-only tokens
  - L-FFN  : activated when processing language-only tokens
  - VL-FFN : activated when processing joint vision-language tokens
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use Flash Attention when available (no mask) to avoid O(N²) memory
        if mask is None and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask, float('-inf'))
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# MoME: Mixture-of-Modality-Experts FFN
# ---------------------------------------------------------------------------

class MoMETransformerLayer(nn.Module):
    """Single transformer layer with three switchable FFN experts."""

    VISION = "vision"
    LANGUAGE = "language"
    VL = "vl"

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        mlp_hidden = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2_v  = nn.LayerNorm(dim)
        self.norm2_l  = nn.LayerNorm(dim)
        self.norm2_vl = nn.LayerNorm(dim)

        make_ffn = partial(Mlp, in_features=dim, hidden_features=mlp_hidden, drop=drop)
        self.ffn_v  = make_ffn()
        self.ffn_l  = make_ffn()
        self.ffn_vl = make_ffn()

    def forward(self, x: torch.Tensor, mode: str = "vl",
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)

        if mode == self.VISION:
            x = x + self.ffn_v(self.norm2_v(x))
        elif mode == self.LANGUAGE:
            x = x + self.ffn_l(self.norm2_l(x))
        else:
            x = x + self.ffn_vl(self.norm2_vl(x))

        return x


# ---------------------------------------------------------------------------
# Vision patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# VLMO backbone (shared transformer with MoME)
# ---------------------------------------------------------------------------

class VLMo(nn.Module):
    """
    Unified VL transformer with Mixture-of-Modality-Experts.

    Supports three operating modes per forward pass:
      - "vision"   : image-only (for vision pre-training stage)
      - "language" : text-only  (for language pre-training stage)
      - "vl"       : joint      (for VL pre-training / fine-tuning)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 vocab_size: int = 30522, max_text_len: int = 40,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_text_len = max_text_len

        # --- Vision side ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_v = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # --- Language side ---
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed_l = nn.Embedding(max_text_len + 2, embed_dim)
        self.token_type_embed = nn.Embedding(2, embed_dim)

        # --- Shared MoME transformer ---
        self.layers = nn.ModuleList([
            MoMETransformerLayer(embed_dim, num_heads, mlp_ratio, drop=drop_rate,
                                 attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed_v, std=0.02)
        nn.init.trunc_normal_(self.cls_token_v, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        B = image.size(0)
        x = self.patch_embed(image)
        cls = self.cls_token_v.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed_v

        for layer in self.layers:
            x = layer(x, mode=MoMETransformerLayer.VISION)

        return self.norm(x)  # (B, 1+num_patches, D)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.word_embed(input_ids) + self.pos_embed_l(pos)

        mask = None
        if attention_mask is not None:
            # (B, 1, 1, L): True where padding
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x = layer(x, mode=MoMETransformerLayer.LANGUAGE, mask=mask)

        return self.norm(x)  # (B, L, D)

    def encode_vl(self, image: torch.Tensor, input_ids: torch.Tensor,
                  attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        B = image.size(0)

        # vision tokens
        v_tokens = self.patch_embed(image)
        cls = self.cls_token_v.expand(B, -1, -1)
        v_tokens = torch.cat([cls, v_tokens], dim=1) + self.pos_embed_v
        N_v = v_tokens.size(1)

        # language tokens
        pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        l_tokens = self.word_embed(input_ids) + self.pos_embed_l(pos)

        x = torch.cat([v_tokens, l_tokens], dim=1)  # (B, N_v+L, D)

        pad_mask = None
        if attention_mask is not None:
            v_mask = torch.ones(B, N_v, device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([v_mask, attention_mask], dim=1)
            pad_mask = (full_mask == 0).unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x = layer(x, mode=MoMETransformerLayer.VL, mask=pad_mask)

        x = self.norm(x)
        return x[:, :N_v], x[:, N_v:]  # (B, N_v, D), (B, L, D)

    def forward(self, image: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                mode: str = "vl"):
        if mode == "vision":
            return self.encode_image(image)
        if mode == "language":
            return self.encode_text(input_ids, attention_mask)
        return self.encode_vl(image, input_ids, attention_mask)


# ---------------------------------------------------------------------------
# Task heads built on top of VLMo backbone
# ---------------------------------------------------------------------------

class VLMoForRetrieval(nn.Module):
    """Dual-encoder retrieval with ITC loss (uses V-FFN / L-FFN experts)."""

    def __init__(self, vlmo: VLMo, embed_dim: int = 256, temp: float = 0.07):
        super().__init__()
        self.vlmo = vlmo
        D = vlmo.embed_dim
        self.vision_proj = nn.Linear(D, embed_dim)
        self.text_proj   = nn.Linear(D, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * temp)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.vlmo.encode_image(image)[:, 0]  # CLS token
        return F.normalize(self.vision_proj(feats), dim=-1)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.vlmo.encode_text(input_ids, attention_mask)[:, 0]
        return F.normalize(self.text_proj(feats), dim=-1)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> dict:
        img_feat  = self.encode_image(image)
        text_feat = self.encode_text(input_ids, attention_mask)

        sim_i2t = img_feat @ text_feat.T / self.temp
        sim_t2i = sim_i2t.T

        B = img_feat.size(0)
        labels = torch.arange(B, device=img_feat.device)
        loss = (F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_t2i, labels)) / 2

        return {"loss": loss, "sim_i2t": sim_i2t, "sim_t2i": sim_t2i}


class VLMoForVQA(nn.Module):
    """Fusion-encoder VQA with classification head (uses VL-FFN expert)."""

    def __init__(self, vlmo: VLMo, num_answers: int):
        super().__init__()
        self.vlmo = vlmo
        D = vlmo.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, num_answers),
        )

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        _, text_feats = self.vlmo.encode_vl(image, input_ids, attention_mask)
        cls_feat = text_feats[:, 0]  # [CLS] of the language side
        logits = self.classifier(cls_feat)

        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.binary_cross_entropy_with_logits(logits, labels)
        return out


def build_vlmo_base(img_size: int = 224) -> VLMo:
    return VLMo(img_size=img_size, patch_size=16, embed_dim=768,
                depth=12, num_heads=12, mlp_ratio=4.0)
