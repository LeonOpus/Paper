"""
Transformer: Attention Is All You Need (Vaswani et al., 2017)
Full encoder-decoder implementation from scratch.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q/k/v: (B, heads, T, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k    = d_model // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self._split_heads(self.w_q(q))
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))

        out, _ = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos   = PositionalEncoding(d_model, dropout, max_len)
        self.scale = math.sqrt(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        x = self.pos(self.embed(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos   = PositionalEncoding(d_model, dropout, max_len)
        self.scale = math.sqrt(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos(self.embed(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, n_layers: int = 6, dropout: float = 0.1,
                 max_len: int = 512, share_embed: bool = True):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len)
        self.decoder = Decoder(vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len)
        self.proj    = nn.Linear(d_model, vocab_size, bias=False)

        if share_embed:
            # Paper Section 3.4: share weights among encoder embed, decoder embed, and pre-softmax
            self.decoder.embed.weight = self.encoder.embed.weight
            self.proj.weight          = self.encoder.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_src_mask(src, pad_idx: int = 0):
        # (B, 1, 1, T_src)
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt, pad_idx: int = 0):
        T = tgt.size(1)
        pad_mask  = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)           # (B,1,1,T)
        causal    = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T,T)
        return pad_mask & causal                                          # (B,1,T,T)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out  = self.encoder(src, src_mask)
        dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.proj(dec_out)                                         # (B, T, vocab)

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask), src_mask

    def decode_step(self, tgt, enc_out, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.proj(dec_out)
