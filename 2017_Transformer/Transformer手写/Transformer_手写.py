import torch
import torch.nn as nn
import math


class Emberddings(nn.Module):
    def __init__(self, vocab_size, d_model):  # 用self之后就可以new直接调用参数
        super().__init__()  # 保证自定义的代码模块是继承于nn.Module的，能被pytorch追踪识别，不写可能也没关系，也有可能会报错
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model  # Embedding会被其他类调用，所以需要把d_model也传进来，这样才能保证d_model的一致性

    def forward(self, x):  # 对外接口，乘以根号d_model
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionnalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(100000.0)) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # 不会作为模型的参数，而是作为模型的缓冲区 # 注册缓冲区的目的是为了在模型的保存和加载过程中保持缓冲区的状态

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def attension(query, key, value, mask=None, dropout=none):  # type: ignore
    d_k = query.size(-1)
    scopes = query @ key.transpose(-2, -1) / math.sqrt(d_k)  # 缩放点积注意力

    if mask is not None:
        scores = scopes.masked_fill(mask == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)

    if dropout:
        attn = dropout(attn)
    return attn @ value, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):  # 传入多少头
        super().__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):  # 传入的是q,k,vdd
        batch_size = query.size(0)

        def transform(x, linear):
            x = linear(x)
            return x.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)

        x, _ = attension(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linear_out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm()))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList(
            [AddNorm(d_model, dropout), AddNorm(d_model, dropout)]
        )

    def forward(self, x, mask):
        out1 = self.sublayer[0](x, lambda y: self.self_attn(x, x, x, mask))
        out2 = self.sublayer[1](out1, self.feed_forward(out1))
        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout):
        self.self_attn_ = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout),
            ]
        )

    def forward(self, x, memory, src_mask, tag_mask):
        out1 = self.sublayer[0](x, lambda y1: self.self_attn(x, x, x, tag_mask))
        out2 = self.sublayer[1](out1, lambda y2: self.cross_attn(out1, memory, memory))
        out3 = self.sublayer[2](out2, lambda y3: self.feed_forward(out2))
        return out3


class Transformer(nn.Module):
    def __init__(
        self, src_vocab, tag_vocab, d_model=512, N=6, head=8, d_ff=2048, dropout=0.1
    ):
        super().__init__()
        self.src_embberding = nn.Sequential(
            Emberddings(src_vocab, d_model), PositionnalEncoding(d_model)
        )

        self.target_embberding = nn.Sequential(
            Emberddings(tag_vocab, d_model), PositionnalEncoding(d_model)
        )

        attn = lambda: MultiHeadAttention(head, d_model, dropout)
        ff = lambda: FeedForward(d_model, d_ff, dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)]
        )

        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, attn(), attn(), ff(), dropout) for _ in range(N)]
        )

        self.out = nn.Linear(d_model, tag_vocab)

        def encoder(self, src, src_mask):
            x = self.src_embed(src)
            for layer in self.encoder:
                x = layer(x, src_mask)
            return x

        def decoder(self, target, memory, src_mask, target_mask):
            x = self.target_embberding(target)
            for layer in self.decoder:
                x = layer(x, memory, src_mask, target_mask)
            return x

        def forward(self, src, target, src_mask=None, target_mask=None):
            memory = self.encoder(src, src_mask)
            out = self.decode(target, memory, src_mask, target_mask)
            return self.out(out)
