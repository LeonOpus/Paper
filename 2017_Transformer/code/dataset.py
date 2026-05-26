"""
WMT14 英德翻译数据集，使用 SentencePiece 分词并按句子数量组批。
"""
import json
import os
import multiprocessing as mp
import torch
from torch.utils.data import Dataset, DataLoader

# worker 进程全局变量，避免每条句子重复加载模型
_sp = None

def _init_worker(sp_model_path):
    global _sp
    import sentencepiece as spm
    _sp = spm.SentencePieceProcessor(model_file=sp_model_path)

def _encode_chunk(args):
    chunk, max_len = args
    pairs = []
    for item in chunk:
        src_ids = _sp.encode(item["src"])[:max_len]
        tgt_ids = _sp.encode(item["tgt"])[:max_len]
        if len(src_ids) > 0 and len(tgt_ids) > 0:
            pairs.append((src_ids, tgt_ids))
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, json_file: str, sp_model_path: str, max_len: int = 128):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.max_len = max_len
        self.bos = sp.bos_id()  # 2
        self.eos = sp.eos_id()  # 3
        self.pad = sp.pad_id()  # 0

        raw = json.load(open(json_file))
        n_workers = mp.cpu_count()
        print(f"分词中，使用 {n_workers} 个 CPU 核心，共 {len(raw)} 句对...", flush=True)

        # 切成 n_workers 块，每个 worker 拿整块处理，避免逐条 pickle 的开销
        chunks = [raw[i::n_workers] for i in range(n_workers)]
        with mp.Pool(n_workers, initializer=_init_worker, initargs=(sp_model_path,)) as pool:
            results = pool.map(_encode_chunk, [(chunk, max_len) for chunk in chunks])

        self.pairs = [pair for chunk in results for pair in chunk]
        print(f"分词完成，有效句对: {len(self.pairs)}", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        src = torch.tensor(src_ids, dtype=torch.long)
        # 解码器输入：<bos> + tgt；目标序列：tgt + <eos>
        tgt_in  = torch.tensor([self.bos] + tgt_ids, dtype=torch.long)
        tgt_out = torch.tensor(tgt_ids + [self.eos], dtype=torch.long)
        return src, tgt_in, tgt_out


def collate_fn(batch, pad_id: int = 0):
    srcs, tgt_ins, tgt_outs = zip(*batch)
    src     = torch.nn.utils.rnn.pad_sequence(srcs,     batch_first=True, padding_value=pad_id)
    tgt_in  = torch.nn.utils.rnn.pad_sequence(tgt_ins,  batch_first=True, padding_value=pad_id)
    tgt_out = torch.nn.utils.rnn.pad_sequence(tgt_outs, batch_first=True, padding_value=pad_id)
    return src, tgt_in, tgt_out


def build_loader(json_file: str, sp_model_path: str, batch_size: int,
                 max_len: int = 128, shuffle: bool = True, num_workers: int = 4):
    ds = TranslationDataset(json_file, sp_model_path, max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
