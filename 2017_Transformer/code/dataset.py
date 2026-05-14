"""
WMT14 EN-DE dataset with SentencePiece tokenization and token-budget batching.
"""
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self, json_file: str, sp_model_path: str, max_len: int = 128):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.max_len = max_len
        self.bos = self.sp.bos_id()  # 2
        self.eos = self.sp.eos_id()  # 3
        self.pad = self.sp.pad_id()  # 0

        raw = json.load(open(json_file))
        self.pairs = []
        for item in raw:
            src_ids = self.sp.encode(item["src"])[:max_len]
            tgt_ids = self.sp.encode(item["tgt"])[:max_len]
            if len(src_ids) > 0 and len(tgt_ids) > 0:
                self.pairs.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        src = torch.tensor(src_ids, dtype=torch.long)
        # decoder input: <bos> + tgt; target: tgt + <eos>
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
