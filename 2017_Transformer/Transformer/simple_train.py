#!/usr/bin/env python
"""
简化版的训练脚本，不依赖 Torchtext
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

import Constants as Constants
from Models import Transformer
from Optim import ScheduledOptim

# 加载演示数据
def load_data():
    with open(config.DEMO_DATA_JSON, 'r') as f:
        data = json.load(f)
    return data

# 简单的数据集
class TranslationDataset(Dataset):
    def __init__(self, examples, src_pad_idx, trg_pad_idx, max_len=100):
        self.examples = examples
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # 填充序列
        src = ex['src'] + [self.src_pad_idx] * (self.max_len - len(ex['src']))
        trg = ex['trg'] + [self.trg_pad_idx] * (self.max_len - len(ex['trg']))

        # 截断
        src = src[:self.max_len]
        trg = trg[:self.max_len]

        return torch.LongTensor(src), torch.LongTensor(trg)

def train_model():
    # 加载数据
    data = load_data()

    # 创建数据集
    train_dataset = TranslationDataset(
        data['train'],
        data['settings']['src_pad_idx'],
        data['settings']['trg_pad_idx'],
        data['settings']['max_len']
    )
    val_dataset = TranslationDataset(
        data['valid'],
        data['settings']['src_pad_idx'],
        data['settings']['trg_pad_idx'],
        data['settings']['max_len']
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    print(f"词汇表大小: {data['settings']['src_vocab_size']}")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = Transformer(
        data['settings']['src_vocab_size'],
        data['settings']['trg_vocab_size'],
        src_pad_idx=data['settings']['src_pad_idx'],
        trg_pad_idx=data['settings']['trg_pad_idx'],
        n_position=100,
        d_word_vec=64,
        d_model=64,
        d_inner=256,
        n_head=2,
        n_layers=1,
        dropout=0.1
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n开始训练...")

    # 训练循环
    model.train()
    for epoch in range(1):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (src, trg) in enumerate(progress_bar):
            src, trg = src.to(device), trg.to(device)

            # 前向传播
            optimizer.zero_grad()

            # 模型输出
            output = model(src, trg[:, :-1])  # 输入去掉最后一个token

            # 计算损失
            loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), trg[:, 1:].contiguous().view(-1))

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} 平均损失: {avg_loss:.4f}')

    # 保存模型
    os.makedirs(config.OUTPUT_BASE, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_BASE, 'simple_model.pth')
    torch.save(model.state_dict(), output_path)
    print(f"\n模型已保存到 {output_path}")

    # 测试生成
    print("\n测试生成:")
    model.eval()
    test_src = torch.LongTensor([[4, 5, 6]]).to(device)  # 测试句子
    with torch.no_grad():
        trg_init = torch.LongTensor([[data['settings']['trg_pad_idx']]]).to(device)
        test_output = model(test_src, trg_init)
        predicted = torch.argmax(test_output, dim=-1)
        print(f"输入: {test_src}")
        print(f"输出: {predicted}")

if __name__ == "__main__":
    if not os.path.exists(config.DEMO_DATA_JSON):
        from demo_data_simple import create_simple_data
        create_simple_data(config.DEMO_DATA_JSON)
    train_model()
