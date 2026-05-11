#!/usr/bin/env python
"""
GPU版本的训练脚本
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

import Constants as Constants
from Models import Transformer
from Optim import ScheduledOptim
import config

# 加载演示数据
def load_data():
    with open(config.DEMO_DATA_JSON, 'r') as f:
        data = json.load(f)
    return data

# 简单的数据集
class TranslationDataset(Dataset):
    def __init__(self, examples, src_vocab, trg_vocab, src_pad_idx, trg_pad_idx, max_len=100):
        self.examples = examples
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
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
        data['src_vocab'],
        data['trg_vocab'],
        data['settings']['src_pad_idx'],
        data['settings']['trg_pad_idx'],
        data['settings']['max_len']
    )
    val_dataset = TranslationDataset(
        data['valid'],
        data['src_vocab'],
        data['trg_vocab'],
        data['settings']['src_pad_idx'],
        data['settings']['trg_pad_idx'],
        data['settings']['max_len']
    )

    print(f"词汇表大小: {data['settings']['src_vocab_size']}")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # 创建模型 - 自动选择GPU或CPU，如果GPU出错则使用CPU
    try:
        # 先尝试GPU
        test_tensor = torch.randn(1, 1).cuda()
        device = torch.device('cuda')
        print(f"使用设备: {device}")
    except RuntimeError:
        # 如果GPU出错，使用CPU
        device = torch.device('cpu')
        print(f"GPU不可用，使用设备: {device}")

    # 使用最小的模型参数
    model = Transformer(
        n_src_vocab=data['settings']['src_vocab_size'],
        n_trg_vocab=data['settings']['trg_vocab_size'],
        src_pad_idx=data['settings']['src_pad_idx'],
        trg_pad_idx=data['settings']['trg_pad_idx'],
        d_word_vec=32,
        d_model=32,
        d_inner=128,
        n_layers=1,
        n_head=2,
        dropout=0.1
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=data['settings']['src_pad_idx'])

    print("\n开始训练...")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    model.train()
    for epoch in range(10):  # 训练10个epoch
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10')

        for batch_idx, (src, trg) in enumerate(progress_bar):
            src, trg = src.to(device), trg.to(device)

            # 前向传播
            optimizer.zero_grad()

            # 预测下一个词
            output = model(src, trg[:, :-1])

            # 计算损失
            loss = criterion(
                output.view(-1, output.size(-1)),
                trg[:, 1:].contiguous().view(-1)
            )

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} 平均损失: {avg_loss:.4f}')

    # 保存模型
        os.makedirs(config.OUTPUT_BASE, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_BASE, 'transformer_model_gpu.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'settings': data['settings']
    }, output_path)

    print(f"\n模型已保存到 {output_path}")

    # 测试生成
    print("\n测试生成:")
    model.eval()
    test_src = torch.LongTensor([[2, 8, 3]]).to(device)  # I love you
    with torch.no_grad():
        # 创建一个起始token用于解码
        trg_init = torch.LongTensor([[2]]).to(device)  # <s>

        # 模型输出
        output = model(test_src, trg_init)
        predicted = torch.argmax(output, dim=-1)

        # 将预测的索引转换为单词
        idx_to_word = {idx: word for word, idx in zip(data['src_vocab'], range(len(data['src_vocab'])))}

        print(f"输入索引: {test_src.squeeze().tolist()}")
        print(f"输入句子: {' '.join([idx_to_word[idx] for idx in test_src.squeeze().tolist()])}")
        print(f"输出索引: {predicted.squeeze().tolist()}")

        # 处理输出
        if predicted.dim() > 1:
            pred_list = predicted.squeeze().tolist()
            if isinstance(pred_list, list):
                print(f"输出句子: {' '.join([idx_to_word[idx] for idx in pred_list])}")
            else:
                print(f"输出单词: {idx_to_word[pred_list]}")
        else:
            pred_item = predicted.item()
            print(f"输出单词: {idx_to_word[pred_item]}")

if __name__ == "__main__":
    # 生成数据（如果不存在）
    if not os.path.exists(config.DEMO_DATA_JSON):
        print("生成演示数据...")
        from demo_data_simple import create_simple_data
        create_simple_data(config.DEMO_DATA_JSON)

    train_model()
