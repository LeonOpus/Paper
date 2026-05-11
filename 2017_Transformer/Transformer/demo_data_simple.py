#!/usr/bin/env python
"""
创建简单的JSON格式演示数据
"""

import json
import os
import Constants as Constants

def create_simple_data(output_path='demo_data_simple.json'):
    """创建简单的JSON格式演示数据"""

    # 简单词汇表
    src_words = [Constants.PAD_WORD, Constants.UNK_WORD, 'I', 'you', 'he', 'she', 'we', 'they', 'love', 'hate']
    trg_words = [Constants.PAD_WORD, Constants.UNK_WORD, 'ich', 'du', 'er', 'sie', 'wir', 'sie', 'liebe', 'hasse']

    # 创建词汇表映射
    src_vocab_to_idx = {word: idx for idx, word in enumerate(src_words)}
    trg_vocab_to_idx = {word: idx for idx, word in enumerate(trg_words)}

    # 训练数据（已经是数字索引）
    train_data = [
        {'src': [2, 8, 3], 'trg': [2, 8, 3]},      # I love you -> ich liebe du
        {'src': [3, 9, 2], 'trg': [3, 9, 2]},      # you hate me -> du hasst mich
        {'src': [4, 6, 8], 'trg': [4, 6, 8]},      # he we love -> er wir lieben
        {'src': [5, 8, 7], 'trg': [5, 8, 7]},      # she love they -> sie liebe sie
        {'src': [6, 8, 4], 'trg': [6, 8, 4]},      # we love he -> wir lieben er
    ]

    # 验证数据
    valid_data = [
        {'src': [7, 8, 3], 'trg': [7, 8, 3]},      # they love you -> sie lieben dich
        {'src': [2, 9, 5], 'trg': [2, 9, 5]},      # I hate she -> ich hasse sie
    ]

    # 设置
    settings = {
        "src_lang": "en",
        "trg_lang": "de",
        "src_pad_idx": 0,
        "trg_pad_idx": 0,
        "src_vocab_size": len(src_words),
        "trg_vocab_size": len(trg_words),
        "max_len": 50
    }

    # 创建数据字典
    data = {
        "settings": settings,
        "src_vocab": src_words,
        "trg_vocab": trg_words,
        "train": train_data,
        "valid": valid_data,
    }

    # 保存为JSON
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"创建简单演示数据成功:")
    print(f"- 训练样本: {len(train_data)}")
    print(f"- 验证样本: {len(valid_data)}")
    print(f"- 源语言词汇表大小: {settings['src_vocab_size']}")
    print(f"- 目标语言词汇表大小: {settings['trg_vocab_size']}")
    print(f"- 数据文件: {output_path}")

if __name__ == "__main__":
    create_simple_data()
