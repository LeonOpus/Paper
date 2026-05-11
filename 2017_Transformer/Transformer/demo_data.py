#!/usr/bin/env python
"""
创建演示数据
"""

import Constants as Constants
import pickle

# 定义全局类
class Settings:
    pass

class Vocab:
    def __init__(self, vocab_dict):
        self.stoi = vocab_dict
        self.itos = {v: k for k, v in vocab_dict.items()}
        self._vocab_dict = vocab_dict

    def __len__(self):
        return len(self._vocab_dict)

class Field:
    def __init__(self):
        self.vocab = None

    def build_vocab(self, words):
        vocab_dict = {}
        for word in words:
            if word not in vocab_dict:
                vocab_dict[word] = len(vocab_dict)
        self.vocab = Vocab(vocab_dict)

def create_demo_data():
    """创建简单的演示数据"""

    # 简单词汇表
    src_words = [Constants.PAD_WORD, Constants.UNK_WORD, 'I', 'you', 'he', 'she', 'we', 'they', 'love', 'hate', 'like', 'see', 'go', 'come', 'home', 'school', 'eat', 'drink', 'sleep', 'work']
    trg_words = [Constants.PAD_WORD, Constants.UNK_WORD, 'ich', 'du', 'er', 'sie', 'wir', 'sie', 'liebe', 'hasse', 'mag', 'sehe', 'gehe', 'kommen', 'nach', 'haus', 'schule', 'esse', 'trinke', 'schlafe', 'arbeite']

    # 创建字段和词汇表
    src_field = Field()
    trg_field = Field()

    src_field.build_vocab(src_words)
    trg_field.build_vocab(trg_words)

    # 创建训练数据
    train_examples = [
        {'src': ['I', 'love', 'you'], 'trg': ['ich', 'liebe', 'du']},
        {'src': ['You', 'hate', 'me'], 'trg': ['du', 'hasst', 'mich']},
        {'src': ['He', 'goes', 'home'], 'trg': ['er', 'geht', 'nach', 'haus']},
        {'src': ['We', 'eat', 'food'], 'trg': ['wir', 'essen', 'essen']},
        {'src': ['They', 'work', 'hard'], 'trg': ['sie', 'arbeiten', 'hart']},
        {'src': ['She', 'likes', 'cats'], 'trg': ['sie', 'mag', 'katzen']},
        {'src': ['I', 'see', 'you'], 'trg': ['ich', 'sehe', 'dich']},
        {'src': ['You', 'come', 'here'], 'trg': ['du', 'kommst', 'her']},
    ]

    # 创建验证数据
    valid_examples = [
        {'src': ['We', 'love', 'music'], 'trg': ['wir', 'lieben', 'musik']},
        {'src': ['He', 'sleeps', 'well'], 'trg': ['er', 'schlaf', 'gut']},
    ]

    # 创建settings对象
    settings = Settings()
    settings.src_lang = "en"
    settings.trg_lang = "de"
    settings.share_vocab = False
    settings.max_len = 10
    settings.min_word_count = 1
    settings.keep_case = False
    settings.src_pad_idx = src_field.vocab.stoi[Constants.PAD_WORD]
    settings.trg_pad_idx = trg_field.vocab.stoi[Constants.PAD_WORD]
    settings.src_vocab_size = len(src_field.vocab)
    settings.trg_vocab_size = len(trg_field.vocab)

    # 载入时动态创建Example对象
    # 直接保存字典形式
    train_data = [{'src': ex['src'], 'trg': ex['trg']} for ex in train_examples]
    valid_data = [{'src': ex['src'], 'trg': ex['trg']} for ex in valid_examples]

    # 创建数据字典
    data = {
        "settings": settings,
        "vocab": {
            "src": src_field,
            "trg": trg_field
        },
        "train": train_data,
        "valid": valid_data,
        "test": []
    }

    # 保存数据
    with open('demo_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    print(f"创建演示数据成功:")
    print(f"- 训练样本: {len(train_examples)}")
    print(f"- 验证样本: {len(valid_examples)}")
    print(f"- 源语言词汇表大小: {settings.src_vocab_size}")
    print(f"- 目标语言词汇表大小: {settings.trg_vocab_size}")

if __name__ == "__main__":
    create_demo_data()