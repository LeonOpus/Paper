"""
BERT 微调 CoNLL-2003 命名实体识别
复现论文: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
对应论文 Table 1 - NER 任务
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config

import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
from seqeval.metrics import f1_score, classification_report

# ── 超参数（来自论文 Appendix A.3）──────────────────────────────────────────
EPOCHS = float(os.environ.get("EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))  # 线性缩放：原32 → 256（×8），LR 同比放大
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1.6e-4"))  # 2e-5 × 8
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "128"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SMOKE_MAX_SAMPLES = int(os.environ.get("SMOKE_MAX_SAMPLES", "0"))
OUTPUT_DIR = os.path.join(config.OUTPUT_BASE, "conll")
LOG_DIR = os.path.join(config.LOG_BASE, "conll")

# NER 标签（CoNLL-2003 的 IOB2 格式）
LABEL_LIST = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# ── 1. 加载数据集 ─────────────────────────────────────────────────────────
print("加载 CoNLL-2003 数据集...")
dataset = load_dataset("conll2003", trust_remote_code=True)
if SMOKE_MAX_SAMPLES > 0:
    for split in dataset.keys():
        keep = min(SMOKE_MAX_SAMPLES, len(dataset[split]))
        dataset[split] = dataset[split].select(range(keep))
    print(f"SMOKE 模式：每个 split 仅保留前 {SMOKE_MAX_SAMPLES} 条样本")
print(dataset)

# ── 2. 加载 Tokenizer（NER 需要 Fast Tokenizer 以对齐子词）────────────────
print(f"\n加载 Tokenizer: {config.MODEL_NAME}")
tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)


# ── 3. 数据预处理（子词对齐标签）─────────────────────────────────────────
def preprocess(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # 特殊 token（[CLS]/[SEP]）忽略
            elif word_id != prev_word_id:
                label_ids.append(label[word_id])  # 每个词的第一个子词
            else:
                label_ids.append(-100)  # 同一词的后续子词忽略
            prev_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


print("预处理数据...")
tokenized = dataset.map(
    preprocess, batched=True, remove_columns=dataset["train"].column_names
)
tokenized.set_format("torch")

# ── 4. 加载模型（Token 分类）─────────────────────────────────────────────
print(f"\n加载模型: {config.MODEL_NAME}")
model = BertForTokenClassification.from_pretrained(
    config.MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


# ── 5. 评估指标（seqeval F1）──────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(preds, labels):
        true_label_row, true_pred_row = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_label_row.append(ID2LABEL[l])
                true_pred_row.append(ID2LABEL[p])
        true_labels.append(true_label_row)
        true_preds.append(true_pred_row)
    return {"f1": f1_score(true_labels, true_preds)}


# ── 6. 训练参数 ───────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=256,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=100,
    dataloader_num_workers=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
    fp16=True,
    report_to="none",
)

# ── 7. 训练 ───────────────────────────────────────────────────────────────
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n开始训练...")
print(f"训练集大小: {len(tokenized['train'])}")
print(f"验证集大小: {len(tokenized['validation'])}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
print(f"论文目标 (BERT-Base CoNLL NER): F1 96.4%\n")

trainer.train()

# ── 8. 最终评估 ───────────────────────────────────────────────────────────
print("\n最终评估（测试集）...")
results = trainer.evaluate(tokenized["test"])
print(f"\n测试集 F1: {results['eval_f1']*100:.2f}%  (论文基准: 96.4%)")

trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
print(f"\n模型已保存到: {OUTPUT_DIR}/best_model")
