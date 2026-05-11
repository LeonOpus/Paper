"""
BERT 微调 MNLI 自然语言推理
复现论文: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
对应论文 Table 1 - GLUE MNLI 任务
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import config

import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ── 超参数（来自论文 Appendix A.3）──────────────────────────────────────────
EPOCHS = float(os.environ.get("EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "128"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SMOKE_MAX_SAMPLES = int(os.environ.get("SMOKE_MAX_SAMPLES", "0"))
OUTPUT_DIR = os.path.join(config.OUTPUT_BASE, "mnli")
LOG_DIR = os.path.join(config.LOG_BASE, "mnli")

# ── 1. 加载数据集 ─────────────────────────────────────────────────────────
print("加载 MNLI 数据集...")
dataset = load_dataset("glue", "mnli")
if SMOKE_MAX_SAMPLES > 0:
    for split in dataset.keys():
        keep = min(SMOKE_MAX_SAMPLES, len(dataset[split]))
        dataset[split] = dataset[split].select(range(keep))
    print(f"SMOKE 模式：每个 split 仅保留前 {SMOKE_MAX_SAMPLES} 条样本")
print(dataset)

# ── 2. 加载 Tokenizer ────────────────────────────────────────────────────
print(f"\n加载 Tokenizer: {config.MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)

# ── 3. 数据预处理（句子对任务）────────────────────────────────────────────
def preprocess(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

print("预处理数据...")
tokenized = dataset.map(preprocess, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized = tokenized.remove_columns(["premise", "hypothesis", "idx"])
tokenized.set_format("torch")

# ── 4. 加载模型（3分类：entailment / neutral / contradiction）────────────
print(f"\n加载模型: {config.MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(
    config.MODEL_NAME,
    num_labels=3,
)

# ── 5. 评估指标 ───────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ── 6. 训练参数 ───────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=1000,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=100,
    max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
    fp16=True,
    report_to="none",
)

# ── 7. 训练 ───────────────────────────────────────────────────────────────
# MNLI 有两个验证集：matched（同领域）和 mismatched（跨领域）
# 论文报告 matched 准确率
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation_matched"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n开始训练...")
print(f"训练集大小: {len(tokenized['train'])}")
print(f"验证集(matched)大小: {len(tokenized['validation_matched'])}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
print(f"论文目标准确率 (BERT-Base MNLI-m): 84.6%\n")

trainer.train()

# ── 8. 最终评估（matched + mismatched）──────────────────────────────────
print("\n最终评估...")
results_m = trainer.evaluate(tokenized["validation_matched"])
results_mm = trainer.evaluate(tokenized["validation_mismatched"])
print(f"\nMatched   准确率: {results_m['eval_accuracy']*100:.2f}%  (论文基准: 84.6%)")
print(f"Mismatched准确率: {results_mm['eval_accuracy']*100:.2f}%  (论文基准: 83.4%)")

trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
print(f"\n模型已保存到: {OUTPUT_DIR}/best_model")
