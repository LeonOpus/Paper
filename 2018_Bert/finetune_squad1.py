"""
BERT 微调 SQuAD v1.1 阅读理解
复现论文: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
对应论文 Table 2 - SQuAD 1.1 任务
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import config

import numpy as np
import collections
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import load_dataset

# ── 超参数（来自论文 Appendix A.3）──────────────────────────────────────────
EPOCHS = float(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))          # 线性缩放：原32 → 128（×4），seq_len=384显存较大
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1.2e-4"))    # 3e-5 × 4
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "384"))
DOC_STRIDE = int(os.environ.get("DOC_STRIDE", "128"))       # 长文档滑动窗口步长
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SMOKE_MAX_SAMPLES = int(os.environ.get("SMOKE_MAX_SAMPLES", "0"))
OUTPUT_DIR = os.path.join(config.OUTPUT_BASE, "squad1")
LOG_DIR = os.path.join(config.LOG_BASE, "squad1")

# ── 1. 加载数据集 ─────────────────────────────────────────────────────────
print("加载 SQuAD v1.1 数据集...")
dataset = None
last_error = None
for ds_name in ("rajpurkar/squad", "squad"):
    try:
        dataset = load_dataset(ds_name)
        print(f"使用数据集源: {ds_name}")
        break
    except Exception as e:
        last_error = e
        print(f"加载失败({ds_name}): {type(e).__name__}: {e}")
if dataset is None:
    raise RuntimeError(f"SQuAD v1.1 加载失败，最后错误: {last_error}")
if SMOKE_MAX_SAMPLES > 0:
    for split in dataset.keys():
        keep = min(SMOKE_MAX_SAMPLES, len(dataset[split]))
        dataset[split] = dataset[split].select(range(keep))
    print(f"SMOKE 模式：每个 split 仅保留前 {SMOKE_MAX_SAMPLES} 条样本")
print(dataset)

# ── 2. 加载 Tokenizer ────────────────────────────────────────────────────
print(f"\n加载 Tokenizer: {config.MODEL_NAME}")
tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)

# ── 3. 训练集预处理 ───────────────────────────────────────────────────────
def preprocess_train(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_SEQ_LEN,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = tokenized.pop("offset_mapping")
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = tokenized.sequence_ids(i)

        # 找到 context 的 token 范围
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        ctx_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        ctx_end = idx - 1

        # 答案不在此 window 内，标为 [CLS]
        if offset[ctx_start][0] > end_char or offset[ctx_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = ctx_start
            while idx <= ctx_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = ctx_end
            while idx >= ctx_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

# ── 4. 验证集预处理 ───────────────────────────────────────────────────────
def preprocess_eval(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_SEQ_LEN,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    tokenized["example_id"] = example_ids
    return tokenized

print("预处理训练集...")
train_dataset = dataset["train"].map(
    preprocess_train, batched=True,
    remove_columns=dataset["train"].column_names
)
print("预处理验证集...")
eval_dataset = dataset["validation"].map(
    preprocess_eval, batched=True,
    remove_columns=dataset["validation"].column_names
)

# ── 5. 加载模型 ───────────────────────────────────────────────────────────
print(f"\n加载模型: {config.MODEL_NAME}")
model = BertForQuestionAnswering.from_pretrained(config.MODEL_NAME)

# ── 6. 后处理：从 logits 还原答案文本 ────────────────────────────────────
def postprocess_predictions(examples, features, raw_predictions, n_best=20, max_answer_len=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        valid_answers = []
        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offset_mapping = features[fi]["offset_mapping"]
            start_indexes = np.argsort(start_logits)[-n_best-1:][::-1].tolist()
            end_indexes = np.argsort(end_logits)[-n_best-1:][::-1].tolist()
            for si in start_indexes:
                for ei in end_indexes:
                    if offset_mapping[si] is None or offset_mapping[ei] is None:
                        continue
                    if ei < si or ei - si + 1 > max_answer_len:
                        continue
                    valid_answers.append({
                        "score": start_logits[si] + end_logits[ei],
                        "text": example["context"][offset_mapping[si][0]:offset_mapping[ei][1]],
                    })
        predictions[example["id"]] = max(valid_answers, key=lambda x: x["score"])["text"] if valid_answers else ""
    return predictions

# ── 7. 评估指标（EM 和 F1）───────────────────────────────────────────────
def normalize_answer(s):
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())

def compute_squad_metrics(predictions, references):
    em_scores, f1_scores = [], []
    for pred_id, pred_text in predictions.items():
        ref_answers = next(r["answers"]["text"] for r in references if r["id"] == pred_id)
        pred_norm = normalize_answer(pred_text)
        best_em, best_f1 = 0, 0
        for ref in ref_answers:
            ref_norm = normalize_answer(ref)
            best_em = max(best_em, int(pred_norm == ref_norm))
            pred_tokens = pred_norm.split()
            ref_tokens = ref_norm.split()
            common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
            num_same = sum(common.values())
            if num_same:
                p = num_same / len(pred_tokens)
                r = num_same / len(ref_tokens)
                best_f1 = max(best_f1, 2 * p * r / (p + r))
        em_scores.append(best_em)
        f1_scores.append(best_f1)
    return np.mean(em_scores) * 100, np.mean(f1_scores) * 100

# ── 8. 训练参数 ───────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=128,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=200,
    dataloader_num_workers=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
    fp16=True,
    report_to="none",
)

# ── 9. 训练 ───────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset.remove_columns(["example_id", "offset_mapping"]),
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

print("\n开始训练...")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(eval_dataset)}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
print(f"论文目标 (BERT-Base SQuAD 1.1): EM 80.8%, F1 88.5%\n")

trainer.train()

# ── 10. 最终评估 ──────────────────────────────────────────────────────────
print("\n最终评估...")
raw_predictions = trainer.predict(eval_dataset.remove_columns(["example_id", "offset_mapping"]))
predictions = postprocess_predictions(
    dataset["validation"], eval_dataset,
    (raw_predictions.predictions[0], raw_predictions.predictions[1])
)
em, f1 = compute_squad_metrics(predictions, dataset["validation"])
print(f"\nEM: {em:.2f}%  (论文基准: 80.8%)")
print(f"F1: {f1:.2f}%  (论文基准: 88.5%)")

trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
print(f"\n模型已保存到: {OUTPUT_DIR}/best_model")
