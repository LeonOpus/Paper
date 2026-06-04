"""
对比微调前后的模型回答
用法: conda run -n 2022_LoRA python practice_qlora/inference.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "/home/leon/Output/2022_LoRA/qwen25-7b/qlora/compliance_qa"

QUESTIONS = [
    "P0违规是什么意思？举个例子。",
    "上市公司购买资产公告中，交易概述章节必须包含哪些要素？",
    "交易对方为自然人时，公告必须披露哪些信息？",
]


def ask(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=300, temperature=0.7, do_sample=True
        )
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )


print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("加载模型 + LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, ADAPTER)

for q in QUESTIONS:
    print("\n" + "=" * 60)
    print(f"问题: {q}")
    print("-" * 60)
    print("[原始模型（adapter 禁用）]")
    with model.disable_adapter():
        print(ask(model, tokenizer, q))
    print("-" * 60)
    print("[微调后（adapter 启用）]")
    print(ask(model, tokenizer, q))
