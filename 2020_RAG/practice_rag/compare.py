"""
四路对比：基础模型 / RAG+基础模型 / 微调模型 / RAG+微调模型
用法: conda run -n 2020_RAG python practice_rag/compare.py

需要先运行 build_index.py 构建向量索引
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag_query import RAGPipeline

LLM_PATH = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/home/leon/Output/2022_LoRA/qwen25-7b/qlora/compliance_qa"

QUESTIONS = [
    "P0违规是什么意思？举个例子。",
    "上市公司购买资产公告中，交易概述章节必须包含哪些要素？",
    "交易对方为自然人时，公告必须披露哪些信息？",
]


def ask_direct(model, tokenizer, question: str) -> str:
    """直接问模型，不带检索上下文"""
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
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )


def main():
    # 初始化 RAG（不加载内置 LLM，由我们自己管理模型）
    print("=" * 70)
    print("初始化 RAG 检索模块（BGE-M3 + ChromaDB）...")
    rag = RAGPipeline(load_llm=False)

    print("\n加载 Qwen2.5-7B-Instruct 基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, dtype=torch.float16, device_map="cuda"
    )

    print("加载 LoRA 微调模型...")
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("全部就绪\n")

    for q in QUESTIONS:
        print("=" * 70)
        print(f"问题: {q}\n")

        # [1] 基础模型，无检索
        print("[1] 基础模型（无检索）")
        print("-" * 40)
        with finetuned_model.disable_adapter():
            print(ask_direct(finetuned_model, tokenizer, q))

        # [2] RAG + 基础模型
        print("\n[2] RAG + 基础模型")
        print("-" * 40)
        with finetuned_model.disable_adapter():
            answer, _ = rag.ask(q, model=finetuned_model, tokenizer=tokenizer)
        print(answer)

        # [3] 微调模型，无检索
        print("\n[3] 微调模型（无检索）")
        print("-" * 40)
        print(ask_direct(finetuned_model, tokenizer, q))

        # [4] RAG + 微调模型
        print("\n[4] RAG + 微调模型（最强组合）")
        print("-" * 40)
        answer, _ = rag.ask(q, model=finetuned_model, tokenizer=tokenizer)
        print(answer)

        print()

    print("=" * 70)
    print("对比完成")


if __name__ == "__main__":
    main()
