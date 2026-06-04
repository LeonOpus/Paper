"""
RAG 查询核心模块：检索相关文档块 + 调用 Qwen 生成回答
"""

import torch
import chromadb
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForCausalLM

CHROMA_DIR = "/home/leon/Output/2020_RAG/chroma_db"
COLLECTION_NAME = "compliance_rules"
BGE_MODEL_PATH = "/home/leon/Model/ModelScope/models/BAAI/bge-m3"
LLM_PATH = "/home/leon/Model/ModelScope/models/Qwen/Qwen2.5-7B-Instruct"
TOP_K = 5


class RAGPipeline:
    def __init__(self, load_llm: bool = True):
        print("加载 BGE-M3 embedding 模型...")
        self.embed_model = BGEM3FlagModel(BGE_MODEL_PATH, use_fp16=True)

        print("连接 ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

        self.llm = None
        self.tokenizer = None
        if load_llm:
            print("加载 Qwen2.5-7B-Instruct...")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
            self.llm = AutoModelForCausalLM.from_pretrained(
                LLM_PATH, torch_dtype=torch.float16, device_map="cuda"
            )
        print("RAG 流水线就绪")

    def retrieve(self, question: str) -> list[str]:
        q_emb = self.embed_model.encode([question])["dense_vecs"]
        results = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=TOP_K,
        )
        return results["documents"][0]

    def generate(self, question: str, context_chunks: list[str],
                 model=None, tokenizer=None) -> str:
        """用指定模型生成回答；未指定时用 self.llm"""
        llm = model if model is not None else self.llm
        tok = tokenizer if tokenizer is not None else self.tokenizer
        assert llm is not None, "未提供生成模型"

        context = "\n\n---\n\n".join(context_chunks)
        system_prompt = (
            "你是一名专业的证券合规审核助手。"
            "请严格根据以下参考资料回答问题，不要编造资料中未提及的内容。"
            "如果参考资料中没有相关信息，请明确说明。"
        )
        user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            output = llm.generate(
                **inputs, max_new_tokens=400, temperature=0.3, do_sample=True
            )
        return tok.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

    def ask(self, question: str, model=None, tokenizer=None) -> tuple[str, list[str]]:
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks, model=model, tokenizer=tokenizer)
        return answer, chunks
