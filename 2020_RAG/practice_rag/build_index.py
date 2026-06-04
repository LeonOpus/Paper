"""
将合规规则文档切片、向量化，存入 ChromaDB
用法: conda run -n 2020_RAG python practice_rag/build_index.py
"""

import os
from pathlib import Path
import chromadb
from FlagEmbedding import BGEM3FlagModel

DOCS_DIR = Path(__file__).parent / "documents"
CHROMA_DIR = "/home/leon/Output/2020_RAG/chroma_db"
COLLECTION_NAME = "compliance_rules"

# BGE-M3 本地模型路径（modelscope 下载后的路径）
BGE_MODEL_PATH = "/home/leon/Model/ModelScope/models/BAAI/bge-m3"


def load_and_chunk(doc_path: Path, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """按段落切片，超长段落再按字符数切分"""
    text = doc_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 超长段落按 chunk_size 滑窗切分
            for i in range(0, len(para), chunk_size - overlap):
                chunk = para[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)

    return [
        {
            "text": chunk,
            "source": doc_path.name,
            "chunk_id": f"{doc_path.stem}_{i}",
        }
        for i, chunk in enumerate(chunks)
    ]


def main():
    print("加载文档...")
    all_chunks = []
    for doc_file in sorted(DOCS_DIR.glob("*.txt")):
        chunks = load_and_chunk(doc_file)
        print(f"  {doc_file.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"总计: {len(all_chunks)} chunks")

    print("\n加载 BGE-M3 embedding 模型...")
    model = BGEM3FlagModel(BGE_MODEL_PATH, use_fp16=True)

    print("计算 embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=32, max_length=512)["dense_vecs"]
    print(f"Embedding shape: {embeddings.shape}")

    print(f"\n写入 ChromaDB: {CHROMA_DIR}")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 重建 collection（幂等）
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    print(f"已写入 {collection.count()} 条向量")
    print("\n索引构建完成，可运行 compare.py 开始对比")


if __name__ == "__main__":
    main()
