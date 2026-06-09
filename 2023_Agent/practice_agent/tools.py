"""
Agent 可调用的工具集
每个工具是一个普通 Python 函数，配有文字描述供模型理解
"""

import chromadb
from FlagEmbedding import BGEM3FlagModel

CHROMA_DIR = "/home/leon/Output/2020_RAG/chroma_db"
BGE_MODEL_PATH = "/home/leon/Model/ModelScope/models/BAAI/bge-m3"

_embed_model = None
_collection = None


def _init_rag():
    global _embed_model, _collection
    if _embed_model is None:
        print("  [工具初始化] 加载 BGE-M3 + ChromaDB...")
        _embed_model = BGEM3FlagModel(BGE_MODEL_PATH, use_fp16=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("compliance_rules")


def search_rules(query: str) -> str:
    """
    在深交所合规规则库中检索与查询最相关的条款。
    适用场景：查询某章节必须披露的要素、违规级别定义、关联交易认定标准等。
    """
    _init_rag()
    emb = _embed_model.encode([query])["dense_vecs"]
    results = _collection.query(query_embeddings=emb.tolist(), n_results=4)
    chunks = results["documents"][0]
    return "\n\n---\n\n".join(chunks)


# 工具注册表：工具名 → 函数
TOOL_REGISTRY = {
    "search_rules": search_rules,
}

# 供 system prompt 使用的工具说明
TOOLS_DESCRIPTION = """\
你可以使用以下工具：

  search_rules(query: str)
    描述：在合规规则数据库中检索相关条款，返回最匹配的规则原文
    示例：search_rules("交易对方为自然人时必须披露哪些信息")
         search_rules("P0违规的认定标准")
         search_rules("关联交易认定标准")

每次只调用一个工具，格式严格如下：

Thought: <当前分析和下一步计划>
Action: search_rules
Action Input: {"query": "检索内容"}

收到 Observation 后继续。所有检查完成后输出：

Thought: <汇总所有发现>
Final Answer:
## 合规审核报告

### 发现的违规项
[列出每项违规，注明 P0/P1/P2 级别和具体缺失内容]

### 审核结论
[总体评价]
"""
