#!/usr/bin/env bash
# 创建 2020_RAG conda 环境并安装 RAG 依赖
set -euo pipefail

ENV_NAME="2020_RAG"

echo "=== 步骤 1: 创建 conda 环境 ${ENV_NAME} (Python 3.11) ==="
if conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "环境已存在，跳过创建"
else
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

echo ""
echo "=== 步骤 2: 安装 PyTorch (CUDA 12.8) ==="
conda run -n "${ENV_NAME}" pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "=== 步骤 3: 安装 RAG 核心依赖 ==="
conda run -n "${ENV_NAME}" pip install \
    transformers \
    sentence-transformers \
    chromadb \
    langchain \
    langchain-community \
    accelerate \
    FlagEmbedding

echo ""
echo "=== 步骤 4: 下载 BGE-M3 embedding 模型 ==="
echo "使用 modelscope 下载 BAAI/bge-m3..."
conda run -n "${ENV_NAME}" modelscope download --model BAAI/bge-m3 > /dev/null 2>&1 &
echo "后台下载中，PID: $!"
echo "可用 'wait' 等待完成，或直接继续（build_index.py 会等模型就绪）"

echo ""
echo "=== 安装完成 ==="
echo "1. 等 BGE-M3 下载完成后运行: conda run -n 2020_RAG python practice_rag/build_index.py"
echo "2. 然后运行: conda run -n 2020_RAG python practice_rag/compare.py"
