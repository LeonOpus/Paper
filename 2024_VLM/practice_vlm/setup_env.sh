#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="2024_VLM"

echo "=== 创建 conda 环境 ${ENV_NAME} ==="
if conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "环境已存在，跳过"
else
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

echo "=== 安装 PyTorch (CUDA 12.8) ==="
conda run -n "${ENV_NAME}" pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 -q

echo "=== 安装 VLM 依赖 ==="
conda run -n "${ENV_NAME}" pip install \
    transformers accelerate \
    qwen-vl-utils \
    Pillow \
    -q

echo "=== 完成 ==="
echo "运行: conda run -n 2024_VLM python practice_vlm/generate_images.py"
echo "然后: conda run -n 2024_VLM python practice_vlm/inference.py"
