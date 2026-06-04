#!/usr/bin/env bash
# 创建 2022_LoRA conda 环境并安装 LLaMA-Factory + QLoRA 依赖
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "${SCRIPT_DIR}")"
LLAMAFACTORY_DIR="${PAPER_DIR}/LLaMA-Factory"
ENV_NAME="2022_LoRA"

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
echo "=== 步骤 3: 安装 LLaMA-Factory ==="
cd "${LLAMAFACTORY_DIR}"
conda run -n "${ENV_NAME}" pip install -e ".[torch,metrics]"

echo ""
echo "=== 步骤 4: 安装 QLoRA 依赖 (bitsandbytes) ==="
conda run -n "${ENV_NAME}" pip install bitsandbytes>=0.43.0

echo ""
echo "=== 验证安装 ==="
conda run -n "${ENV_NAME}" python -c "
import torch
import bitsandbytes
from llamafactory.train import run_exp
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
print('LLaMA-Factory: OK')
"

echo ""
echo "=== 安装完成 ==="
echo "运行训练: bash practice_qlora/train.sh"
