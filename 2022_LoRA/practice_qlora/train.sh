#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "${SCRIPT_DIR}")"
LLAMAFACTORY_DIR="${PAPER_DIR}/LLaMA-Factory"
CONFIG="${SCRIPT_DIR}/configs/qwen25_1.5b_qlora_sft.yaml"

# 加载环境变量
source "${PAPER_DIR}/env.sh"

# 检查 conda 环境
ENV_NAME="2022_LoRA"
if ! conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "错误: conda 环境 '${ENV_NAME}' 不存在，请先运行 setup_env.sh"
    exit 1
fi

# 进入 LLaMA-Factory 目录运行（LlamaFactory 要求从其根目录执行）
cd "${LLAMAFACTORY_DIR}"

echo "=========================================="
echo "  QLoRA 微调: Qwen2.5-1.5B-Instruct"
echo "  数据集: compliance_qa (25 samples)"
echo "  配置: ${CONFIG}"
echo "=========================================="

conda run -n "${ENV_NAME}" --no-capture-output \
    python -m llamafactory.train "${CONFIG}"
