#!/bin/bash
# 设置transformer环境的脚本

echo "设置Transformer环境..."

# 创建虚拟环境
python -m venv transformer_env
echo "虚拟环境创建完成"

# 激活虚拟环境
source transformer_env/bin/activate
echo "虚拟环境已激活"

# 升级pip
pip install --upgrade pip

# 安装必要依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm numpy

echo "环境设置完成！"
echo "使用方法："
echo "1. 运行 source setup_transformer_env.sh 激活环境"
echo "2. 然后运行 python train_gpu.py 使用GPU训练"