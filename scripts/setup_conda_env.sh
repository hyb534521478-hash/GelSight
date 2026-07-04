#!/usr/bin/env bash
set -e
ENV_NAME=${1:-gelsight_force}
conda create -n "$ENV_NAME" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# GPU版PyTorch建议优先按本机CUDA版本安装；下面cu121适配多数CUDA 12环境。
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
echo "环境创建完成: conda activate $ENV_NAME"
