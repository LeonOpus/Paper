#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2017_Transformer/env.sh"

cd "/home/lnick/GitHub/Paper/2017_Transformer/Transformer"

conda run -n "${PAPER_ID}" env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" python simple_train.py
