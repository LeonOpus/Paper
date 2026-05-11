#!/usr/bin/env bash
set -euo pipefail

export ALBEF_PROJECT_ROOT="/home/leon/GitHub/Paper/2021_ALBEF"
export ALBEF_CODE_ROOT="${ALBEF_PROJECT_ROOT}/code"
export ALBEF_DATA_ROOT="/home/leon/DataSet/HuggingFace/2021_ALBEF"
export ALBEF_OUTPUT_ROOT="/home/leon/Output/2021_ALBEF"
export ALBEF_CONDA_ENV="2021_ALBEF"

export PYTHONNOUSERSITE=1

export HF_HOME="/home/leon/Model/HuggingFace"
export HF_HUB_CACHE="/home/leon/Model/HuggingFace/hub"
export HF_DATASETS_CACHE="/home/leon/DataSet/HuggingFace"
export HF_ENDPOINT="https://hf-mirror.com"
