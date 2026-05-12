#!/usr/bin/env bash
set -euo pipefail

export ALBEF_PROJECT_ROOT="${HOME}/GitHub/Paper/2021_ALBEF"
export ALBEF_CODE_ROOT="${ALBEF_PROJECT_ROOT}/code"
export ALBEF_DATA_ROOT="${HOME}/DataSet/HuggingFace/2021_ALBEF"
export ALBEF_OUTPUT_ROOT="${HOME}/Output/2021_ALBEF"
export ALBEF_CONDA_ENV="2021_ALBEF"

export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-${HOME}/Model/HuggingFace}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HOME}/DataSet/HuggingFace"
export HF_ENDPOINT="https://hf-mirror.com"
