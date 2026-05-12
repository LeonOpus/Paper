#!/usr/bin/env bash
set -euo pipefail

export VLMO_PROJECT_ROOT="${HOME}/GitHub/Paper/2022_VLMO"
export VLMO_CODE_ROOT="${VLMO_PROJECT_ROOT}/code"
export VLMO_DATA_ROOT="${HOME}/DataSet/HuggingFace/2022_VLMO"
export VLMO_OUTPUT_ROOT="${HOME}/Output/2022_VLMO"
export VLMO_CONDA_ENV="2022_VLMO"

export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-${HOME}/Model/HuggingFace}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HOME}/DataSet/HuggingFace"
export HF_ENDPOINT="https://hf-mirror.com"
