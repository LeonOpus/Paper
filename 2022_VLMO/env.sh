#!/usr/bin/env bash
set -euo pipefail

export VLMO_PROJECT_ROOT="/home/leon/GitHub/Paper/2022_VLMO"
export VLMO_CODE_ROOT="${VLMO_PROJECT_ROOT}/code"
export VLMO_DATA_ROOT="/home/leon/DataSet/HuggingFace/2022_VLMO"
export VLMO_OUTPUT_ROOT="/home/leon/Output/2022_VLMO"
export VLMO_CONDA_ENV="2022_VLMO"

export PYTHONNOUSERSITE=1

export HF_HOME="/home/leon/Model/HuggingFace"
export HF_HUB_CACHE="/home/leon/Model/HuggingFace/hub"
export HF_DATASETS_CACHE="/home/leon/DataSet/HuggingFace"
export HF_ENDPOINT="https://hf-mirror.com"
