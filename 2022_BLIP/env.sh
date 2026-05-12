#!/usr/bin/env bash
set -euo pipefail

export BLIP_PROJECT_ROOT="${HOME}/GitHub/Paper/2022_BLIP"
export BLIP_CODE_ROOT="${BLIP_PROJECT_ROOT}/code"
export BLIP_DATA_ROOT="${HOME}/DataSet/HuggingFace/2022_BLIP"
export BLIP_OUTPUT_ROOT="${HOME}/Output/2022_BLIP"
export BLIP_CONDA_ENV="2022_BLIP"

export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-${HOME}/Model/HuggingFace}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HOME}/DataSet/HuggingFace"
export HF_ENDPOINT="https://hf-mirror.com"

export MODELSCOPE_CACHE="${HOME}/Model/ModelScope"
export MS_DATASETS_CACHE="${HOME}/DataSet/ModelScope"

# COCO images are shared with VLMO
export COCO_IMAGE_ROOT="${HOME}/DataSet/HuggingFace/2022_VLMO/data"
