#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PAPER_ID="$(basename "${SCRIPT_DIR}")"

export HF_HOME="${HF_HOME:-${HOME}/Model/HuggingFace}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MODELSCOPE_CACHE="${HOME}/Model/ModelScope"
export PYTHONNOUSERSITE=1

export OUTPUT_DIR="${HOME}/Output/${PAPER_ID}/output"
export LOG_DIR="${HOME}/Output/${PAPER_ID}/logs"

mkdir -p "${HF_HUB_CACHE}" "${OUTPUT_DIR}" "${LOG_DIR}" \
         "${HOME}/DataSet/HuggingFace/${PAPER_ID}/data"

if [[ $- == *i* ]] && command -v conda &>/dev/null; then
    conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "${PAPER_ID}" && \
    [[ "${CONDA_DEFAULT_ENV:-}" != "${PAPER_ID}" ]] && conda activate "${PAPER_ID}" 2>/dev/null || true
fi
