#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PAPER_ID="$(basename "${SCRIPT_DIR}")"

: "${PAPER_ROOT:=${HOME}/GitHub/Paper/${PAPER_ID}}"
: "${PAPER_DATA:=${HOME}/DataSet/HuggingFace/${PAPER_ID}}"
export PAPER_ROOT PAPER_DATA

export HF_HOME="${HF_HOME:-${HOME}/Model/HuggingFace}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/DataSet/HuggingFace}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export OUTPUT_ROOT="${HOME}/Output/${PAPER_ID}"
export OUTPUT_DIR="${OUTPUT_ROOT}/output"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" \
         "${TRANSFORMERS_CACHE}" "${PAPER_DATA}/data" "${OUTPUT_ROOT}/logs" "${OUTPUT_DIR}"

if [[ "${PAPER_AUTO_ACTIVATE_CONDA:-1}" == "1" ]] && [[ $- == *i* ]]; then
  if command -v conda >/dev/null 2>&1; then
    if conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "${PAPER_ID}"; then
      if [[ "${CONDA_DEFAULT_ENV:-}" != "${PAPER_ID}" ]]; then
        conda activate "${PAPER_ID}" >/dev/null 2>&1 || conda activate "${PAPER_ID}"
      fi
    fi
  fi
fi
