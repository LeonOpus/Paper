#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

conda activate "${VLMO_CONDA_ENV}"

python "${VLMO_CODE_ROOT}/Retrieval.py" \
    --config  "${VLMO_CODE_ROOT}/configs/Retrieval_flickr.yaml" \
    --output_dir "${VLMO_OUTPUT_ROOT}/retrieval_flickr" \
    "$@"
