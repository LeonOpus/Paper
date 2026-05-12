#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
conda run -n "${BLIP_CONDA_ENV}" --no-capture-output \
    python "${BLIP_CODE_ROOT}/eval_retrieval.py" \
    --config "${BLIP_PROJECT_ROOT}/configs/retrieval_coco.yaml" \
    "$@"
