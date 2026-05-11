#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2021_ALBEF/env.sh"

mkdir -p "${ALBEF_DATA_ROOT}/annotations" \
         "${ALBEF_DATA_ROOT}/images" \
         "${ALBEF_DATA_ROOT}/cache" \
         "${ALBEF_DATA_ROOT}/checkpoints" \
         "${ALBEF_OUTPUT_ROOT}/logs" \
         "${ALBEF_OUTPUT_ROOT}/checkpoints" \
         "${ALBEF_OUTPUT_ROOT}/runs"

# ALBEF expects relative data/*.json paths in default configs.
ln -sfn "${ALBEF_DATA_ROOT}/annotations" "${ALBEF_CODE_ROOT}/data"

echo "ALBEF paths are ready."
echo "code:   ${ALBEF_CODE_ROOT}"
echo "data:   ${ALBEF_DATA_ROOT}"
echo "output: ${ALBEF_OUTPUT_ROOT}"

