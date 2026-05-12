#!/usr/bin/env bash
# Download BLIP-format COCO Karpathy split annotations (val 5K + test 5K)
# Images are already available at $COCO_IMAGE_ROOT from the VLMO project
set -euo pipefail

source "$(dirname "$0")/env.sh"

DATA_DIR="${BLIP_DATA_ROOT}/data"
mkdir -p "${DATA_DIR}"

BASE_URL="https://storage.googleapis.com/sfr-vision-language-research/datasets"

for split in coco_karpathy_train coco_karpathy_val coco_karpathy_test; do
    dest="${DATA_DIR}/${split}.json"
    if [ -f "${dest}" ]; then
        echo "[skip] ${split}.json already exists"
    else
        echo "[download] ${split}.json"
        wget -q --show-progress -O "${dest}" "${BASE_URL}/${split}.json"
    fi
done

echo "Done. Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
