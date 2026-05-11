#!/usr/bin/env bash
# Download and prepare annotation JSONs for COCO retrieval and Flickr30K.
# Images need to be downloaded separately (see README_repro.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DATA_DIR="${VLMO_DATA_ROOT}/data"
mkdir -p "${DATA_DIR}/coco/images" "${DATA_DIR}/flickr30k/images"

# ---- COCO annotations (same format as ALBEF) ----
# Source: https://storage.googleapis.com/sfr-vision-language-research/datasets/
BASE_URL="https://storage.googleapis.com/sfr-vision-language-research/datasets"

for split in coco_train coco_val coco_test flickr30k_train flickr30k_val flickr30k_test; do
    DEST="${DATA_DIR}/${split}.json"
    if [[ -f "${DEST}" ]]; then
        echo "Already exists: ${DEST}"
    else
        echo "Downloading ${split}.json ..."
        wget -q "${BASE_URL}/${split}.json" -O "${DEST}"
    fi
done

echo ""
echo "Annotations ready at: ${DATA_DIR}"
echo ""
echo "Next steps:"
echo "  1. Download COCO 2014 images → ${DATA_DIR}/coco/images/"
echo "     train2014/ val2014/ test2015/"
echo "  2. Download Flickr30K images → ${DATA_DIR}/flickr30k/images/"
echo "  3. Run: bash setup_paths.sh"
echo "  4. Run: bash run_retrieval_coco.sh"
