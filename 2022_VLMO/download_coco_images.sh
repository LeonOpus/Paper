#!/usr/bin/env bash
# Download COCO 2014 images (train ~13GB, val ~6GB).
# Runs in background; safe to interrupt and resume with -c flag.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

IMG_DIR="${VLMO_DATA_ROOT}/data/coco/images"
mkdir -p "${IMG_DIR}"

WGET_OPTS=(--tries=10 --timeout=30 --retry-connrefused -c)

echo "Downloading COCO train2014 (~13GB) ..."
wget "${WGET_OPTS[@]}" \
    "http://images.cocodataset.org/zips/train2014.zip" \
    -O "${IMG_DIR}/train2014.zip"

echo "Downloading COCO val2014 (~6GB) ..."
wget "${WGET_OPTS[@]}" \
    "http://images.cocodataset.org/zips/val2014.zip" \
    -O "${IMG_DIR}/val2014.zip"

echo "Extracting ..."
unzip -q -n "${IMG_DIR}/train2014.zip" -d "${IMG_DIR}/"
unzip -q -n "${IMG_DIR}/val2014.zip"   -d "${IMG_DIR}/"

echo "COCO images ready at ${IMG_DIR}"
