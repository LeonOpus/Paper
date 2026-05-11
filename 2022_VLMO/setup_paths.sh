#!/usr/bin/env bash
# Create symlinks from code/data/ → actual data directory so YAML configs
# can reference relative paths like "data/coco_train.json".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

CODE_DATA="${VLMO_CODE_ROOT}/data"
REAL_DATA="${VLMO_DATA_ROOT}/data"

mkdir -p "${REAL_DATA}"

if [[ -L "${CODE_DATA}" ]]; then
    echo "Symlink already exists: ${CODE_DATA}"
else
    ln -s "${REAL_DATA}" "${CODE_DATA}"
    echo "Created: ${CODE_DATA} -> ${REAL_DATA}"
fi
