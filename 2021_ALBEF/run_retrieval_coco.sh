#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2021_ALBEF/env.sh"
bash "/home/lnick/GitHub/Paper/2021_ALBEF/setup_paths.sh"

missing=0
for f in \
  "${ALBEF_DATA_ROOT}/annotations/coco_train.json" \
  "${ALBEF_DATA_ROOT}/annotations/coco_val.json" \
  "${ALBEF_DATA_ROOT}/annotations/coco_test.json" \
  "${ALBEF_DATA_ROOT}/checkpoints/ALBEF.pth"; do
  if [[ ! -f "${f}" ]]; then
    echo "缺失文件: ${f}"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "请先执行: bash /home/lnick/GitHub/Paper/2021_ALBEF/download_albef_assets.sh"
  exit 1
fi

cd "${ALBEF_CODE_ROOT}"

conda run -n "${ALBEF_CONDA_ENV}" env PYTHONNOUSERSITE=1 python Retrieval.py \
  --config ./configs_local/Retrieval_coco_local.yaml \
  --output_dir "${ALBEF_OUTPUT_ROOT}/runs/retrieval_coco" \
  --checkpoint "${ALBEF_DATA_ROOT}/checkpoints/ALBEF.pth"
