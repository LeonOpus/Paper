#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2021_ALBEF/env.sh"
bash "/home/lnick/GitHub/Paper/2021_ALBEF/setup_paths.sh"
bash "/home/lnick/GitHub/Paper/2021_ALBEF/prepare_coco_quick.sh"

missing=0
for f in \
  "${ALBEF_DATA_ROOT}/annotations/coco_quick_train.json" \
  "${ALBEF_DATA_ROOT}/annotations/coco_quick_val.json" \
  "${ALBEF_DATA_ROOT}/annotations/coco_quick_test.json" \
  "${ALBEF_DATA_ROOT}/checkpoints/ALBEF.pth"; do
  if [[ ! -f "${f}" ]]; then
    echo "缺失文件: ${f}"
    missing=1
  fi
done

if [[ ! -d "${ALBEF_DATA_ROOT}/images/coco/val2014" ]]; then
  echo "缺失目录: ${ALBEF_DATA_ROOT}/images/coco/val2014"
  missing=1
fi

if [[ "${missing}" -ne 0 ]]; then
  echo "请先完成 quick 数据准备。"
  exit 1
fi

cd "${ALBEF_CODE_ROOT}"

conda run -n "${ALBEF_CONDA_ENV}" env PYTHONNOUSERSITE=1 python Retrieval.py \
  --config ./configs_local/Retrieval_coco_quick_local.yaml \
  --output_dir "${ALBEF_OUTPUT_ROOT}/runs/retrieval_coco_quick" \
  --checkpoint "${ALBEF_DATA_ROOT}/checkpoints/ALBEF.pth"
