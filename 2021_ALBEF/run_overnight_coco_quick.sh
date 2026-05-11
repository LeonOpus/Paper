#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2021_ALBEF/env.sh"
LOG_DIR="${ALBEF_OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${ALBEF_OUTPUT_ROOT}/runs/retrieval_coco_quick"

echo "[$(date '+%F %T')] overnight quick run started"

# 为了确保早上能看到结果，使用更小子集。
export COCO_QUICK_TRAIN_CAPTIONS="${COCO_QUICK_TRAIN_CAPTIONS:-4000}"
export COCO_QUICK_VAL_CAPTIONS="${COCO_QUICK_VAL_CAPTIONS:-500}"
export COCO_QUICK_TEST_CAPTIONS="${COCO_QUICK_TEST_CAPTIONS:-500}"

prepare_ok=0
for i in $(seq 1 50); do
  echo "[$(date '+%F %T')] prepare attempt ${i}"
  if bash "/home/lnick/GitHub/Paper/2021_ALBEF/prepare_coco_quick.sh"; then
    prepare_ok=1
    break
  fi
  echo "[$(date '+%F %T')] prepare failed, sleep 60s then retry"
  sleep 60
done

if [[ "${prepare_ok}" -ne 1 ]]; then
  echo "[$(date '+%F %T')] prepare failed too many times"
  exit 1
fi

cd "${ALBEF_CODE_ROOT}"
train_ok=0
for i in 1 2 3; do
  echo "[$(date '+%F %T')] train attempt ${i}"
  if conda run -n "${ALBEF_CONDA_ENV}" env PYTHONNOUSERSITE=1 python Retrieval.py \
      --config ./configs_local/Retrieval_coco_quick_local.yaml \
      --output_dir "${ALBEF_OUTPUT_ROOT}/runs/retrieval_coco_quick" \
      --checkpoint "${ALBEF_DATA_ROOT}/checkpoints/ALBEF.pth"; then
    train_ok=1
    break
  fi
  echo "[$(date '+%F %T')] train attempt ${i} failed, sleep 60s then retry"
  sleep 60
done

if [[ "${train_ok}" -eq 1 ]]; then
  echo "[$(date '+%F %T')] SUCCESS"
  touch "${ALBEF_OUTPUT_ROOT}/runs/retrieval_coco_quick/DONE_OK"
  exit 0
fi

echo "[$(date '+%F %T')] FAILED"
touch "${ALBEF_OUTPUT_ROOT}/runs/retrieval_coco_quick/DONE_FAIL"
exit 1
