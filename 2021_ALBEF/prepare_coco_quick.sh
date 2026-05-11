#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2021_ALBEF/env.sh"

COCO_ROOT="${ALBEF_DATA_ROOT}/images/coco"
ANN_ROOT="${ALBEF_DATA_ROOT}/annotations"
mkdir -p "${COCO_ROOT}" "${ANN_ROOT}"

ZIP_URL="${COCO_VAL2014_URL:-http://images.cocodataset.org/zips/val2014.zip}"
ZIP_PATH="${COCO_ROOT}/val2014.zip"

echo "[1/3] 下载 COCO val2014（断点续传）"
if [[ -f "${ZIP_PATH}" ]] && ! unzip -tq "${ZIP_PATH}" >/dev/null 2>&1; then
  echo "检测到损坏压缩包，删除后重下: ${ZIP_PATH}"
  rm -f "${ZIP_PATH}"
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  curl -fL --retry 30 --retry-all-errors --retry-delay 2 -C - -o "${ZIP_PATH}" "${ZIP_URL}"
else
  echo "已存在: ${ZIP_PATH}"
fi

echo "[2/3] 解压 val2014 图片"
cd "${COCO_ROOT}"
unzip -tq "${ZIP_PATH}" >/dev/null
unzip -n "${ZIP_PATH}" >/dev/null

echo "[3/3] 生成 quick 版标注（仅使用 val2014 图片）"
python - <<'PY'
import json
import os
import random

ann_root = "/home/lnick/DataSet/2021_ALBEF/annotations"

train_n = int(os.environ.get("COCO_QUICK_TRAIN_CAPTIONS", "12000"))
val_n = int(os.environ.get("COCO_QUICK_VAL_CAPTIONS", "2000"))
test_n = int(os.environ.get("COCO_QUICK_TEST_CAPTIONS", "2000"))
seed = int(os.environ.get("COCO_QUICK_SEED", "42"))
rng = random.Random(seed)

def load(name):
    with open(os.path.join(ann_root, name), "r") as f:
        return json.load(f)

def save(name, data):
    with open(os.path.join(ann_root, name), "w") as f:
        json.dump(data, f)

train = [x for x in load("coco_train.json") if x["image"].startswith("val2014/")]
val = [x for x in load("coco_val.json") if x["image"].startswith("val2014/")]
test = [x for x in load("coco_test.json") if x["image"].startswith("val2014/")]

def pick(data, n):
    if n >= len(data):
        return data
    idx = list(range(len(data)))
    rng.shuffle(idx)
    idx = idx[:n]
    return [data[i] for i in idx]

train_q = pick(train, train_n)
val_q = pick(val, val_n)
test_q = pick(test, test_n)

save("coco_quick_train.json", train_q)
save("coco_quick_val.json", val_q)
save("coco_quick_test.json", test_q)

print("coco_quick_train.json:", len(train_q))
print("coco_quick_val.json:", len(val_q))
print("coco_quick_test.json:", len(test_q))
PY

echo "准备完成：可以运行 run_retrieval_coco_quick.sh"
