#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2021_ALBEF/env.sh"

mkdir -p "${ALBEF_DATA_ROOT}/annotations" "${ALBEF_DATA_ROOT}/checkpoints"

DATA_URL="${ALBEF_DATA_URL:-https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz}"
PRETRAIN_JSON_URL="${ALBEF_PRETRAIN_JSON_URL:-https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip}"
CKPT_URL="${ALBEF_CKPT_URL:-https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth}"

# inherit: 使用当前 shell 代理；direct: 禁用代理；auto: 先 inherit 再 direct
PROXY_MODE="${ALBEF_PROXY_MODE:-auto}"

_set_proxy_mode() {
  local mode="$1"
  if [[ "${mode}" == "direct" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
    unset http_proxy https_proxy all_proxy no_proxy
  fi
}

_download_one() {
  local url="$1"
  local out="$2"
  local mode="$3"

  _set_proxy_mode "${mode}"
  echo "[download] mode=${mode} url=${url}"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 50 --retry-all-errors --retry-delay 2 -C - -o "${out}" "${url}"
    return 0
  fi

  wget --tries=50 --waitretry=2 --timeout=30 --read-timeout=30 --retry-connrefused -c -O "${out}" "${url}"
}

download_with_fallback() {
  local url="$1"
  local out="$2"

  if [[ "${PROXY_MODE}" == "inherit" || "${PROXY_MODE}" == "direct" ]]; then
    _download_one "${url}" "${out}" "${PROXY_MODE}"
    return
  fi

  # auto
  if ! _download_one "${url}" "${out}" "inherit"; then
    echo "[warn] inherit 模式下载失败，切到 direct 重试: ${out}"
    _download_one "${url}" "${out}" "direct"
  fi
}

check_min_size() {
  local file="$1"
  local min_bytes="$2"
  if [[ ! -f "${file}" ]]; then
    echo "[error] 缺失文件: ${file}" >&2
    return 1
  fi
  local size
  size="$(stat -c %s "${file}")"
  if (( size < min_bytes )); then
    echo "[error] 文件过小，疑似未下载完整: ${file} (${size} bytes)" >&2
    return 1
  fi
}

pushd "${ALBEF_DATA_ROOT}/annotations" >/dev/null
download_with_fallback "${DATA_URL}" "data.tar.gz"
download_with_fallback "${PRETRAIN_JSON_URL}" "json_pretrain.zip"

check_min_size "data.tar.gz" 100000000
check_min_size "json_pretrain.zip" 1000000

tar -tzf data.tar.gz | grep -q "flickr30k_train.json"
tar -xzf data.tar.gz
unzip -o json_pretrain.zip >/dev/null
popd >/dev/null

pushd "${ALBEF_DATA_ROOT}/checkpoints" >/dev/null
download_with_fallback "${CKPT_URL}" "ALBEF.pth"
check_min_size "ALBEF.pth" 10000000
popd >/dev/null

for f in \
  "${ALBEF_DATA_ROOT}/annotations/flickr30k_train.json" \
  "${ALBEF_DATA_ROOT}/annotations/flickr30k_val.json" \
  "${ALBEF_DATA_ROOT}/annotations/flickr30k_test.json" \
  "${ALBEF_DATA_ROOT}/annotations/coco_train.json" \
  "${ALBEF_DATA_ROOT}/annotations/coco_val.json" \
  "${ALBEF_DATA_ROOT}/annotations/coco_test.json" \
  "${ALBEF_DATA_ROOT}/checkpoints/ALBEF.pth"; do
  [[ -f "${f}" ]] || { echo "[error] 缺失关键文件: ${f}" >&2; exit 1; }
done

echo "ALBEF 资产下载并校验完成。"
