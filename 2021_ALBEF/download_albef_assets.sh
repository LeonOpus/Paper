#!/usr/bin/env bash
set -euo pipefail

source "/home/lnick/GitHub/Paper/2021_ALBEF/env.sh"

mkdir -p "${ALBEF_DATA_ROOT}/annotations" "${ALBEF_DATA_ROOT}/checkpoints"

# Optional mirror override:
#   ALBEF_DATA_URL
#   ALBEF_PRETRAIN_JSON_URL
#   ALBEF_CKPT_URL
DATA_URL="${ALBEF_DATA_URL:-https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz}"
PRETRAIN_JSON_URL="${ALBEF_PRETRAIN_JSON_URL:-https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip}"
CKPT_URL="${ALBEF_CKPT_URL:-https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth}"

# Proxy mode:
#   inherit (default): use current shell proxy env.
#   direct: disable proxy for this script.
PROXY_MODE="${ALBEF_PROXY_MODE:-inherit}"
WGET_COMMON_OPTS=(--tries=30 --timeout=30 --read-timeout=30 --waitretry=3 --retry-connrefused -c)
if [[ "${PROXY_MODE}" == "direct" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset http_proxy https_proxy all_proxy no_proxy
  WGET_COMMON_OPTS+=(--no-proxy --no-check-certificate)
fi

cd "${ALBEF_DATA_ROOT}/annotations"

# Official annotation packages from ALBEF README.
wget "${WGET_COMMON_OPTS[@]}" \
  -O data.tar.gz \
  "${DATA_URL}"

wget "${WGET_COMMON_OPTS[@]}" \
  -O json_pretrain.zip \
  "${PRETRAIN_JSON_URL}"

tar -xzf data.tar.gz
unzip -o json_pretrain.zip

cd "${ALBEF_DATA_ROOT}/checkpoints"
wget "${WGET_COMMON_OPTS[@]}" \
  -O ALBEF.pth \
  "${CKPT_URL}"

echo "ALBEF annotation json and checkpoint downloaded."
