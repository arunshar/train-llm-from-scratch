#!/usr/bin/env bash
set -euo pipefail

ROOT=/ephemeral
PROJECT_ROOT=/msi
PY="${PROJECT_ROOT}/envs/pytorch-2.14.0/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "missing staged Python environment: $PY" >&2
  exit 1
fi

cd /workspace
export PYTHONPATH=/workspace
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TIKTOKEN_CACHE_DIR="${PROJECT_ROOT}/tiktoken_cache"

mkdir -p "${ROOT}/data/pile_raw" "${ROOT}/ckpts" "${ROOT}/logs"

if [[ "$(cat "${ROOT}/DATA_STATUS" 2>/dev/null)" == "READY" && -f "${ROOT}/data/pile_train.h5" && -f "${ROOT}/data/pile_dev.h5" ]]; then
  echo "bounded Pile subset already prepared"
  cat "${ROOT}/data_manifest.json"
  exit 0
fi

"$PY" scripts/prepare_pretrain_data.py \
  --split val \
  --raw_dir "${ROOT}/data/pile_raw" \
  --out "${ROOT}/data/pile_dev.h5"

"$PY" scripts/prepare_pretrain_data.py \
  --split train \
  --num_shards 1 \
  --raw_dir "${ROOT}/data/pile_raw" \
  --out "${ROOT}/data/pile_train.h5"

"$PY" hpc/make_400m_data_manifest.py --root "$ROOT"
printf 'READY\n' > "${ROOT}/DATA_STATUS"
cat "${ROOT}/data_manifest.json"
