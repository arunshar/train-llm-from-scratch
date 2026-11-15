#!/usr/bin/env bash
set -euo pipefail

: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the source revision}"

ROOT=/ephemeral
PROJECT_ROOT=/msi
PY="${PROJECT_ROOT}/envs/pytorch-2.14.0/bin/python"
CKPT="${ROOT}/ckpts/base_pretrained.pt"

if [[ ! -x "$PY" ]]; then
  echo "missing staged Python environment: $PY" >&2
  exit 1
fi
if [[ "$(cat "${ROOT}/DATA_STATUS" 2>/dev/null)" != "READY" ]]; then
  echo "400M data workspace is not READY" >&2
  exit 1
fi

cd /workspace
export PYTHONPATH=/workspace
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TIKTOKEN_CACHE_DIR="${PROJECT_ROOT}/tiktoken_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '%s\n' "$EXPECTED_COMMIT" > "${ROOT}/source_commit.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader > "${ROOT}/gpu.txt"

resume_args=()
if [[ -f "$CKPT" ]]; then
  resume_args=(--resume "$CKPT")
  echo "resuming 400M pretrain from $CKPT"
fi

"$PY" scripts/pretrain_base.py \
  --config configs/pretrain.json \
  --device cuda \
  --amp_dtype bf16 \
  --use_wandb false \
  --save_every 200 \
  --log_dir "${ROOT}/logs" \
  --train_path "${ROOT}/data/pile_train.h5" \
  --dev_path "${ROOT}/data/pile_dev.h5" \
  --out_ckpt "$CKPT" \
  "${resume_args[@]}"

"$PY" hpc/verify_400m_pretrain.py \
  --ckpt "$CKPT" \
  --out "$ROOT" \
  --device cuda
