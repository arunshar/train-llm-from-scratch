#!/usr/bin/env bash
#SBATCH --job-name=tllm-bootstrap
#SBATCH --account=shekhars
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${REMOTE_CODE_ROOT:?REMOTE_CODE_ROOT must name the staged clean checkout}"
: "${PROJECT_ROOT:?PROJECT_ROOT must name the persistent scratch project root}"

IMAGE_NAME=pytorch-2.14.0-cuda12.6-cudnn9-runtime.sif
IMAGE="${PROJECT_ROOT}/images/${IMAGE_NAME}"
ENV_ROOT="${PROJECT_ROOT}/envs/pytorch-2.14.0"
LOCK="${REMOTE_CODE_ROOT}/hpc/requirements.lock"

mkdir -p \
  "${PROJECT_ROOT}/apptainer_cache" \
  "${PROJECT_ROOT}/envs" \
  "${PROJECT_ROOT}/hf_cache" \
  "${PROJECT_ROOT}/images" \
  "${PROJECT_ROOT}/runs" \
  "${PROJECT_ROOT}/tiktoken_cache"

export APPTAINER_CACHEDIR="${PROJECT_ROOT}/apptainer_cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"

if [[ ! -f "$IMAGE" ]]; then
  temporary_image="${IMAGE}.partial.${SLURM_JOB_ID}"
  apptainer pull \
    "$temporary_image" \
    docker://pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
  mv "$temporary_image" "$IMAGE"
fi

image_digest=$(sha256sum "$IMAGE" | awk '{print $1}')
lock_digest=$(sha256sum "$LOCK" | awk '{print $1}')
printf 'image_sha256=%s\nlock_sha256=%s\n' "$image_digest" "$lock_digest"

if [[ -e "$ENV_ROOT" && ! -f "${ENV_ROOT}/requirements.sha256" ]]; then
  echo "removing incomplete environment: $ENV_ROOT"
  rm -rf "$ENV_ROOT"
fi

if [[ -e "$ENV_ROOT" ]]; then
  recorded_digest=$(cat "${ENV_ROOT}/requirements.sha256")
  if [[ "$recorded_digest" != "$lock_digest" ]]; then
    echo "existing environment was built from a different lock file" >&2
    exit 1
  fi
else
  # Official PyTorch runtime images omit ensurepip. Create the venv without
  # pip, then install through the image interpreter so packages land in
  # sys.prefix (the venv) while still seeing torch from system site-packages.
  apptainer exec \
    --bind "${PROJECT_ROOT}:/msi" \
    --bind "${REMOTE_CODE_ROOT}:/workspace:ro" \
    "$IMAGE" \
    bash -lc '
      set -euo pipefail
      python -m venv --without-pip --system-site-packages /msi/envs/pytorch-2.14.0
      /msi/envs/pytorch-2.14.0/bin/python -m pip install \
        --require-hashes \
        -r /workspace/hpc/requirements.lock
    '
  printf '%s\n' "$lock_digest" > "${ENV_ROOT}/requirements.sha256"
fi

apptainer exec \
  --bind "${PROJECT_ROOT}:/msi" \
  --bind "${REMOTE_CODE_ROOT}:/workspace:ro" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace
    /msi/envs/pytorch-2.14.0/bin/python -c "
import datasets
import h5py
import numpy
import tiktoken
import torch
print(\"python environment imports passed\")
print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)
"
    /msi/envs/pytorch-2.14.0/bin/python -m pip freeze > /msi/environment.freeze.txt
  '

sha256sum "$IMAGE" > "${PROJECT_ROOT}/container.sha256"
printf 'STAGING_PASS\n' > "${PROJECT_ROOT}/BOOTSTRAP_STATUS"
