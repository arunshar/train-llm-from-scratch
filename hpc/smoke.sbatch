#!/usr/bin/env bash
#SBATCH --job-name=tllm-smoke
#SBATCH --account=shekhars
#SBATCH --partition=msigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00

set -euo pipefail

: "${REMOTE_CODE_ROOT:?REMOTE_CODE_ROOT must name the staged clean checkout}"
: "${PROJECT_ROOT:?PROJECT_ROOT must name the persistent scratch project root}"
: "${RUN_ROOT:?RUN_ROOT must name the fresh smoke run directory}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the source revision}"

IMAGE="${PROJECT_ROOT}/images/pytorch-2.14.0-cuda12.6-cudnn9-runtime.sif"
ENV_ROOT="${PROJECT_ROOT}/envs/pytorch-2.14.0"

if [[ "$(cat "${PROJECT_ROOT}/BOOTSTRAP_STATUS" 2>/dev/null)" != "STAGING_PASS" ]]; then
  echo "bootstrap has not reached STAGING_PASS" >&2
  exit 1
fi
if [[ ! -f "$IMAGE" || ! -x "${ENV_ROOT}/bin/python" ]]; then
  echo "staged image or Python environment is missing" >&2
  exit 1
fi
if [[ "$(git -C "$REMOTE_CODE_ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "remote checkout does not match EXPECTED_COMMIT" >&2
  exit 1
fi
if [[ -n "$(git -C "$REMOTE_CODE_ROOT" status --porcelain)" ]]; then
  echo "remote checkout is not clean" >&2
  exit 1
fi

apptainer exec \
  --nv \
  --cleanenv \
  --env "EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
  --bind "${PROJECT_ROOT}:/msi" \
  --bind "${RUN_ROOT}:/ephemeral" \
  --bind "${REMOTE_CODE_ROOT}:/workspace:ro" \
  --pwd /workspace \
  "$IMAGE" \
  bash /workspace/hpc/run_smoke.sh
