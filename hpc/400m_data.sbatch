#!/usr/bin/env bash
#SBATCH --job-name=tllm-400m-data
#SBATCH --account=shekhars
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

: "${REMOTE_CODE_ROOT:?REMOTE_CODE_ROOT must name the staged clean checkout}"
: "${PROJECT_ROOT:?PROJECT_ROOT must name the persistent scratch project root}"
: "${WORKSPACE:?WORKSPACE must name the persistent 400M workspace}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the source revision}"

IMAGE="${PROJECT_ROOT}/images/pytorch-2.14.0-cuda12.6-cudnn9-runtime.sif"
ENV_ROOT="${PROJECT_ROOT}/envs/pytorch-2.14.0"

if [[ "$(cat "${PROJECT_ROOT}/BOOTSTRAP_STATUS" 2>/dev/null)" != "STAGING_PASS" ]]; then
  echo "bootstrap has not reached STAGING_PASS" >&2
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "staged image is missing: $IMAGE" >&2
  exit 1
fi
bash "${REMOTE_CODE_ROOT}/hpc/verify_image_pin.sh" --image "$IMAGE" >/dev/null
if [[ ! -f "${ENV_ROOT}/pyvenv.cfg" || ! -L "${ENV_ROOT}/bin/python" ]]; then
  echo "staged Python environment is missing: $ENV_ROOT" >&2
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

mkdir -p "$WORKSPACE"

# Do not use --cleanenv here. HuggingFace HTTPS needs host TLS state.
apptainer exec \
  --bind "${PROJECT_ROOT}:/msi" \
  --bind "${WORKSPACE}:/ephemeral" \
  --bind "${REMOTE_CODE_ROOT}:/workspace:ro" \
  --pwd /workspace \
  "$IMAGE" \
  bash /workspace/hpc/run_400m_data.sh
