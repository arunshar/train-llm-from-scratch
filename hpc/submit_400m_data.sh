#!/usr/bin/env bash
set -euo pipefail

REMOTE_CODE_ROOT=$(git rev-parse --show-toplevel)
EXPECTED_COMMIT=$(git rev-parse HEAD)
PROJECT_ROOT=${PROJECT_ROOT:-"/scratch.global/${USER}/train-llm-from-scratch"}
WORKSPACE=${WORKSPACE:-"${PROJECT_ROOT}/workspaces/400m"}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to submit from a dirty remote checkout" >&2
  exit 1
fi
if [[ "$(cat "${PROJECT_ROOT}/BOOTSTRAP_STATUS" 2>/dev/null)" != "STAGING_PASS" ]]; then
  echo "bootstrap must be verified before the 400M data job" >&2
  exit 1
fi
bash "${REMOTE_CODE_ROOT}/hpc/verify_image_pin.sh" \
  --sidecar "${PROJECT_ROOT}/container.sha256" >/dev/null
if (( $(squeue -u "$USER" -h | wc -l) >= 1 )); then
  echo "project concurrency ceiling is one active job" >&2
  exit 1
fi

mkdir -p "${WORKSPACE}/data-logs"

job_id=$(sbatch \
  --parsable \
  --output="${WORKSPACE}/data-logs/slurm-%j.out" \
  --error="${WORKSPACE}/data-logs/slurm-%j.err" \
  --export="ALL,REMOTE_CODE_ROOT=${REMOTE_CODE_ROOT},PROJECT_ROOT=${PROJECT_ROOT},WORKSPACE=${WORKSPACE},EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
  "${REMOTE_CODE_ROOT}/hpc/400m_data.sbatch")

printf '%s\n' "$job_id" > "${WORKSPACE}/data_job_id.txt"
printf 'data_job_id=%s\n' "$job_id"
printf 'workspace=%s\n' "$WORKSPACE"
printf 'expected_commit=%s\n' "$EXPECTED_COMMIT"
