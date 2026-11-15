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
  echo "bootstrap must be verified before the 400M pretrain job" >&2
  exit 1
fi
if [[ "$(cat "${WORKSPACE}/DATA_STATUS" 2>/dev/null)" != "READY" ]]; then
  echo "400M data job must reach READY before pretrain" >&2
  exit 1
fi
bash "${REMOTE_CODE_ROOT}/hpc/verify_image_pin.sh" \
  --sidecar "${PROJECT_ROOT}/container.sha256" >/dev/null
if (( $(squeue -u "$USER" -h | wc -l) >= 1 )); then
  echo "project concurrency ceiling is one active job" >&2
  exit 1
fi

run_id="400m-pretrain-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_COMMIT:0:12}"
run_root="${PROJECT_ROOT}/runs/${run_id}"
mkdir "$run_root"

cat > "${run_root}/run_envelope.txt" <<EOF
run_id=${run_id}
envelope=400m-pretrain-stage1
expected_commit=${EXPECTED_COMMIT}
account=shekhars
partition=msigpu
gres=gpu:h100:1
cpus_per_task=8
memory=64G
wall_time=24:00:00
workspace=${WORKSPACE}
maximum_concurrent_jobs=1
success_status=PRETRAIN_CKPT_VERIFIED
EOF

job_id=$(sbatch \
  --parsable \
  --output="${run_root}/slurm-%j.out" \
  --error="${run_root}/slurm-%j.err" \
  --export="ALL,REMOTE_CODE_ROOT=${REMOTE_CODE_ROOT},PROJECT_ROOT=${PROJECT_ROOT},WORKSPACE=${WORKSPACE},EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
  "${REMOTE_CODE_ROOT}/hpc/400m_pretrain.sbatch")

printf '%s\n' "$job_id" > "${run_root}/job_id.txt"
printf 'pretrain_job_id=%s\n' "$job_id"
printf 'run_root=%s\n' "$run_root"
printf 'workspace=%s\n' "$WORKSPACE"
