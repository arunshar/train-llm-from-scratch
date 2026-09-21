#!/usr/bin/env bash
set -euo pipefail

REMOTE_CODE_ROOT=$(git rev-parse --show-toplevel)
EXPECTED_COMMIT=$(git rev-parse HEAD)
PROJECT_ROOT=${PROJECT_ROOT:-"/scratch.global/${USER}/train-llm-from-scratch"}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to submit from a dirty remote checkout" >&2
  exit 1
fi
if [[ "$(cat "${PROJECT_ROOT}/BOOTSTRAP_STATUS" 2>/dev/null)" != "STAGING_PASS" ]]; then
  echo "bootstrap must be verified before the smoke job" >&2
  exit 1
fi
if (( $(squeue -u "$USER" -h | wc -l) >= 1 )); then
  echo "project concurrency ceiling is one active job" >&2
  exit 1
fi

run_id="smoke-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_COMMIT:0:12}"
run_root="${PROJECT_ROOT}/runs/${run_id}"
mkdir "$run_root"

cat > "${run_root}/run_envelope.txt" <<EOF
run_id=${run_id}
expected_commit=${EXPECTED_COMMIT}
account=shekhars
partition=msigpu
qos=partition-default-common_agate
gres=gpu:h100:1
cpus_per_task=8
memory=64G
wall_time=00:30:00
maximum_concurrent_jobs=1
maximum_retry_count=1
artifact_root=${run_root}
success_status=SMOKE_RESULT_VERIFIED
EOF

job_id=$(sbatch \
  --parsable \
  --output="${run_root}/slurm-%j.out" \
  --error="${run_root}/slurm-%j.err" \
  --export="ALL,REMOTE_CODE_ROOT=${REMOTE_CODE_ROOT},PROJECT_ROOT=${PROJECT_ROOT},RUN_ROOT=${run_root},EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
  "${REMOTE_CODE_ROOT}/hpc/smoke.sbatch")

printf '%s\n' "$job_id" > "${run_root}/job_id.txt"
printf 'smoke_job_id=%s\n' "$job_id"
printf 'run_root=%s\n' "$run_root"
