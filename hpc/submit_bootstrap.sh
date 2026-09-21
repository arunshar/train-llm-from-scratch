#!/usr/bin/env bash
set -euo pipefail

REMOTE_CODE_ROOT=$(git rev-parse --show-toplevel)
EXPECTED_COMMIT=$(git rev-parse HEAD)
PROJECT_ROOT=${PROJECT_ROOT:-"/scratch.global/${USER}/train-llm-from-scratch"}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to stage from a dirty remote checkout" >&2
  exit 1
fi
if (( $(squeue -u "$USER" -h | wc -l) >= 1 )); then
  echo "project concurrency ceiling is one active job" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/bootstrap-logs"

job_id=$(sbatch \
  --parsable \
  --output="${PROJECT_ROOT}/bootstrap-logs/slurm-%j.out" \
  --error="${PROJECT_ROOT}/bootstrap-logs/slurm-%j.err" \
  --export="ALL,REMOTE_CODE_ROOT=${REMOTE_CODE_ROOT},PROJECT_ROOT=${PROJECT_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
  "${REMOTE_CODE_ROOT}/hpc/bootstrap.sbatch")

printf 'bootstrap_job_id=%s\n' "$job_id"
printf 'expected_commit=%s\n' "$EXPECTED_COMMIT"
printf 'project_root=%s\n' "$PROJECT_ROOT"
