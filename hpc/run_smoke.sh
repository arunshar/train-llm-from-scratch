#!/usr/bin/env bash
set -euo pipefail

: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the source revision}"

ROOT=/ephemeral
PROJECT_ROOT=/msi
PY="${PROJECT_ROOT}/envs/pytorch-2.14.0/bin/python"
IMAGE="${PROJECT_ROOT}/images/pytorch-2.14.0-cuda12.6-cudnn9-runtime.sif"

if [[ ! -x "$PY" ]]; then
  echo "missing staged Python environment: $PY" >&2
  exit 1
fi

cd /workspace
export PYTHONPATH=/workspace
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TIKTOKEN_CACHE_DIR="${PROJECT_ROOT}/tiktoken_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

failure_status() {
  printf 'SMOKE_FAILED\n' > "${ROOT}/RESULT_STATUS"
}
trap failure_status ERR

actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "source commit mismatch: got $actual_commit, expected $EXPECTED_COMMIT" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "remote source tree is not clean" >&2
  git status --short >&2
  exit 1
fi

printf '%s\n' "$actual_commit" > "${ROOT}/source_commit.txt"
sha256sum "$IMAGE" > "${ROOT}/container_sha256.txt"
"$PY" -m pip freeze > "${ROOT}/environment.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader > "${ROOT}/gpu.txt"

"$PY" hpc/make_smoke_fixtures.py --root "$ROOT"

"$PY" tests/test_post_training_smoke.py
"$PY" tests/test_rl_math.py
PYTHONPATH=. "$PY" tests/test_checkpoint_resume.py
PYTHONPATH=. "$PY" tests/verify_rl_optimizes.py

COMMON=(
  --device cuda
  --amp_dtype bf16
  --log_dir "${ROOT}/logs"
  --use_wandb false
)

"$PY" scripts/pretrain_base.py \
  --config configs/smoke/pretrain.json \
  "${COMMON[@]}" \
  --train_path "${ROOT}/data/pile_train.h5" \
  --dev_path "${ROOT}/data/pile_dev.h5" \
  --out_ckpt "${ROOT}/ckpts/base_pretrained.pt"

"$PY" scripts/train_sft.py \
  --config configs/smoke/sft.json \
  "${COMMON[@]}" \
  --pretrained_ckpt "${ROOT}/ckpts/base_pretrained.pt" \
  --data_path "${ROOT}/data/sft_packed.h5" \
  --out_ckpt "${ROOT}/ckpts/sft.pt"

"$PY" scripts/train_reward.py \
  --config configs/smoke/reward.json \
  "${COMMON[@]}" \
  --sft_ckpt "${ROOT}/ckpts/sft.pt" \
  --pref_path "${ROOT}/data/preferences.jsonl" \
  --out_ckpt "${ROOT}/ckpts/reward.pt"

"$PY" scripts/train_dpo.py \
  --config configs/smoke/dpo.json \
  "${COMMON[@]}" \
  --sft_ckpt "${ROOT}/ckpts/sft.pt" \
  --pref_path "${ROOT}/data/preferences.jsonl" \
  --out_ckpt "${ROOT}/ckpts/dpo.pt" \
  --loss_type dpo

"$PY" scripts/train_ppo.py \
  --config configs/smoke/ppo.json \
  "${COMMON[@]}" \
  --sft_ckpt "${ROOT}/ckpts/sft.pt" \
  --prompt_path "${ROOT}/data/rl_prompts_train.jsonl" \
  --reward_source verifier \
  --out_ckpt "${ROOT}/ckpts/ppo_upstream.pt"

"$PY" scripts/train_grpo.py \
  --config configs/smoke/grpo.json \
  "${COMMON[@]}" \
  --sft_ckpt "${ROOT}/ckpts/sft.pt" \
  --prompt_path "${ROOT}/data/rl_prompts_train.jsonl" \
  --curriculum_path "${ROOT}/data/arithmetic_prompts.jsonl" \
  --out_ckpt "${ROOT}/ckpts/grpo_upstream.pt"

"$PY" scripts/train_ppo.py \
  --config configs/smoke/ppo.json \
  "${COMMON[@]}" \
  --sft_ckpt "${ROOT}/ckpts/dpo.pt" \
  --prompt_path "${ROOT}/data/rl_prompts_train.jsonl" \
  --reward_source rm \
  --reward_ckpt "${ROOT}/ckpts/reward.pt" \
  --out_ckpt "${ROOT}/ckpts/ppo_sequential.pt"

"$PY" scripts/train_grpo.py \
  --config configs/smoke/grpo.json \
  "${COMMON[@]}" \
  --sft_ckpt "${ROOT}/ckpts/ppo_sequential.pt" \
  --prompt_path "${ROOT}/data/rl_prompts_train.jsonl" \
  --curriculum_path "${ROOT}/data/arithmetic_prompts.jsonl" \
  --out_ckpt "${ROOT}/ckpts/grpo_sequential.pt"

EXPECTED_COMMIT="$EXPECTED_COMMIT" "$PY" hpc/verify_smoke.py \
  --root "$ROOT" \
  --device cuda

trap - ERR
