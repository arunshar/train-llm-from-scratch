# MSI execution harness

This directory stages a reduced 13M end-to-end smoke before any larger training run. The smoke exercises the shared pretraining, SFT, reward, and DPO stages, then verifies both requested branches:

1. Upstream topology: PPO and GRPO each start from `sft.pt`.
2. Sequential topology: PPO starts from `dpo.pt` with the reward model, then GRPO starts from the PPO checkpoint.

The smoke validates mechanics and checkpoint compatibility. It does not establish model quality.

## Local admission

Create an isolated environment outside the repository:

```bash
uv venv ~/.venvs/train-llm-from-scratch --python 3.12
uv pip install --python ~/.venvs/train-llm-from-scratch/bin/python -e .
```

Run the self-contained checks:

```bash
PY=~/.venvs/train-llm-from-scratch/bin/python
PYTHONPATH=. "$PY" tests/test_post_training_smoke.py
PYTHONPATH=. "$PY" tests/test_rl_math.py
PYTHONPATH=. "$PY" tests/test_checkpoint_resume.py
PYTHONPATH=. "$PY" tests/verify_rl_optimizes.py
PYTHONPATH=. "$PY" tests/test_hpc_image_pin.py
PYTHONPATH=. "$PY" tests/test_hpc_400m_envelope.py
```

## Source staging

The remote tree must be a clean copy of an Arun-authored commit. From the Mac:

```bash
test -z "$(git status --porcelain)"
rsync -az \
  --exclude .venv \
  --exclude __pycache__ \
  ./ agate:~/train-llm-from-scratch/
ssh agate 'git -C ~/train-llm-from-scratch status --short --branch'
```

The submission scripts require a clean tree, resolve its exact commit, and record that SHA in each run envelope.

## Bootstrap

The bootstrap is a CPU Slurm job. It pulls the pinned PyTorch image, records the SIF hash, creates a persistent environment under scratch, and installs the hashed dependency lock.

```bash
ssh agate 'cd ~/train-llm-from-scratch && bash hpc/submit_bootstrap.sh'
```

Monitor only the returned job ID:

```bash
ssh agate 'squeue -j JOB_ID'
ssh agate 'sacct -j JOB_ID --format=JobID,JobName%18,State,ExitCode,Elapsed,MaxRSS'
```

Read both bootstrap logs and verify that `/scratch.global/arunshar/train-llm-from-scratch/BOOTSTRAP_STATUS` contains `STAGING_PASS`.

## One-H100 smoke

Submit only after bootstrap verification:

```bash
ssh agate 'cd ~/train-llm-from-scratch && bash hpc/submit_smoke.sh'
```

Submit and smoke both refuse unless the staged SIF digest matches `hpc/expected_image.sha256`. That pin is the image hashed by bootstrap job 1503077 and reused by smoke job 1503299. Submit checks the recorded sidecar on the login node. The smoke job rehashes the SIF on the compute node.

The smoke requests one H100, 8 CPUs, 64 GB of memory, and 30 minutes. It writes to a fresh directory under:

```text
/scratch.global/arunshar/train-llm-from-scratch/runs/
```

The run is accepted only when `RESULT_STATUS` contains `SMOKE_RESULT_VERIFIED` and `smoke_result.json` records all eight checkpoints.

## 400M stage 1 (data then pretrain)

This envelope is not automatic after smoke. It is also not the full 50k-step recipe and not SFT/RL.

Bounded data is Pile-uncopyrighted `val.jsonl.zst` (338 MB) plus `train/00.jsonl.zst` (11.2 GB). Remaining shards stay excluded. Data is a CPU `msismall` job. Pretrain is one H100 for 24 hours with `--resume`. Completing `train_steps=50000` in one job is not the success criterion. A loadable 400M checkpoint with finite logits is.

Commit and sync before submit. The remote checkout must stay clean.

```bash
ssh agate 'cd ~/train-llm-from-scratch && bash hpc/submit_400m_data.sh'
# after DATA_STATUS=READY and the queue is empty
ssh agate 'cd ~/train-llm-from-scratch && bash hpc/submit_400m_pretrain.sh'
```

Workspace:

```text
/scratch.global/arunshar/train-llm-from-scratch/workspaces/400m
```

## Scale-up gate

Do not start SFT, reward, DPO, PPO, or GRPO from this envelope. Upstream post-training stages do not fully resume, and `msigpu` MaxTime is 24 hours. Do not download Pile shards 01 through 29 unless Arun opens a later data envelope.
