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

The smoke requests one H100, 8 CPUs, 64 GB of memory, and 30 minutes. It writes to a fresh directory under:

```text
/scratch.global/arunshar/train-llm-from-scratch/runs/
```

The run is accepted only when `RESULT_STATUS` contains `SMOKE_RESULT_VERIFIED` and `smoke_result.json` records all eight checkpoints.

## Scale-up gate

Do not submit the 400M configuration after a smoke pass automatically. First add bounded data manifests, robust resume coverage, stage-specific Slurm envelopes, and independent evaluation. The MSI partition has a 24-hour job ceiling, while the upstream post-training stages do not support full resume.
