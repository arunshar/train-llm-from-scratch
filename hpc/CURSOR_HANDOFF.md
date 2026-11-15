# Cursor chat handoff: train-llm-from-scratch on MSI

Date: 2026-09-20 (evening PT)
Owner: Arun Sharma (arunshar, shekhars)
Source chat: tesla-phone-screen-2026 session `f45ee787-0ecc-4a5e-89f8-e01267bb7cb3`
Resume in: Cursor window opened on `/Users/arunsharma/code/train-llm-from-scratch`

Read this file before acting. `hpc/LEDGER.md` is the short status. This file is the operating context.

## Status line

Reduced 13M one-H100 smoke is verified. Arun opened a 400M envelope. Stage 1 is bounded Pile val plus train shard 00, then 400M pretrain on one H100 with resume. The harness is committed on branch `hpc/400m-stage1-envelope` (draft PR #1) and is not merged to `main`. No 400M job has been submitted.

MSI checkout: `4b76c110284ec76d428294731a79032e9922a62b` (detached HEAD, still clean). Do not rsync until Arun merges the branch and asks.
Smoke that actually ran: commit `be91e7c70ce97eb7cc621a93bafbae90d9e6a001`, job `1503299`.
Staging: `BOOTSTRAP_STATUS=STAGING_PASS`.
Smoke: `RESULT_STATUS=SMOKE_RESULT_VERIFIED`.
Queue: empty at handoff time. Re-check with `squeue -u arunshar`.

## Who this window is for

This repo is the fork `arunshar/train-llm-from-scratch` of `FareedKhan-dev/train-llm-from-scratch`.
The Tesla phone-screen harness (`~/code/tesla-phone-screen-2026`) is a different project. Do not edit it from this window.

Arun asked to run this project from the console in this window. Local CPU tests are allowed. MSI training is not allowed on the login node.

## Rails

- No em dashes in any output, comments, or commits. Split into two sentences instead. No semicolon splices as a dash substitute.
- Commits are Arun-only. Author Arun Sharma `sharm485@umn.edu`. Never add `Co-Authored-By`.
- Commit and push only when Arun asks.
- MSI: never run training, pip, or heavy Python on the login node. Use `srun` or `sbatch` only.
- Concurrency ceiling: one job for this project. `hpc/submit_*.sh` refuse if `squeue -u $USER` is nonempty.
- Remote checkout must stay clean. Submission scripts refuse a dirty tree.
- Smoke is mechanics and checkpoint compatibility, not model quality.
- Do not download Pile shards 01 through 29. Stage 1 may download val.jsonl.zst and train/00.jsonl.zst only, after a clean committed checkout.
- Do not start SFT, DPO, PPO, or GRPO in this envelope.
- Do not claim a 24h job finished the 50k-step 400M recipe.

## Paths

| Role | Path |
| --- | --- |
| Local repo | `/Users/arunsharma/code/train-llm-from-scratch` |
| Local venv | `~/.venvs/train-llm-from-scratch` (Python 3.12.13) |
| MSI SSH host | `agate` (ControlMaster to `login.msi.umn.edu`, user `arunshar`) |
| MSI code | `/users/1/arunshar/train-llm-from-scratch` (detached HEAD, must stay clean) |
| MSI scratch | `/scratch.global/arunshar/train-llm-from-scratch` (~13G) |
| Image | `.../images/pytorch-2.14.0-cuda12.6-cudnn9-runtime.sif` |
| Env | `.../envs/pytorch-2.14.0` |
| Verified run | `.../runs/smoke-20260921T010805Z-be91e7c70ce9` |

SSH: `ssh agate`. Re-auth Duo if ControlMaster is dead (`ssh -O check agate`).

## What shipped

Harness under `hpc/`:

- `requirements.in` / hashed `requirements.lock` (Linux manylinux, no torch; torch comes from the SIF)
- `make_smoke_fixtures.py`: synthetic data under the run root (`CONTEXT_LENGTH=256`, `VOCAB_SIZE=50304`, `SEED=1337`)
- `verify_smoke.py`: eight checkpoints, finite logits, writes `smoke_result.json`
- `bootstrap.sbatch` + `submit_bootstrap.sh`: `msismall`, pull SIF, venv, hashed pip
- `smoke.sbatch` + `submit_smoke.sh` + `run_smoke.sh`: one H100, Apptainer `--nv --cleanenv`
- `expected_image.sha256` + `verify_image_pin.sh`: refuse bootstrap/smoke if the SIF digest drifts from job `1503077` / `1503299`
- Binds: `PROJECT_ROOT:/msi`, `RUN_ROOT:/ephemeral`, `CODE:/workspace:ro` so upstream `/ephemeral` hardcodes work

Apptainer layout:

- Host scratch -> `/msi`
- Fresh run dir -> `/ephemeral`
- Git checkout (read-only) -> `/workspace`

## Job history (2026-09-20)

| Job | What | Result |
| --- | --- | --- |
| 1502898 | bootstrap, `msismall` `acn26` | FAILED. Runtime SIF has no `ensurepip`. |
| 1503077 | bootstrap retry, `n64` | COMPLETED `STAGING_PASS`. Torch `2.14.0+cu126`. Image sha256 `254b32301756f7c63bd636984dc954ae4d0d59d86cd3e600aa6bcae13cbb9ce5`. |
| 1503261 | smoke | FAILED in 4s. Host `-x` on venv `bin/python` follows a symlink into the SIF. |
| 1503273 | smoke | FAILED exit 127. `--cleanenv` plus runtime image has no `git`. |
| 1503299 | smoke, `msigpu` `e3`, 2 min 16 s | COMPLETED `SMOKE_RESULT_VERIFIED`. NVIDIA H100, driver `580.178.04`. |

Fixes now on `main`:

- `0404908` venv `--without-pip --system-site-packages`, install lock via image interpreter
- `2e5125c` host check uses `pyvenv.cfg` and the python symlink, not `-x`
- `be91e7c` in-container git skipped; `smoke.sbatch` already pins HEAD on the host
- `4b76c11` ledger update only (docs). Does not change the verified smoke SHA.

## Verified smoke artifacts

Run dir: `/scratch.global/arunshar/train-llm-from-scratch/runs/smoke-20260921T010805Z-be91e7c70ce9`

Eight checkpoints in `ckpts/`:

- `base_pretrained.pt` (pretrain)
- `sft.pt` (sft)
- `reward.pt` (reward)
- `dpo.pt` (dpo_dpo)
- `ppo_upstream.pt` / `grpo_upstream.pt` from `sft.pt`
- `ppo_sequential.pt` from `dpo.pt` plus RM
- `grpo_sequential.pt` from `ppo_sequential.pt`

Also: `RESULT_STATUS`, `smoke_result.json`, `gpu.txt`, `source_commit.txt`, `container_sha256.txt`, Slurm out/err.

CPU tests inside that job also passed: `test_post_training_smoke.py`, `test_rl_math.py`, `test_checkpoint_resume.py`, `verify_rl_optimizes.py`.

## How Arun can run in this window

Local CPU (safe, no GPU, no MSI):

```bash
cd /Users/arunsharma/code/train-llm-from-scratch
PY=~/.venvs/train-llm-from-scratch/bin/python
PYTHONPATH=. "$PY" tests/test_post_training_smoke.py
PYTHONPATH=. "$PY" tests/test_rl_math.py
PYTHONPATH=. "$PY" tests/test_checkpoint_resume.py
PYTHONPATH=. "$PY" tests/verify_rl_optimizes.py
PYTHONPATH=. "$PY" tests/test_hpc_image_pin.py
PYTHONPATH=. "$PY" tests/test_hpc_400m_envelope.py
```

MSI inspect (login node is fine for these):

```bash
ssh agate
cd ~/train-llm-from-scratch
git status --short --branch
cat /scratch.global/arunshar/train-llm-from-scratch/BOOTSTRAP_STATUS
cat /scratch.global/arunshar/train-llm-from-scratch/runs/smoke-20260921T010805Z-be91e7c70ce9/RESULT_STATUS
cat /scratch.global/arunshar/train-llm-from-scratch/runs/smoke-20260921T010805Z-be91e7c70ce9/smoke_result.json
```

Do not run `python scripts/pretrain_base.py` or similar on `agate` login. Do not resubmit bootstrap or smoke unless Arun opens a new envelope.

If he authorizes the committed 400M stage-1 data job:

```bash
ssh agate 'cd ~/train-llm-from-scratch && bash hpc/submit_400m_data.sh'
```

## Upstream gotchas that the harness works around

Training scripts hardcode `/ephemeral` paths and `HF_HOME`. Do not rewrite those for MSI. Bind `/ephemeral` to the run directory instead.

The SIF python is 3.12. The venv `bin/python` is a symlink to `/usr/bin/python` inside the container. Host `-x` is a lie. Inside Apptainer the same path is executable.

`pip` overlaying the venv warned that image `spin` wants `click<8.4` while the lock has `click==8.5.0`. Imports still passed. Training does not use `spin`.

`project.yaml` still says `maximum_total_jobs: 2`. Actual jobs used: two bootstraps (one fail, one pass) plus three smokes (two fail, one pass). That field is documentation. The live ceiling is one concurrent job.

## Open decisions (Arun only)

1. Review and merge draft PR #1, then authorize `submit_400m_data.sh`.
2. After DATA_STATUS=READY, authorize the first 24h pretrain job.
3. Leave remaining Pile shards and post-training closed.

## First move for the new chat

1. Confirm cwd is `/Users/arunsharma/code/train-llm-from-scratch`.
2. Restate the status line in a few sentences.
3. Ask whether Arun wants local tests, MSI inspect, a harness change, or a new Slurm envelope.
4. Do not submit jobs, commit, or push until he asks.
