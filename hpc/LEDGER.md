# Train LLM From Scratch MSI Ledger

## Status

Reduced 13M one-H100 smoke is verified on MSI. Arun opened a 400M envelope. Stage 1 is bounded Pile val plus train shard 00, then 400M pretrain on one H100 with resume. The harness is committed on branch `hpc/400m-stage1-envelope` and is not yet merged to `main`. No 400M Slurm job has been submitted.

Pinned smoke source: `be91e7c70ce97eb7cc621a93bafbae90d9e6a001`
Remote checkout: `/users/1/arunshar/train-llm-from-scratch` (detached HEAD, clean at `4b76c11`)
Scratch project: `/scratch.global/arunshar/train-llm-from-scratch` (13G)
Staging: `BOOTSTRAP_STATUS=STAGING_PASS`
Smoke: `RESULT_STATUS=SMOKE_RESULT_VERIFIED`

## CURRENT NEXT ACTION

Wait for Arun to review and merge the 400M stage-1 harness (and the image pin) to `main` before any rsync or sbatch. After a clean remote checkout of the merged commit, the first live job is `bash hpc/submit_400m_data.sh` only. Do not submit pretrain until `DATA_STATUS=READY`. Do not download Pile shards 01-29. Do not start SFT/RL.

## BLOCKERS

Live 400M submit is blocked on Arun merging the harness branch to `main`. The remote checkout must stay clean.

## Completed log

- 2026-11-15 (harness committed, not merged): Committed the image pin and the 400M stage-1 harness on branch `hpc/400m-stage1-envelope` for review. The six local CPU test scripts passed. Nothing ran on MSI and no sbatch was submitted.
- 2026-09-20 (400M stage-1 harness, not submitted): Opened a bounded envelope after Arun asked. Data is Pile val (338 MB) plus train/00 (11.2 GB). Pretrain is one H100, 24h, `--resume`, success is a verified checkpoint rather than 50k steps. Files live under `hpc/submit_400m_*.sh`. No sbatch. Commit required before the remote tree can stay clean and still contain these scripts.
- 2026-09-20 (image pin, local only): Local CPU suite passed again. MSI inspect confirmed empty queue, clean detached HEAD `4b76c11`, `STAGING_PASS`, and `SMOKE_RESULT_VERIFIED` for job `1503299`. Pinned SIF digest `254b32301756f7c63bd636984dc954ae4d0d59d86cd3e600aa6bcae13cbb9ce5` in `hpc/expected_image.sha256`. Submit checks the sidecar. Bootstrap and smoke rehash the image on a compute node.
- 2026-09-20 (smoke verified): Job `1503299` on `msigpu` node `e3`, one H100, 2 min 16 s, commit `be91e7c`. Eight checkpoints written under `runs/smoke-20260921T010805Z-be91e7c70ce9`. GPU `NVIDIA H100` driver `580.178.04`, torch `2.14.0+cu126`, CUDA `12.6`. Image sha256 `254b32301756f7c63bd636984dc954ae4d0d59d86cd3e600aa6bcae13cbb9ce5`.
- 2026-09-20 (bootstrap staging): Job `1503077` on `msismall` node `n64` completed `STAGING_PASS` after job `1502898` failed because the PyTorch runtime image has no `ensurepip`.
- 2026-09-20 (harness fixes on the smoke path): `0404908` venv `--without-pip`, `2e5125c` host-side symlink check, `be91e7c` skip in-container git under Apptainer `--cleanenv`.
- 2026-09-20 (remote clone): Cloned `arunshar/train-llm-from-scratch` to `/users/1/arunshar/train-llm-from-scratch`.
- 2026-09-20 (local harness validation): Validated every shell script with `bash -n` and ShellCheck, parsed `project.yaml`, compiled the Python helpers, generated all nine deterministic fixtures in a temporary directory, and confirmed every training command accepts the smoke arguments.
- 2026-09-20 (MSI discovery): Verified user `arunshar`, group and account `shekhars`, partition `msigpu`, H100 GRES, 24-hour partition ceiling, Apptainer availability, scratch headroom, and an active reusable SSH connection.
- 2026-09-20 (local admission): Created an isolated Python 3.12 environment and passed the post-training smoke, RL math, checkpoint resume, and PPO/GRPO reward-improvement checks.
- 2026-09-20 (project selection): Selected `arunshar/train-llm-from-scratch`, local development with MSI sync, both upstream and sequential experiment topologies, and a reduced model before any scale-up.
