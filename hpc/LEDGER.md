# Train LLM From Scratch MSI Ledger

## Status

Reduced 13M one-H100 smoke is verified on MSI. Both topologies completed on synthetic fixtures. The full 400M pipeline remains excluded.

Pinned source: `be91e7c70ce97eb7cc621a93bafbae90d9e6a001`
Remote checkout: `/users/1/arunshar/train-llm-from-scratch` (detached HEAD, clean)
Scratch project: `/scratch.global/arunshar/train-llm-from-scratch` (13G)
Staging: `BOOTSTRAP_STATUS=STAGING_PASS`
Smoke: `RESULT_STATUS=SMOKE_RESULT_VERIFIED`

## CURRENT NEXT ACTION

Wait for Arun before any 400M or full-data job. The smoke envelope is closed. Do not submit another Slurm job unless Arun opens a new envelope.

## BLOCKERS

None for the reduced smoke. Scale-up is a policy gate, not a technical blocker.

## Completed log

- 2026-09-20 (smoke verified): Job `1503299` on `msigpu` node `e3`, one H100, 2 min 16 s, commit `be91e7c`. Eight checkpoints written under `runs/smoke-20260921T010805Z-be91e7c70ce9`. GPU `NVIDIA H100` driver `580.178.04`, torch `2.14.0+cu126`, CUDA `12.6`. Image sha256 `254b32301756f7c63bd636984dc954ae4d0d59d86cd3e600aa6bcae13cbb9ce5`.
- 2026-09-20 (bootstrap staging): Job `1503077` on `msismall` node `n64` completed `STAGING_PASS` after job `1502898` failed because the PyTorch runtime image has no `ensurepip`.
- 2026-09-20 (harness fixes on the smoke path): `0404908` venv `--without-pip`, `2e5125c` host-side symlink check, `be91e7c` skip in-container git under Apptainer `--cleanenv`.
- 2026-09-20 (remote clone): Cloned `arunshar/train-llm-from-scratch` to `/users/1/arunshar/train-llm-from-scratch`.
- 2026-09-20 (local harness validation): Validated every shell script with `bash -n` and ShellCheck, parsed `project.yaml`, compiled the Python helpers, generated all nine deterministic fixtures in a temporary directory, and confirmed every training command accepts the smoke arguments.
- 2026-09-20 (MSI discovery): Verified user `arunshar`, group and account `shekhars`, partition `msigpu`, H100 GRES, 24-hour partition ceiling, Apptainer availability, scratch headroom, and an active reusable SSH connection.
- 2026-09-20 (local admission): Created an isolated Python 3.12 environment and passed the post-training smoke, RL math, checkpoint resume, and PPO/GRPO reward-improvement checks.
- 2026-09-20 (project selection): Selected `arunshar/train-llm-from-scratch`, local development with MSI sync, both upstream and sequential experiment topologies, and a reduced model before any scale-up.
