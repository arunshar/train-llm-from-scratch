# Train LLM From Scratch MSI Ledger

## Status

The fork is cloned locally at commit `b995104cff1dd488cdcc1517bc284649d9252d88`. The self-contained CPU test gate passes in the Python 3.12 environment at `~/.venvs/train-llm-from-scratch`. MSI transport and read-only scheduler discovery pass. The local MSI harness passes syntax, lint, fixture, YAML, and CLI validation. It has not been committed, synced, or submitted.

## CURRENT NEXT ACTION

Arun reviews and commits the `hpc/` harness. Then sync the clean committed tree to MSI and submit the bootstrap staging job.

## BLOCKERS

The project policy reserves commits and pushes for Arun. Remote staging and Slurm submission remain blocked until the harness is reviewed and committed. The submission scripts then pin the clean commit in each run envelope.

The full 400M pipeline remains excluded from the current run envelope. The first authorized execution tier is a reduced 13M one-H100 smoke covering both requested topology choices.

## Completed log

- 2026-09-20 (local harness validation): Validated every shell script with `bash -n` and ShellCheck, parsed `project.yaml`, compiled the Python helpers, generated all nine deterministic fixtures in a temporary directory, and confirmed every training command accepts the smoke arguments.
- 2026-09-20 (MSI discovery): Verified user `arunshar`, group and account `shekhars`, partition `msigpu`, H100 GRES, 24-hour partition ceiling, Apptainer availability, scratch headroom, and an active reusable SSH connection. No jobs were submitted.
- 2026-09-20 (local admission): Created an isolated Python 3.12 environment and passed the post-training smoke, RL math, checkpoint resume, and PPO/GRPO reward-improvement checks.
- 2026-09-20 (project selection): Selected `arunshar/train-llm-from-scratch`, local development with MSI sync, both upstream and sequential experiment topologies, and a reduced model before any scale-up.
