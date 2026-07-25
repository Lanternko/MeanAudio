# Phase-8 Qwen bucket quarter backlog

## Scope

Durable sequential chain:

| Tier | Order | Arm | Execution | Claim role |
|---|---:|---|---|---|
| Primary | 1 | No-Q | Fresh/resume official-Qwen adapter | Primary control |
| Primary | 2 | K=2 balanced | `REUSE=k2_balanced_historical` | Primary K-resolution |
| Primary | 3 | K=5 balanced | Fresh/resume bucket arm | Primary K-resolution |
| Primary | 4 | K=10 balanced | Fresh/resume bucket arm | Primary K-resolution |
| Backup | 1 | K=3 balanced | Fresh/resume bucket arm | Backup K-resolution |
| Backup | 2 | K=5 fixed | Fresh/resume bucket arm | Diagnostic strategy comparison against K=5 balanced |
| Backup | 3 | K=10 fixed | `REUSE=k10_fixed_historical` | Historical reference |

K=5 fixed is a diagnostic/backup arm. It must not be used as part of the
primary K-resolution claim.

Every arm is quarter scale: Stage 1 has 100,000 updates, Stage 2 has 50,000
additional updates (final checkpoint iteration 150,000), and training contains
exactly 251,599 rows.

## No-Q adapter

`scripts/training_pipelines/execute_phase8_qwen_noq_arm_eval.sh` is the
official-aligned Qwen No-Q adapter. It deliberately does not use
`train_pipeline_phase8_halfq_quarter.sh`, whose No-Q arm is based on the
legacy/LP cache.

The adapter is bound to:

- `phase8_qwen_meansim_k2_balanced.tsv`
- `phase8_qwen_official_matched_npz_cache_train.txt`
- `phase8_qwen_official_matched_npz`
- the passed grid manifest, NPZ manifest, and exhaustive Qwen cache audit
- `use_q_conditioning=false` in both stages

The TSV retains K=2 balanced `q_level` values, but the model ignores them. This
keeps caption, row order, cache filenames, NPZ tensors, seed, optimizer,
schedule, and training scale matched to K=2 balanced while isolating the Q
conditioning route.

## Durability and gates

The chain is implemented in
`scripts/training_pipelines/sequence_phase8_qwen_bucket_quarter_backlog.sh`.

- A nonblocking `flock` prevents duplicate backlog schedulers.
- Before each GPU arm, the scheduler waits for the GPU gate.
- NVML (`nvidia-smi`) is the primary process query.
- If NVML fails or returns malformed data, the gate falls back to `/proc`
  command-line and open NVIDIA-device inspection.
- A matching GPU process is busy; unreadable process identity is unknown.
  Busy and unknown both wait. An NVML error alone is never treated as idle;
  the gate proceeds only after the process fallback itself completes cleanly.
- Each arm runs through `run_with_experiment_report.sh`.
- Existing reports are parsed and checked against experiment identity,
  quarter scale, full MusicCaps hash/row count, training audit, contract,
  model hashes, protocols, and metric endpoints before being skipped.
- A malformed or stale existing report fails closed instead of silently
  skipping.
- `set -euo pipefail` makes a failed primary arm terminal. Backup arms are
  unreachable until all four primary reports validate again.
- K=2 balanced and K=10 fixed always enter the existing execute script with
  historical reuse, so its audit/model/TSV-equivalence validation is mandatory
  and no retraining occurs.

This is a shared host. The queue, watcher, repair controller, and every agent
are forbidden from rebooting/shutting down the host, reloading NVIDIA modules,
restarting shared services, changing system packages, or signaling another
user's process. Driver compatibility is process-local: the queue validates the
loaded `595.71.05` kernel module and hash-pinned `595.71.05` user-owned
`libcuda`/`libnvidia-ml`, then prepends that directory only to its own
`LD_LIBRARY_PATH`. CUDA and NCCL must pass a functional probe before an arm
starts.

The local watcher performs routine polling with zero LLM calls. A new hard
incident is fingerprinted and handed to the durable repair controller. The
controller allows one low-cost Luna repair proposal in an isolated worktree,
then requires a fresh SOL approval bound to the exact commit, diff, incident,
repair command, and rollback command. Approved commands are short and bounded;
the supervisor—not the repair command—resumes the immutable queue. The repair
is closed only after deterministic iteration/checkpoint progress is observed.

For newly trained arms, `EXPERIMENT_RUN_MODE=fresh` rejects existing
artifacts. `EXPERIMENT_RUN_MODE=resume` accepts only artifacts that satisfy the
underlying immutable contract and checkpoint iteration checks. The No-Q
adapter saves checkpoint/EMA state every 25,000 iterations. Existing bucket
arms retain their established pipeline checkpoint behavior.

## Safe commands

Print the exact queue without data scans, locks, GPU queries, or GPU work:

```bash
DRY_RUN=true \
  bash scripts/training_pipelines/sequence_phase8_qwen_bucket_quarter_backlog.sh
```

Run all CPU/data/historical-reuse preflights and stop before the GPU gate:

```bash
PREFLIGHT_ONLY=true EXPERIMENT_RUN_MODE=resume \
  bash scripts/training_pipelines/sequence_phase8_qwen_bucket_quarter_backlog.sh
```

Inspect the GPU gate once without launching training:

```bash
GPU_CHECK_ONLY=true EXPERIMENT_RUN_MODE=resume \
  bash scripts/training_pipelines/sequence_phase8_qwen_bucket_quarter_backlog.sh
```

The actual chain is intentionally not launched by this implementation task.
When authorized, use `fresh` only if all new-arm artifact paths are empty;
otherwise use the reviewed `resume` mode:

```bash
EXPERIMENT_RUN_MODE=resume POLL_SECONDS=60 \
  bash scripts/training_pipelines/sequence_phase8_qwen_bucket_quarter_backlog.sh
```

## Reports

Per-arm reports are written below `/home/kojiek/logs`:

- `phase8_qwen_bucket_quarter_noq_FINAL_METRICS.json`
- `phase8_qwen_bucket_quarter_k{2,3,5,10}_{balanced|fixed}_FINAL_METRICS.json`
  for arms present in the queue

After all seven reports validate, the chain writes
`phase8_qwen_bucket_quarter_backlog_FINAL_METRICS.json` atomically. Its
`claim_policy` records that K=5 fixed is diagnostic-only.
