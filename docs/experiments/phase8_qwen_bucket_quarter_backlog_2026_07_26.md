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

## 2026-09-14：評估補跑排入 p2 隊尾（054）

五個 S2 EMA（NoQ、K=3/5/10 balanced、K=5 fixed）從 07-26~30 就在磁碟上，但 `cfg0_eval_runtime` 裡沒有任何一個的 canonical 報告。

- queue seat `054_qwen_bucket_quarter_eval_backfill.sh`，在 053 → 052 之後
- contract `docs/experiments/qwen_bucket_quarter_eval_backfill_contract.json`，action `scripts/training_pipelines/qwen_bucket_quarter_eval_backfill.sh`
- 只做評估，不訓練。CFG0：NoQ 跑 noq；每個 Q arm 跑 q9（主）+ q0（檢查模型有沒有在看 code），共 9 格。CFG3+neg：NoQ 跑 noq、Q arm 只跑 q9，共 5 格。估 ~6.5 GPU-h、~15 GB 音檔
- 判讀（預先寫死，floor CLAP 0.0042）：q9 vs q0 差 < 0.0084 = 忽略 code；Q arm q9 vs NoQ ±0.0084 = 平手；K 之間差 > 0.0084 才算有差（只比 balanced）；兩個協定同號才下結論
- **Provenance caveat**：這批 run 從 `phase8_qwen_official_matched_npz` 讀文字特徵，該目錄 08-22 已被原地覆寫成 Caption 2.0。EMA 本身有效，但數字無法從現在的 NPZ 重現，也**不能**放進 c2p0 表比較

## 2026-09-14：054 結果（14/14 cell PASS）

原始 guest 在生成前 exit 2（checkpoint root 漏列 `/mnt/HDD/kojiek/MeanAudio_exps`；NoQ arm id 應為 `noq_cfg0_noq`）。另一 session 以 `harn/qwen_bucket_backfill_recovery_20260914/` 修復重排，科學設定不變，5 個 EMA sha 全符。CFG3 cell 用 `phase4_eval.py` 逐檔 CLAP（非 b32），只能在 054 內部比，不可與 b32 數字混比。語料 provenance caveat 同上：不可與 c2p0 表比。單一訓練 seed。

| arm | CFG0 CLAP | CE | CU | PC | PQ | CFG3+fid8 CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|---|---|---|---|---|
| NoQ | **0.1724** | 6.480 | 6.902 | 5.379 | 6.647 | **0.1969** | 6.753 | 7.243 | 4.689 | 7.078 |
| k3_bal q9 | 0.1597 | 6.275 | 6.630 | 5.456 | 6.297 | 0.1800 | 7.104 | 7.300 | 4.660 | 7.042 |
| k3_bal q0 | 0.1550 | 6.008 | 6.487 | 5.384 | 6.169 | — | | | | |
| k5_bal q9 | 0.1565 | 6.545 | 6.766 | 5.545 | 6.624 | 0.1767 | 6.802 | 7.154 | 4.799 | 6.980 |
| k5_bal q0 | 0.1508 | 6.285 | 6.642 | 5.483 | 6.536 | — | | | | |
| k10_bal q9 | 0.1556 | 6.169 | 6.393 | 5.398 | 6.136 | 0.1513 | 6.259 | 6.799 | 4.558 | 6.622 |
| k10_bal q0 | 0.1560 | 6.014 | 6.338 | 5.401 | 6.122 | — | | | | |
| k5_fixed q9 | 0.1541 | 5.963 | 6.411 | 5.425 | 6.028 | 0.1865 | 6.568 | 6.963 | 4.793 | 6.692 |
| k5_fixed q0 | 0.1277 | 5.643 | 6.070 | 5.482 | 5.731 | — | | | | |

判讀（門檻 = 同協定訓練 seed 底線 2×；CLAP CFG0 0.0084）：
1. **q 碼在 CLAP 上被忽略（balanced arm）**：q9−q0 = k3 +0.0047、k5 +0.0057、k10 −0.0004，皆 <0.0084。AES 有小幅一致 q9>q0（k3 CU +0.14 / PQ +0.13、k5 CU +0.12 超門檻），僅 CFG0 單協定。k5_fixed q9−q0 +0.026 是 fixed 分桶的 q0 支撐集問題，不計。
2. **Q 輸 NoQ（兩協定一致）**：CFG0 所有 Q arm 低 0.013–0.018；CFG3 低 0.010–0.046。PQ：k5_bal / k3_bal 在兩協定內與 NoQ 平手，k10 / k5_fixed 明確較低。
3. **K 排名**：CLAP CFG0 三個 balanced 相差 0.0041 → 不可排；CFG3 k10 低 ~0.027，但兩協定不一致 → CLAP 不宣稱。**PQ 兩協定一致 k10 最差**（CFG0 −0.16 vs k3、−0.49 vs k5；CFG3 −0.42 / −0.36，均 >2× 底線）。
結論：與 048/049（c2p0 true-random）一致——Q 傷 CLAP、q 碼無 CLAP 響應；新增一點：細分到 K=10 在 AES 上有害。收線。
