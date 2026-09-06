# Phase-8 end-to-end quarter baseline vs aligned half-Q

## Correct scale

The historical pipeline is Stage 1 `400k` plus Stage 2 `200k`. The reduced
experiment scales each stage independently:

| Arm | Stage 1 | Stage 2 | Q routing |
|---|---:|---:|---|
| Quarter No-Q baseline | 100k | 50k | Q disabled in both stages; effective q10 |
| Quarter Half-Q | 100k | 50k | aligned q0/q9 enabled in both stages |

Both arms train from scratch. They use the same initialization seed, TSV,
row order, cache, batch size, LR, optimizer schedule, NoMask setting, and
single-caption features. The baseline sees the same q0/q9 TSV but ignores its
`q_level` column, so the conditioning route is the controlled difference.

The previous proposal that reused a completed 400k Stage-1 checkpoint and
trained only 50k Stage-2 updates was an S2-only pilot, not an end-to-end
quarter-scale experiment, and was stopped before it created any checkpoint or
evaluation artifact.

## Half-Q construction and alignment

The binary split is computed from raw actual-clip
`credibility_analysis.mean_similarity`, not from the coarse q3–q9 labels.
Rows are ranked by `(mean_similarity, source_id)`:

```text
q0: lower 125,799 rows
q9: upper 125,800 rows
```

The observed boundary is:

```text
lower max = 0.7884269833564759
upper min = 0.7884277522563934
```

Before either arm starts, all 251,599 rows are rechecked through:

```text
catalog id -> cache/NPZ manifest -> unique source JSONL id
-> five-caption MeanSimilarity -> historical floor(x*10) Q
-> balanced binary rank
```

The gate fails on missing, duplicate, reused, or exact-versus-`_0` ambiguous
ids, hash drift, or cache/manifest mismatch.

## Metrics

All metrics use the full 5,521-prompt MusicCaps test set.

| Scope | Models evaluated | Protocol |
|---|---|---|
| Stage 1 | No-Q, Half-Q q9, Half-Q q0 | FluxAudio, FM25, CFG 4.5 |
| Global | No-Q, Half-Q q9, Half-Q q0 | MeanAudio, MeanFlow1, CFG 0.5 |

Each endpoint records CLAP plus `aes_CE`, `aes_CU`, `aes_PC`, and `aes_PQ`.
The primary Half-Q endpoint is q9; q0 is the binary-axis diagnostic. Stage-1
and global results are reported separately because their inference protocols
differ.

## Runtime

Historical measured throughput suggests:

- Stage 1 100k: roughly 3–3.5 hours per arm;
- Stage 2 50k: roughly 1 hour 40 minutes per arm;
- three full Stage-1 FM25 evaluations: roughly 8–9 hours total;
- three global MeanFlow1 evaluations: roughly 35 minutes total.

The full two-arm chain is therefore expected to take roughly 18–20 hours after
the GPU becomes available. The durable queue waits for pre-existing MeanAudio
GPU work and sends a Discord report on success, failure, or interruption.

## Launch

```bash
tmux new-session -d -s p8_halfq_quarter_e2e \
  "cd /home/kojiek/MeanAudio && \
   scripts/run_with_experiment_report.sh \
     --experiment phase8_halfq_quarter_e2e \
     --report /home/kojiek/logs/phase8_halfq_quarter_e2e_FINAL_METRICS.json \
     --log /home/kojiek/logs/phase8_halfq_quarter_e2e_sequence.log \
     -- bash scripts/training_pipelines/sequence_phase8_halfq_quarter.sh"
```
