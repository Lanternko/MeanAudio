# Phase-8 quarter baseline vs aligned half-Q

## Registered comparison

Both global arms start from the same completed catalog-matched No-Q Stage-1
checkpoint at iteration 400,000 and use the same data order, seed, LR, batch
size, NoMask setting, single-caption cache, and 50,000 Stage-2 updates.

| Arm | Stage-2 conditioning | Endpoint |
|---|---|---:|
| Quarter No-Q baseline | Q disabled, effective q10 | iteration 450,000 |
| Half-Q | lower MeanSimilarity rank half q0; upper half q9 | iteration 450,000 |

The exact No-Q baseline EMA already exists in the preserved 200k training
trajectory at `ema_ckpts/0.450000.pt`; it is evaluated directly rather than
retrained. Its Hydra contract is required to match seed `14159265`, LR `1e-4`,
batch `8`, accumulation `1`, NoMask, and `use_q_conditioning=false`.

## Half-Q construction and alignment

The binary split is computed from raw
`credibility_analysis.mean_similarity`, not from the coarse q3–q9 labels.
Rows are sorted by `(mean_similarity, source_id)` and split by rank:

```text
q0: lower 125,799 rows
q9: upper 125,800 rows
```

Before writing or accepting the TSV, the builder verifies all 251,599 rows:

```text
catalog id -> unique original JSONL id -> five-caption MeanSimilarity
-> historical floor(mean_similarity * 10) label -> balanced binary rank
```

It also checks the aligned parent/source hashes and fails on missing,
duplicate, reused, or exact-versus-`_0` ambiguous ids. The observed boundary is
`0.7884269833564759 < 0.7884277522563934`.

At the Stage-1 to Stage-2 transition, q0–q9 are initialized bit-exact from the
trained No-Q q10 row. This prevents random Q embeddings from confounding the
binary comparison.

## Metrics

Metrics cover the complete 5,521-prompt MusicCaps test set:

| Scope | Protocol |
|---|---|
| Stage 1 | FluxAudio, native 25-step Flow Matching, CFG 4.5, No-Q |
| Global baseline | MeanAudio, one-step MeanFlow, CFG 0.5, No-Q |
| Global half-Q | MeanAudio, one-step MeanFlow, CFG 0.5, q9 primary and q0 diagnostic |

Each scope records CLAP plus `aes_CE`, `aes_CU`, `aes_PC`, and `aes_PQ`.
Stage-1 FM25 and global MeanFlow1 are reported side by side but are not treated
as the same inference protocol.

The completed Stage-1 metric is reused with a pinned file hash; current CLAP is
`0.1909`.

## Launch

The durable queue first runs a CPU-only preflight, waits until unrelated
MeanAudio GPU work finishes, and then runs the global baseline evaluation,
half-Q training, and q9/q0 global evaluations:

```bash
tmux new-session -d -s p8_halfq_quarter \
  "cd /home/kojiek/MeanAudio && \
   scripts/run_with_experiment_report.sh \
     --experiment phase8_halfq_quarter \
     --report /home/kojiek/logs/phase8_halfq_qpilot_s2_50000_FINAL_METRICS.json \
     --log /home/kojiek/logs/phase8_halfq_quarter_sequence.log \
     -- bash scripts/training_pipelines/sequence_phase8_halfq_quarter.sh"
```

Final metrics are written to:

```text
/home/kojiek/logs/phase8_halfq_qpilot_s2_50000_FINAL_METRICS.json
```
