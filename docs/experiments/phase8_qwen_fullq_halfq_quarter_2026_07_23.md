# Phase-8 Qwen Full-Q vs Half-Q quarter experiment

This experiment starts only after `phase8_halfq_quarter_e2e` has produced its
final Stage-1 and global metrics report.

## Matched design

- Full-Q arm: Stage 1 100k from scratch, then Stage 2 50k.
- Half-Q arm: Stage 1 100k from scratch, then Stage 2 50k.
- Both stages in both arms use Q conditioning.
- Both arms use the same 251,599 audio clips, official Qwen captions, Qwen text
  cache, row order, seed, optimizer, and schedule.
- The only treatment difference is Q granularity:
  - Full-Q is `clamp(floor(actual_clip_mean_similarity * 10), 0, 9)`.
  - Half-Q is a balanced rank split of the same actual-clip MeanSimilarity:
    lower half q0, upper half q9.

## Fail-closed alignment gates

Before either arm starts, the queue:

1. re-resolves all 251,599 catalog clip IDs against the original five-caption
   MeanSimilarity JSONL;
2. reverifies the historical Full-Q formula and the balanced Half-Q split;
3. checks every Qwen row against the official Qwen JSON by parsed Jamendo track;
4. requires the prior exhaustive Qwen NPZ audit, including all embedded
   `clip_id` and `caption_sha256` checks and the semantic matched-vs-shuffled
   gate;
5. verifies deterministic distributed NPZ provenance probes against the
   selected Qwen TSV;
6. confirms Full-Q and Half-Q TSVs differ only in `q_level`.

No training starts if any gate fails.

## Metrics

Both model stages are measured on all 5,521 MusicCaps prompts:

- Stage 1: FluxAudio, 25 flow-matching steps, CFG 4.5.
- Global: MeanAudio, one MeanFlow step, CFG 0.5.
- Primary comparison: Half-Q q9 minus Full-Q q9.
- Within-support diagnostics: Full-Q q9 versus q6, and Half-Q q9 versus q0.

The immutable result report is:

`/home/kojiek/logs/phase8_qwen_fullq_halfq_quarter_e2e_FINAL_METRICS.json`

The launch is wrapped by `scripts/run_with_experiment_report.sh`, so success,
failure, or interruption is reported to the configured Discord webhook.
