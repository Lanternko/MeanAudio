# scripts/

Helper scripts for training, evaluation, preprocessing, and analysis.
Canonical entrypoints (`train.py`, `eval.py`, `infer.py`, `train_pipeline.sh`) stay at the repo root.

All shell scripts here `cd "$WORK_DIR"` (= `$HOME/MeanAudio`) before invoking `train.py`/`eval.py`, so they work from any cwd. Invoke as `bash scripts/<subdir>/<script>.sh`.

---

## Subdirectories

### `training_pipelines/` — experiment-specific train pipelines

Variants of `train_pipeline.sh` for specific Phase experiments. Each is a self-contained two-stage runner (S1 → migrate → S2 → eval). Canonical generic pipeline remains at repo root as `train_pipeline.sh`.

| Script | Experiment |
|--------|------------|
| `train_pipeline_destructured.sh` | Generic destructured variant (untracked, in-progress 2026-05-08) |
| `train_pipeline_p4v2_qwen.sh` | P4V2-Qwen — Qwen single-cap BC, NoQ |
| `train_pipeline_p7v1_qwen.sh` | P7V1-Qwen — Qwen single-cap Random, +Q |
| `train_pipeline_p8_qwen.sh` | P8-Qwen — Qwen single-cap Random, NoQ |
| `train_pipeline_phase8_bugfix_rerun.sh` | Phase 8 rerun after q=None bug fix |
| `train_pipeline_phase8v4_q.sh` | P8V4 + Q (S2-only Q variant, `[consistency=X.XX]` prefix) |
| `train_pipeline_phase9_5_v1.sh` | P9.5 V1 — Qwen task-framed multi-cap, NoQ |
| `train_pipeline_phase9_5_v2.sh` | P9.5 V2 — Qwen multi-cap + Q (SKIPPED, V1 failed gate) |
| `train_pipeline_phase9_v1.sh` | P9 V1 — LP-MC multi-cap random, NoQ |
| `train_pipeline_phase9_v1_ablation_s1fixed_s2multi.sh` | S1 fixed-caption + S2 multi-cap ablation |
| `train_pipeline_phase9_v1_bugfix_rerun.sh` | P9 V1 rerun after q=None bug fix |
| `train_pipeline_phase9_v1_s2_salvage.sh` | S2-only salvage from corrupted run |
| `train_pipeline_phase9_v2.sh` | P9 V2 — multi-cap + Q (full Q) |
| `train_pipeline_phase9_v2_bugfix.sh` | P9 V2 rerun after q=None bug fix |

### `eval/` — eval batch scripts

Multi-q sweeps, baseline reruns, p100/p090 ablations. Each calls `python eval.py` then `~/research/meanaudio_eval/phase4_eval.py` for CLAP/AES/PE-AV metrics.

| Script | Purpose |
|--------|---------|
| `eval_p7v1_qsweep_musiccaps.sh` | P7V1 q=0..9 sweep on MusicCaps |
| `eval_p8_noq_baseline_rerun.sh` | P8 NoQ baseline rerun (clean comparison) |
| `eval_p8_q10_sanity.sh` | P8 null-token (q=10) sanity check |
| `eval_p8_qsweep_musiccaps.sh` | P8 q=0..9 sweep on MusicCaps |
| `eval_p8v4_noq_p090_backfill_prefixed_ref.sh` | P8V4 NoQ p=0.90 (in-support) backfill |
| `eval_p8v4_noq_p100.sh` | P8V4 NoQ p=1.00 (OOD edge) |
| `eval_p8v4_noq_p100_peav.sh` | P8V4 NoQ p=1.00 with PE-AV metrics |
| `eval_p8v4_noq_qsweep_control_musiccaps.sh` | P8V4 NoQ q-sweep control |
| `eval_p8v4_q.sh` | P8V4 +Q q-sweep on both benchmarks |
| `eval_phase9_v1_bugfix_rerun_jamendo.sh` | P9 V1 bugfix rerun on Jamendo |

### `preprocess/` — data prep helpers

Caption sampling, text re-extraction, TSV generation, A/B normalization.

| Script | Purpose |
|--------|---------|
| `gen_phase8v5_tsv.py` | Generate Phase 8 V5 training TSV |
| `reextract_text_phase8v4.py` | Re-encode Qwen2-Audio captions into NPZ text_features |
| `reextract_text_phase8v5.py` | Re-encode Phase 8 V5 captions |
| `sample_musiccaps_v3.py` | Sample MusicCaps prompts for v3 subjective A/B |
| `sample_musiccaps_v3_extend.py` | Extend v3 prompt set |
| `write_metadata_ab.py` | Write `metadata.json` for subjective A/B |
| `write_metadata_v3.py` | Write `metadata.json` for v3 A/B |
| `normalize_ab.py` | Peak-normalize subjective A/B WAVs to −1 dBFS |
| `normalize_ab_v3.py` | Same, v3 |

### `analysis/` — post-hoc metric & probe analysis

| Script | Purpose |
|--------|---------|
| `aes_subjective_v4.py` | Audiobox Aesthetics scoring for subjective_ab_v3 |
| `clap_subjective_v4.py` | CLAP scoring for subjective_ab_v3 |
| `probe_battery_results.json` | 375-state determinism/steering probe results (gitignored) |

### `legacy/` — kept-for-reference scripts

One-off or superseded scripts retained for traceability of past experiments.

| Script | Purpose |
|--------|---------|
| `babysit_fullpipe.sh` | Watchdog for full S1→S2→eval pipeline |
| `chain_after_salvage.sh` | Chain runs after S2 salvage |
| `disk_cleanup_plan.sh` | Disk-cleanup plan (executed 2026-04-18) |
| `probe_v1_steering.sh` | Steering-probe v1 driver |
| `strict_bc_audit.sh` | Best-consensus (BC) caption audit |

### `runs/` — disposable one-off run scripts (gitignored)

`run_*.sh` files were already gitignored at repo root. Moved into this subdir; pattern in `.gitignore` still matches at any depth. Use as scratchpad for tmux-launched jobs.

### `flowmatching/`, `meanflow/` — minimal Flow-Matching / MeanFlow demo runners

Pre-existing demo runners (`train_`, `eval_`, `infer_`). Untouched.

### `train_mini.sh`, `extract_audio_latents.sh`

Pre-existing utilities at scripts root. Untouched.

---

## Adding a new script

1. Pick the right subdir (training_pipeline, eval, preprocess, analysis, or legacy).
2. Make sure the script starts with `set -eo pipefail` and either `cd "$HOME/MeanAudio"` or uses absolute paths.
3. Add a one-line entry to the table above.
4. If it's a one-off / disposable, drop it in `runs/` (already gitignored).
