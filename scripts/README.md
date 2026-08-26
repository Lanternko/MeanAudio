# scripts/

Helper scripts for training, evaluation, preprocessing, and analysis.
Canonical entrypoints (`train.py`, `eval.py`, `infer.py`, `train_pipeline.sh`) stay at the repo root.

All shell scripts here `cd "$WORK_DIR"` (= `$HOME/MeanAudio`) before invoking `train.py`/`eval.py`, so they work from any cwd. Invoke as `bash scripts/<subdir>/<script>.sh`.

---

## Subdirectories

### `training_pipelines/` — experiment-specific train pipelines

Variants of `train_pipeline.sh` for specific Phase / caption experiments. Each is a self-contained two-stage runner (S1 → migrate → S2 → eval). Canonical generic pipeline remains at repo root as `train_pipeline.sh`.

#### Historical Phase / Qwen / EXP

| Script | Experiment |
|--------|------------|
| `train_pipeline_destructured.sh` | EXP-A style destructured LP-MC |
| `train_pipeline_p4v2_qwen.sh` | P4V2-Qwen — Qwen single-cap BC, NoQ |
| `train_pipeline_p7v1_qwen.sh` | P7V1-Qwen — Qwen single-cap Random, +Q |
| `train_pipeline_p8_qwen.sh` | P8-Qwen — Qwen single-cap Random, NoQ |
| `train_pipeline_phase8_bugfix_rerun.sh` | **Active 2026-07-16** — P8 NoQ retrain after q=None→10 fix |
| `train_pipeline_phase8v4_q.sh` | P8V4 + Q (S2-only Q variant, `[consistency=X.XX]` prefix) |
| `train_pipeline_phase9_5_v1.sh` | **Historical invalid cache; 0.0609 excluded** |
| `train_pipeline_phase9_5_v2.sh` | **Historical invalid cache/gate; skipped** |
| `train_pipeline_phase9_v1.sh` | **Historical invalid cache; do not rerun/cite** |
| `train_pipeline_phase9_v1_ablation_s1fixed_s2multi.sh` | **Historical invalid multi-cap cache** |
| `train_pipeline_phase9_v1_bugfix_rerun.sh` | **Historical only — invalid misaligned multi-cap cache; do not rerun/cite** |
| `train_pipeline_phase9_v1_s2_salvage.sh` | **Historical invalid S1; do not use for attribution** |
| `train_pipeline_phase9_v2.sh` | **Historical invalid cache; do not rerun/cite** |
| `train_pipeline_phase9_v2_bugfix.sh` | **Historical invalid cache; 0.0403 excluded** |
| `train_pipeline_qwen_slot0_masked.sh` | EXP-B style Qwen slot-0 fixed |
| `train_pipeline_expH_rewrite.sh` | EXP-H — Qwen→LP-MC-style rewrite（collapsed MC CLAP 0.0617） |
| `train_pipeline_slice10_from_tsv.sh` | Generic slice10 TSV → S1/S2 helper |

#### Music Flamingo / LP-MC controls

| Script | Experiment |
|--------|------------|
| `train_pipeline_music_flamingo_10k.sh` | MF original verbose 10k NoQ |
| `train_pipeline_music_flamingo_100k.sh` | MF original verbose 100k NoQ |
| `train_pipeline_music_flamingo_short_rewrite_10k.sh` | A1 MF-short-rewrite 10k |
| `train_pipeline_music_flamingo_short_direct_10k.sh` | A2 MF-short-direct 10k |
| `train_pipeline_music_flamingo_short_direct_100k.sh` | A3 MF-short-direct 100k |
| `train_pipeline_music_flamingo_static_random_3cap_10k.sh` | A4 MF-static-random-3cap 10k |
| `train_pipeline_music_flamingo_static_random_3cap_100k.sh` | A5 MF-static-random-3cap 100k |
| `train_pipeline_music_flamingo_expanded_3cap_100k.sh` | A6 MF-expanded-3cap 100k-audio / 300k-caption |
| `train_pipeline_lpmc_10k_control.sh` | LP-MC 10k control matched to MF recipe |
| `train_pipeline_lpmc_100k_control.sh` | LP-MC 100k control matched to MF recipe |

#### Queue / monitor helpers

| Script | Purpose |
|--------|---------|
| `schedule_mfstatic3cap10k.sh` | Queue A4 |
| `schedule_mfstatic3cap100k_after_eval.sh` | Queue A5 after prior eval |
| `schedule_mfexpanded3cap100k.sh` | Queue A6 |
| `schedule_music_flamingo_todos.sh` | MF ablation batch queue |
| `schedule_music_flamingo_open_todos.sh` | Remaining MF eval-only queue |
| `monitor_lpmc10k_then_lpmc100k.sh` | Chain LPMC 10k → 100k |
| `monitor_lpmc100k_then_mfshort10k.sh` | Chain LPMC100k → MF-short 10k |
| `monitor_mf10k_then_next.sh` | Chain after MF10k |

Root orchestration for paper baseline: `../run_phase8_bugfix_full.sh`（re-extract clean NPZ → `train_pipeline_phase8_bugfix_rerun.sh`）。

### `eval/` — eval batch scripts

Multi-q sweeps, baseline reruns, Music Flamingo prompt-style evals. Each calls `python eval.py` then `~/research/meanaudio_eval/phase4_eval.py` (and often PE-AV).

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
| `eval_phase9_v1_bugfix_rerun_jamendo.sh` | P9 V1 bugfix rerun on Jamendo（數字已失效） |
| `eval_mfstatic3cap10k_shortdirect_jamendo.sh` | A4 ckpt × short-direct MF prompts |
| `eval_mfstatic3cap10k_original_mfcap_jamendo.sh` | A4b original verbose MF prompts |
| `eval_mfshort100k_direct_shortdirect_jamendo.sh` | A3 matched short-direct prompts |
| `eval_mfshort100k_direct_mfstyle_jamendo.sh` | A3 × original MF-style prompts |
| `eval_mfexpanded3cap100k_shortdirect_jamendo.sh` | A6 eval-only short-direct MF prompts |
| `eval_lpmc100k_shortdirect_mfcap_jamendo.sh` | C1 LPMC100k × short-direct MF prompts |
| `reverse_control_mf400_prompts.sh` | Reverse control with MF prompts |

### `preprocess/` — data prep helpers

Caption sampling, text re-extraction, TSV generation, Music Flamingo prep, A/B normalization.

| Script | Purpose |
|--------|---------|
| `gen_phase8v5_tsv.py` | Generate Phase 8 V5 training TSV |
| `reextract_text_phase7_clean.py` | Clean P8 bugfix NPZ text re-extract (no prefix) |
| `reextract_text_phase8v4.py` | Re-encode Qwen2-Audio / P8V4 captions into NPZ text_features |
| `reextract_text_phase8v5.py` | Re-encode Phase 8 V5 captions |
| `make_phase8_shuffled_q_tsv.py` | Deterministic q-only permutation for the Phase8 S2 Real-Q vs Shuffled-Q control |
| `sample_musiccaps_v3.py` | Sample MusicCaps prompts for v3 subjective A/B |
| `sample_musiccaps_v3_extend.py` | Extend v3 prompt set |
| `write_metadata_ab.py` | Write `metadata.json` for subjective A/B |
| `write_metadata_v3.py` | Write `metadata.json` for v3 A/B |
| `normalize_ab.py` | Peak-normalize subjective A/B WAVs to −1 dBFS |
| `normalize_ab_v3.py` | Same, v3 |
| `music_flamingo_jamendo_slice_caption.py` | Music Flamingo captioning driver |
| `music_flamingo_jamendo_slice10_10k.sh` | MF 10k slice caption job |
| `prepare_music_flamingo_slice10_train.py` | Build MF slice10 train TSV/NPZ inputs |
| `prepare_music_flamingo_static_random_3cap.py` | A4/A5 static-random 3-cap TSV |
| `prepare_music_flamingo_expanded_3cap.py` | A6 expanded 3-cap (300k-caption) prep |
| `prepare_lpmc_slice10_control.py` | LP-MC control matched to MF IDs |
| `rewrite_music_flamingo_short_captions.py` | A1 post-hoc short rewrite |
| `build_music_flamingo_lpmc_slice_review.py` | MF vs LP-MC slice review helper |

### `analysis/` — post-hoc metric & probe analysis

| Script | Purpose |
|--------|---------|
| `aes_subjective_v4.py` | Audiobox Aesthetics scoring for subjective_ab_v3 |
| `clap_subjective_v4.py` | CLAP scoring for subjective_ab_v3 |
| `compute_slice_caption_clap.py` | Slice-level caption CLAP comparison |
| `qwen_slice10_caption.py` | Qwen captions on MF slice10 IDs |
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

1. Pick the right subdir (training_pipelines, eval, preprocess, analysis, or legacy).
2. Make sure the script starts with `set -eo pipefail` and either `cd "$HOME/MeanAudio"` or uses absolute paths.
3. Add a one-line entry to the table above.
4. If it's a one-off / disposable, drop it in `runs/` (already gitignored).

## Text overlay：新 arm 一律先查能不能複用（2026-08-26）

Text overlay 只存 T5 特徵，**一份 251,599 檔的單 caption overlay = 76 GiB**。
新增 arm 前先判斷它是「新 caption 來源」還是「既有 caption 的選擇規則」：

| 情況 | 做法 | 成本 |
|---|---|---|
| 選擇規則（bestof3 / worstof3 / random pick / balanced…）<br>caption 已在某個 stacked overlay 的 slot 裡 | `add_cap_index_column.py` 產 `cap_index` 欄<br>+ contract 指 stacked overlay + `cap_index_column` | ~30 MB、數秒 |
| 真的是新 caption 來源（新 prompt / 新 captioner） | 才編碼新 overlay | 76 GiB、數小時 |

判斷方法：算 caption 的 `sha256(caption.encode('utf-8'))`，看它在不在目標 stacked overlay 的
`caption_sha256`（0-d 逗號串接）之內。全中就能複用。

- 現有 stacked overlay：`~/text_overlays/true_random`，slot order **slot0, slot1, slot3**（**不含 slot2**）
- loader 支援：`ExtractedAudio(cap_index_fixed=N)` 或 `cap_index_column='cap_index'`，
  帶 `caption_sha256` binding guard，配錯會直接報錯
- 已套用範例：022 `..._fair013_bestof3_capidx_train.tsv`
- 2026-08-26 依此刪除 `worst013` / `fake_random` 兩份重複 overlay（回收 152 GB），
  紀錄見 `docs/experiments/text_overlay_dedup_2026_08_26.json`
