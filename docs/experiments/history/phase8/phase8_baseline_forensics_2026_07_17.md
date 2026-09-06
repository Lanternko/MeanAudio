# Phase 8 baseline forensics (2026-07-17)

## Outcome

The July `phase8_bugfix_rerun` result (`MusicCaps CLAP 0.0615`) is not a clean
Phase-8 baseline.  Its rebuilt cache paired Phase-7 row captions with audio
latents selected by `npz_cache_train.txt`, but that cache list is not aligned
to the Phase-7 TSV IDs.

The historical `0.1851` output is real, but its old label
`Random-NoQ baseline` is inaccurate.  The April run consumed the text features
already stored in each original NPZ and also consumed `q_level` despite
`use_q_conditioning=false`.

## Decisive evidence

1. Historical Phase-4 preprocessing extracted the original NPZ cache with
   `_QUARANTINED_meanaudio_captions.tsv` (then named
   `meanaudio_captions.tsv`).  The authoritative NPZ-index-to-caption catalog
   survives as `_QUARANTINED_npz.tsv`.
2. Phase-7 and Phase-8 training logs point at the original NPZ directory; their
   pipelines did not re-encode text after changing the Hydra TSV.
3. Comparing Phase-7 rows to the original extraction catalog through
   `npz_cache_train.txt` gives:

   - selected rows: 251,599
   - ID matches after normalizing the extraction `_0` suffix: 2
   - mismatches: 251,597 (**99.9992%**)

   Example: Phase-7 row 0 is `94_1317594_segment_0`, but the cache selects
   `33.npz`, whose extraction-catalog ID is `00_1014400_segment_2_0`.
4. A fixed 128-row probe decoded the same cached audio latents once and scored
   those identical reconstructions against the two competing caption maps:

   | Caption reference | CLAP |
   |---|---:|
   | Original NPZ extraction catalog (audio/text matched) | **0.2047** |
   | July clean Phase-7 row mapping (audio/text mismatched) | **0.0442** |

   Delta: `+0.1605`; the matched mapping is 4.6x higher.
5. Re-evaluation of the first 100 generated files with the same current
   evaluator also separated the checkpoints: historical audio `0.1910`, July
   rerun audio `0.0734`.  Therefore the evaluator did not manufacture the
   historical gap.
6. At historical commit `d59e3bbc`, `runner_meanflow.py` passed
   `data['q_level']` whenever present and did not consult
   `use_q_conditioning`.  The Phase-7 TSV has `q_level`, so the April run was
   effectively Q-conditioned during training.  The current runner correctly
   respects the flag, which is another historical/current difference.
7. A deterministic medium gate sampled 4,096 filenames across the full cache
   (seed `20260717`).  The four cache-position quartiles contributed
   `1043/970/1059/1024` rows.  Exhaustive validation passed for all 4,096 rows:

   - source/output audio mean and std were bit-identical;
   - TSV, cache, manifest, embedded clip ID, caption hash, and historical Q
     provenance all agreed;
   - 512 randomly selected decoded latents scored **CLAP 0.2014** against the
     catalog-matched captions, consistent with the 128-row result (`0.2047`).
8. The exact configurable launcher then completed a short end-to-end gate on
   those 4,096 rows: 100 Stage-1 updates, Stage-1-to-Stage-2 migration, 100
   Stage-2 updates, EMA synthesis, 64 generations, and CLAP evaluation.  Live
   configs confirmed Q conditioning enabled and the legacy 77-token NoMask
   path.  The short-run generated-audio CLAP (`-0.0664`) is intentionally not
   a quality result: 200 updates from random initialization cannot measure
   convergence.  It only proves the entire execution path is wired and
   resumable.

## Mask and stage controls already ruled out

- Same-checkpoint Mask/NoMask inference and 2k repair probes did not recover
  CLAP; NoMask was usually worse.
- Stage-1 and Stage-2 mask probes independently rejected the mask hypothesis.
- The stage-switch detector had a real stale-string bug, now fixed, but direct
  Stage-1/Stage-2 loss equivalence for FluxAudio was exact because FluxAudio
  ignores `r`.

## Reproduction path

- `scripts/preprocess/rebuild_phase8_legacy_npz.py` reconstructs the actual
  historical NPZ audio/text pairing from `_QUARANTINED_npz.tsv`, embeds
  `clip_id`, `caption_sha256`, `catalog_index`, and preserves the historical
  row-position `q_level` provenance.
- `scripts/training_pipelines/train_pipeline_phase8_legacy_repro.sh` uses the
  catalog-matched cache, Q conditioning, and the legacy NoMask path.  It is
  intentionally blocked until the full provenance-backed cache exists.
- `scripts/training_pipelines/train_pipeline_phase8_legacy_medium_gate.sh`
  reproduces the end-to-end 4,096-row integration gate without launching a
  long experiment.
- `scripts/monitor_phase8_legacy_repro.py` is the fail-closed live watcher.  It
  verifies the immutable cache/gate hashes, live Hydra configuration, process
  liveness, log freshness, finite loss/gradient values, and hard runtime
  errors; an alert terminates the training process group.
- `scripts/training_pipelines/run_phase8_legacy_guarded.sh` owns the complete
  state machine: full cache build, structural gate, decoded-audio semantic
  gate, Stage 1, migration, Stage 2, and MusicCaps evaluation.
- `scripts/audit_phase8_legacy_repro.py` is the independent completion audit.
  It does not accept the mere existence of `metrics.txt`: both resumable
  checkpoints and final EMA weights must load, the Stage-1/Stage-2 configs
  must match the legacy target, all 5,521 unique MusicCaps FLACs must be
  present/readable, the NoQ+NoMask evaluation arguments must be recorded, and
  all CLAP/AES values must be finite.  A CLAP delta greater than 0.03 from the
  historical 0.1851 is held for investigation rather than declared complete.
- `train_pipeline_phase8_bugfix_rerun.sh` now rejects NPZs without embedded
  clip and caption provenance.  Agreement between a TSV, a cache list, and a
  manifest is no longer accepted as proof of audio/text alignment.

## Guarded full run status

The guarded full cache rebuild completed with 251,599 files.  Exhaustive
row/file validation plus a 4,102-row deep probe passed, and the independent
512-row decoded-cache semantic gate scored CLAP **0.1998** (required minimum
0.15).  The full 400k Stage-1 + 200k Stage-2 pipeline was then launched
automatically.  Its first 20k resumable checkpoint was deserialized on CPU and
contained finite model, optimizer, scheduler, and EMA state at exactly
iteration 20,000.  Training remains under the watcher; final baseline
completion is intentionally unclaimed until the strict completion audit
passes after Stage 2 and MusicCaps evaluation.

The primary MusicCaps evaluation generates and evaluates all **5,521** TSV
records.  The pipeline's `--num_samples 2048` argument does not truncate CLAP
or AES in `phase4_eval.py`; it applies only to optional FAD, which this run does
not request.

## Addendum (2026-07-19): the `--no_q` evaluation was invalid, not the training

The completed run's pipeline evaluation reported MusicCaps CLAP **0.0134**
(CE 3.17, PQ 4.88) and the strict audit correctly held it.  Investigation
showed the collapse was produced by the evaluation arguments, not by the
model:

1. The shared pipeline (`train_pipeline_phase8_bugfix_rerun.sh`) hardcoded
   `--no_q` in both eval blocks regardless of `USE_Q_CONDITIONING`.  The
   legacy repro trains with Q enabled in both stages, so its evaluation used
   the null token `q=10`.
2. Under the fixed runners, `q=10` occupies a different position than in the
   historical code.  Stage 1 (`runner_flowmatching.py`) passes the row
   `q_level` for every sample, including text-dropped CFG rows, so `q=10`
   never occurs in Stage 1.  Stage 2 (`mean_flow.py:157`) uses
   `q=torch.full_like(q, 10)` only in the CFG-unconditional pass, which also
   uses null text.  The repro model therefore learns `q=10` as an
   "unconditional generation" marker.
3. Historically the reverse held: the broken April Stage-1 runner never passed
   `q`, so FluxAudio filled `q=10` for all 400k Stage-1 updates **with real
   text**, making it the universal default embedding, and the Stage-2
   unconditional pass filled 9 (the fill-9 bug), never 10.  That is why the
   historical `--no_q` evaluation (0.1851) worked and the repro's did not.
4. Paired probe, same checkpoint, same 512 MusicCaps prompts:
   `--quality_level 9` → CLAP **0.1716** (CE 5.49, PQ 6.57);
   `--no_q` → CLAP **0.0101** (CE 3.18, PQ 4.88).
5. Full-set corrected evaluation (5,521 records, `--quality_level 9`,
   NoMask): CLAP **0.1684**, CE 5.36, CU 6.59, PC 4.79, PQ 6.49.

The faithful historical comparison for the corrected condition is the
2026-04-17 `--quality_level 9` measurement **0.1907**, not the `--no_q`
0.1851, whose legacy semantics (`q=10` as universal default) cannot be
reproduced under the fixed runners.  Delta `0.1684 − 0.1907 = −0.0223` is
within the audit's ±0.03 review threshold, and its magnitude matches the
established Stage-1-effective-q-training penalty (~0.02) from the P7 full-Q
control (2026-04-24), which the repro incurs (fixed Stage-1 runner trains q)
and the historical run did not.

Corrections applied:

- `train_pipeline_phase8_bugfix_rerun.sh` now derives the eval q flag from
  `USE_Q_CONDITIONING` (`true` → `--quality_level 9`, `false` → `--no_q`), so
  the legacy repro and the catalog-matched NoQ variants are each evaluated
  consistently with their training.
- `audit_phase8_legacy_repro.py` now compares against 0.1907 and requires
  `'quality_level': 9` / `'no_q': False` in the recorded eval arguments.
- The invalid `--no_q` artifacts were preserved as
  `eval_output/phase8_legacy_repro_stage2_200000_musiccaps_noq_invalid` (plus
  the matching `metrics/` dir and `*_eval_noq_invalid.log`), and the canonical
  evaluation was regenerated under the corrected arguments.

Lesson recorded in memory (`project_legacy_repro_noq_eval_trap_2026_07_18.md`):
when reproducing a bug-era run, every flag must be translated to the semantics
it had in the historical code, not copied by name.
