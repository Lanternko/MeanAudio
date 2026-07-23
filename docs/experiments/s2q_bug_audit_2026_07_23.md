# S2Q bug audit and reduced-scale rerun plan (2026-07-23)

## Verdict

The `0.1426` Stage-2 Real-Q result is **not** a valid test of aligned
MeanSimilarity-Q. Two independent implementation/data faults are active:

1. The TSV called Real-Q carries historical **row-position Q**, while its
   audio/caption rows were repaired to the actual NPZ catalog. The Q label
   therefore belongs to another clip for most rows.
2. The No-Q Stage-1 to Q Stage-2 migration preserved untrained random
   `q_embed[0..9]` rows instead of initializing them from the trained No-Q
   `q_embed[10]` row.

The poor result can therefore be explained without concluding that correct
MeanSimilarity-Q is harmful.

## Data-path evidence

The original MeanSimilarity-Q rule is:

```text
q_level = clamp(floor(mean_similarity * 10), 0, 9)
```

It is a fixed-width score discretization, not equal-frequency deciles. This was
verified exactly on all 218,977 Phase-7 rows whose original JSONL id was
recoverable: 218,977/218,977 labels matched the formula.

Recomputing Q from `credibility_analysis.mean_similarity` for the **actual
catalog clip id** in `phase8_legacy_catalog_train.tsv` gives:

| Check | Result |
|---|---:|
| Rows | 251,599 |
| Existing row-position Q equals actual-clip Q | 61,865 (24.59%) |
| Q labels requiring repair | 189,734 (75.41%) |
| Absolute Q error at least 2 | 86,867 (34.53%) |
| Mean absolute Q error | 1.220 |
| Pearson correlation: row-position Q vs actual-clip Q | 0.00195 |
| Pearson correlation: shuffled control vs actual-clip Q | 0.00112 |

The supposed Real-Q assignment is therefore statistically indistinguishable
from a shuffled assignment with respect to the actual clip. The nearly
identical histogram hid the problem.

Correct actual-clip Q histogram:

```text
q3=1, q4=306, q5=11,056, q6=49,174,
q7=74,184, q8=76,562, q9=40,316
```

There are no q0, q1, or q2 training samples, and only one q3 sample. A reported
0-to-9 quality axis is therefore not fully supported by this dataset.

### Manual row checks

| Row | Actual clip | Old Q | Correct Q | mean_similarity |
|---:|---|---:|---:|---:|
| 0 | `00_1014400_segment_2_0` | 8 | 6 | 0.653873 |
| 1 | `00_1014400_segment_3_0` | 8 | 6 | 0.626500 |
| 2 | `00_1014400_segment_4_0` | 8 | 5 | 0.599680 |
| 3 | `00_101900_segment_0_0` | 7 | 8 | 0.814610 |
| 4 | `00_101900_segment_1_0` | 7 | 5 | 0.598334 |
| 16 | `00_1022300_segment_4_0` | 8 | 4 | 0.494747 |
| 18 | `00_1027900_segment_1_0` | 5 | 9 | 0.903619 |

The full manifest retains 40 examples with the training caption and all source
candidate captions for manual review.

## Q injection and q=10 semantics

The current intended routing is:

```text
effective_q =
    row.q_level,  if use_q_conditioning is true
    10,           otherwise
```

For the proposed two-stage design:

```text
Stage 1: use_q_conditioning=false -> q10 for every row
Transition: copy q10 exactly into q0..q9
Stage 2: use_q_conditioning=true  -> corrected per-row q3..q9
```

`q=10` can represent No-Q **only when that model/path trained q10 with real
text**, as the current No-Q Stage 1 does. It is not universally interchangeable
with `--no_q` for a Q-trained model. In a Q-trained run, q10 is out of the
per-row support and is used only in the detached unconditional MeanFlow target.

The machine can easily distinguish q0 and q10 because they are different
embedding rows. In the old S2-only Real-Q transition:

| Checkpoint observation | Result |
|---|---:|
| No-Q S1 `L2(q0, q10)` | 31.768 |
| Real-Q S2 q0/q1/q2 change from No-Q S1 | exactly 0 |
| Real-Q S2 q3 change from No-Q S1 | 0.011 |
| Real-Q S2 q6/q7/q8/q9 changes | 6.527 / 5.406 / 5.686 / 5.286 |

Thus q0..q9 were not a neutral quality axis at the transition; they were
mostly random vectors far from the trained No-Q prior. The model also has no
way to discover that a Q label belongs to the wrong audio/caption row. Alignment
must be guaranteed by the dataloader input.

## GitHub/current-code comparison

`origin/main` was refreshed and is at
`fbf061d9bd95ef40bf3d94e1aa510d3ef8d95a9b` (same as local `HEAD`). The dirty
workspace was preserved.

Relevant historical differences:

- April Stage 1 did not pass `q_level` into FluxAudio; fixed by `39f8769`.
- Historical MeanAudio filled `q=None` with q9 instead of the null q10; fixed by
  `e214f0c`.
- Historical MeanFlow read `data['q_level']` whenever present and ignored the
  nominal No-Q flag; current runners respect `use_q_conditioning`.
- These historical semantics explain why copying an old flag name is not enough
  to reproduce an old condition.

The current model routing itself passes Q to both Stage-1 and Stage-2 networks.
The newly identified failures are the repaired-catalog/Q provenance mismatch
and the unneutral S1-to-S2 Q initialization.

## Repairs and artifacts

- `scripts/preprocess/align_meansim_q_to_catalog.py`
  - resolves Q from the actual catalog clip id;
  - validates the historical formula;
  - writes a hash/provenance manifest and manual-review examples;
  - never overwrites its input.
- Corrected aligned TSV:
  `/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train_meansim_aligned.tsv`
- Corrected manifest:
  `/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train_meansim_aligned.manifest.json`
- Corrected shuffled control:
  `/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train_meansim_aligned_q_shuffled_seed424242.tsv`
- `migrate_stage1_to_stage2_ckpt.py --q-init copy-null`
  copies q10 exactly into q0..q9 for online and both EMA tracks.
- Both training runners now record `train/q_fraction/0` through
  `train/q_fraction/10` in TensorBoard/log output, exposing the effective Q
  support instead of relying on the launcher contract.

## Quarter-scale experiment

`scripts/training_pipelines/train_pipeline_phase8_meansim_q_pilot.sh` implements
a bounded pilot:

- reuse the completed 400k No-Q Stage 1 (`q10`);
- run only 50k Stage-2 updates (one quarter of the old 200k S2);
- initialize q0..q9 exactly from q10;
- compare corrected aligned Q with a fixed-seed shuffled-Q control;
- evaluate q9 and q6, plus diagnostic q10 and q0.

Both arms pass full preflight without starting a GPU process:

```bash
ARM=aligned  PREFLIGHT_ONLY=true bash scripts/training_pipelines/train_pipeline_phase8_meansim_q_pilot.sh
ARM=shuffled PREFLIGHT_ONLY=true bash scripts/training_pipelines/train_pipeline_phase8_meansim_q_pilot.sh
```

The pilot was prepared but not launched because unrelated training/evaluation
jobs are currently active.

## Decision order

1. Run the corrected 50k aligned/shuffled pilot with neutral Q initialization.
2. Interpret Q information as useful only if aligned beats both shuffled and
   the matched No-Q reference on paired prompts.
3. Only after the repaired 10-level pathway is measured, test a two-bin axis.
   Because the historical mapping is fixed-width and leaves q0..q2 empty, the
   two-bin test should use a median/quantile split with balanced support, not
   `floor(mean_similarity * 2)`.
