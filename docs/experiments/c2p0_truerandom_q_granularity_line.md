# Q resolution on a rotating caption pool (013 true-random × K∈{3,10})

**Status (2026-09-09):** 048 / 049 authored, contracts pre-registered, both
accepted by `accept_guest`, installed in `p2/pending` behind 047. Nothing has
trained yet.

## The question

Two knobs have never been crossed in this project:

- every **Q** arm conditioned on a single, fixed caption per clip;
- every **rotation** arm (012, 013, 034, 046) trained with **NoQ**.

So we do not know whether a per-clip quality code still carries signal once the
caption attached to that clip changes every epoch — or whether the rotation
already supplies whatever regularisation Q was providing.

The ablation is over the **resolution** of the code, not its presence:

| Arm | Grid source | q codes | Rows per code |
|---|---|---|---|
| 048 `qk3b` | `phase8_qwen_meansim_k3_balanced.tsv` | `{0, 5, 9}` | 83,866 / 83,866 / 83,867 |
| 049 `qk10b` | `phase8_qwen_meansim_k10_balanced.tsv` | `{0…9}` | 25,159–25,160 each |

Both come unchanged from the 2026-07-24 bucket grid: same signal (actual-clip
`credibility_analysis.mean_similarity`), same `q_code_policy =
round-half-up(index*9/(K-1))` with the endpoints pinned at q0/q9, same 251,599
rows. Only the number of cuts moves.

## Why `balanced` and not `fixed`

`fixed` is equal-width on [0,1], and the signal is not uniform there:

| | occupied codes | note |
|---|---|---|
| `k3_fixed` | 2 of 3 | **q0 has zero rows** |
| `k10_fixed` | 7 of 10 | = the historical Full-Q decile TSV |
| `k3_balanced` | 3 of 3 | |
| `k10_balanced` | 10 of 10 | |

Under `fixed`, occupancy collapses by a *different* amount at each K — exactly
the confound a K ablation must not carry. `balanced` is equal-frequency, so both
arms are fully occupied and K is the only moving part. (`k5_balanced` vs
`k5_fixed` was run against each other on the single-caption corpus for precisely
this reason; that stays a diagnostic, not part of this claim.)

## Data

`scripts/preprocess/build_c2p0_truerandom_q_tsvs.py` copies `q_level`
row-for-row from the grid TSVs onto `k3_true_random_train.tsv`. Nothing is
recomputed. The join is positional and guarded by per-row id equality across all
251,599 rows (the two files were independently verified to carry the same ids in
the same order), plus a histogram-equality check against the grid manifest.
Outputs and hashes: `docs/experiments/c2p0_truerandom_q_tsvs.manifest.json`.
Zero new overlay bytes — the 3-caption stack at `~/text_overlays/true_random` is
reused as-is.

## Comparators

| Arm | Scale | Conditioning | CLAP |
|---|---|---|---|
| 013 true-random NoQ | quarter | `--no_q` | **produced by 048 Step 0** |
| 013 true-random NoQ | full | `--no_q` | 0.2221 |
| 012 true-random NoQ | quarter | `--no_q` | 0.2053 |
| c2p0 slot0 + Q k3_balanced (single caption) | full | `--quality_level 9` | 0.2126 |

The same-scale NoQ number does not exist: the 013 true-random NoQ arm was
trained at quarter on 2026-08-26 but only ever evaluated at full. 048 Step 0
evaluates that existing EMA — no training, ~25 min — so both Q arms have an
honest budget-matched control.

## Decision rule (pre-registered)

Primary metric: MusicCaps 5521 / MeanFlow 25 / CFG 0 / seed 42 / NoMask / full
precision / `--quality_level 9`, `clap_score`. CFG0 training-seed floor for CLAP
is 0.0042, so the band is ±0.0084.

1. **Q-response first.** `|q9 − q0| ≤ 0.0084` ⇒ the model ignored the code.
   Two arms that both ignore their code cannot be ranked by resolution.
2. **vs NoQ.** `≥ NoQ_quarter + 0.0084` ⇒ Q buys something on top of rotation.
   Inside the band ⇒ rotation already supplies it. `≤ −0.0084` ⇒ conditioning on
   a per-clip code while the caption rotates hurts.
3. **Cross-arm.** K=10 beats K=3 only if their q9 CLAP differ by more than
   0.0084. Inside that band, the answer is *resolution does not matter here* —
   which is the finding, not a win for either arm.

The four AES metrics are reported alongside CLAP against their own 2× floors
(CE 0.1343, CU 0.052, PC 0.0554, PQ 0.0523).

## Known caveats (recorded before launch)

- **S1-Q never trains `q_embed[10]`** (memory
  `project_q_null_token_never_trained_bug`). Stage 2's MeanFlow CFG target uses
  q=10 as its unconditional code, so both arms carry the same untrained-null
  contamination. Shared across K=3 and K=10, so it does not confound the
  resolution comparison — but neither absolute number is cleanly comparable to a
  NoQ arm without this stated.
- **`q_level` is a property of the clip, not of the rotated caption.** A clip
  whose three captions disagree carries the same code whichever is drawn this
  epoch. That is the only way to cross the two knobs at zero encoding cost, and
  it is an assumption, not a measurement.
- **S1 100k = 3.18 epochs**, so the rotation reaches 3·(1−(2/3)^3.18) = 2.19 of
  3 captions per clip. Inherited from every quarter rotation arm.
- The grid's signal is Qwen n-caption `mean_similarity`.
  `project_mean_sim_interpretation_hypothesis` and
  `reference_p7v1_q_support_gating` both argue it is a coarse support-set marker
  rather than a quality measure, so a null here is evidence about *resolution*,
  not about quality conditioning in general.

## Adjacent, not part of this line

The single-caption Qwen bucket quarter grid
(`docs/experiments/phase8_qwen_bucket_quarter_backlog_2026_07_26.md`) has
**trained EMAs for `k3_balanced`, `k5_balanced`, `k10_balanced` and `noq`, but
CFG0 numbers only for `k5_balanced` and `noq`.** The K-resolution claim on that
corpus is two ~25-min evaluations away from existing on disk. That is the
single-caption counterpart to this line and would make the 2×2 (rotation ×
resolution) readable, but it is not scheduled here.

## Files

| | |
|---|---|
| Data builder | `scripts/preprocess/build_c2p0_truerandom_q_tsvs.py` |
| Data manifest | `docs/experiments/c2p0_truerandom_q_tsvs.manifest.json` |
| Shared action | `scripts/training_pipelines/c2p0_truerandom_q_action.sh` |
| Wrappers | `scripts/training_pipelines/c2p0_truerandom_q{k3,k10}_quarter.sh` |
| Contracts | `docs/experiments/c2p0_truerandom_q{k3,k10}_quarter_cfg0_contract.json` |
| Queue | `scripts/queue_candidates/04{8,9}_c2p0_truerandom_q{k3,k10}_quarter.sh` |
