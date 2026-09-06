# Fulltrack Q3 PQ audit

Audit ID: `FTQ3-AUDIT-A-v1`  
Envelope: `FTQ3-PQ-AUDIT-v1` (`3679e29b14e760aec4f74c283a73744fe3cc3bf74e1545134f928c836f2c9843`)

## Scope and protocol

This is an additive, evidence-only audit. It does not edit or retire any historical
contract, report, checkpoint, result table, queue entry, or runtime state.

The historical result under review is the Q-conditioned fulltrack checkpoint at
`q=9`, evaluated on MusicCaps 5,521 with MeanFlow 25, literal CFG 0, generation
seed 42, NoMask, and full precision:

| CLAP | CE | CU | PC | PQ |
|---:|---:|---:|---:|---:|
| 0.1821 | 6.8458 | 7.1468 | 5.3016 | 6.9337 |

## Findings

| ID | Verdict | Claim | Evidence and limitation |
|---|---|---|---|
| F01 | REFUTED | Stale-audio contamination explains this historical evaluation run. | The frozen eval log has exactly one `Eval args`, 5,521 `Audio saved` records, and 5,521 unique saved FLAC paths. This refutes that specific explanation for this run, but does not prove byte-level reproducibility. |
| F02 | INCONCLUSIVE | The historical evaluation is reproduced at byte level. | The report and aggregate metrics remain, but the generated audio was deleted and no per-audio hash manifest was preserved. |
| F03 | CONFIRMED | The historical Caption 2.0/slot012 contract identity was a provenance mislabel and the current contract now records the correction. | The bound checkpoint's training log uses `phase8_qwen_meansim_k3_balanced.tsv`: a Q-conditioned upstream track-level caption corpus, distinct from the per-segment Caption 2.0 slot0 Q3 checkpoint. The current contract now identifies the fulltrack Q3 eval and preserves the historical run identifiers only as identifiers. The historical metric vector remains unchanged; this audit makes no authorship claim for the concurrent correction. |
| F04 | CONFIRMED | The fulltrack corpus has a track-to-10-second caption granularity mismatch. | One upstream track caption was broadcast to retained segments while each row represents the segment's first approximately 10 seconds. This is not cross-track mismatch. |
| F05 | REFUTED | Available evidence proves captions crossed between different tracks. | The mapper keys captions by the same local track identity before broadcasting within that track. Historical row-level NPZ binding remains a separate uncertainty. |
| F06 | INCONCLUSIVE | The historical TSV-to-NPZ text binding was correct for every training row. | The run used `require_text_overlay=false`, and the original NPZ directory was later overwritten in place. Current files can neither prove nor disprove the historical binding. |
| F07 | REFUTED | The Q pathway was unused or the Q3 result came from random, uninitialized quality embeddings. | The MeanFlow runner consumes `q_level` when Q conditioning is enabled. The migration record proves q0–q9 were bit-exact copies of trained q10 before Stage 2. This does not prove Q caused the gain. |
| F08 | REFUTED | Same-track caption sharing alone is sufficient to produce the PQ gain. | The R-Shared quarter control lowered CLAP from 0.1916 to 0.1703 and changed PQ from 6.3911 to 6.3836 rather than raising it. Rich-caption/NoQ/quarter/MF1 differences leave interactions unresolved. |
| F09 | INFERENCE | The table is consistent with an aesthetic-versus-prompt-adherence tradeoff and possible optimization toward a no-reference aesthetic predictor. | After deduplicating by `exp_label` (16 unique rows), Pearson correlations with PQ are CE 0.969184, CU 0.987095, PC 0.459229, and CLAP −0.126996; Worst arms repeatedly trade lower CLAP for higher PQ. Audiobox Aesthetics defines CE, CU, PC, and PQ as independent no-reference per-item axes, not prompt-alignment metrics. Correlation and model scores do not establish causality, human preference, or metric gaming. |
| F10 | INCONCLUSIVE | Fulltrack Q3 is a reproducible causal advantage that should be adopted. | The unusually high historical canonical metric is useful for hypothesis generation, but training provenance is not reproducible, the mechanism is unresolved, and no matched fresh training replication exists. |

## Metric pattern

The table was deduplicated by `exp_label` before correlation. PQ is nearly
collinear with the Audiobox CE and CU axes in these 16 arms, while its association
with CLAP is slightly negative. This makes PQ alone an unsafe optimization target:
it can improve while prompt adherence worsens.

The primary [Audiobox Aesthetics paper](https://arxiv.org/abs/2502.05139v1) and
the pinned [Audiobox repository README](https://github.com/facebookresearch/audiobox-aesthetics/tree/2618e9d451b456e9328b39495b5e6234678aa550)
describe CE (Content Enjoyment), CU (Content Usefulness), PC (Production
Complexity), and PQ (Production Quality) as no-reference per-item axes. Neither
source makes them a replacement for prompt alignment.

## Bounded conclusion

An evaluation bug is not demonstrated. A contract-label provenance defect and
track-to-10-second granularity mismatch are confirmed. Historical row-level
binding and the causal mechanism remain unresolved. The PQ 6.9337 result may
guide hypotheses, but it does not establish a reproducible causal advantage.

## Prohibited claims

- `historical_training_alignment_valid`
- `cross_track_mismatch_confirmed`
- `byte_level_eval_reproduced`
- `fulltrack_causal_advantage`
- `non_fulltrack_pq_over_6_9_achieved`

## Next experiments

1. Repeat fulltrack Q3 q9 and per-segment slot0 Q3 q9 canonically from fresh,
   no-skip output directories.
2. Evaluate the historical fulltrack NoQ checkpoint with canonical CFG 0.
3. Evaluate fulltrack Q3 q0 only as a named secondary, noncanonical diagnostic.
4. Bind checkpoint, TSV, argv, code, environment, all ID-to-audio SHA-256 values,
   metric counts, report, and cleanup evidence; fail closed on any mismatch.
5. Require a freshly bound, reproducible training control before any causal claim.

## Frozen evidence

The machine-readable companion
[`fulltrack_q3_pq_audit_2026_08_28.json`](fulltrack_q3_pq_audit_2026_08_28.json)
contains the exact absolute paths and SHA-256 values for both checkpoints, the
MusicCaps TSV, historical report/log, migration record, result table, R-Shared
control, and phase-status evidence. It also pins Audiobox Aesthetics paper v1
and repository commit `2618e9d451b456e9328b39495b5e6234678aa550`.

During acceptance, the pre-existing untracked evaluation contract changed from
SHA-256 `333d455c…` to `23a7654e…` at 2026-08-28 18:17:51 +08:00. The new content
records the same provenance correction as F03. This concurrent external change
was preserved; this audit makes no authorship claim for it.
