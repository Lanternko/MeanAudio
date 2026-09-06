# CFG0 retrospective evidence and comparison

Generated: `2026-08-22T05:13:13.012391+00:00`

## Disposition

All nine evaluations are classified as `retrospective_operationally_complete_comparable`.
No GPU rerun is required for the comparison table below. This bundle preserves the
surviving report, metrics, log, checkpoint, and current input hashes. It does not
claim that the historical runs passed the later strict HARN acceptance flow.

## Metrics

| Rank | Label | CLAP | AES-CE | AES-CU | AES-PC | AES-PQ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `phase8_qwen_caption2p0_bestof3_noq_quarter_musiccaps_mf25_cfg0_noq` | 0.2129 | 6.2368 | 6.7327 | 5.1212 | 6.5316 |
| 2 | `phase8_qwen_caption2p0_fair013_bestof3_noq_quarter_musiccaps_mf25_cfg0_noq` | 0.2114 | 6.2046 | 6.6693 | 5.1490 | 6.4793 |
| 3 | `phase8_qwen_caption2p0_slot1_noq_quarter_musiccaps_mf25_cfg0_noq` | 0.2047 | 6.3008 | 6.7593 | 5.1632 | 6.5668 |
| 4 | `phase8_qwen_caption10s_multisent_noq_quarter_musiccaps_mf25_cfg0_noq` | 0.2029 | 6.1185 | 6.7031 | 5.0350 | 6.5364 |
| 5 | `phase8_qwen_caption2p0_slot2_noq_quarter_musiccaps_mf25_cfg0_noq` | 0.2017 | 6.2071 | 6.7487 | 5.0814 | 6.5623 |
| 6 | `phase8_qwen_caption2p0_fair013_worstof3_noq_quarter_musiccaps_mf25_cfg0_noq` | 0.1985 | 6.4061 | 6.8835 | 5.2172 | 6.6789 |
| 7 | `phase8_qwen_caption2p0_fair013_k3_quarter_musiccaps_mf25_cfg0_q9` | 0.1966 | 5.9310 | 6.4984 | 5.2635 | 6.3988 |
| 8 | `phase8_qwen_caption2p0_worstof3_noq_quarter_musiccaps_mf25_cfg0_noq` | 0.1957 | 6.4072 | 6.8399 | 5.3208 | 6.6398 |
| 9 | `phase8_qwen_caption2p0_qwen3cap_k3_quarter_musiccaps_mf25_cfg0_q9` | 0.1894 | 5.7757 | 6.4003 | 5.0264 | 6.2127 |

## Evidence boundary

For every cell, its log contains exactly 5,521 unique `Audio saved` IDs; that ID set
exactly matches the MusicCaps TSV, and the metric evaluator loaded 5,521 records.
The logged CFG, steps, seed, conditioning, NoMask, and precision also match across all
nine cells. The five finite metrics,
checkpoint, metrics file, report, and complete eval log survive and are SHA-256 bound
in the JSON bundle.

The generated FLAC files do not survive, and the executed wrapper was not hash-bound
at launch. These are provenance limitations, not evidence that the computation failed.
This record therefore supports within-table comparison while retaining the historical
execution caveat.

Machine-readable evidence: `/home/kojiek/MeanAudio/docs/experiments/results/phase8/cfg0_retrospective_evidence_2026_08_22.json`
