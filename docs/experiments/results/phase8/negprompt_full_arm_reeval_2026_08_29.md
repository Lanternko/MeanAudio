# Negative-prompt full-arm re-evaluation (2026-08-29)

13 arms, MusicCaps 5,521, MeanFlow 25 steps, **CFG 1.5**, seed 42, NoMask, full precision,
negative prompt = `low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi`.

Secondary protocol, **not** the canonical CFG 0 contract. Cells are comparable to each other and,
per clip, to the CFG 0 run in `novocal_reeval/`; they are not comparable to the historical CFG 0 table.

Script: `scripts/eval/negprompt_reeval_full_arms.py`. Per-clip scores kept under
`~/nvme_experiment_artifacts/meanaudio/negprompt_reeval/<arm>.json`.

Reproduction check: `c2p0_slot0_full_noq` returned PQ 7.2366 / dCLAP +0.0304 / dCE +0.6369,
matching the 2026-08-28 full-scale confirmation exactly.

## Full set

| arm | PQ cfg0 | PQ neg | dPQ | CLAP cfg0 | CLAP neg | dCLAP | CE cfg0 | CE neg | dCE | crest_min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c2p0_slot0_full_noq` | 6.5793 | **7.2366** | +0.6572 | 0.2201 | 0.2505 | +0.0304 | 6.2870 | 6.9239 | +0.6369 | 3.28 |
| `c2p0_slot0_q5_full_q9` | 6.5730 | **7.2272** | +0.6542 | 0.2235 | 0.2513 | +0.0278 | 6.3352 | 6.9016 | +0.5664 | 2.61 |
| `c2p0_slot0_q5_full_q0` | 6.5396 | **7.2129** | +0.6734 | 0.2212 | 0.2511 | +0.0299 | 6.2674 | 6.8725 | +0.6051 | 2.54 |
| `c2p0_fair013_worst_full` | 6.7195 | **7.1824** | +0.4629 | 0.2195 | 0.2438 | +0.0242 | 6.4162 | 6.8446 | +0.4284 | 2.81 |
| `c2p0_slot0_q3_full_q9` | 6.5437 | **7.1692** | +0.6255 | 0.2190 | 0.2493 | +0.0303 | 6.2474 | 6.8232 | +0.5758 | 2.43 |
| `c2p0_slot0_q3_full_q0` | 6.5197 | **7.1597** | +0.6400 | 0.2172 | 0.2487 | +0.0315 | 6.1960 | 6.7975 | +0.6015 | 2.34 |
| `c2p0_fair013_k3_full_q9` | — | **7.1343** | — | — | 0.2410 | — | — | 6.7718 | — | 2.69 |
| `c2p0_slot0_full_seed27182818` | 6.5270 | **7.1222** | +0.5951 | 0.2234 | 0.2517 | +0.0283 | 6.1527 | 6.7220 | +0.5694 | 2.01 |
| `c2p0_fair013_best_full` | 6.4670 | **7.0745** | +0.6074 | 0.2299 | 0.2620 | +0.0321 | 6.1644 | 6.7789 | +0.6145 | 2.65 |
| `fulltrack_q3_full_q9` | 6.9337 | **7.0020** | +0.0683 | 0.1870 | 0.1926 | +0.0056 | 6.8458 | 6.7970 | -0.0488 | 2.32 |
| `c2p0_slot2_full_noq` | 6.5124 | **6.9943** | +0.4819 | 0.2143 | 0.2382 | +0.0239 | 6.0703 | 6.4847 | +0.4144 | 2.61 |
| `fulltrack_noq_full` | 6.8586 | **6.9553** | +0.0967 | 0.1845 | 0.1928 | +0.0083 | 6.7252 | 6.7180 | -0.0072 | 2.57 |
| `p7v1_fullq_control_q9` | 6.5580 | **6.8611** | +0.3031 | 0.1860 | 0.2057 | +0.0197 | 5.8506 | 6.1536 | +0.3031 | 2.04 |

## Ranking inversion

PQ order under CFG 0:

1. `fulltrack_q3_full_q9` (6.9337)
2. `fulltrack_noq_full` (6.8586)
3. `c2p0_fair013_worst_full` (6.7195)
4. `c2p0_slot0_full_noq` (6.5793)
5. `c2p0_slot0_q5_full_q9` (6.5730)
6. `p7v1_fullq_control_q9` (6.5580)
7. `c2p0_slot0_q3_full_q9` (6.5437)
8. `c2p0_slot0_q5_full_q0` (6.5396)
9. `c2p0_slot0_full_seed27182818` (6.5270)
10. `c2p0_slot0_q3_full_q0` (6.5197)
11. `c2p0_slot2_full_noq` (6.5124)
12. `c2p0_fair013_best_full` (6.4670)

PQ order under the negative-prompt protocol:

1. `c2p0_slot0_full_noq` (7.2366)
2. `c2p0_slot0_q5_full_q9` (7.2272)
3. `c2p0_slot0_q5_full_q0` (7.2129)
4. `c2p0_fair013_worst_full` (7.1824)
5. `c2p0_slot0_q3_full_q9` (7.1692)
6. `c2p0_slot0_q3_full_q0` (7.1597)
7. `c2p0_fair013_k3_full_q9` (7.1343)
8. `c2p0_slot0_full_seed27182818` (7.1222)
9. `c2p0_fair013_best_full` (7.0745)
10. `fulltrack_q3_full_q9` (7.0020)
11. `c2p0_slot2_full_noq` (6.9943)
12. `fulltrack_noq_full` (6.9553)
13. `p7v1_fullq_control_q9` (6.8611)

The two fulltrack arms hold the top two PQ slots at CFG 0 and fall to 10th and 12th of 13 here.
CLAP ordering is essentially unchanged -- c2p0 already led on CLAP -- so the inversion is specific
to the aesthetics metrics, not a uniform shift.

## Paired delta by prompt subset

lofi-prompt n=1969, clean-prompt n=3552, split on low-fidelity vocabulary in the MusicCaps caption.

| arm | dPQ lofi | dPQ clean | dCE lofi | dCE clean |
|---|---:|---:|---:|---:|
| `c2p0_slot0_full_noq` | +0.6772 | +0.6461 | +0.7144 | +0.5939 |
| `c2p0_slot0_q5_full_q9` | +0.6929 | +0.6328 | +0.6508 | +0.5196 |
| `c2p0_slot0_q5_full_q0` | +0.7078 | +0.6542 | +0.6914 | +0.5572 |
| `c2p0_fair013_worst_full` | +0.4846 | +0.4508 | +0.4920 | +0.3931 |
| `c2p0_slot0_q3_full_q9` | +0.6518 | +0.6108 | +0.6497 | +0.5349 |
| `c2p0_slot0_q3_full_q0` | +0.6637 | +0.6269 | +0.6765 | +0.5599 |
| `c2p0_slot0_full_seed27182818` | +0.6301 | +0.5758 | +0.6741 | +0.5113 |
| `c2p0_fair013_best_full` | +0.6018 | +0.6105 | +0.7054 | +0.5641 |
| `fulltrack_q3_full_q9` | +0.0434 | +0.0821 | -0.0969 | -0.0221 |
| `c2p0_slot2_full_noq` | +0.4851 | +0.4801 | +0.4762 | +0.3801 |
| `fulltrack_noq_full` | +0.0687 | +0.1123 | -0.0525 | +0.0180 |
| `p7v1_fullq_control_q9` | +0.2754 | +0.3184 | +0.3075 | +0.3006 |

The lofi-vs-clean gap is small (roughly 0.03-0.05 PQ on the c2p0 arms), so at full scale the
fidelity-targeted subset is **not** where most of the gain sits. On both fulltrack arms the sign
reverses: the intervention helps *less* on the subset its wording targets. This weakens, but does
not overturn, the 512-prompt reading that roughly 30% of the effect is fidelity semantics -- the U2
reverse-direction control is still the stronger evidence for directionality.

## Saturation

Crest factor and clipping over the first 64 clips of each arm. Clipping is 0.0 everywhere.
`crest_min` is a worst-case, not a mean, and two arms sit at the historical <2.0 saturation line:
`c2p0_slot0_full_seed27182818` 2.01 and `p7v1_fullq_control_q9` 2.04. Confirm this before any move
to cfg 2.5.

