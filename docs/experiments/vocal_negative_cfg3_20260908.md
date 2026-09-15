# Fidelity8 + vocal terms: preregistered paired evaluation

Operator request: design and queue a no-vocal negative-prompt experiment; compare
`vocals`, `singing`, `choir`, or all three. Append as P2 044 after the reserved 043 slot (currently held), without changing
existing experiments. Contract: `vocal_negative_cfg3_20260908_contract.json`.

Five arms: canonical fidelity8 control, fidelity8 + vocals, fidelity8 + singing,
fidelity8 + choir, fidelity8 + vocals, singing, choir. The four additions are named
secondary protocols; they do not replace the canonical benchmark.

Use the existing c2p0 slot0 full NoQ checkpoint from the preceding negative-prompt
experiments. Hold checkpoint, 5,521 ordered MusicCaps prompts, generation seed 42,
MeanFlow 25, literal CFG 3, NoMask, full precision and scorer batches of 32 fixed.
Generate all 5,521 prompts independently for every arm (27,605 clips total), then
aggregate the same per-clip scores within preregistered groups. No historical
scores are reused. Identical prompt order preserves the generation noise pairing.

Primary group: 2,535 captions without terms matching the existing vocal regex.
Secondary groups: full set and 2,986 captions with matching terms. The frozen TSV,
regex and ID lists are hash-bound in the contract. This is a caption proxy, not
verified absence of vocals in reference or generated audio. Do not infer audible
vocal removal from CLAP or AES alone.

Primary outcome: paired AES PQ delta against fidelity8 on the primary group.
Report CLAP and AES CE/CU/PC/PQ deltas with paired bootstrap intervals (10,000
resamples, seed 20260908). For the four treatment-control comparisons use 98.75%
intervals (Bonferroni). Label a candidate metric improvement only if PQ lower
bound > 0 and CLAP lower bound > -0.01. Report all candidates, including failures;
no automatic winner promotion. Secondary group comparisons and CE/CU/PC are
exploratory. Different absolute subgroup means are not evidence of treatment gain.
Single checkpoint and generation seed limit generalization.

Save per-clip metrics, audio hashes and RMS/crest/centroid/clipping diagnostics.
A PQ improvement does not by itself establish perceptual improvement; report
signal changes alongside it. Retain reports and summary, delete only registered
transient audio after complete finite metrics, valid mono 16kHz audio and matching
provenance are verified. Resume skips only hash-valid complete reports and
regenerates an incomplete cell from seed 42.

HARN includes input/storage gates, exact P2 process ownership, pause handling,
independent deterministic child monitoring, terminal receipts, and queue handoff.
Storage estimate: 8 GB peak additional, hard floor 63,687,091,200 bytes. No model
calls on healthy polls. No training, checkpoint mutation or repair is authorized.

## Results (completed 2026-09-09)

Artifacts: `~/nvme_experiment_artifacts/meanaudio/vocal_negative_20260908/`
(`summary.json` + five per-arm reports). All 5 × 5,521 clips were generated and
scored on 2026-09-09 between 08:18 and 11:03. The queue launcher file is still
listed under `p2/held/044_vocal_negative_cfg3.sh`, but the result files are complete.

### Preregistered decision

`summary.json` gives **`not_demonstrated` for all four treatments.** None meets
"PQ lower bound > 0 and CLAP lower bound > −0.01" on the primary group.

### Primary group: captions without vocal terms (n = 2,535), Δ vs fidelity8, 98.75% (Bonferroni)

| Arm | PQ | CLAP | Verdict |
|---|---|---|---|
| + vocals | −0.045 [−0.066, −0.024] | −0.0011 [−0.0033, +0.0012] | PQ worse |
| + singing | −0.025 [−0.040, −0.011] | +0.0016 [+0.0002, +0.0030] | PQ worse |
| + choir | −0.181 [−0.209, −0.156] | −0.0077 [−0.0103, −0.0051] | PQ worse |
| + all three | −0.177 [−0.205, −0.149] | −0.0085 [−0.0115, −0.0057] | PQ worse |

In this group CE and PC also drop in every arm: CE −0.11 to −0.57, PC −0.19 to −0.78,
with PC hit hardest.

### Secondary groups (exploratory, 98.75%)

| Arm | Vocal captions (n=2,986) PQ / CLAP | Full 5,521 PQ / CLAP |
|---|---|---|
| + vocals | +0.018 [+0.002, +0.033] / −0.0007 | −0.011 [−0.024, +0.002] / −0.0009 |
| + singing | +0.018 [+0.008, +0.028] / +0.0024 | −0.002 [−0.011, +0.007] / +0.0020 |
| + choir | −0.097 / −0.0068 | −0.136 / −0.0072 |
| + all three | −0.094 / −0.0074 | −0.132 / −0.0079 |

### Absolute values (full 5,521) and signal

| Arm | CLAP | CE | PC | PQ | RMS dB (sd) | centroid Hz |
|---|---|---|---|---|---|---|
| fidelity8 | 0.2605 | 7.211 | 5.106 | 7.599 | −17.9 (3.9) | 1584 |
| + vocals | 0.2596 | 7.010 | 4.729 | 7.588 | −18.8 (6.4) | 1491 |
| + singing | 0.2625 | 7.141 | 4.960 | 7.598 | −16.5 (4.3) | 1521 |
| + choir | 0.2533 | 6.951 | 4.648 | 7.463 | −17.1 (6.8) | 1815 |
| + all three | 0.2526 | 6.807 | 4.461 | 7.467 | −18.4 (7.9) | 1681 |

Clipped fraction is 0 in every arm.

### Observation layer

1. **Adding vocal terms to fidelity8 does not raise PQ on instrumental captions.**
   All four arms lower it. `choir` and the three-term combination cost ~0.18 PQ and
   ~0.008 CLAP.
2. **Small PQ gains appear only in the secondary group of vocal captions**
   (`vocals` / `singing` +0.018). That is exploratory, and against the 2× floor it
   is noise-level. It is not evidence of vocal removal.
3. **`singing` is the only arm with a CLAP lower bound above 0** (+0.0016 / +0.0024
   / +0.0020 across the three groups), but it fails the PQ gate. It is the most
   neutral of the four.
4. **Loudness spread grows.** With vocals / choir / all three, the RMS sd rises from
   3.9 dB to 6.4–7.9 dB, meaning some clips get much quieter or louder. `choir` also
   pushes the spectral centroid up by ~230 Hz. PC/CE losses may partly come from
   this signal change. Check against the listening test before interpreting.
5. **No vocal-presence detection was run.** Whether vocals were actually removed is
   still unknown. Per the preregistration, do not infer audible vocal removal from
   CLAP/AES.
