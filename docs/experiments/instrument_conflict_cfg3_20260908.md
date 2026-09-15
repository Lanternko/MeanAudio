# Per-caption instrument-conflict ablation (P2 045)

Operator approved append-only registration after the current queue tail, 044.
This is a separately named secondary protocol, not the canonical MusicCaps-5521
benchmark. Existing queue entries and scientific contracts remain unchanged.

Use 3,207 MusicCaps captions containing explicit instrument nouns and no explicit
negation token, in original source order. The frozen 23-family lexicon, selection
rules and per-row assignments are in the hash-bound `assignments.json` runtime
input. No LLM annotation is used. Generic bass, brass, strings and keys are not
inferred as specific instruments; guitar includes electric/acoustic/bass guitar.

For every eligible caption, select one mentioned instrument with seed 20260908.
Generate three paired arms:

| Arm | Negative prompt |
|---|---|
| fidelity8 | Registered eight fidelity terms |
| fidelity8_conflict | fidelity8 + that row's selected mentioned instrument |
| fidelity8_unmentioned | fidelity8 + that row's fixed unmentioned instrument |

All three arms use the same captions, order, c2p0 slot0 full NoQ checkpoint,
MeanFlow 25, literal CFG 3, generation seed 42, NoMask and full precision. Each
arm regenerates from its own identically seeded RNG stream. A dedicated evaluator
snapshot reads per-row negatives; existing `eval.py` and prior evaluations are
not edited. Scoring uses original captions, CLAP/AES batches of 32.

Attempting to match the entire negative-word frequency distribution between
conflict and unmentioned arms is infeasible with the fixed mention constraints.
The deterministic fallback balances unmentioned terms across 23 instruments
(139–140 uses each). The target distribution is naturally guitar-heavy (1,428),
then piano (508), drums (303), flute (133), violin (131), and smaller groups.
This remaining lexical-frequency confound must be disclosed. Per-target results
and equal-instrument macro means accompany population averages; neither removes
all lexical confounding. Unmentioned does not prove acoustically absent.

Automated primary contrast: original-caption CLAP(conflict) minus
CLAP(unmentioned), paired 10,000-bootstrap seed 20260908, two-sided 95% interval.
Upper bound below zero supports a semantic-interference proxy. It does not prove
instrument removal. Both treatment-vs-fidelity8 contrasts, AES CE/CU/PC/PQ,
instrument strata and signal statistics are exploratory; no metric winner is
promoted. Save all per-clip scores, generated-audio hashes and RMS/crest/centroid/
clipping diagnostics.

Retain 183 instrument-stratified blind triplets (up to eight per instrument,
seed 20260909; 549 audio files). Randomize ABC independently per caption, keep
assignment key separate from the rating CSV, and ask for target presence
(0 absent/1 uncertain/2 present), prominence (0–4), and quality (1–5). Audible
suppression remains unconfirmed until blind ratings are collected. Queue
completion means generation, validated metrics and listening pack are ready;
it does not fabricate ratings or block later compute while waiting for them.

Compute budget: 9,621 generated clips; prior same-checkpoint MF25/CFG3 logs show
1,024 clips in 5.6–6.4 minutes. Generation estimate is 53–60 minutes; total GPU
work including scoring is provisionally 1.5–3 hours, excluding queue wait,
preemption and resource holds. Peak additional disk budget is 5 GB; hard floor
63,687,091,200 bytes. Retain blind audio/reports; clean only this run's validated
transient audio. Partial cells restart from seed 42. Complete reports must pass
provenance/ID/finite-metric/audio-hash checks before reuse.

The resource-owning parent verifies its exact P2 seat, monitors the child and
capacity, handles P1 pause and HUP/INT/TERM, and leaves storage holds pollable.
Notifications are idempotent; notification failure prevents promotion. The P2
host owns terminal-to-next scheduling. No training or automatic repair is added.

## Results (completed 2026-09-09)

Queue terminal: `p2/done/045_instrument_conflict_cfg3.terminal.json`. Artifacts:
`~/nvme_experiment_artifacts/meanaudio/instrument_conflict_20260908/`
(`summary.json`, three per-arm reports, `listening/`). Decision recorded in
`summary.json`: `semantic_interference_proxy: supported`,
`audible_instrument_suppression: pending_blind_listening`, `promotion: none`.

### Preregistered primary contrast

Original-caption CLAP, conflict − unmentioned, n = 3,207, paired bootstrap 95%:
**−0.0069 [−0.0087, −0.0051]**. Upper bound < 0, so the semantic-interference
proxy is **supported**. That supports a proxy only. It does not show the instrument
was removed.

### Arm aggregates (n = 3,207)

| Arm | CLAP | CE | CU | PC | PQ | RMS dB | crest mean |
|---|---|---|---|---|---|---|---|
| fidelity8 | 0.2701 | 7.541 | 7.781 | 5.204 | 7.747 | −17.76 | 6.71 |
| fidelity8_conflict | 0.2519 | 7.338 | 7.656 | 5.271 | 7.592 | −16.56 | 5.97 |
| fidelity8_unmentioned | 0.2588 | 7.299 | 7.594 | 5.115 | 7.457 | −15.11 | 5.26 |

Clipped fraction is 0 in every arm.

### Exploratory paired contrasts (population, 95%)

| Contrast | CLAP | PQ | CE | CU | PC |
|---|---|---|---|---|---|
| conflict − fidelity8 | −0.0181 [−0.0198, −0.0165] | −0.155 [−0.171, −0.139] | −0.203 | −0.125 | +0.066 |
| unmentioned − fidelity8 | −0.0112 [−0.0129, −0.0096] | −0.290 [−0.307, −0.273] | −0.242 | −0.188 | −0.090 |
| conflict − unmentioned | −0.0069 [−0.0087, −0.0051] | +0.135 [+0.115, +0.156] | +0.039 | +0.062 | +0.156 |

Equal-instrument macro means: conflict − unmentioned CLAP −0.0088 / PQ +0.080;
conflict − fidelity8 CLAP −0.0317 / PQ −0.189; unmentioned − fidelity8 CLAP
−0.0229 / PQ −0.269.

### Per-target (conflict − unmentioned CLAP, 95%)

- CI clearly below 0: flute −0.038 (n=133), saxophone −0.037 (56), violin −0.027 (131),
  clarinet −0.024 (26), marimba −0.024 (26), trombone −0.019 (29), trumpet −0.015 (87),
  piano −0.013 (508).
- CI covers 0: guitar −0.0003 (n=1,428, 45% of rows), drums −0.0007 (303), and most
  of the small strata.
- Only positive CI: tuba +0.041 [+0.002, +0.088] (n=12).

### Observation layer

1. **Adding any instrument name to fidelity8 hurts every automated metric except PC.**
   Both treatment arms lose CLAP and PQ against fidelity8. Adding an *unmentioned*
   instrument costs PQ more (−0.29) than adding the *mentioned* one (−0.15).
2. **The CLAP penalty is specific to the caption instrument, but weakly.** The primary
   contrast is ~0.007 CLAP, about 0.8× the CFG0 2× seed floor (0.0084), measured
   under a different protocol. Within this paired design the interval is well away
   from 0. The effect sits in wind/string/keyboard families. Guitar, the largest
   target and the one with a lexical confound, shows ~0.
3. **Loudness confound.** RMS rises from −17.8 dB (fidelity8) to −16.6 (conflict)
   to −15.1 (unmentioned), and mean crest falls from 6.7 to 6.0 to 5.3. The PQ gaps
   between arms follow this loudness/crest ordering, so they cannot be read as
   instrument-specific (cf. the 2026-08-31 negprompt loudness confound).
4. **Lexical frequency confound still applies** (`control_frequency_exact_match:
   false`): conflict negatives are 45% "guitar", while unmentioned negatives spread
   evenly over 23 instruments.

### Blind listening

The pack is ready: 183 triplets / 549 files in `listening/`, with `ratings.csv`
separate from the key. **As of 2026-09-11, 0/549 rows are rated.** Whether the
target instrument is audibly suppressed stays unconfirmed until ratings come in.
Do not claim "the negative prompt removes the instrument" from CLAP alone.
