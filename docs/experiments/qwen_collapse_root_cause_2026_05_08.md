# Qwen-trained collapse — root-cause diagnostic synthesis (2026-05-08)

## Standing puzzle

All MeanAudio models trained on **Qwen2.5-Omni-3B captions** collapse to MusicCaps CLAP ≈ 0.06 with steering ratio < 0.12, regardless of single/multi-cap or Q/NoQ configuration. LP-MC counterparts reach CLAP 0.18–0.20 with steering 0.9–1.7. This document audits 5 hypotheses (H1–H4 + H7/H9) using cheap diagnostics (no new training).

## Observation table

| Train | Caption | Q | MC CLAP | Jamendo s42 CLAP | max steering ratio |
|---|---|---|---|---|---|
| P7 V1 | LP-MC single | +Q | **0.197** | — | **1.702** |
| P8 | LP-MC single | NoQ | **0.185** | 0.1409 (s42) | **1.723** |
| P9 V1 | LP-MC multi-5 | NoQ | 0.0650 | 0.0650 (s42) | 0.147 |
| P9 V2 | LP-MC multi-5 | +Q | 0.0403 | 0.0403 | 0.056 |
| **P8-Qwen** | **Qwen single** | NoQ | **0.0611** | 0.0582 | **0.120** |
| **P7V1-Qwen** | **Qwen single** | +Q | **0.0687** | 0.0598 | **0.057** |
| **P9.5 V1** | **Qwen multi-5** | NoQ | **0.0609** | 0.0597 | **0.044** |
| **P4V2-Qwen** | **Qwen single (BC-selected)** | NoQ | **0.0611** | **0.0596** | TBD |

P8-Qwen collapses despite being single-cap → multi-cap is not a necessary condition for collapse; **Qwen single-cap training is sufficient to reproduce the collapse under this recipe**.

## Cross-prompt eval matrix (Jamendo seed=42 2048 same audio)

| Train | Caption type | LP-MC prompt | Qwen prompt | Δ Qwen−LP | Status |
|---|---|---|---|---|---|
| P8 (LP-MC NoQ single) | LP-MC writing 4-task random | 0.1409 | **0.2246** | **+0.084** | text-conditioning healthy |
| P9V1 (LP-MC NoQ multi-5) | LP-MC 5-cap | 0.0650 | 0.0837 | +0.019 | collapsed |
| P9V2 (LP-MC +Q multi-5) | LP-MC 5-cap | 0.0403 | 0.0622 | +0.022 | collapsed |
| P8-Qwen (Qwen NoQ single) | Qwen single | 0.0582 | 0.0776 | +0.019 | collapsed |
| P7V1-Qwen (Qwen +Q single) | Qwen single | 0.0598 | 0.0791 | +0.019 | collapsed |
| P9.5 V1 (Qwen NoQ multi-5) | Qwen 5-cap | 0.0597 | 0.0799 | +0.020 | collapsed |
| **P4V2-Qwen (Qwen NoQ BC-single)** | **Qwen single (best caption-selected)** | **0.0596** | **0.0801** | **+0.020** | **collapsed** |
| **P-LPMC-destructured (EXP-A)** | **LP-MC boilerplate stripped** | **0.0561** | **0.0781** | **+0.022** | **collapsed (induced)** |

**G1+G2 (5/8) — All 5 collapsed models converge to +0.019–0.022 Qwen-prompt boost (extremely tight). Healthy P8 has +0.084 (4x larger).** This is a clean signature: a +0.02 universal boost from Qwen prompts for collapsed models, distinguishable from a +0.08+ boost for text-conditioning-healthy models. Multi-cap LP-MC and ALL Qwen-trained variants converge to the same +0.019–0.020 Qwen-prompt boost. The +0.02 boost is now interpreted as a metric-level artifact (Qwen prompts may be slightly easier to score high CLAP under HTSAT-base) rather than a model-quality differentiator.

**Kill shot for H1 (Codex 5/7 prompt mismatch):** LP-MC trained healthy model + Qwen prompts → 0.2246 (higher than LP-MC + LP-MC eval). Qwen prompts carry usable conditioning information at inference. Multi-cap LP-MC and Qwen-trained models BOTH stay collapsed regardless of prompt source, getting only the universal +0.02 metric boost. The +0.02 residual is NOT the dominant cause for either class of collapse.

**Important new finding (G1):** Multi-cap LP-MC collapse (P9V1, +0.019 Qwen boost) and single-cap Qwen collapse (P8-Qwen, +0.019 Qwen boost) yield the SAME cross-prompt signature. This argues for a shared failure mechanism between "multi-cap supervision" and "Qwen-style supervision" — distinct caption-source mismatch and multi-cap-only hypotheses cannot be the full story. Whatever causes one likely causes the other.

## Caption-vs-audio raw CLAP similarity (Jamendo seed=42 n=2048)

| Captioner | mean self-sim | mean inter-cap | discrim | R@1 | R@10 |
|---|---|---|---|---|---|
| Qwen mean-of-5 | 0.356 | 0.167 | 0.189 | 3.0% | 14.7% |
| Qwen slot 0 | 0.311 | 0.130 | **0.181** | **2.1%** | **12.6%** |
| LP-MC random 1-of-4 | 0.249 | 0.121 | 0.128 | 0.7% | 5.8% |

Qwen single-caption (slot 0) is more aligned (+0.062), more discriminating (R@1 3x higher), and has higher discrim (+0.053) than LP-MC random-1-of-4. **H2 (Qwen captions are generic / low-specificity) is FALSIFIED.**

## Pipeline integrity (B2/B3/B4/B5 + B2.5)

| Diag | Coverage | Result | Conclusion |
|---|---|---|---|
| B2 — NPZ CLAP cache vs fresh forward | 10 random | cos = 1.0000 (10/10) | ✅ no CLAP cache corruption |
| B2.5 — NPZ T5 cache vs fresh forward | 8 random | pool_cos = 1.0000, frame_min = 1.0000 | ✅ no T5 cache corruption |
| B3 — TSV id↔caption alignment | 60 random | 60/60 caption ∈ Qwen JSONL 5-slot | ✅ no row shift |
| B4 — T5 max_length=77 truncation | 5K LP-MC, 5K Qwen | LP-MC 19.7%, Qwen 0.1% truncated | ✅ Qwen NOT truncation-limited (LP-MC truncates more, trains better) |
| B5 — Stage-2 final loss | 5 runs | 0.986 ± 0.001 (within 0.1%) | ✅ all runs converge similarly; loss does not distinguish |

**H3 (pipeline bug specific to Qwen) FALSIFIED across 5 independent checks.**

## Caption anatomy (E1/E2/E3 on full 251K)

**Important pre-analysis correction (5/8 06:45):** P8 (LP-MC NoQ baseline) was trained on
**phase7_v1_train.tsv** (LP-MC writing 4-task random, "the low quality recording features..." prefix style),
not phase4_train.tsv (mu-llama-style "begins with a / final segment"). Confirmed via
phase8_stage2_200000.log line 1. P7V1 also used phase7_v1_train.tsv, just with `use_q_conditioning=True`.

So the canonical "LP-MC trains well" baseline = LP-MC writing task with the `"the low quality recording"`
boilerplate prefix, NOT the mu-llama temporal-narrative template.

E1 vocabulary (analysis on phase7_v1_train.tsv = the actual baseline TSV):
- Qwen unigram vocab: **4,955** (vs LP-MC 3,993, +24%)
- Qwen bigram vocab: **50,291** (vs LP-MC 35,162, +43%)
- LP-MC top-50 trigram coverage: **99.5%** of captions (heavy boilerplate)
- Qwen top-50 trigram coverage: 76.9%
- LP-MC top trigram: "the low quality" appears in 35% of captions
- LP-MC bigrams: "low quality" (10,042 / 25k), "quality recording" (10,034), "recording features" (8,798)
- Qwen top trigram: "music features a" appears in 12% of captions

E2 multi-cap intra-audio Jaccard 4-gram (Qwen 5 captions per audio):
- mean **0.106**, median 0.097, only 0.6% of pairs > 0.3 → 5 captions are genuinely diverse per audio

E3 string-level uniqueness:
- Qwen all 5: 99.9% unique strings
- LP-MC: 95.4% unique (4.6% duplicates)

→ **Qwen has more vocabulary diversity AND more discriminating captions**, but trains worse.

## Inter-caption embedding distribution (F)

For 2048 random audio captions:

| Embedding space | Qwen slot 0 off-diag | Qwen mean-of-5 | LP-MC | Δ |
|---|---|---|---|---|
| CLAP (cond signal) | **0.395** | 0.605 | **0.299** | **+0.10** |
| T5 (cross-attn input) | 0.770 | 0.918 | 0.785 | ≈ 0 |

**Asymmetry**: in CLAP space Qwen captions cluster more tightly together (off-diag +0.10). In T5 space the distributions are essentially identical. CLAP is used in MeanAudio as the conditional embedding `text_features_c` (typically dropped 10–20% for CFG), T5 is the cross-attention sequence.

## What is left standing

- ✗ H1 (eval prompt mismatch): residual +0.02, not main cause
- ✗ H2 (Qwen captions generic): falsified — Qwen IS more specific
- ✗ H3 (pipeline bug — NPZ/TSV/truncation): falsified across 5 checks
- ◐ H7/H9 (text feature clustering at training):
  - CLAP cond shows +0.10 clustering for Qwen → plausible CFG signal weakening
  - T5 cross-attn shows no difference → does NOT explain through T5 channel
  - Mixed evidence; needs intervention experiment to falsify

## Open hypothesis (not yet tested by intervention)

**H10 — LP-MC writing-task boilerplate is a feature, not a bug, for training.** The repeated prefix "the low quality recording features a [X]" (35% of P8 training captions) creates a stable template the model can use as anchor; the variable part inside the template maps to acoustic features. Qwen's higher vocabulary diversity removes this anchor. (Note this is opposite to standard "diversity helps" intuition.)

**H11 — Qwen captions describe "mood / atmosphere / style" more than acoustic specifics.** Even with high embedding-space discrimination (R@1 = 3% vs 0.7% LP-MC), the *content* may lean abstract ("vibrant and upbeat", "soothing and folk-inspired") whereas LP-MC describes acoustic structure ("Eb major", "127 BPM", "shimmering hi hats"). Acoustic-level content might give the diffusion model a more direct learnable signal.

**H12 — CLAP cond clustering bottleneck (from F1).** CLAP embedding off-diag +0.10 for Qwen → during CFG training, the conditional pass cond-vector and unconditional pass null-vector difference is smaller in expectation → weaker classifier-free gradient → model relies on audio prior more. T5 doesn't carry this asymmetry, so cross-attn alone isn't enough to compensate.

## Recommended next experiments (ranked)

### EXP-A (cheapest, GPU 1.5 day): P-LPMC-NoBoilerplate retraining
Strip the LP-MC "low quality recording features a" prefix (and similar templated openers from the
4 LP-MC writing tasks) from all training captions in `phase7_v1_train.tsv`. Retrain P8 with same
config. If MC CLAP drops from 0.185 toward Qwen-level → H10 confirmed (boilerplate was the anchor).

### EXP-B (1.5 day): P-Qwen-Boilerplate retraining
Prepend a fixed boilerplate ("the low quality recording features a ") to every Qwen caption. Retrain. If MC CLAP recovers from 0.06 toward LP-MC-level → H10 confirmed from the other direction.

### EXP-C (1.5 day, simplest): P-Qwen-MC-mixed
50/50 random sample LP-MC vs Qwen captions per training step. Tests whether LP-MC's structure can rescue the Qwen-only collapse. Intermediate result expected if H10 partial.

### EXP-D (cheap, eval only): Cross-attention activation inspection
Load P8 (LP-MC) and P8-Qwen ckpts. Feed identical prompt batch through both. Compare cross-attention weight entropy / norm. If Qwen-trained model has near-uniform cross-attn (model ignores text) → confirms the model never learned to attend.

### EXP-E (medium, eval only): Same-audio cross-caption training-batch contrast
Sample 256 random training batches (8 each), compute (T5 || CLAP) embedding pairwise distance within batch. If Qwen batches have systematically smaller intra-batch distance than LP-MC → confirms gradient-signal magnitude difference.

EXP-A and EXP-B are the most surgical (single intervention, clear hypothesis). EXP-D is the cheapest informative test (no retraining needed). Recommend running **EXP-D first (this week, after P4V2-Qwen finishes)** to lock in interpretation, then EXP-A or EXP-B based on findings.

## Closing note (claim discipline per memory ref. `reference_claim_classification_2026_04_21.md`)

- **Observation**: Qwen-trained models all collapse to ~0.06 MC CLAP regardless of cap-source single/multi or Q/NoQ. Caption-vs-audio raw CLAP higher for Qwen. CLAP inter-caption clustering higher for Qwen (+0.10). T5 inter-caption clustering identical.
- **High-confidence inference**: Pipeline integrity and prompt-side mismatch are NOT the cause. Caption "genericness" is not the cause (Qwen is more specific).
- **Low-confidence inference (currently)**: CLAP cond clustering or boilerplate-anchor effects are plausible mechanisms but not yet causally verified.
- **Forbidden until intervention**: claiming any specific mechanism is THE cause; claiming "Qwen captions are bad for training" without scope qualification; claiming LP-MC structure is necessary.


## EXP-A RESULT (5/9 morning) — H10 confirmed

Trained P-LPMC-destructured on phase7_v1_train_destructured.tsv (boilerplate stripped) with
same Stage 1 + Stage 2 600K iter pipeline as P8 baseline:

| Metric | P-destructured | P8 baseline | Δ |
|---|---|---|---|
| MusicCaps CLAP | **0.0608** | 0.1851 | **−67%** |
| Jamendo s42 CLAP | **0.0561** | 0.1409 | −60% |
| Stage 2 final loss | 0.9862 | 0.9867 | ≈ same |
| AES quality (CE/CU/PC/PQ) | 5.92/6.54/5.33/6.49 | normal | normal |

**Stripping the LP-MC boilerplate template ("the low quality recording features..." 45% prefix)
collapses healthy P8 (0.185) directly into Qwen-cluster (0.061).** Loss does not differ — the
model still converges on reconstruction objective, but loses text-conditioning capability.

→ **H10 confirmed (EXP-A direction)**: The full LP-MC writing-task style — which is correlated with the boilerplate prefix — is necessary in tested settings for healthy text conditioning. Without it, the model collapses regardless of caption content quality. Note: EXP-C later showed the prefix string alone is insufficient; the necessary property is the full LP-MC writing-task style, not the prefix string in isolation.

This result is consistent with H11 (Qwen 5-task framing variance hypothesis): both LP-MC
boilerplate-stripped and Qwen 5-task-mixed share the property of lacking the LP-MC writing-task
style, and both collapse to the ~0.06 cluster.

| Training data | Inductive anchor stability | Result |
|---|---|---|
| LP-MC + "low quality recording" prefix (45%) | high (stable anchor) | healthy 0.185 |
| **LP-MC stripped (EXP-A)** | **none** | **collapsed 0.061** |
| Qwen 5-task BC-selected | mixed across 5 framings | collapsed 0.061 |
| LP-MC multi-cap (P9V1) | per-step random of 5 caps | collapsed 0.065 |

→ All structurally-unstable training collapses to ~0.06 plateau.

**Next: EXP-B (Qwen-Slot0-Fixed)** — force ALL 251K audio to use Qwen slot 0 caption only,
test whether H11 (5-task variance is the Qwen collapse cause) holds:
  - MC CLAP recovers → H11 confirmed; Qwen captions can train healthy if framing fixed
  - Still ~0.06 → Qwen caption content lacks anchor regardless of framing uniformity


## EXP-B RESULT (2026-05-11 evening) — H11 falsified

Trained Qwen-Slot0-Fixed (`p_qwen_slot0_stage2_200000`) on `qwen_slot0_train.tsv` (all 251K
audio forced to Qwen slot 0 caption — same single framing throughout). Same Stage 1 + Stage 2
600K iter pipeline as all other Qwen reruns. NPZ regen on `/home/kojiek/exps_nvme/npz_qwen_slot0`
using the original audio mean/std (verified identical to `npz_phase8v4` source) + fresh T5 + CLAP
for slot 0 caption.

| Metric | EXP-B (Qwen-Slot0-Fixed) | All other collapsed Qwen runs |
|---|---|---|
| MusicCaps CLAP (LP prompt) | **0.0615** | 0.0596–0.0687 |
| Jamendo s42 CLAP (LP prompt) | **0.0608** | 0.0582–0.0598 |
| Jamendo s42 CLAP (Qwen prompt) | **0.0812** | 0.0776–0.0837 |
| **Qwen-prompt boost (Δ)** | **+0.0204** | +0.019–0.022 |
| Stage 2 final loss | 0.9866 | ~0.987 (all converge) |
| AES quality (CE/CU/PC/PQ) | 5.87/6.59/5.34/6.47 | normal |

**Forcing ALL 251K audio onto Qwen slot 0 (same caption framing throughout) DOES NOT rescue
collapse. EXP-B joins the +0.06 / +0.020 cluster as the 7th collapsed Qwen-trained model
(8th counting EXP-A).**

→ **H11 FALSIFIED**: 5-task framing variance is NOT the cause of Qwen-trained collapse.
Removing the variance entirely (single slot throughout 251K) does not help — model still
collapses to the same plateau.

Whatever property of Qwen captions causes collapse, it is **present in each individual
caption slot**, not in the heterogeneity across slots. Collapse traces to intrinsic properties
of the Qwen caption distribution itself, not to mixing-induced inductive instability.

## Updated cross-prompt eval matrix (full audit, n=8 collapsed + 1 healthy)

| Train | Caption type | LP-MC prompt | Qwen prompt | Δ Qwen−LP | Status |
|---|---|---|---|---|---|
| P8 (LP-MC NoQ single) | LP-MC writing 4-task random | 0.1409 | **0.2246** | **+0.084** | text-conditioning healthy |
| P9V1 (LP-MC NoQ multi-5) | LP-MC 5-cap | 0.0650 | 0.0837 | +0.019 | collapsed |
| P9V2 (LP-MC +Q multi-5) | LP-MC 5-cap | 0.0403 | 0.0622 | +0.022 | collapsed |
| P8-Qwen (Qwen NoQ single) | Qwen single | 0.0582 | 0.0776 | +0.019 | collapsed |
| P7V1-Qwen (Qwen +Q single) | Qwen single | 0.0598 | 0.0791 | +0.019 | collapsed |
| P9.5 V1 (Qwen NoQ multi-5) | Qwen 5-cap | 0.0597 | 0.0799 | +0.020 | collapsed |
| P4V2-Qwen (Qwen NoQ BC-single) | Qwen single (BC-selected) | 0.0596 | 0.0801 | +0.020 | collapsed |
| P-LPMC-destructured (EXP-A) | LP-MC boilerplate stripped | 0.0561 | 0.0781 | +0.022 | collapsed (induced) |
| **EXP-B Qwen-Slot0-Fixed** | **Qwen slot-0 forced (same framing 251K)** | **0.0608** | **0.0812** | **+0.020** | **collapsed (H11 dead)** |

**G2 finding (8/9 models)**: 8 collapsed models, all with +0.019–0.022 Qwen-prompt boost
(extremely tight). Healthy P8 is alone at +0.084 (4× larger). The +0.02 metric artifact
signature holds across ALL collapsed training conditions: multi-cap LP-MC, multi-cap Qwen,
single-cap Qwen, BC-selected Qwen, slot-fixed Qwen, AND boilerplate-stripped LP-MC. Caption
source / multi-vs-single / framing variance / Q conditioning are all orthogonal to whether
collapse happens — they don't isolate. The boundary is binary: either the training data
provides a stable inductive template anchor (P8 only, in our 9-model audit) or it doesn't.

## What is left standing after EXP-A + EXP-B

- ✗ H1 eval prompt mismatch — falsified
- ✗ H2 Qwen captions generic — falsified (Qwen is more specific)
- ✗ H3 pipeline bug — falsified (5 checks)
- ✓ H10 LP-MC boilerplate as inductive anchor — **confirmed** (EXP-A 0.0608 from baseline 0.1851)
- ✗ **H11 Qwen 5-task framing variance** — **falsified** (EXP-B 0.0615 with single-slot uniformity)
- ◐ H7/H9/H12 CLAP cond clustering at training — still plausible mechanism, untested by intervention

## What this jointly implies (claim discipline)

- **Observation**: Boilerplate stripping LP-MC collapses healthy 0.185 → 0.061 (EXP-A).
  Forcing single Qwen slot framing on 251K still collapses 0.062 (EXP-B). Same +0.02 metric
  artifact present in both, and in all 8 collapsed models.
- **High-confidence inference**: The presence of a stable, repeatable caption template
  (e.g., LP-MC's "the low quality recording features..." prefix in 35–45% of train captions)
  is a necessary inductive anchor in THIS training setup (Stage 1 FluxAudio + Stage 2 MeanFlow,
  Jamendo 251K, T5+CLAP encoders). Removing the anchor from LP-MC collapses it. Adding
  "structural uniformity" alone to Qwen (slot 0 only — same prompt framing, no template prefix)
  does NOT add an anchor — Qwen captions are lexically/syntactically diverse enough that they
  do not form a learnable template even at single-slot granularity.
- **Low-confidence inference**: The exact "anchor" property is not yet isolated. Candidates:
  (a) repeated n-gram prefix giving stable initial cross-attn key distribution,
  (b) constrained vocabulary giving stable T5 embedding subspace,
  (c) reduced CLAP cond clustering during training,
  (d) consistent template enabling the model to learn the "fill-in" semantic mapping in the
  variable part of the template.
- **Forbidden until intervention**: claiming any one mechanism among (a)–(d) is THE cause;
  claiming Qwen-style captioning is "bad for training" in absolute terms (it is bad for this
  particular pretraining recipe + scale; could be fine elsewhere); claiming H11 was wrong to
  test — it was a clean prediction that produced clean falsification, and it is now the
  strongest single piece of evidence narrowing the causal space.

## Recommended next experiments after EXP-A + EXP-B

### EXP-C (mid cost, 1.5 day GPU): P-Qwen-Boilerplate
Prepend the LP-MC boilerplate ("the low quality recording features a ") to every Qwen caption.
This is the inverse of EXP-A: if H10's "anchor template" is the whole story, MC CLAP should
recover toward 0.185. If MC CLAP stays ~0.06, anchor template alone is not enough — need
additional caption-content properties.

### EXP-D (cheap, eval-only): Cross-attention activation inspection
Load P8 vs EXP-A vs EXP-B ckpts. Feed identical prompt batch through all three. Compare
cross-attention entropy / norm. If collapsed models show near-uniform cross-attn (model
ignores text channel), confirms text channel is not being used at inference.

### EXP-E (cheap, eval-only): Same-audio cross-caption training-batch contrast
Compute T5+CLAP embedding pairwise distance distribution within LP-MC vs Qwen training batches
(no audio, just text). If Qwen has systematically smaller intra-batch text-feature distance →
quantifies the H12 CLAP-clustering bottleneck without retraining.

EXP-D / EXP-E are eval-only and cheap (hours, not days). EXP-C is the natural next intervention
to isolate "anchor template" vs "caption content".


## EXP-D2 PATHWAY PROBE (2026-05-12 evening) — text projection activation magnitude collapse

**Status**: preliminary mechanistic probe; **needs weight-norm + S1/S2-stage confirmation before
treating as established mechanism**. Observations are stable across 8 prompts × 3 timesteps,
but root cause (learned weight shrinkage vs activation cancellation vs distillation effect)
is not yet localized.

### Observation table (4 models, n=8 prompts, t∈{0.2, 0.5, 0.8}, averaged)

| Model | CLAP cond_proj output ‖x‖ | T5 text_input_proj output ‖x‖ | a2t attn mass | attn output norm | \|gate_msa\| |
|---|---|---|---|---|---|
| **P8 healthy** | **6.924** | **160.66** | 0.253 | 59.7 | 1.42 |
| EXP-A LP-MC strip | 0.136 (–98%) | 5.84 (–96%) | 0.309 | 45.5 | 1.10 |
| EXP-B Qwen slot-0 | 0.105 (–98%) | 2.26 (–99%) | 0.313 | 27.7 | 0.95 |
| EXP-C Qwen+prefix | 0.070 (–99%) | 0.27 (–99.8%) | 0.333 | 25.9 | 0.85 |

Input-space inter-prompt cos (sanity, before any projection): CLAP 0.338, T5 per-token 0.360.

### What this shows (claim-discipline tier 1, can write)

- Collapsed models still assign **nontrivial cross-attention mass** to text tokens
  (0.25–0.33, even higher than healthy P8).
- The failure is NOT a routing/gating issue and is NOT "model ignores text" at the
  attention-weight level.
- Collapsed models show **catastrophic shrinkage of text-projection activations**
  along BOTH the CLAP pooled-cond pathway and the T5 token-projection pathway,
  by factors of **50–600× vs healthy P8**.
- Cross-attention operates over nearly-muted text representations.
- H12 (CLAP inter-prompt cos clustering bottleneck) is NOT supported — P8 has the
  HIGHEST cond inter-prompt cos (0.75) and remains healthy.

### Cannot write yet (needs confirmation)

- "The model learned to zero out text projections during training" — this is a
  plausible inference but distinguishes between two mechanisms:
  - (M1) Projection **weights** shrink to near-zero (learned shortcut)
  - (M2) Projection weights are normal but produce small outputs due to LayerNorm
    cancellation, bias offsets, or input direction mismatch
- Whether the projection shrinkage happens in **Stage 1 (FluxAudio)** or only after
  **Stage 2 (MeanFlow distillation)** is unknown. If S1 is already collapsed → caption
  regime makes base flow matching unlearnable. If S2 only → MeanFlow distillation
  objective amplifies the ignore-text shortcut.

### Sanity checks queued (cheap, eval-only)

1. **Projection weight-norm audit**: dump ‖text_cond_proj.weight‖, ‖text_input_proj.weight‖
   (and biases) for all 4 S2 checkpoints + 4 S1 checkpoints. Confirms M1 vs M2.
2. **S1 vs S2 activation magnitudes**: same forward-pass probe on S1 (FluxAudio) ema_final
   ckpts. Localizes collapse to Stage 1 or Stage 2.

### Implications if mechanism confirmed

- Paper section "Mechanism of collapse" becomes much stronger: a measurable, single-table
  signature distinguishing healthy from 9/10 collapsed models.
- EXP-F (originally "Qwen rewrite to LP-MC style") splits into 4 candidate interventions
  depending on where shrinkage happens:
  - Stage-1 magnitude penalty / projection norm regularizer
  - Frozen text-projection initialization with large weights
  - LP-MC-style content rewriting
  - LP-MC anchor caption mixing

### Probe artifacts

- Script: `~/research/meanaudio_training/exp_d2_pathway_probe.py`
- Raw results: `~/research/meanaudio_training/exp_d2_pathway_results.json`


## EXP-C RESULT (2026-05-12 evening) — prefix string alone not sufficient (H12 implicit falsified)

Trained P-Qwen-Boilerplate (`p_qwen_slot0_boilerplate_stage2_200000`) on `qwen_slot0_boilerplate_train.tsv`
(all 251K Qwen slot-0 captions prepended with "The low quality recording features a ", first letter
lowercased). Same Stage 1+2 600K iter pipeline. NPZ regen via T5+CLAP on boilerplate-prepended
captions, audio mean/std from `npz_phase8v4`.

| Metric | EXP-C (Qwen+boilerplate) | EXP-B (Qwen slot-0, no boilerplate) | P8 (LP-MC healthy) |
|---|---|---|---|
| MusicCaps CLAP (LP prompt) | **0.0580** | 0.0615 | 0.1851 |
| Jamendo s42 CLAP (LP prompt) | **0.0554** | 0.0608 | 0.1409 |
| Qwen-jamendo CLAP (Qwen prompt) | **0.0747** | 0.0812 | 0.2246 |
| **Qwen-prompt boost (Δ)** | **+0.0193** | +0.0204 | +0.084 |
| Stage 2 final loss | 0.9860 | 0.9866 | 0.9867 |
| AES quality (CE/CU/PC/PQ) | 5.73/6.46/5.27/6.35 | 5.87/6.59/5.34/6.47 | normal |

**Prepending the LP-MC boilerplate string to Qwen captions does NOT rescue collapse.**
Qwen-prompt boost +0.019 falls squarely in the collapsed cluster (+0.019–0.022) — not near
healthy P8 (+0.084). EXP-C is the 9th collapsed model in the audit (9/10 training configurations).

→ **H12 (implicit) FALSIFIED**: The LP-MC anchor effect cannot be explained by the prefix
string alone. The FULL LP-MC writing-task style is necessary — constrained vocabulary throughout
the caption, consistent sentence structure, and the "fill-in" template framing that makes the
variable semantic content learnable. Grafting only the prefix onto a Qwen-style body fails.

## Updated cross-prompt eval matrix (full audit, n=9 collapsed + 1 healthy)

| Train | Caption type | LP-MC prompt | Qwen prompt | Δ Qwen−LP | Status |
|---|---|---|---|---|---|
| P8 (LP-MC NoQ single) | LP-MC writing 4-task random | 0.1409 | **0.2246** | **+0.084** | **healthy** |
| P9V1 (LP-MC NoQ multi-5) | LP-MC 5-cap | 0.0650 | 0.0837 | +0.019 | collapsed |
| P9V2 (LP-MC +Q multi-5) | LP-MC 5-cap | 0.0403 | 0.0622 | +0.022 | collapsed |
| P8-Qwen (Qwen NoQ single) | Qwen single | 0.0582 | 0.0776 | +0.019 | collapsed |
| P7V1-Qwen (Qwen +Q single) | Qwen single | 0.0598 | 0.0791 | +0.019 | collapsed |
| P9.5 V1 (Qwen NoQ multi-5) | Qwen 5-cap | 0.0597 | 0.0799 | +0.020 | collapsed |
| P4V2-Qwen (Qwen NoQ BC-single) | Qwen single (BC-selected) | 0.0596 | 0.0801 | +0.020 | collapsed |
| P-LPMC-destructured (EXP-A) | LP-MC boilerplate stripped | 0.0561 | 0.0781 | +0.022 | collapsed (induced) |
| EXP-B Qwen-Slot0-Fixed | Qwen slot-0 forced (same framing 251K) | 0.0608 | 0.0812 | +0.020 | collapsed (H11 dead) |
| **EXP-C Qwen+Boilerplate** | **Qwen slot-0 + LP-MC prefix prepended** | **0.0554** | **0.0747** | **+0.019** | **collapsed (H12 dead)** |

**G3 finding (9/10 models)**: The +0.019–0.022 collapsed cluster now holds across 9 training
configurations spanning: multi-cap LP-MC, all Qwen variants, boilerplate-stripped LP-MC, and
boilerplate-prepended Qwen. The only separator is whether the FULL LP-MC writing-task style
(not just prefix, not just structural uniformity) is present.

## What is left standing after EXP-A + EXP-B + EXP-C

- ✗ H1 eval prompt mismatch — falsified
- ✗ H2 Qwen captions generic — falsified (Qwen is more specific)
- ✗ H3 pipeline bug — falsified (5 checks)
- ✓ H10 LP-MC boilerplate as inductive anchor — **confirmed** (EXP-A)
- ✗ H11 Qwen 5-task framing variance — **falsified** (EXP-B)
- ✗ **H12 (implicit) prefix-alone sufficient** — **falsified** (EXP-C, MC CLAP 0.058)
- ◐ H7/H9/H12-mechanism CLAP cond clustering — still plausible, untested by intervention

## Narrowed anchor hypothesis (post EXP-C)

The "inductive anchor" requires more than the prefix string. Candidates for what makes the full
LP-MC style effective:
- **(a) Constrained vocabulary throughout**: LP-MC top-50 trigrams cover 99.5% of captions;
  the variable semantic part is embedded in a predictable grammatical frame
- **(b) Acoustic-structure-grounded content**: LP-MC describes concrete acoustic specifics
  ("Eb major", "shimmering hi hats", "driving bassline") vs Qwen's mood/style abstractions
- **(c) Consistent sentence-level template**: "The low quality recording features a [X] [Y], [Z]."
  — a fixed scaffold that makes the fill-in mapping learnable
- Prepending the prefix to Qwen (EXP-C) tests only (c)-head, not (a) or (b). Result: (c)-head
  alone is insufficient → (a) and/or (b) also contribute.

**Next candidate experiments:**
- **EXP-F (short, 1 day)**: Qwen captions rewritten to LP-MC acoustic-structure style
  (replace mood language with acoustic specifics). Tests hypothesis (b).
- **EXP-D/E**: eval-only probes (cross-attn entropy, intra-batch text distance) — cheapest
  path to mechanism discrimination without retraining.


## EXP-D3 WEIGHT + STAGE AUDIT (2026-05-13) — S1-origin; MLP dominant

**Status**: ✅ confirmed mechanism localization; updates EXP-D2 from "preliminary" to "established observation."

### Weight-norm audit (M1 vs M2)

| Cond | Stage | cond_proj ‖W‖_2 | cond_proj avg-mag | text_proj ‖W‖_2 | text_proj avg-mag |
|---|---|---|---|---|---|
| P8_healthy | S1 | 135.6 | 0.05134 | 178.4 | 0.05162 |
| P8_healthy | S2 | **146.4** | **0.05460** | **207.2** | **0.06010** |
| EXP_A_LPMCstripped | S1 | 110.7 | 0.03796 | 106.0 | 0.03083 |
| EXP_A_LPMCstripped | S2 | 112.2 | 0.03805 | 106.6 | 0.03081 |
| EXP_B_Qwen_slot0 | S1 | 104.2 | 0.03513 | 104.1 | 0.03024 |
| EXP_B_Qwen_slot0 | S2 | 104.9 | 0.03542 | 104.7 | 0.03063 |
| EXP_C_Qwen_boilerplate | S1 | 103.6 | 0.03423 | 103.4 | 0.02979 |
| EXP_C_Qwen_boilerplate | S2 | 103.3 | 0.03405 | 103.5 | 0.02972 |

**M1 verdict (weight shrinkage)**: Collapsed models have ~25–30% smaller weight L2 vs P8, NOT the ~99% collapse seen in activations. **M1 is a minor contributor at most.** The activation collapse cannot be explained by learned weight shrinkage alone.

**M2 dominant**: Weights are similar; the activation collapse must be driven by how the MLP stack inside each projection processes its inputs.

### Stage attribution (S1 vs S2 activation magnitudes)

| Cond | Stage | CLAP cond: Linear-only ‖x‖ | CLAP cond: Full ‖x‖ | T5: Linear-only ‖x‖ | T5: Full ‖x‖ |
|---|---|---|---|---|---|
| P8_healthy | S1 | 1.772 | 7.772 **(×4.4)** | 4.900 | 78.78 **(×16.1)** |
| P8_healthy | **S2** | **1.689** | **6.924** **(×4.1)** | **5.762** | **160.66** **(×27.9)** |
| EXP_A | S1 | 1.656 | **1.115 (÷1.5)** | 2.815 | **0.903 (÷3.1)** |
| EXP_A | S2 | 1.036 | **0.136 (÷7.6)** | 4.543 | **5.838 (mild)** |
| EXP_B | S1 | 1.495 | **0.482 (÷3.1)** | 2.678 | **1.304 (÷2.1)** |
| EXP_B | S2 | 1.058 | **0.105 (÷10.1)** | 2.997 | **2.256 (÷1.3)** |
| EXP_C | S1 | 1.546 | **0.558 (÷2.8)** | 2.161 | **0.448 (÷4.8)** |
| EXP_C | S2 | 0.993 | **0.070 (÷14.2)** | 2.099 | **0.267 (÷7.9)** |

**Key pattern**:
- P8 (healthy): Linear output → MLP **amplifies** (×4–28). First Linear carries ~1.7 magnitude; MLP boosts to 7–161.
- Collapsed models: Linear output is **similar** (0.9–1.7 range, comparable to P8 first-linear). But MLP **attenuates** (÷1.5–14) rather than amplifying.
- **The collapse is not in the first Linear layer — it is in the MLP/LayerNorm stack inside the projection module.**

**Stage attribution**:
- **Collapse is S1-origin** (FluxAudio stage). All 3 collapsed conditions already show cond_full <<< P8 after S1 alone:
  - EXP-A S1 cond_full = 1.11 (vs P8 S1 = 7.77)
  - EXP-B S1 cond_full = 0.48 (vs P8 S1 = 7.77)
  - EXP-C S1 cond_full = 0.56 (vs P8 S1 = 7.77)
- Stage 2 (MeanFlow distillation) **deepens** the collapse further (all 3 drop from S1→S2 cond_full) but did NOT originate it.

### Updated mechanistic summary (EXP-D2 + EXP-D3)

The caption regime in Stage 1 (FluxAudio) is sufficient to produce both the learned-weight-mild-shrinkage (M1, −25%) and the MLP-attenuation M2 pattern. The MLP inside the projection modules **reverses polarity** — from amplifying (as in P8) to attenuating. Stage 2 distillation further deepens this attenuation but does not create it.

This points toward the projection MLPs learning an active suppression of text signal during flow-matching training on anchor-unstable caption distributions. The first linear still produces reasonable magnitudes (0.9–1.7 at S2), so the problem is squarely in the learned nonlinear processing that follows.

### Artifacts

- Script: `~/research/meanaudio_training/exp_d3_weight_and_stage_audit.py`
- Results: `~/research/meanaudio_training/exp_d3_audit_results.json`


## EXP-D4 NULL RESULT (2026-05-13) — projection transplant makes things worse

**Status**: ✅ null result confirmed across all 3 models. Rules out projection-only fix as viable EXP-F.

### Experiment design

Transplanted ALL 10 projection keys (`text_cond_proj.*` + `text_input_proj.*`) from healthy P8 S2 into each collapsed model's S2 ckpt, leaving all joint_blocks / fused_blocks / q_embed / t_embed / r_embed untouched. Sanity probe confirmed projection activations now match P8 exactly (cond_mag = 6.924, text_mag = 160.66 for all 3 patched models).

### Results (MusicCaps CLAP, n=5521)

| Model | Original CLAP | Transplant CLAP | Δ |
|---|---|---|---|
| EXP-A LP-MC-stripped | **0.0608** | 0.0536 | **−12%** |
| EXP-B Qwen-Slot0-Fixed | **0.0615** | 0.0452 | **−27%** |
| EXP-C Qwen+Boilerplate | **0.0580** | 0.0550 | **−5%** |

All 3 transplants are **worse** than originals. Despite the projections now producing P8-level text activation magnitudes, performance degrades.

### Interpretation

The transplant injects 50–600× louder text signals into a downstream network (joint_blocks + fused_blocks) that **was trained from scratch on muted text representations**. The entire downstream weight distribution co-adapted during S1+S2 training to expect near-zero text inputs. Suddenly replacing the projections with P8-scale output causes prediction errors because the joint_blocks cannot interpret the new signal distribution.

**Projection collapse = symptom, not cause.** Downstream blocks are not able to use the restored P8-scale text projections without retraining, consistent with co-adaptation to muted text representations during S1+S2. The gate_msa finding (collapsed models |1.10–0.85| vs P8's |1.42|) is consistent with reduced text-pathway influence, though this does not directly prove the blocks "ignore" text. The whole network adapted together to the muted-text regime during training. Fixing only the output end of the projection modules (while leaving all downstream layers intact) breaks the co-adaptation.

### What this rules out for EXP-F

- ❌ **Projection-norm regularization at training time** (regularizing the projection weights won't fix downstream co-adaptation)
- ❌ **Frozen large-magnitude text-projection init** alone (same reasoning: downstream layers adapt around any fixed init if the caption regime is wrong)
- These interventions might help *if also* fixing the caption distribution — but they are NOT sufficient on their own as EXP-F candidates.

### What EXP-F must target

The co-adaptation is throughout the entire joint model, and it originates in Stage 1. The only viable repair path is **retraining from scratch (or from S1 init) with a caption distribution that provides the missing anchor stability**. Candidate interventions:

1. **LP-MC caption mixing in S1** — add some % LP-MC boilerplate-structured captions alongside Qwen to provide anchor signal during FluxAudio training
2. **S1 training-data anchor injection** — use LP-MC as the training caption source for S1, switching to Qwen only at S2 (or keeping LP-MC throughout)
3. **Qwen captions rewritten to LP-MC acoustic-structure style** (EXP-F reframed) — if rewritten Qwen retains the anchor property, this is the cleanest path

The transplant result is a strong positive finding for the research narrative: it provides mechanistic proof that the collapse is embedded in the joint attention blocks throughout the network, not localized to the text projection modules.

### Artifacts

- Transplant script: `~/research/meanaudio_training/exp_d4_transplant.py`
- Eval chain: `~/research/meanaudio_training/exp_d4_eval_chain.sh`
- Sanity results: `~/research/meanaudio_training/exp_d4_sanity_results.json`
- Eval outputs: `~/MeanAudio/eval_output/metrics/exp_{a,b,c}_p8proj_transplant_no_q_musiccaps/metrics.txt`


## EXP-F RESULT (2026-05-14) — 50% LP-MC/Qwen mixing did not rescue (G4)

**Status**: ✅ complete. MC CLAP **0.0610** — collapsed, no recovery.

### Intervention

Per-audio 50/50 caption source assignment (seed=42): each of 251,599 audio clips randomly assigned LP-MC (phase7_v1_train.tsv) or Qwen slot-0 caption. Mixed NPZ via symlinks. Actual split: 49.9% LP-MC / 50.1% Qwen. S1 400K + S2 200K, NoQ (identical config to EXP-A/B/C).

### Result

| Condition | Caption source | MC CLAP | Status |
|---|---|---|---|
| P8 (100% LP-MC NoQ) | LP-MC 4-task random | **0.1851** | healthy |
| EXP-A (LP-MC stripped) | LP-MC no-boilerplate | 0.0608 | collapsed |
| EXP-B (100% Qwen slot-0 NoQ) | Qwen slot-0 | 0.0615 | collapsed |
| EXP-C (Qwen+boilerplate prefix) | Qwen + LP-MC prefix | 0.0580 | collapsed |
| **EXP-F (50% LP-MC + 50% Qwen NoQ)** | **LP-MC + Qwen 50-50** | **0.0610** | **collapsed** |

EXP-F at 0.0610 is at the centroid of the collapsed cluster, indistinguishable from 100% Qwen training.

### Interpretation (G4 finding)

50% LP-MC exposure at S1 does not shift the optimization out of the Qwen-collapse attractor. LP-MC boilerplate density in EXP-F is ~20% (half of P8's ~45%), which appears below the phase-transition threshold needed to establish anchor conditioning. Possible explanations (not yet distinguished):
1. Qwen's higher vocab diversity dominates gradient signal even at 50% share
2. Per-audio random alternation prevents stable template formation across steps
3. ~20% anchor boilerplate density is below a phase-transition threshold (P8 had ~45%)

### 10-model audit summary (complete)

| Model | MC CLAP | Status |
|---|---|---|
| P8 LP-MC (healthy) | 0.1851 | healthy |
| All 9 others | 0.0580–0.0690 | collapsed |

**In the EXP/Qwen-collapse audit, all tested non-healthy-control variants collapsed** (MC CLAP 0.058–0.069; P7 V1 LP-Rnd-Q is not in this universe and remains healthy). Among tested configurations, only full LP-MC writing-task style has produced healthy conditioning; tested partial substitutes (prefix string alone, 50-50 mixing, single-slot Qwen uniformity) did not.

### Paper wording

> Merely mixing 50% LP-MC-style anchor captions is insufficient; 50% Qwen exposure during Stage-1 produces the same co-adaptation collapse as 100% Qwen training (MC CLAP 0.061 vs 0.062).

### Artifacts

- Eval: `~/MeanAudio/eval_output/metrics/p_expf_50mix_stage2_200000_no_q_musiccaps/metrics.txt`
- Logs: `~/logs/exp_f.log`, `~/logs/p_expf_50mix_stage2_200000.log`
- Data prep: `~/research/meanaudio_training/exp_f_mix_data_prep.py`
- TSV: `~/eval_tsvs_p100/exp_f_50mix_train.tsv`
- NPZ: `~/exps_nvme/npz_expf_50mix/`

---

## EXP-G DESIGN — LP-MC S1 → Qwen S2 (stage-localization test)

**Status**: queued. Pending PM approval.

### Motivation

EXP-A~F established that anchor formation must occur during training, and that it is absent in all Qwen-trained variants. EXP-F showed 50% LP-MC mixing at S1 is insufficient. The one untested factor is **stage localization**: does the anchor need to persist across *all* of S1, or is it enough to have formed it in S1 and then switch to Qwen in S2?

### Intervention

Reuse P8 healthy LP-MC S1 checkpoint (`phase8_stage2_200000`'s S1 half) — no new S1 training needed. Run S2 (200K iter) with Qwen slot-0 captions (same NPZ as EXP-B: `npz_qwen_slot0`), NoQ. Eval MusicCaps + Jamendo s42 LP prompt + Qwen prompt.

Cost: ~6.7h GPU (S2 only) + ~11 min eval. Cheapest remaining stage-isolation probe.

### Design

```
S1:  P8 LP-MC NoQ FluxAudio (400K) ← reuse existing ckpt, NO retraining
Migrate: standard S1→S2 migration
S2:  Qwen slot-0 captions, NoQ, 200K iter (cumulative 600K)
Eval: MusicCaps (n=5521) + Jamendo s42 (n=2048) LP + Qwen prompts
Probe: steering ratio (same 24-wav battery as prior probes)
```

Use Qwen slot-0 (not BC): keeps the cleanest single-framing control, consistent with EXP-B.

### Interpretation thresholds

| MC CLAP | Interpretation |
|---|---|
| 0.15–0.18 | S1 anchor formation is the dominant factor; Qwen S2 can be absorbed once anchor is in place |
| 0.09–0.15 | Partial: S1 anchor provides some protection but S2 Qwen regime partially erodes it |
| ~0.06 | Qwen caption regime in S2 is sufficient to collapse conditioning regardless of S1 anchor |

### Pre-cleared paper wording

**If CLAP ≥ 0.12**: "Stage-1 anchor formation under LP-MC writing-task supervision appears sufficient to preserve text conditioning even when Stage-2 training uses Qwen captions."

**If CLAP ~0.06**: "The LP-MC anchor formed in Stage 1 does not protect against caption-regime co-adaptation during Stage 2 training on Qwen captions."

Do NOT write either result as proof of a specific mechanism; report as behavior-level observation.

### Script template

```bash
EXP=p_expg_lpmcs1_qwens2
S1_ITER=400000; S2_ITER=200000; S_TOTAL=600000
S1_CKPT=exps/phase8_stage2_200000/...  # P8 S1 ckpt_last.pth path TBD

python migrate_stage1_to_stage2_ckpt.py \
    --s1_ckpt "${S1_CKPT}" \
    --s2_out  "exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ckpt_last.pth"

python set_training_stage.py --stage 2
torchrun --nproc_per_node=1 --master_port=23465 train.py \
    data=meanaudio model=meanaudio_s exp_id=${EXP}_stage2_${S2_ITER} \
    num_iterations=${S_TOTAL} lr_schedule_steps=[999999,999999] \
    batch_size=8 +accumulation_steps=1 learning_rate=1e-4 num_workers=4 \
    +use_rope=False +use_wandb=False +use_q_conditioning=false \
    val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
    data.AudioCaps_npz.tsv=<qwen_slot0_train.tsv> \
    ++data.AudioCaps_npz.npz_dir=/home/kojiek/exps_nvme/npz_qwen_slot0 \
    ++data.AudioCaps_npz.gt_cache=/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt \
    ++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4 \
    ++data.AudioCaps_val_npz.gt_cache=null
```
