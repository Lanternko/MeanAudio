# Qwen-trained collapse — root-cause diagnostic synthesis (2026-05-08)

## Standing puzzle

All MeanAudio models trained on **Qwen2.5-Omni-3B captions** collapse to MusicCaps CLAP ≈ 0.06 with steering ratio < 0.12, regardless of single/multi-cap or Q/NoQ configuration. LP-MC counterparts reach CLAP 0.18–0.20 with steering 0.9–1.7. This document audits 5 hypotheses (H1–H4 + H7/H9) using cheap diagnostics (no new training).

## Observation table

| Train | Caption | Q | MC CLAP | max steering ratio |
|---|---|---|---|---|
| P7 V1 | LP-MC single | +Q | **0.197** | **1.702** |
| P8 | LP-MC single | NoQ | **0.185** | **1.723** |
| P9 V1 | LP-MC multi-5 | NoQ | 0.0650 | 0.147 |
| P9 V2 | LP-MC multi-5 | +Q | 0.0403 | 0.056 |
| **P8-Qwen** | **Qwen single** | NoQ | **0.0611** | **0.120** |
| **P7V1-Qwen** | **Qwen single** | +Q | **0.0687** | **0.057** |
| **P9.5 V1** | **Qwen multi-5** | NoQ | **0.0609** | **0.044** |

P8-Qwen collapses despite being single-cap → multi-cap is not a necessary condition for collapse; **Qwen captions alone trigger it**.

## Cross-prompt eval matrix (Jamendo seed=42 2048 same audio)

| Train | Caption type | LP-MC prompt | Qwen prompt | Δ Qwen−LP | Status |
|---|---|---|---|---|---|
| P8 (LP-MC NoQ single) | LP-MC writing 4-task random | 0.1409 | **0.2246** | **+0.084** | text-conditioning healthy |
| **P9V1 (LP-MC NoQ multi-5)** | LP-MC 5-cap | 0.0650 | **0.0837** | **+0.019** | collapsed (~Qwen-level) |
| P8-Qwen (Qwen NoQ single) | Qwen single | 0.0582 | 0.0776 | +0.019 | collapsed |
| P7V1-Qwen (Qwen +Q single) | Qwen single | 0.0598 | 0.0791 | +0.019 | collapsed |
| P9.5 V1 (Qwen NoQ multi-5) | Qwen 5-cap | 0.0597 | 0.0799 | +0.020 | collapsed |

**G1 (5/8 06:50) addition — P9V1 + Qwen prompts = 0.0837.** Multi-cap LP-MC and ALL Qwen-trained variants converge to the same +0.019–0.020 Qwen-prompt boost. The +0.02 boost is now interpreted as a metric-level artifact (Qwen prompts may be slightly easier to score high CLAP under HTSAT-base) rather than a model-quality differentiator.

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
