# 10-Model Collapse Audit: What We Ruled Out

**Date**: 2026-05-20  
**Purpose**: Summary table for professor meeting — all tested interventions on Qwen-content / LP-MC-perturbed models, organized by hypothesis tested.

> ⛔ **2026-07-16 correction**：P9 V1/V2 and P9.5 V1 multi-cap caches
> used positional audio lookup instead of `npz_cache_train.txt`; those
> checkpoints are excluded. Statements below involving P9/P9.5 as evidence
> for multi-cap effects or a shared LP-multi/Qwen mechanism are withdrawn.
> The valid single-cap, fixed-selection, mix, rewrite, and stage-localization
> experiments remain behavioral evidence about the tested Qwen regime.

---

## Post-Audit Caveat: Masked T5 Cache Bug Found Later

This audit ruled out row alignment, stale CLAP/T5 values, truncation, q wiring, and prompt mismatch. It did **not** cover the T5 padding mask. A later diagnostic found that MeanAudio cached 77 T5 hidden states but discarded the tokenizer `attention_mask`, so joint attention treated padding states as valid text tokens during training.

Patch implemented 2026-05-20:
- new NPZ writers save `text_attention_mask`
- the dataset loads the mask when present
- FM/MF training and eval pass the mask to the network
- joint latent+text attention masks padded text keys

The 10-model audit remains valid as a behavioral map of collapsed checkpoints, but "pipeline bug ruled out" should now be read narrowly: the previous checks ruled out cache value/ordering bugs, not the missing-mask training bug.

Follow-up status on 2026-05-21:

- Masked Qwen slot0 NPZ regenerated at `~/exps_nvme/npz_qwen_slot0_masked`
- Counts match exactly: TSV rows = `gt_cache` rows = NPZ files = 251,599
- Schema audit passed: `mean`, `std`, `text_features`, `text_features_c`, `text_attention_mask`
- Masked rerun launched as `p_qwen_slot0_masked_stage1_400000` / `p_qwen_slot0_masked_stage2_200000`
- Training log confirms `Loaded text attention masks: [251599, 77]`
- Latest observed progress: Stage 1 at roughly 205k / 400k iterations; Stage 2 not started

This follow-up is deliberately raw Qwen slot0, not cleaned Qwen. If it recovers, the missing-mask bug is a dominant causal factor. If it remains collapsed, the audit's caption-distribution hypotheses stay active after the cache/model interface bug is fixed.

---

## Healthy Baselines

| Model | Caption corpus | Architecture | MC CLAP | Steering ratio |
|---|---|---|---|---|
| **P8 / LP-Rnd-NoQ** | LP-MC original (251K, 4-task random) | single-cap, NoQ | **0.1851** | 0.91–1.72 |
| **P7V1 / LP-Rnd-Q** | LP-MC original (same) | single-cap, +Q (MeanSim signal) | **0.1975** | 0.10–0.21 |

> Steering ratio = (change in output when prompt changes) / (change from noise alone). Healthy models: > 0.9. Collapsed models: < 0.15.

---

## 10 Collapsed Models

### Group A — Qwen Caption Content (4 runs)
> **Question**: Is the collapse caused by something about the Qwen caption *content* (abstraction level, selection strategy, number of captions)?

| Experiment | What was varied | MC CLAP | Steering ratio | Hypothesis tested | Verdict |
|---|---|---|---|---|---|
| **P8-Qwen / Qwen-Rnd-NoQ** | Caption source: Qwen single-cap random. Architecture identical to P8. | 0.0611 | 0.06–0.12 | "Qwen low-conditioning behavior requires multi-cap" | ❌ Falsified for Qwen; does not estimate multi-cap effect |
| **P4V2-Qwen / Qwen-BC-NoQ** | Caption selection: best-consensus (highest pairwise sim across 5 slots) | 0.0611 | — | "Random-cap selection strategy drives Qwen collapse" | ❌ Falsified — best-consensus collapses identically |
| **P7V1-Qwen / Qwen-Rnd-Q** | Architecture identical to P7V1 (+Q conditioning with Qwen-local MeanSim) | 0.0687 | 0.04–0.06 | "Q conditioning rescues Qwen collapse" | ❌ Falsified — +Q does not help |
| **P9.5 V1 / Qwen-Multi-NoQ** | **INVALID: misaligned cache** | ~~0.0609~~ | 0.02–0.04 | multi-cap comparison | **Excluded** |

**Corrected finding**: Valid Qwen single-cap random/best-consensus and NoQ/+Q
controls remain in the low-CLAP range. Single vs multi-cap is not evaluated by
this audit after excluding P9.5.

---

### Group B — LP-MC Surface Style Perturbation (4 runs)
> **Question**: Which surface features of LP-MC captions are responsible for healthy training?

| Experiment | What was varied | MC CLAP | Steering ratio | Hypothesis tested | Verdict |
|---|---|---|---|---|---|
| **EXP-A / LP-MC-Stripped** | Removed all LP-MC boilerplate prefixes ("The low quality recording features a...") from healthy P8 training data | 0.0608 | — | "LP-MC content alone is sufficient; the template opener is superfluous" | ✅ Confirmed H10 — stripping the template *induces collapse* in otherwise healthy LP-MC training |
| **EXP-B / Qwen-Slot0-Fixed** | All 251K audio pinned to Qwen slot-0 caption (removes 5-task framing variance) | 0.0615 | — | "Qwen's multi-task framing variance (5 different task framings per audio) is the collapse cause" | ❌ Falsified — single-framing Qwen still collapses; collapse is in each caption, not in the variety |
| **EXP-C / Qwen+LP-MC-Prefix** | Prepended "The low quality recording features a " to every Qwen slot-0 caption | 0.0580 | — | "The LP-MC opening template string is the key anchor; adding it to Qwen is sufficient" | ❌ Falsified — prefix alone insufficient; the variable body (Qwen-style) still causes collapse |
| **EXP-H / Qwen→LP-MC-Rewrite** | Full acoustic-style rewrite of Qwen captions using LLM (Qwen2.5-Omni-3B), CLAP semantic preservation 99.2% | 0.0617 | — | "Qwen collapse = caption surface style mismatch (abstract mood vs acoustic structure); LP-MC acoustic-style rewrite fixes it" | ❌ Falsified — full style transfer with 99.2% semantic preservation still collapses |

**Finding (Group B combined)**: The LP-MC "anchor" is not reducible to any single surface feature. The opening template is *necessary* (EXP-A: removing it breaks healthy training) but *not sufficient alone* (EXP-C: adding it to Qwen does not fix collapse). Full LP-MC surface style rewrite (EXP-H) also fails. The property that makes LP-MC trainable does not live in surface style features accessible to LLM-based rewriting.

---

### Group C — Caption Distribution Mixing (2 runs)
> **Question**: Can we preserve the LP-MC anchor by mixing LP-MC captions into Qwen training, or by using LP-MC for one stage?

| Experiment | What was varied | MC CLAP | Steering ratio | Hypothesis tested | Verdict |
|---|---|---|---|---|---|
| **EXP-F / 50-50-Mix** | Per-audio random assignment: 50% LP-MC / 50% Qwen slot-0 (actual split 49.9% / 50.1%) | 0.0610 | — | "Mixing 50% LP-MC anchor captions alongside Qwen prevents collapse" | ❌ Falsified — 50% anchor (half of P8's ~45% boilerplate density) is below the threshold needed |
| **EXP-G / LP-S1→Qwen-S2** | Stage-separated: S1 trained with healthy P8 LP-MC weights; S2 trained with Qwen slot-0 captions | 0.0679 | 0.07–0.10 | "LP-MC Stage-1 anchor formation is sufficient; Qwen can be used safely in Stage 2" | ❌ Falsified — Stage-2 Qwen training erodes the LP-MC anchor; PE-AV score = −0.034 (negative) |

**Finding (Group C)**: The LP-MC anchor must be present throughout *both* training stages and at sufficient density. Diluting it 50% or limiting it to Stage 1 is insufficient; the model co-adapts to the Qwen caption regime during the Qwen-exposed stage regardless of prior LP-MC exposure.

---

## Summary: MC CLAP Cluster View

```
Healthy LP-MC training
  P8 / LP-Rnd-NoQ         ████████████████████  0.1851
  P7V1 / LP-Rnd-Q         ████████████████████  0.1975

Valid low-CLAP cluster (0.058–0.069) — 9 models
  EXP-C  Qwen+prefix                  ▌  0.0580
  EXP-A  LP-MC stripped               ▌  0.0608
  P9.5 V1  Qwen-Multi-NoQ       INVALID / excluded
  EXP-F  50-50 mix                    ▌  0.0610
  P8-Qwen  Qwen-Rnd-NoQ               ▌  0.0611
  P4V2-Qwen  Qwen-BC-NoQ              ▌  0.0611
  EXP-B  Qwen-Slot0-Fixed             ▌  0.0615
  EXP-H  Qwen→LP-MC rewrite           ▌  0.0617
  EXP-G  LP-MC S1→Qwen S2             ▌  0.0679
  P7V1-Qwen  Qwen-Rnd-Q               ▌  0.0687
```

Nine valid non-LP-MC-original configurations fall in a tight cluster at MC
CLAP 0.058–0.069 across caption selection, Q conditioning, style transfer, and
data mixing. The excluded multi-cap run cannot support a multi/single claim.

---

## Cross-Prompt Signature (Shared Collapse Mechanism)

Each model was evaluated with both LP-MC prompts and Qwen prompts on the same generated audio (Jamendo seed=42, n=2048). The Δ = Qwen prompt CLAP − LP-MC prompt CLAP:

| Model | LP prompt CLAP | Qwen prompt CLAP | Δ | Status |
|---|---|---|---|---|
| **P8 / LP-MC healthy** | 0.141 | **0.225** | **+0.084** | **healthy** |
| P9 V1 (LP-MC multi-5) | ~~0.065~~ | ~~0.084~~ | ~~+0.019~~ | **INVALID / excluded** |
| P8-Qwen / Qwen-Rnd-NoQ | 0.058 | 0.078 | +0.019 | collapsed |
| P7V1-Qwen / Qwen-Rnd-Q | 0.060 | 0.079 | +0.019 | collapsed |
| P9.5 V1 / Qwen-Multi-NoQ | ~~0.060~~ | ~~0.080~~ | ~~+0.020~~ | **INVALID / excluded** |
| P4V2-Qwen / Qwen-BC-NoQ | 0.060 | 0.080 | +0.020 | collapsed |
| EXP-A / LP-MC-Stripped | 0.056 | 0.078 | +0.022 | collapsed |
| EXP-B / Qwen-Slot0-Fixed | 0.061 | 0.081 | +0.020 | collapsed |
| EXP-C / Qwen+Boilerplate | 0.055 | 0.075 | +0.019 | collapsed |

All 8 collapsed models: **Δ = +0.019 to +0.022** (extremely tight).  
Healthy P8: **Δ = +0.084** (4× larger).

The +0.02 Qwen-prompt boost remains observable in several valid low-CLAP
checkpoints, but the excluded LP/Qwen multi-cap rows can no longer establish a
shared multi-cap mechanism or universality across all conditions.

---

## Mechanistic Evidence (EXP-D Series, eval-only probes)

Text projection activation magnitudes (n=8 prompts × 3 timesteps):

| Model | CLAP cond_proj ‖x‖ | T5 text_proj ‖x‖ | MLP behavior |
|---|---|---|---|
| **P8 healthy** | **6.92** (×4.1 vs linear-only) | **160.7** (×27.9) | **amplifies** signal |
| EXP-A LP-MC-Stripped | 0.14 (÷7.6) | 5.8 (÷3.1) | **attenuates** signal |
| EXP-B Qwen-Slot0-Fixed | 0.11 (÷10.1) | 2.3 (÷1.3) | **attenuates** signal |
| EXP-C Qwen+Boilerplate | 0.07 (÷14.2) | 0.27 (÷7.9) | **attenuates** signal |

Key findings:
- Collapsed models still assign **nontrivial cross-attention weight** to text tokens (0.25–0.33); the failure is NOT "the model ignores text"
- The failure is in the **text projection MLP**: healthy model amplifies text signals ×4–28×; collapsed models attenuate ÷2–14×
- This collapse is **S1-origin** (FluxAudio stage); S2 deepens it but does not originate it
- Transplanting healthy P8 text projections into collapsed models makes performance **worse** (−5% to −27%), because all downstream joint_blocks co-adapted to muted text representations during training — the collapse is distributed throughout the network

---

## Qwen Caption Quality Sanity (Alignment Audit, 2026-05-17)

To confirm that Qwen caption content is not intrinsically misaligned with audio:

| Condition | Diag CLAP sim | Shuffled sim | R@10 |
|---|---|---|---|
| **Qwen slot-0 captions** | 0.291 | 0.134 | **11.9%** |
| LP-MC captions (reference) | 0.309 | 0.134 | **12.1%** |
| Random baseline | — | — | ~0.49% |

**Qwen captions are semantically aligned with audio at LP-MC level** (R@10 11.9% vs 12.1% LP-MC; both 24× above random). Audio-caption mismatch is ruled out as the collapse cause.

---

## What the Evidence Rules Out

| Claim | Evidence |
|---|---|
| "Collapse is caused by Qwen captions being semantically misaligned with audio" | ❌ Alignment audit: Qwen R@10 = 11.9%, comparable to LP-MC 12.1% |
| "Qwen low-CLAP behavior requires multi-cap random-pick" | ❌ Valid Qwen single-cap P8-Qwen is already low (0.0611); this says nothing about the causal effect of multi-cap itself |
| "Collapse is caused by caption selection strategy" | ❌ Random (0.0611) = best-consensus (0.0611) |
| "Adding Q conditioning to Qwen-trained models fixes collapse" | ❌ Qwen-Rnd-Q (0.0687) = Qwen-Rnd-NoQ (0.0611) |
| "The LP-MC opening template string is the key anchor" | ❌ Prepending it to Qwen (EXP-C, 0.0580) does not rescue; but removing it from LP-MC (EXP-A, 0.0608) does cause collapse — template is necessary but not sufficient |
| "Full LP-MC acoustic surface style is the key anchor" | ❌ EXP-H full rewrite (99.2% CLAP-preserved, LP-MC acoustic vocabulary, target length) collapses at 0.0617 |
| "50% LP-MC anchor data prevents co-adaptation" | ❌ EXP-F (50-50 mix, 0.0610) collapses same as 100% Qwen |
| "LP-MC Stage-1 anchor survives Qwen Stage-2 training" | ❌ EXP-G (LP-MC S1→Qwen S2, 0.0679, PE-AV score = −0.034) collapses |
| "All relevant pipeline bugs were ruled out" | ❌ **Withdrawn**: the later canonical-mapping audit found the P9/P9.5 multi-cap cache bug; claims for unaffected runs must be scoped per cache |
| "Collapse is caused by train/eval prompt distribution mismatch (EXP-H trained on EXP-H captions, evaluated with LP-MC/MusicCaps prompts)" | ❌ **Cross-prompt eval (2026-05-20)**: EXP-H model evaluated with in-distribution EXP-H captions (from training set) gives CLAP **0.0317** — *lower* than 0.0617 (MC LP-MC prompts) and 0.0589 (JM LP-MC prompts). In-distribution prompts make CLAP worse, not better. |

---

## EXP-H Cross-Prompt Eval: In-Distribution Prompts (2026-05-20)

**Question (P1)**: All Qwen-trained models were evaluated using LP-MC / MusicCaps prompts, not Qwen-style captions. Could the low CLAP (~0.06) be an artifact of train/eval prompt distribution mismatch rather than genuine conditioning failure?

**Method**:
- Sampled 2048 rows from the EXP-H *training* set (`expH_rewrite_train.tsv`, seed=42)
- Generated audio with the EXP-H S2 EMA checkpoint using these EXP-H-style captions as prompts
- Evaluated CLAP(generated audio, EXP-H caption) using `phase4_eval.py --tsv expH_rewrite_crosseval_2048.tsv`
- Note: 0% overlap with seed42 test set (test/train use disjoint audio IDs)

**Result**:

| Eval condition | Prompt type | CLAP | AES_CE | AES_PQ |
|---|---|---|---|---|
| MusicCaps LP-MC prompts (n=5521) | Out-of-distribution | 0.0617 | 5.9605 | 6.5167 |
| Jamendo seed42 LP-MC prompts (n=2048) | Out-of-distribution | 0.0589 | 5.9205 | 6.4751 |
| **Training-set EXP-H captions (n=2048)** | **In-distribution** | **0.0317** | 5.9200 | 6.4751 |

**Interpretation**: Using in-distribution prompts gives CLAP = 0.0317 — *lower* than the already-collapsed 0.0617/0.0589. If the model had learned EXP-H-style conditioning and simply failed to generalize to LP-MC prompts, in-distribution CLAP would be higher. Instead it is lower. The collapsed model generates generic audio; more specific/detailed EXP-H captions (describing instrumentation, tempo, texture explicitly) score worse against generic audio than the somewhat-generic LP-MC MusicCaps prompts do.

AES scores are nearly identical across all three conditions (CE: 5.92–5.96, PQ: 6.47–6.52), confirming the same underlying audio quality/naturalness regardless of prompt — consistent with a model that ignores the text prompt and generates similarly-sounding generic audio for all inputs.

**Verdict**: Train/eval prompt distribution mismatch hypothesis **DEFINITIVELY FALSIFIED**. The collapse is genuine regardless of prompt style.

---

## Embedding Distribution Analysis (2026-05-20)

Script: `~/research/meanaudio_training/compare_caption_embedding_distributions.py`  
Results: `~/research/meanaudio_training/embedding_dist_results.json`  
n=2000 random captions per corpus, T5 (flan-t5-large mean-pool, 1024-dim) + CLAP (HTSAT-base, 512-dim).

### T5 distribution (cross-attention input)

| Corpus | mean ‖x‖ | offdiag cos | PCA top-5% |
|---|---|---|---|
| LP-MC | 0.877 | 0.779 | 48.2% |
| Qwen | 0.922 | 0.770 | 34.8% |
| EXP-H | 0.897 | 0.845 | 43.5% |

Inter-corpus centroid cos (T5): LP-MC↔Qwen=0.926 / **LP-MC↔EXP-H=0.967** / Qwen↔EXP-H=0.947  
1-NN (T5): EXP-H query → 99.0% self, 1.0% Qwen, 0.0% LP-MC

EXP-H is its own tight T5 cluster, centroid closer to LP-MC than Qwen (+0.041). The surface template rewrite shifted T5 representation toward LP-MC.

### CLAP distribution (conditioning signal `text_features_c`)

| Corpus | offdiag cos | PCA top-5% |
|---|---|---|
| LP-MC | **0.298** (most diverse) | 41.4% |
| Qwen | 0.397 | 51.5% |
| EXP-H | 0.376 | **55.7%** (most concentrated) |

Inter-corpus centroid cos (CLAP): LP-MC↔Qwen=0.907 / LP-MC↔EXP-H=0.918 / **Qwen↔EXP-H=0.929**  
1-NN (CLAP): Qwen query → 41.0% land in EXP-H; EXP-H query → 27.0% land in Qwen; LP-MC is isolated (only 9.5–10.0% of Qwen/EXP-H neighbors come from LP-MC).

EXP-H CLAP distribution is Qwen-like: geometrically nearer to Qwen (cos 0.929) than LP-MC (0.918), higher intra-corpus clustering (0.376 vs LP-MC 0.298), and more PCA-concentrated (55.7% vs LP-MC 41.4%).

### Interpretation

**In T5 space: EXP-H → LP-MC-like. In CLAP space: EXP-H → Qwen-like.**

This asymmetry is by design: EXP-H rewrote surface structure (which T5 encodes faithfully) while preserving semantic content (which CLAP captures, CLAP diag 99.2%). The result is LP-MC template wrapping Qwen semantics — T5 sees the template, CLAP sees the content.

FluxAudio uses `text_features_c` (CLAP) as the primary conditioning signal. EXP-H's CLAP distribution is Qwen-like: higher intra-corpus clustering → weaker per-sample gradient signal → text-projection pathway learns to attenuate rather than amplify, consistent with EXP-D3 mechanism. LP-MC's lower CLAP clustering (0.298) makes each audio's conditioning vector more distinctive → stronger gradient → stable conditioning.

**This supports H12 (CLAP clustering bottleneck) from the root-cause doc**, which was previously "plausible but untested by intervention." EXP-H collapsed at Qwen CLAP-clustering level, while LP-MC at its own lower level remains healthy. The rewrite could not change CLAP distribution because CLAP semantic preservation was its design constraint.

**Claim discipline**: This is supporting evidence for H12, not causal proof. A direct causal test would require training with LP-MC CLAP embeddings but Qwen T5 embeddings (or vice versa), which would require a different NPZ encoding scheme.

---

## Working Hypothesis

> Although Qwen captions exhibit valid audio-caption alignment and can be rewritten into LP-MC-like acoustic surface form without losing CLAP alignment, models trained on these rewritten captions still collapse to the same low-CLAP regime. The embedding distribution analysis (2026-05-20) provides a mechanistic candidate: EXP-H shifted T5 distribution toward LP-MC (centroid cos 0.967 vs Qwen's 0.926), but CLAP conditioning distribution remained Qwen-like (EXP-H ↔ Qwen cos 0.929, closer than EXP-H ↔ LP-MC 0.918). LP-MC's distinctively low intra-corpus CLAP clustering (offdiag cos 0.298 vs Qwen/EXP-H 0.376–0.397) may provide stronger per-sample conditioning gradients, keeping the CLAP projection pathway from collapsing during Stage 1 training.

**Most consistent explanation (supporting evidence, not causal proof)**:
The CLAP conditioning pathway (`text_features_c`) is the critical dimension for healthy text conditioning. LP-MC captions are more diverse in CLAP embedding space than Qwen/EXP-H, which creates stronger gradient signal for the text-projection MLP during Stage 1. This supports H12 ("CLAP cond clustering bottleneck") from the root-cause diagnostic, now with EXP-H as a direct data point: a model trained with LP-MC T5 embeddings but Qwen-like CLAP embeddings still collapses.

**Remaining open question**: Can training stability be recovered by reducing CLAP clustering to LP-MC levels while keeping Qwen semantic content? This would require re-encoding training captions using LP-MC CLAP embeddings with Qwen T5 embeddings (a mixed NPZ scheme not yet tested). The alternative is that LP-MC corpus-level properties beyond CLAP clustering (e.g., audio granularity match, vocabulary constraint, or unknown structural factors) are the true necessary condition.

---

## Not to Write

- "Qwen captions are bad for TTA generally" — they are bad for *this particular* S1+S2 recipe at *this* scale
- "LP-MC style is fully understood" — we know it's necessary but cannot isolate which property within LP-MC style is causally sufficient
- "Acoustic rewrite proves content mismatch" — EXP-H falsified the content-mismatch framing; the failure persists despite 99.2% semantic preservation
- "V2 multi-cap Qwen training will also fail" — true operationally (P9.5 V2 was skipped on sequential gate), but this is a behavior prediction, not a claimed mechanism

---

*Experiments completed 2026-05-09 through 2026-05-19. The 2026-07-16 audit
retains 9 valid configurations and excludes P9.5 V1 because its multi-cap
cache was misaligned.*
