# EXP-H: Qwen-to-LP-MC-style Rewrite

**Status**: spec v1 (2026-05-18)  
**Hypothesis**: Qwen captions are semantically aligned with audio (Groups 1–5 audit PASS) but collapse training. If the semantic content is preserved but the sentence style is transformed to LP-MC acoustic-structure format, the collapse should be avoided.

---

## Motivation

All Qwen variants tested (random/BC/slot0/boilerplate/prefix/50-50 mix) collapsed to MC CLAP ~0.06. The retrieval audit confirms Qwen captions ARE semantically aligned. Therefore the failure is in **caption style / training-regime learnability**, not content.

LP-MC specific style features (from corpus analysis of 251,599 captions):
- 40% open with "The low quality recording features a..."
- 18% open with "This is a [genre] piece/song."
- 9.5% open with "This audio contains..."
- Mean length: **42.5 words** (vs Qwen ~15-20 words)
- 59% contain "melody", 58% "bass" → dense acoustic vocabulary
- Concrete instrument names: kick, snare, hi hat, synth pad, arpeggiated X
- Fixed mood appendages: "It sounds [adj]." / "The recording is noisy and in mono."

Qwen style features:
- Prose-like, abstract mood: "creating a vibrant and upbeat mood"
- Genre labels + BPM/key metadata
- Short (~15-20 words)
- Some junk (JSON, prompt injection)

EXP-H directly tests: **same semantic content, LP-MC training-friendly style → does collapse disappear?**

---

## Experiment Design

### Rewrite Generation

**Script**: `~/research/meanaudio_training/gen_expH_rewrites.py`  
**Model**: Qwen2.5-Omni-3B in text-only chat mode (already in env, proven capable)  
**Input**: Qwen slot0 caption (from `phase9_omni_captions.jsonl`)  
**Output**: LP-MC acoustic-structure style rewrite  
**Style constraints**:
- Preserve semantic content (instruments, genre, mood, tempo)
- Use LP-MC opening pattern ("The low quality recording features a..." or "This is a [genre] piece.")
- Concrete instrument vocabulary (bass guitar, synth pad, punchy kick, shimmering hi hats, etc.)
- Avoid abstract mood prose as main content (can appear as suffix)
- Target 25–50 words
- Do NOT add instrument content absent in original caption

**Output TSV**: `~/eval_tsvs_p100/expH_rewrite_train.tsv` (id, caption, q_level=5)

### Sanity Phase (10K rewrite)

Before full training, do statistical validation on 10K random clips:

1. **top-50 trigram coverage** vs LP-MC baseline
2. **acoustic keyword density** (fraction containing melody/bass/drum/synth/kick/snare)
3. **caption length distribution** (target: 25–50 words, p50 ≥ 30)
4. **vocab/bigram entropy** (should approach LP-MC, away from Qwen)
5. **CLAP diagonal sim** (should stay ≥ Qwen slot0 = 0.312; if it drops below 0.20, style is destroying content)

Compare to 3 baselines:
- LP-MC (target)
- Qwen slot0 (input)
- EXP-C Qwen+prefix (failed control)

Gate: if CLAP diagonal ≥ 0.20 AND top-50 trigram coverage ≥ 50% of LP-MC → proceed to full training.

### Full Training (if sanity passes)

Training recipe: same as P8 (S1 400K + S2 200K, NoQ, random single-cap)  
NPZ: regenerate with T5+CLAP encoding of rewritten captions  
Exp name: `p_expH_rewrite`  
Eval: MC CLAP + Jamendo seed42 2048

### Expected results

| Outcome | Interpretation |
|---|---|
| MC CLAP ≥ 0.15 | Style is the bottleneck: LP-MC template makes Qwen content learnable |
| MC CLAP ~0.06 | Style alone not sufficient; deeper learnability issue |
| MC CLAP ≥ 0.18 (LP-MC level) | Full recovery: Qwen content + LP-MC style = LP-MC quality |

---

## Few-shot Prompt Design

8 Qwen→LP-MC example pairs. Cover: electronic, rock, classical, jazz, ambient, metal, folk, world.

Examples emphasize:
- Instrument extraction from abstract Qwen prose
- LP-MC opening pattern adoption
- Length expansion (15w → 35-45w)
- Concrete vocabulary substitution (mood → instrument list)

See `gen_expH_rewrites.py` for full prompt.

---

## File Locations

| File | Path |
|---|---|
| Rewrite script | `~/research/meanaudio_training/gen_expH_rewrites.py` |
| 10K sanity script | `~/research/meanaudio_training/expH_sanity_stats.py` |
| Rewrite TSV (full) | `~/eval_tsvs_p100/expH_rewrite_train.tsv` |
| Sanity result | `~/research/meanaudio_training/expH_sanity_results.json` |

---

## Status Log

- [ ] 10K rewrite generation  ← **NEXT: run gen_expH_rewrites.py**
- [ ] Sanity statistics (trigram, CLAP, length)
- [ ] Gate decision
- [ ] Full 251K rewrite
- [ ] NPZ generation
- [ ] Training
- [ ] Eval

**Scripts ready (2026-05-18)**:
- `gen_expH_rewrites.py` — Qwen2.5-Omni-3B text-only rewriter, 8 few-shot examples, batch=32
- `expH_sanity_stats.py` — trigram/CLAP/length/entropy gate checker
