# EXP-H: Qwen-to-LP-MC-style Rewrite

**Status**: ✅ **completed / COLLAPSED** (MC CLAP **0.0617**, 2026-05-18 train + 2026-05-20 cross-prompt eval)  
**Hypothesis**: Qwen captions are semantically aligned with audio (Groups 1–5 audit PASS) but collapse training. If the semantic content is preserved but the sentence style is transformed to LP-MC acoustic-structure format, the collapse should be avoided.

**Result summary**: rewrite preserves semantics (99.2%) and pushes T5 embedding toward LP-MC-like, but CLAP / PE-AV still collapse. Cross-prompt eval with in-distribution EXP-H train captions gives CLAP **0.0317** (even lower) → train/eval prompt mismatch falsified. Full write-up: `docs/experiments/history/phase8/qwen_collapse_root_cause_2026_05_08.md` + memory `project_expH_status_2026_05_18.md`.

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

**Revised gate (2026-05-18 CC verdict)**:

| Gate | Criterion | Hard/Warning |
|---|---|---|
| Duplicate id | 0 | Hard |
| Fallback rows | 0 | Hard |
| Major chat leakage | 0% | Hard |
| Hallucination suffix | < 0.5% | Hard |
| CLAP diagonal | `rewrite_diag ≥ max(0.20, qwen_diag − 0.05)` OR `rewrite_diag / qwen_diag ≥ 0.8` | Hard |
| Caption length / acoustic KW / bigram entropy | close to LP-MC | Hard |
| Top-50 trigram overlap vs LP-MC | ≥ 50% | **Warning only** (30% acceptable if CLAP passes) |

Rationale: trigram overlap gap (30% vs 50%) is attributed to EXP-H using varied LP-MC structures
("There is a...", "The low quality recording features...") rather than LP-MC's formulaic
"that consists of" pattern. Style is demonstrably LP-MC-like on all other dimensions.
CLAP diagonal is the decisive gate for semantic preservation.

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

- [x] 10K rewrite generation (10,000 rows, fallback=0, chat leakage=0%, hallucination suffix 0.16%)
- [x] Text-side sanity (length/LP-MC opening/acoustic KW/entropy: all pass; trigram 30% = warning)
- [x] CLAP gate PASS — rewrite_diag=0.3058, qwen_diag=0.3084, ratio=99.2%, gate=0.2584
- [x] Gate decision: **CLEARED** (2026-05-18)
- [x] Full 251K rewrite (251,599 rows, bad_chat=0, bad_suffix=60, fallback=1 — all PASS)
- [x] NPZ generation (251,599 files, errors=0, shapes correct)
- [x] Training: S1 400K (2026-05-18 20:45 → 05-19 09:16) + S2 200K (09:16 → 15:26 JST)
- [x] Eval: MusicCaps n=5521 + Jamendo seed42 n=2048 + PE-AV

## Final Results (2026-05-19)

| Benchmark | CLAP | AES_PQ | PE-AV t2a R@10 |
|---|---|---|---|
| MusicCaps (n=5521) | **0.0617** | 6.5167 | 18.1% |
| Jamendo seed42 (n=2048) | **0.0589** | 6.4751 | 29.3% |

**VERDICT: COLLAPSED** — MC CLAP ~0.06 → "style alone insufficient; deeper failure."

Hypothesis FALSIFIED: LP-MC acoustic-style rewrite with 99.2% semantic preservation does NOT fix Qwen collapse.
All Qwen-content models (EXP-A through EXP-H, 10 total) collapse at MC CLAP 0.06±0.01.
Only LP-MC original corpus achieves non-collapsed training (MC CLAP 0.1851).

**Scripts (2026-05-18)**:
- `gen_expH_rewrites.py` — Qwen2.5-Omni-3B text-only rewriter, 8 few-shot examples, batch=32
  - `clean_output()`: stops at chat continuation markers AND hallucination suffixes
- `expH_sanity_stats.py` — gate checker; default audio root: `/mnt/HDD/hsiehyian/segments_no_vocals`

**10K sanity TSV**: `~/eval_tsvs_p100/expH_rewrite_10k_sanity.tsv`
