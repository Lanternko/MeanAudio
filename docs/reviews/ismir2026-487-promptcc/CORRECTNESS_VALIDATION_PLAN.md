# PromptCC Correctness Validation Plan

**Status:** Deferred — no human-rating resources are currently available.

**2026-07-16 execution note:** A clean, bug-fixed NoQ baseline rerun is in
progress before the random-bin control. This is required because the historical
NoQ result used a mismatched q-null-token path and may overstate the PromptCC
gain. The clean Stage-1 checkpoint will be reused for the later random-bin
Stage-2 control.

## P0 TODO: clean all-five-caption / true-random control

**Tooling status 2026-07-16:** checklist items 1–3 are implemented in
`~/research/meanaudio_training/gen_multicap_npz.py` and
`validate_multicap_npz.py`. Generation now requires the canonical gt_cache,
writes mapped filenames plus a caption-hash manifest and attention masks, and
each NPZ embeds `clip_id`, `row_index`, and `caption_sha256`. The validator
defaults to a full provenance + mapped `mean/std` exact-equality audit. The
clean cache has not been generated yet because the GPU is occupied by the P8
clean baseline; items 4–6 remain pending.

CPU metadata preflight completed on the real 251,599-row corpus: zero missing
caption ids, every TSV caption belongs to its five-caption pool, row 0 maps to
`33.npz`, and the v2 manifest is prepared at
`/mnt/HDD/kojiek/phase9_multicap_npz_clean/MANIFEST.tsv` (32 MB). Disk
preflight reports about 630 GB free and about 413 GB required. Caption encoding
has not started and must wait for the active P8 GPU run.

Canonical clean-cache commands:

```bash
python ~/research/meanaudio_training/gen_multicap_npz.py \
  --tsv /mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv \
  --jsonl ~/research/music_cleaning/results_20260119_043407.jsonl \
  --gt-cache /mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt \
  --src-npz ~/research/meanaudio_training/npz_phase7_clean \
  --out-npz /mnt/HDD/kojiek/phase9_multicap_npz_clean --resume

python ~/research/meanaudio_training/validate_multicap_npz.py \
  --tsv /mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv \
  --jsonl ~/research/music_cleaning/results_20260119_043407.jsonl \
  --gt-cache /mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt \
  --src-npz ~/research/meanaudio_training/npz_phase7_clean \
  --npz-dir /mnt/HDD/kojiek/phase9_multicap_npz_clean --alignment-checks all
```

The historical Phase 9 true-random result (MusicCaps CLAP 0.0650) is invalid as
evidence about multi-caption training. The historical multi-caption cache
writer paired TSV row `i` with audio statistics from `src_npz/i.npz`, whereas
the canonical training manifest maps row `i` through `npz_cache_train.txt`
(for example, row 0 maps to `33.npz`). An audit on 2026-07-16 found zero exact
matches between the sequential filename and the canonical filename across all
251,599 training rows. The preflight validator checked counts, filenames,
keys, and tensor shapes, but not TSV--caption--audio semantic alignment.

The same writer was used for Phase 9.5 V1, whose pipeline also loaded the
multi-caption cache positionally. Its 0.0609 result is therefore excluded too;
the Qwen JSONL corpus remains valid, but that multi-cap checkpoint does not.

Consequences and claim restriction:

- The 0.0650 checkpoint learned from systematically mismatched audio--caption
  pairs; its high AES but low CLAP is compatible with learning the marginal
  music distribution while ignoring text.
- The bug-fixed Phase 9 rerun still reused the affected Stage-1 checkpoint and
  the same misaligned multi-caption cache.
- Until a clean rerun is complete, do not cite 0.0650 as evidence that exposing
  all five captions or true-random caption sampling harms TTM training.

Required rerun checklist:

1. Patch the multi-caption cache writer to require
   `npz_cache_train.txt` (or another explicit row-to-NPZ manifest) and copy
   audio `mean`/`std` from the mapped source filename.
2. Save and propagate `text_attention_mask`; use the corrected q-null-token
   and CFG clone paths.
3. Add a mandatory alignment audit that checks TSV ID/caption membership and
   exact audio `mean`/`std` equality against the mapped source NPZ. Counts and
   shapes alone are insufficient.
4. Retrain both Stage 1 and Stage 2 from scratch. The historical Stage-1
   checkpoint must not be reused.
5. Compare clean true-random and static-random models using the same caption
   pool, code revision, optimizer-step budget, inference settings, and NoQ
   configuration. Report MusicCaps and MTG-Jamendo CLAP/PE-AV/AES plus the
   same-seed prompt-steering probe.
6. If compute permits, separately test explicit five-caption expansion or
   deterministic cycling; dynamic random sampling and a five-times-expanded
   dataset are related but not identical interventions.

## Current conclusion

The current experiments do not establish that the PromptCC score measures
caption correctness. The score measures the stochastic self-agreement of one
captioning model in text-embedding space. High agreement can reflect a stable
correct description, but it can also reflect a stable error; low agreement can
reflect captioner noise, or several different but valid descriptions of the
same music.

Accordingly, the current paper should not claim that PromptCC:

- determines whether a caption is correct or trustworthy;
- detects hallucinated captions;
- directly measures audio-grounded caption quality; or
- teaches the TTM model how much to trust an individual caption.

A claim supported by the current evidence is:

> PromptCC measures the stochastic self-agreement of a captioning model, and
> this agreement metadata is empirically useful as an auxiliary condition in
> the evaluated LP-MusicCaps--MeanAudio training pipeline.

## Work possible without human evaluation

These analyses provide supporting evidence but must not be presented as proof
of correctness:

1. Measure the relationship between PromptCC and audio--text CLAP/PE-AV scores.
2. Use the unused MTG-Jamendo tags to test whether instruments, genres, vocals,
   and moods mentioned in captions are supported by the metadata. Tags are for
   analysis only and remain excluded from TTM training.
3. Compare the clip-level PromptCC score against per-caption centrality, because
   the current training procedure randomly selects one of five captions while
   assigning all five the same clip-level score.
4. Report high-agreement/low-audio-alignment and low-agreement/high-audio-
   alignment counterexamples to demonstrate the limits of the score.
5. Add a random-bin control with the same bin-frequency distribution as the
   true PromptCC labels. This tests whether the gain comes from consistency
   semantics or merely from an additional embedding/data partition.
6. If compute permits, repeat the baseline, PromptCC, and random-bin control
   across multiple training seeds.

Automated models or metadata are only proxies. They cannot replace human
groundedness judgments and should be described as convergent diagnostics.

## Deferred human validation

When human-rating resources become available, sample 400--600 clips across
PromptCC score quantiles and retain all five captions per clip. At least three
blinded listeners should independently rate:

- audio-grounded correctness;
- relevance;
- specificity;
- presence of hallucinated attributes; and
- whether a caption is different from the others but still valid.

Primary analyses should include:

- correlation between PromptCC and mean caption correctness;
- correlation between PromptCC and hallucination rate;
- rates of high-PromptCC/incorrect and low-PromptCC/correct counterexamples;
- comparison with per-caption centrality, CLAP, and PE-AV; and
- a mixed-effects model with participant and clip as random effects.

This experiment can establish whether PromptCC is a useful proxy for a
specified notion of correctness on the evaluated population. It cannot prove
that agreement and correctness are universally equivalent.

## Interpretation if correctness is not validated

A negative result would not make PromptCC useless. It would instead show that
the contribution is a training-utility result rather than a caption-correctness
result. The research remains useful if true PromptCC labels outperform both the
no-conditioning baseline and a frequency-matched random-bin control. The paper
should then frame the mechanism as unresolved and restrict generalization to
the tested captioner, backbone, and dataset.
