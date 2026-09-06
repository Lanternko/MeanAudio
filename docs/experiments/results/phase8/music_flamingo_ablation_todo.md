# Music Flamingo Caption Ablations

Context: Music Flamingo captions are strong but verbose. The current MeanAudio
text path uses `max_length=77` with truncation, so the ablations below test
whether shorter, MeanAudio-friendly captions recover the LP-MC gap.

**Status 2026-07-16**: A1–A6 and C1 complete. All metrics under
`eval_output/metrics/`. Paper-facing LP-MC primary baseline remains P7 V1;
MF series is a captioner / style exploration track.

## TODO

- [x] **A1: MF-short-rewrite 10k** — completed 2026-05-27
  - Input: existing `music_flamingo_slice10_10k` captions.
  - Method: compress each caption to 35-60 words with genre/style,
    instruments/sounds, rhythm/energy, vocals, and mood early in the text.
  - Constraint: avoid hard truncation; verify FLAN-T5 token length stays within
    the 77-token MeanAudio conditioning window where possible.
  - Train: same IDs and recipe as MF10k/LPMC10k (`S1=100k`, `S2=50k`, NoQ).
  - Eval: MusicCaps + Jamendo heldout CLAP/AES + PE-AV.
  - Result 2026-05-27: MC CLAP `0.1381`, MC PE-AV `0.0102`; Jamendo CLAP
    `0.1534`, Jamendo PE-AV `0.0862`. AES improved strongly, but alignment
    worsened vs original MF10k. Deterministic keyword compression is too lossy.

- [x] **A2: MF-short-direct 10k** — completed 2026-05-29
  - Input: original 10s audio slices.
  - Method: ask Music Flamingo directly for compact 35-50 word training
    captions with concrete acoustic nouns first.
  - Train/eval: same as A1.
  - Pipeline: `scripts/training_pipelines/train_pipeline_music_flamingo_short_direct_10k.sh`
  - Result 2026-05-29: MC CLAP `0.1720`, MC PE-AV `0.0299`; Jamendo CLAP
    `0.1639`, Jamendo PE-AV `0.1005`. Direct short prompting recovers much of
    the alignment lost by post-hoc rewrite, with lower AES than rewrite.

- [x] **A3: MF-short-direct 100k** — completed 2026-05-31
  - Input: original 10s audio slices from `_QUARANTINED_phase4_train.tsv`.
  - Method: same direct compact Music Flamingo prompt as A2.
  - Train: 100k NoQ recipe (`S1=200k`, `S2=100k`).
  - Eval: MusicCaps + Jamendo heldout CLAP/AES + PE-AV.
  - Pipeline: `scripts/training_pipelines/train_pipeline_music_flamingo_short_direct_100k.sh`
  - Result 2026-05-31: MC CLAP `0.1840`, MC PE-AV `0.0411`; Jamendo CLAP
    `0.1748`, Jamendo PE-AV `0.1133`. Improves MusicCaps alignment over
    MF100k/LPMC100k, but CE/PQ remain lower than rewrite/static-random regimes.

- [x] **A4: MF-static-random-3cap 10k** — completed 2026-06-01
  - Input: same 10k IDs with three fixed caption sources: original MF,
    short-direct MF, and short-aesthetic MF.
  - Method: Phase7-like static random single-caption pick per ID
    (`multi_cap=False`), not dynamic multi-cap.
  - Train: 10k NoQ recipe (`S1=100k`, `S2=50k`).
  - Eval: MusicCaps + Jamendo heldout CLAP/AES + PE-AV.
  - Pipeline: `scripts/training_pipelines/train_pipeline_music_flamingo_static_random_3cap_10k.sh`
  - Queue: `scripts/training_pipelines/schedule_mfstatic3cap10k.sh`
  - Result 2026-06-01: MC CLAP `0.1752`, MC PE-AV `0.0300`; Jamendo CLAP
    `0.1713`, Jamendo PE-AV `0.1051`, Jamendo CE `6.2852`, Jamendo PQ
    `6.7009`. Static caption diversity restores strong aesthetic scores.
  - Eval-only follow-up 2026-06-02: same checkpoint on short-direct MF Jamendo
    prompts completed. Jamendo CLAP `0.1752`, PE-AV `0.1215`, CE `5.9246`,
    PQ `6.5074`. Short-direct prompts improve alignment but reduce aesthetic
    scores vs LPMC/Jamendo prompts.

- [x] **A4b: MF-static-random-3cap 10k original-MF eval-only** — completed 2026-06-04
  - Input: same checkpoint as A4
    (`mfstatic3cap10k_noq_fast_stage2_50000_ema_final.pth`).
  - Eval set: same Jamendo holdout IDs as A4.
  - Method: generate eval captions with Music Flamingo `slice10_v1` original
    verbose prompt, then eval-only against the A4 checkpoint.
  - Purpose: complete the prompt-style triangle for A4:
    LPMC/Jamendo vs short-direct MF vs original verbose MF.
  - Pipeline: `scripts/eval/eval_mfstatic3cap10k_original_mfcap_jamendo.sh`
  - Result: Jamendo CLAP `0.1752`, PE-AV `0.0916`, CE `5.9230`, PQ `6.5491`.
    Original verbose MF prompts match short-direct CLAP but sit between
    short-direct and in-train-style prompts on PE-AV / AES.

- [x] **A5: MF-static-random-3cap 100k** — completed 2026-06-04
  - Input: same 100k IDs with three fixed caption sources: original MF,
    short-direct MF, and short-aesthetic MF.
  - Method: generate missing `short-aesthetic 100k`, then apply the same
    Phase7-like static random single-caption pick per ID as A4.
  - Train: 100k NoQ recipe (`S1=200k`, `S2=100k`).
  - Eval: MusicCaps + Jamendo heldout CLAP/AES + PE-AV.
  - Pipeline: `scripts/training_pipelines/train_pipeline_music_flamingo_static_random_3cap_100k.sh`
  - Queue: `scripts/training_pipelines/schedule_mfstatic3cap100k_after_eval.sh`
  - Result 2026-06-04: completed via HDD-backed symlink exps. MC CLAP
    `0.1824`, MC PE-AV `0.0419`; Jamendo CLAP `0.1806`, Jamendo PE-AV
    `0.1159`, Jamendo CE `6.1458`, Jamendo PQ `6.6872`.

- [x] **A6: MF-expanded-3cap 100k-audio / 300k-caption** — completed
  - Input: same 100k audio IDs as A5 with all three Music Flamingo caption
    variants exposed: original, short-direct, short-aesthetic.
  - Method: prepare a 300k caption TSV plus 100k clip TSV, then use
    `extract_audio_latents.py --multi_caption` only during NPZ extraction.
    The resulting NPZ cache is ordinary single-caption rows and training still
    uses `multi_cap=False`. The former claim that this "avoids the Phase9
    dynamic multi-cap failure mode" was withdrawn on 2026-07-16 because the
    Phase9 cache was audio-caption misaligned.
  - Train: 100k-audio/300k-caption NoQ recipe (`S1=200k`, `S2=100k`).
  - Eval: MusicCaps + Jamendo heldout CLAP/AES + PE-AV, followed by eval-only
    on the existing short-direct MF Jamendo prompt set.
  - Pipeline: `scripts/training_pipelines/train_pipeline_music_flamingo_expanded_3cap_100k.sh`
  - Queue: `scripts/training_pipelines/schedule_mfexpanded3cap100k.sh`
  - Result:
    - MusicCaps: CLAP `0.1726`, PE-AV `0.0359`, CE `5.4998`, PQ `6.2944`
    - Jamendo (in-train 3cap style): CLAP `0.1766`, PE-AV `0.1138`, CE `6.0900`, PQ `6.7043`
    - Jamendo short-direct eval-only: CLAP `0.1859`, PE-AV `0.1337`, CE `6.0893`, PQ `6.6986`
  - Read: A6 does **not** beat A5 on MusicCaps CLAP (`0.1726` vs `0.1824`) or
    matched Jamendo CLAP (`0.1766` vs `0.1806`). Expanded multi-caption exposure
    did not help beyond A5's static style mixture; useful part of A4/A5 looks
    closer to style mixture / regularization than repeated multi-caption exposure.

- [x] **C1: LPMC100k on short-direct MF Jamendo prompts** — completed 2026-06-04
  - Input: `lpmc100k_noq_stage2_100000_ema_final.pth`.
  - Eval TSV:
    `music_flamingo_slice10_100k_short_direct_mfcap_jamendo_holdout2048.tsv`.
  - Method: eval-only, no new captions needed; use the exact same short-direct
    MF prompt set used by MF-short-direct 100k / A6 short-direct eval.
  - Purpose: close the fairness gap in the 100k comparison.
  - Pipeline: `scripts/eval/eval_lpmc100k_shortdirect_mfcap_jamendo.sh`
  - Result: Jamendo CLAP `0.2044`, PE-AV `0.1329`, CE `6.0594`, PQ `6.6853`.
    LP-MC-trained model still leads on short-direct MF prompts for CLAP
    (vs A3/A6 short-direct eval ~0.17–0.19), supporting that training caption
    distribution remains the dominant factor.

## Decision Gate

- If A1 closes most of the LPMC10k gap, the main issue is caption length/style.
- If only A2 improves, Music Flamingo needs short-caption prompting at audio
  caption time rather than post-hoc compression.
- If neither improves, the mismatch is deeper than verbosity/truncation.

### Verdict (post A1–A6 + C1)

1. **A1 post-hoc rewrite fails** on alignment; **A2 direct short prompt** is the
   right way to shorten MF captions.
2. **A3 short-direct 100k** reaches MC CLAP ~0.184 — competitive with A5 but
   weaker aesthetics than static-random mixtures.
3. **A4/A5 static-random 3-cap** restores AES while keeping alignment; **A6
   expanded 300k-caption rows do not improve over A5** on the main metrics.
4. **C1**: LPMC100k still beats MF-trained models under the same short-direct
   MF eval prompts on CLAP → training captioner/style gap remains primary.
