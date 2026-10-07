# AES extended campaign — frozen before scoring

## Questions

1. Does the noise penalty depend on foreground instrument, register, and SoundFont?
2. Does PQ or CE penalize melody order destruction and misaligned functional harmony when note inventory, note count, timing and held duration are controlled?
3. Is lower background energy specifically beneficial, or does removing a known musical sustained bus reduce the score?
4. Can the published 10-second window aggregation detect the order of three 10-second blocks?
5. Do effects replicate across an independently seeded set of compositions and transfer to the existing karaoke song clips?

## MIDI factorial

64 deterministic composition templates, 32 discovery and 32 validation. Each includes a chord-aware lead, diatonic backing triads and bass, 96/112/128/144 BPM, major/minor modes, eight tonics, six progression patterns, varying rhythm and velocity. The fixed 30-second timeline contains approximately 28 seconds of music. Templates are composed programmatically; they are not independent human songs or training runs. The two partitions are frozen replication sets, with no score-based selection.

Two SoundFonts (GeneralUser GS, FluidR3 GM), TinySoundFont, no added effects. Main factorial: six foreground patches × three whole-score octave registers × seven conditions (clean, white and pink at SNR 40/30/20 dB). Other backing instruments remain guitar and acoustic bass. Instrument effects therefore refer to the foreground within this arrangement, not a universal instrument ranking. Noise is added to the mono16 waveform; unlike the earlier pilot, the set SNR is the AES input SNR before overall gain normalization.

Each template/bank has 149 cases, totaling 19,072 thirty-second audio files and 38,144 raw/normalized four-axis file scores, corresponding to 114,432 ten-second network windows. Stored audio is float32 mono16, resampled from stereo44.1 synthesis. The actual MIDI is read back to schedule rendering. Released voices are reset and discarded for 100 ms between renders; initial 490 ms must remain silent for clean synthesis.

## Controlled contrasts and limits

- Melody shuffle/reversal preserves the lead pitch histogram, velocity, every onset and duration. Repeated single note preserves onsets/durations but changes the pitch histogram.
- Backing bar rotation/shuffle preserves the backing pitch histogram, velocity and held duration, while changing its alignment with the lead. Bass moves with triads. Some shuffled progressions may remain musically acceptable. Chromatic backing shifts are separate compound interventions that change pitch classes.
- Arpeggiation delays chord voices while preserving each note's original duration and pitch. Polyphonic overlap changes; it is not a pure ordering-only intervention.
- Lead duration ×0.3/×1.7 deliberately changes held duration and release/overlap. Attack ramp and release trimming operate on a rendered lead bus, retaining MIDI and backing audio, but changing waveform energy/envelope. These are not hypothetical artifact-free changes.

Artifact-check note recorded during scoring: the ×1.7 case can overlap repeated pitches on the same MIDI channel. TinySoundFont note-off may release concurrent voices; its scheduled held duration is not guaranteed to equal the isolated acoustic held duration. Retain this planned compound case transparently, but do not use it to infer a pure sustain-duration mechanism. All other inventory contrasts must pass exact MIDI event checks.
- Split two/four instruments assigns the same lead events to different patches with the backing fixed.
- A known pad bus adds the same backing triads at amplitude 0.35. Pad −6/−12/off compare against this pad reference, not against the score with no added pad. This tests a musical sustained sound, not noise.
- Block reversal/rotation permutes exact 10-second sample blocks. Its raw score should be invariant under the published nonoverlapping 10-second average. Matched LUFS can change slightly because loudness measurement uses overlapping gating blocks at the new boundaries. No fades are inserted into this exact permutation control.
- Gain −6 dB should become effectively identical after normalization.

## Scoring and inference

Use the same pinned Audiobox-Aesthetics checkpoint as previous campaigns, four axes PQ/CE/CU/PC. Primary: remeasured gain-only −23 LUFS on the mono16 waveform. Secondary: raw render loudness. No compression, clipping or peak limiting. Record effective model precision/autocast, not only the predictor's unused precision label. Pin checkpoint hashes, code and design hashes, all audio/MIDI hashes.

Bootstrap templates, averaging bank/context replicates within template; 10,000 bootstrap samples. Report discovery, validation, pooled and SoundFont strata. Report all planned conditions, including null or reversed effects. Primary validation hypotheses are seven prespecified contrasts × four axes; paired sign-flip tests with Holm family-wise correction. Confidence intervals remain exploratory, not simultaneous. Factorial interactions are differences of paired noise effects, not observational correlations with timbre.

## Transfer and missing data

The planned generated-audio transfer uses 64 shared prompt filenames across three training seeds, control/N100 and CFG0/CFG3 fidelity negative arms, selected by SHA256 independent of outcomes. Original generation audio is absent from its archived directories at preflight. This branch is explicitly marked unavailable in design.json; old per-clip scores cannot substitute for edited audio. No model regeneration or canonical MeanAudio evaluation is launched by this sidecar. A later restored source would require a separately frozen source manifest before intervention scoring.

Karaoke transfer reuses the previously frozen 23 verse/chorus clips and their 4 eligible overlapping 30-second versions; section labels remain model predictions awaiting human verification. The paired primary inference clusters by song. Known noise additions, local sample order, EQ and low-energy gain attenuation are defined in a separate pre-score transfer manifest. EQ/quiet-frame changes cannot be described as verified noise removal.

## Compute and completion

Resume by atomically saved template manifests and append-only JSONL scores with audio hash verification. Record actual CUDA-event forward time and wall time, peak allocated VRAM and session score count; forward time is not total GPU-host occupancy or a utilization-weighted GPU-hours measurement. Do not rerun identical predictions just to increase GPU time. All final completion requires expected score coverage, controlled MIDI/event checks, normalization checks, identity/permutation controls, transfer results and reports.

Official primary references: [AES implementation](https://github.com/facebookresearch/audiobox-aesthetics), [AES paper](https://arxiv.org/html/2502.05139v1). Human preference and unreasonable metric bias still require independent blind listening; no such ratings have been collected here.
