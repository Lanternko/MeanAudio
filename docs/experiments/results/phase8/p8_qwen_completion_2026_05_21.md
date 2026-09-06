# P8-Qwen Completion Note

Date: 2026-05-21

## Summary

P8-Qwen is complete. The training, metric evals, PE-AV evals, and steering backfill all finished, but the result is a collapse.

The important valid conclusion is that Qwen single-cap NoQ has low prompt
conditioning under the P8 recipe. **2026-07-16 correction**：P9.5's multi-cap
cache was misaligned, so this result can no longer be phrased as "removing
multi-cap did not recover" or as an isolated single-vs-multi comparison.

## Experiment

| Field | Value |
| --- | --- |
| Experiment id | `p8_qwen_stage2_200000` |
| Setup | Qwen single-cap random pick, NoQ |
| Baseline being matched | P8 LP-MC random single-cap NoQ |
| Stage 1 | FluxAudio, 400k iters |
| Stage 2 | MeanFlow, 200k iters |
| Train TSV | `/mnt/HDD/kojiek/phase4_jamendo_data/qwen_singlecap_random_train.tsv` |
| Train NPZ | `/home/kojiek/phase9_5_random_singlecap_npz` |

## Completion Evidence

| Artifact | Evidence |
| --- | --- |
| Stage 1 checkpoint | `exps_nvme/p8_qwen_stage1_400000/p8_qwen_stage1_400000_ema_final.pth` |
| Stage 2 checkpoint | `exps_nvme/p8_qwen_stage2_200000/p8_qwen_stage2_200000_ema_final.pth` |
| Stage 1 log | `~/logs/p8_qwen_stage1_400000.log` ends with synthesized EMA saved |
| Stage 2 log | `~/logs/p8_qwen_stage2_200000.log` ends with synthesized EMA saved |
| Backfill log | `~/logs/p8_qwen_backfill.log` ends with `backfill done` |
| Active process check | No active P8-Qwen / Qwen training or eval process was found on 2026-05-21 |

The Stage 1 final EMA was written on 2026-05-05 22:34. The Stage 2 final EMA was written on 2026-05-06 11:36. The PE-AV and steering backfill completed on 2026-05-06 15:43.

## Metrics

| Eval | CLAP | PE-AV mean | t2a R@10 | a2t R@10 | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| MusicCaps | 0.0611 | -0.036807 | 0.254 | 0.217 | collapsed |
| Jamendo s42, LP prompt | 0.0582 | 0.008300 | 0.439 | 0.342 | collapsed |
| Jamendo s42, Qwen prompt | 0.0776 | 0.084845 | 0.195 | 0.439 | collapsed |
| Healthy P8 control, Qwen prompt | 0.2246 | 0.193157 | 10.303 | 10.791 | healthy |

Saved metric files:

- `MeanAudio/eval_output/metrics/p8_qwen_stage2_200000_musiccaps/metrics.txt`
- `MeanAudio/eval_output/metrics/p8_qwen_stage2_200000_musiccaps_peav.json`
- `MeanAudio/eval_output/metrics/p8_qwen_stage2_200000_jamendo_s42/metrics.txt`
- `MeanAudio/eval_output/metrics/p8_qwen_stage2_200000_jamendo_s42_peav.json`
- `MeanAudio/eval_output/metrics/p8_qwen_stage2_200000_qwen_random_jamendo_s42/metrics.txt`
- `MeanAudio/eval_output/metrics/p8_qwen_stage2_200000_qwen_random_jamendo_s42_peav.json`
- `MeanAudio/eval_output/metrics/phase8_stage2_200000_qwen_random_jamendo_s42/metrics.txt`
- `MeanAudio/eval_output/metrics/phase8_stage2_200000_qwen_random_jamendo_s42_peav.json`

## Steering Probe

The steering probe is noise-dominant, not prompt-dominant:

| Pair | Ratio |
| --- | ---: |
| Instrument | 0.120 |
| Vocals | 0.055 |
| Drums | 0.040 |
| Density | 0.033 |

Reference interpretation:

- Healthy P8 LP-MC single-cap: roughly 0.91-1.72, prompt-dominant.
- Collapsed models: below roughly 0.15, noise-dominant.
- P8-Qwen max ratio is 0.120, so it falls in the collapsed cluster.

## Interpretation

P8-Qwen gives the clean single-cap control for the Qwen rerun:

1. The model finished training and evaluation; this is not an incomplete-run artifact.
2. Qwen single-cap NoQ lies in the same low CLAP region as other valid Qwen single-cap/selection controls; the invalid Qwen multi-cap row is excluded.
3. Qwen-prompt eval gives only the known small metric lift, from 0.0582 to 0.0776 on Jamendo s42, while healthy P8 reaches 0.2246 on the same Qwen-prompt eval.
4. The steering probe confirms the CLAP result: prompt changes produce much smaller output movement than noise changes.

Therefore, the supported paper wording is only that the tested Qwen caption
regime is associated with muted-text conditioning even in a single-cap NoQ
setup. The effect of multi-cap random-pick remains unknown pending a clean rerun.

Do not overclaim that Qwen captions are inherently unusable for audio generation. The supported claim is narrower: under this two-stage MeanAudio training recipe and these data encodings, Qwen single-cap training did not learn healthy prompt conditioning.
