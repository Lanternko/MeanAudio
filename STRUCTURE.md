# MeanAudio Repo Structure

One-page map of the codebase. For setup / training commands see `CLAUDE.md`; for experiment status see `docs/experiments/phase_status.md`.

```
MeanAudio/
├── train.py                          # main training entrypoint (S1 or S2 per --variant)
├── eval.py                           # audio generation entrypoint (--variant + --use_meanflow)
├── infer.py                          # single-prompt inference (no --no_q flag — see CLAUDE.md NEVER)
├── demo.py                           # simple demo
├── test_meanaudio.py                 # minimal smoke test
├── set_training_stage.py             # patch runner to FluxAudio (S1) or MeanAudio (S2)
├── migrate_stage1_to_stage2_ckpt.py  # S1→S2 ckpt converter (needs ckpt_last.pth, not ema_final.pth)
├── train_pipeline.sh                 # CANONICAL two-stage pipeline (S1 → migrate → S2 → eval)
├── run_phase8_bugfix_full.sh         # orchestration: clean text re-extract → P8 NoQ bug-free retrain
│
├── CLAUDE.md                         # MUST READ — onboarding, NEVER list, q-flag rule, GPU policy
├── README.md / LICENSE / pyproject.toml
├── EXPERIMENT_LOG.md                 # early cumulative numbers (prefer docs/experiments/best_results.md)
├── STRUCTURE.md                      # this file
├── .gitignore
│
├── meanaudio/                        # the package
│   ├── model/                        # networks.py, mean_flow.py, flow_matching.py (DO NOT edit MeanAudio class)
│   │                                 #   + text_attention_mask in joint attention / mean pooling
│   ├── data/                         # extracted_audio.py, data_setup.py, eval/, extraction/
│   ├── ext/                          # external dependencies vendored
│   ├── runner_meanflow.py            # main training loop (S2)
│   ├── runner_flowmatching.py        # FluxAudio S1 runner
│   ├── eval_utils.py                 # generate_mf / generate_fm
│   └── utils/
│
├── config/                           # hydra configs (base, eval, train, data/, hydra/)
├── sets/                             # latent mean/std, test TSVs
├── data/                             # local symlinks to external data (mostly gitignored)
├── docs/                             # all experiment / metric / meeting notes
│   ├── experiments/                  # phase_status, best_results, MF ablations, qwen audits, etc.
│   ├── meetings/                     # 2026-MM-DD prof discussion notes
│   ├── eval/                         # subjective_prompts, etc.
│   ├── metrics/                      # audiobox_aesthetics, etc.
│   ├── literature/                   # Literature_Insights
│   └── reviews/                      # ISMIR 2026 paper 487 archive (ismir2026-487-promptcc/)
│
├── scripts/                          # all helper scripts (see scripts/README.md)
│   ├── training_pipelines/           # ~35 experiment-specific train pipelines (P4–P9, MF, EXP-H, …)
│   ├── eval/                         # ~17 eval batch scripts (q-sweeps, MF, baselines)
│   ├── preprocess/                   # caption sampling, MF prep, text re-extraction, A/B norm
│   ├── analysis/                     # subjective AES/CLAP scorers, slice CLAP, probe results
│   ├── legacy/                       # superseded but kept-for-reference (babysit, audit)
│   ├── runs/                         # disposable run_*.sh (gitignored)
│   ├── flowmatching/, meanflow/      # minimal demo runners
│   └── train_mini.sh, extract_audio_latents.sh
│
├── training/                         # training-related utilities (kept from upstream)
├── av-benchmark → .external/av-benchmark
│                                      # compatibility symlink used by eval scripts
│
├── .archive/                         # hidden historical / local low-priority material
│   ├── legacy/                       # former archive/: fix_scripts/, old_outputs/, old_scripts/
│   ├── source-backups/2026-05-31/    # former scattered *.bak / *.bak2 source snapshots
│   ├── generated-output/             # old local output/ WAV samples
│   └── wandb-offline/                # old local W&B offline runs
├── .external/                        # hidden external checkouts
│   └── av-benchmark/                 # AV evaluation toolkit (gitignored)
├── .side_projects/                   # hidden non-MeanAudio side projects
│
├── exps/ → /home/kojiek/exps_nvme    # symlink — all checkpoints on NVMe
├── eval_output/ → /mnt/HDD/...       # symlink — generated audio outputs on HDD
└── weights/                          # CLAP/T5/model weights (mostly gitignored)
```

## External paths (not in repo)

| Path | Contents |
|------|----------|
| `~/exps_nvme/` | training checkpoints (linked as `exps/`) |
| `/mnt/HDD/kojiek/MeanAudio_eval_output/` | generated audio (linked as `eval_output/`) |
| `/mnt/HDD/kojiek/MeanAudio_eval_output_OLD/` | archived old eval outputs (52 GB, moved 2026-05-16) |
| `/mnt/HDD/kojiek/phase4_jamendo_data/` | training TSVs and NPZ（部分 LP-MC 檔有 `_QUARANTINED_*` prefix） |
| `/home/kojiek/research/meanaudio_eval/phase4_eval.py` | CLAP/AES/PE-AV metric script |
| `/home/kojiek/research/meanaudio_training/` | NPZ writers, multi-cap tools, EXP scripts |
| `~/venvs/dac/` | primary Python env |
| `~/venvs/music_flamingo/` | Music Flamingo captioning env |

## Conventions

- **Canonical pipeline**: `bash train_pipeline.sh` from repo root. Experiment-specific variants live in `scripts/training_pipelines/`.
- **Eval q-flag**: `--no_q` for NoQ-trained models, `--quality_level N` for Q-trained. Mixing pollutes — see CLAUDE.md.
- **4-token naming** (paper-facing): `{Caption}-{Sel}-{Q}` (e.g. `LP-Rnd-Q`). Phase IDs internal-only.
- **Tmux for >5 min jobs**. Chain stages with `&&` plus `set -eo pipefail` (see `feedback_pipefail_silent_crash_2026_04_22.md`).
- **Never edit `meanaudio/model/networks.py:MeanAudio`** (Stage 2 architecture).
- **Never touch main repo from `.claude/worktrees/`** — always operate in `~/MeanAudio/`.
- **Generated clutter** belongs in gitignored or hidden paths (`output/`, `wandb/`, `.archive/generated-output/`, `.archive/wandb-offline/`). Source backups belong in `.archive/source-backups/`.
- **Multi-cap NPZ**: must use `npz_cache_train.txt` mapping + v2 manifest validation. Old Phase 9 caches are invalid (2026-07-16).
