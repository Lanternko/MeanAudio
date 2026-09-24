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
│   ├── runs/                         # disposable run_*.sh (gitignored; root one-offs moved to runs/root_oneoffs_2026-07_09/)
│   ├── flowmatching/, meanflow/      # minimal demo runners
│   └── train_mini.sh, extract_audio_latents.sh
│
├── research/                         # moved in from ~/research/meanaudio_* on 2026-09-24 (old paths are symlinks)
│   ├── training/                     # NPZ writers, multi-cap tools, EXP-A~H scripts, caption10s_pipeline
│   │                                 #   npz_phase7_clean/ npz_phase8v4/ outputs/ are data (gitignored, DO NOT delete)
│   └── eval/                         # phase4_eval.py (frozen, sha-bound), peav_eval.py, RTF benchmarks
├── runtime/                          # closed runtime dirs moved in (gitignored): mf_legacydup_quarter_runtime, eval_tsvs_p100
├── smoke_data/                       # small smoke/probe TSVs (merged with ~/smoke_data, now a symlink here)
├── deliverables/                     # listening packs / zips (gitignored)
├── workspace/                        # symlink index to live data outside the repo (gitignored) — see table below
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
│   ├── av-benchmark/                 # AV evaluation toolkit (gitignored)
│   ├── audio-ab-test/                # GitHub Pages blind A/B test (own repo; ~/audio-ab-test is a symlink)
│   └── ICME26-ATTM-GC-FluxAudio/     # ATTM challenge reference repo (~/reference-repos/... is a symlink)
│
├── exps/ → /home/kojiek/exps_nvme    # symlink — all checkpoints on NVMe
├── eval_output/ → /mnt/HDD/...       # symlink — generated audio outputs on HDD
└── weights/                          # CLAP/T5/model weights (mostly gitignored)
```

## Data outside the repo（`workspace/` 有同名 symlink）

這些目錄被 queue contract／腳本用絕對路徑大量引用（部分驗證閘會比對字串路徑），**原地保留不搬**，只從 `workspace/` 提供入口。

| `workspace/` 入口 | 實際路徑 | 內容 |
|------|------|------|
| `exps_nvme`（另有 `exps/`） | `~/exps_nvme/` | 訓練 checkpoint（NVMe 工作區） |
| `hdd_exps` | `/mnt/HDD/kojiek/meanaudio_exps` | checkpoint 歸檔區（exps_nvme symlink 目標，不可刪） |
| `eval_output_nvme` | `~/eval_output_nvme/` | 標準 eval（`mc_mf25_eval.sh`）輸出 |
| `eval_output_home` | `~/eval_output/` | MF recaption 輸出（實體目錄，不是 symlink） |
| `hdd_eval_output`（另有 `eval_output/`） | `/mnt/HDD/kojiek/MeanAudio_eval_output` | 舊 eval 音檔 |
| `text_overlays` | `~/text_overlays/` | caption arm 的 T5/CLAP overlay（hardlink 共用） |
| `logs` | `~/logs/` | 訓練／queue log |
| `gpu_queue` | `~/gpu_queue/` | p1/p2 queue、contracts、notifier |
| `nvme_experiment_artifacts` | `~/nvme_experiment_artifacts/meanaudio/` | 各 sweep／ladder／probe 的輸出 |
| `cfg0_eval_runtime` | `~/cfg0_eval_runtime/` | CFG0 harness 的 bindings／reports |
| `qwen_bucket_backfill_cfg3_fidelity8_runtime` | `~/qwen_bucket_backfill_cfg3_fidelity8_runtime/` | 054 backfill runtime |
| `hdd_jamendo_data` | `/mnt/HDD/kojiek/phase4_jamendo_data/` | 訓練／eval TSV 與 NPZ |

其他：`scripts/eval/eval_metrics.py` 是 **canonical** metric script；`research/eval/phase4_eval.py` 是凍結的舊版（歷史 contract 以 `/home/kojiek/research/meanaudio_eval/phase4_eval.py` 綁 sha，該路徑現為 symlink）。環境：`~/venvs/dac/`（主要）、`~/venvs/music_flamingo/`。

## Conventions

- **Canonical pipeline**: `bash train_pipeline.sh` from repo root. Experiment-specific variants live in `scripts/training_pipelines/`.
- **Eval q-flag**: `--no_q` for NoQ-trained models, `--quality_level N` for Q-trained. Mixing pollutes — see CLAUDE.md.
- **4-token naming** (paper-facing): `{Caption}-{Sel}-{Q}` (e.g. `LP-Rnd-Q`). Phase IDs internal-only.
- **Tmux for >5 min jobs**. Chain stages with `&&` plus `set -eo pipefail` (see `feedback_pipefail_silent_crash_2026_04_22.md`).
- **Never edit `meanaudio/model/networks.py:MeanAudio`** (Stage 2 architecture).
- **Never touch main repo from `.claude/worktrees/`** — always operate in `~/MeanAudio/`.
- **Generated clutter** belongs in gitignored or hidden paths (`output/`, `wandb/`, `.archive/generated-output/`, `.archive/wandb-offline/`). Source backups belong in `.archive/source-backups/`.
- **Multi-cap NPZ**: must use `npz_cache_train.txt` mapping + v2 manifest validation. Old Phase 9 caches are invalid (2026-07-16).
