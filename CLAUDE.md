# MeanAudio — Claude Code 上手指南

## 專案概覽

**MeanAudio** 是一個文字驅動的音訊生成系統，採用兩階段訓練架構：
- **Stage 1**：FluxAudio（Flow Matching，單向 ODE）
- **Stage 2**：MeanAudio（Mean Flow，更快推理）

目前進行到 **Phase 6 V2**：quality-aware conditioning（q_embed，0~10 的品質信號從 Stage 1 就開始訓練）。

---

## 目錄結構

```
MeanAudio/
├── meanaudio/
│   ├── model/
│   │   ├── networks.py        # FluxAudio / MeanAudio 模型定義（最常改）
│   │   ├── mean_flow.py       # MeanFlow / FluxLoss，含 loss() / fn()（最常改）
│   │   └── flow_matching.py   # FlowMatching ODE solver
│   ├── runner_meanflow.py     # 訓練迴圈（最常改）
│   ├── data/
│   │   └── extracted_audio.py # Dataset，__getitem__ 回傳 q_level
│   └── eval_utils.py          # generate_mf() / generate_fm()
├── train.py                   # torchrun 入口
├── eval.py                    # 評估入口（--quality_level, --use_meanflow）
├── infer.py                   # 單句推理
├── train_pipeline.sh          # 主訓練腳本（Stage 1 → migrate → Stage 2 → eval）
├── set_training_stage.py      # 切換 Stage 1 / Stage 2 模式（patch runner）
├── migrate_stage1_to_stage2_ckpt.py  # S1→S2 checkpoint 轉換
├── config/                    # Hydra 設定（model / data）
├── sets/                      # TSV eval sets（test-audiocaps.tsv 等）
├── scripts/                   # 參考腳本（eval_meanflow.sh 等）
└── archive/                   # 舊輸出和一次性腳本（不需要動）
```

---

## 環境

```bash
source ~/venvs/dac/bin/activate   # 啟動 Python 環境
export CUDA_VISIBLE_DEVICES=0      # 單 GPU 訓練
```

---

## Git / GitHub

- **Remote**: `https://github.com/Lanternko/MeanAudio.git`
- **Branch 策略**: 直接在 `main` 開發，每個 Phase 結束後加 tag
- **Git identity**: `lanternko <jerry86012@gmail.com>`
- **Auth**: token 已存在 credential store，直接 `git push` 即可

Commit message 格式：
```
phase6_v2: 簡短描述

- file.py: 做了什麼
- other.py: 做了什麼
```

---

## 訓練流程

```bash
# 開新 tmux session
tmux new -s phase6v2

# 啟動完整訓練（Stage 1 → migrate → Stage 2 → eval CLAP+FAD）
cd ~/MeanAudio && source ~/venvs/dac/bin/activate && bash train_pipeline.sh
```

`train_pipeline.sh` 的參數區塊（只需改這裡）：
```bash
EXP_PREFIX="phase6_v2"   # 實驗名稱
S1_ITERATIONS=400000      # Stage 1 steps
S2_ITERATIONS=200000      # Stage 2 steps
LEARNING_RATE=1e-4
```

---

## 關鍵架構：Quality Conditioning（Phase 6 V2+）

**q_embed**（`nn.Embedding(11, hidden_dim)`）：
- index 0~9 = 品質等級（0 最差，9 最好）
- index 10 = null token（unconditional pass 用）

**FluxAudio** (`networks.py`)：
- `self.q_embed` 在 `__init__` 中定義
- `predict_flow(... q=None)` → `q=None` 時填充 null token（index 10）
- `forward(... q=None)` 傳 q 給 predict_flow

**FluxLoss** (`mean_flow.py`)：
- `loss(... q=None)` → conditional pass 用 `q`，unconditional pass 用 null token

**Stage 切換** (`set_training_stage.py`)：
- `--stage 1`：patch runner 使用 FluxAudio + FluxLoss
- `--stage 2`：patch runner 使用 MeanAudio + MeanFlow

---

## Checkpoint 遷移（S1 → S2）

`migrate_stage1_to_stage2_ckpt.py` 做三件事：
1. 複製 `t_embed → r_embed`（Stage 2 新增的 embed）
2. **保留 Stage 1 已訓練的 q_embed**（Phase 6 V2+ 行為；舊 checkpoint 才隨機初始化）
3. 清除 optimizer / scheduler state

---

## Eval

```bash
# Stage 2 eval（MeanAudio，1 step）
python eval.py \
    --variant meanaudio_s \
    --model_path exps/EXPNAME/EXPNAME_ema_final.pth \
    --output eval_output/EXPNAME_q9/audio \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.9 --quality_level 9 \
    --tsv sets/test-audiocaps.tsv --full_precision

# CLAP + FAD
python av-benchmark/evaluate.py \
    --gt_cache ./data/audiocaps/test-features \
    --pred_audio eval_output/EXPNAME_q9/audio \
    --pred_cache eval_output/EXPNAME_q9/cache \
    --audio_length=10 --recompute_pred_cache \
    --skip_video_related \
    --output_metrics_dir=eval_output/EXPNAME_q9
```

---

## 注意事項

- **不要動 `meanaudio/model/networks.py` 裡的 MeanAudio 類別**（Stage 2 架構），只改 FluxAudio
- `run_*.sh` 已在 `.gitignore`，臨時腳本可用這個命名規則避免 commit
- `exps/`、`eval_output/`、`*.pth`、`*.flac` 都在 `.gitignore`，不會進 git
- 資料路徑：`/mnt/HDD/kojiek/phase4_jamendo_data/`
