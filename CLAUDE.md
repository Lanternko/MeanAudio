# MeanAudio — Claude Code 上手指南

## 專案概覽

**MeanAudio** 是一個文字驅動的音訊生成系統，採用兩階段訓練架構：
- **Stage 1**：FluxAudio（Flow Matching，單向 ODE）
- **Stage 2**：MeanAudio（Mean Flow，更快推理）

目前進行到 **Phase 6 V2**：quality-aware conditioning（q_embed，0~10 的品質信號從 Stage 1 就開始訓練）。

**下一個方向（Phase 7 候選）**：
1. Native q inference（每個 clip 用自己的 q_level）— 教授假設 FAD 可改善
2. 引入 Meta Audiobox Aesthetics 自動評分取代 FAD（若和聽感更相關）

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

### 客觀評估（CLAP + FAD，與 baseline 比較用）

**TSV 選擇**：
- `phase4_test.tsv`：只有 `id` + `caption`，搭配 `--quality_level N` 使用
- `phase6_test.tsv`：同一批 clip + `q_level` 欄位，native_q inference 用（不傳 `--quality_level`）

**Step 1：生成音訊（Jamendo test set，90k 筆，約 2 小時）**
```bash
# 固定 q_level（如 q=9）
python eval.py \
    --variant meanaudio_s \
    --model_path exps/EXPNAME/EXPNAME_ema_final.pth \
    --output eval_output/EXPNAME_jamendo/audio \
    --tsv /mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --quality_level 9 \
    --full_precision \
    2>&1 | tee ~/logs/EXPNAME_eval.log

# native_q（每個 clip 用自己的 q_level）
python eval.py \
    --variant meanaudio_s \
    --model_path exps/EXPNAME/EXPNAME_ema_final.pth \
    --output eval_output/EXPNAME_native_q_jamendo/audio \
    --tsv /mnt/HDD/kojiek/phase4_jamendo_data/phase6_test.tsv \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --full_precision \
    2>&1 | tee ~/logs/EXPNAME_native_q_eval.log
```

**Step 2：計算 metrics（CLAP + FAD）**
```bash
python ~/research/meanaudio_eval/phase4_eval.py \
    --gen_dir eval_output/EXPNAME_jamendo/audio \
    --exp_name EXPNAME \
    --num_samples 2048 \
    2>&1 | tee ~/logs/EXPNAME_metrics.log
```

結果存在 `eval_output/metrics/EXPNAME/metrics.txt`。

### 主觀評估（人耳，三首固定 prompt）

25 steps + cfg_strength 3.5，音質比單步好：

```bash
python infer.py \
    --variant meanaudio_s \
    --model_path exps/EXPNAME/EXPNAME_ema_final.pth \
    --output eval_output/EXPNAME_subjective \
    --encoder_name t5_clap --text_c_dim 512 \
    --use_meanflow --num_steps 25 --cfg_strength 3.5 \
    --quality_level 9 --full_precision \
    --prompt "PROMPT"
```

五首固定 prompt（`--output_name` 命名規則：`<phase>_<style>`）：
- **鋼琴**：`This is a piano cover of a glam metal music piece. The piece is being played gently on a keyboard with a grand piano sound. There is a calming, relaxing atmosphere in this piece.`
- **重金屬**：`This is the recording of a heavy metal music piece. There is a male vocalist singing melodically in the lead. The main tune is being played by the distorted electric guitar while the bass guitar is playing in the background. The rhythmic background consists of a simple acoustic drum beat. The atmosphere is aggressive.`
- **Lo-Fi 民謠**：`The low quality recording features a live performance of a folk song that consists of an arpeggiated electric guitar melody played over groovy bass, punchy snare and shimmering cymbals. It sounds energetic and the recording is noisy and in mono.`
- **EDM**：`This is an electronic dance music piece. There is a synth lead playing the main melody. The beat consists of a kick drum, clap, hi-hat and synthesized bass. The atmosphere is energetic and euphoric.`
- **Cinematic**：`This is a cinematic orchestral piece. There are strings playing a sweeping melody with brass accents. The piece builds in intensity with a dramatic crescendo. The atmosphere is epic and emotional.`

下載到本機（Mac terminal）：
```bash
scp -P 22 -r kojiek@140.122.184.29:~/MeanAudio/eval_output/EXPNAME_subjective/ ~/Downloads/EXPNAME_subjective/
```

### Baseline 數字（Jamendo test set，n=2048）

| Phase | CLAP ↑ | FAD ↓ |
|-------|--------|-------|
| phase4_v1 | 0.1957 | 1.5548 |
| phase4_v2 | 0.1929 | 1.5853 |
| phase6_v1 (q=9) | 0.1898 | 1.7628 |
| **phase6_v2 (q=9)** | **0.2139** | 2.5849 |
| phase6_v2 q=0 | TBD | TBD |
| phase6_v2 q=4 | TBD | TBD |
| phase6_v2 q=7 | TBD | TBD |
| phase6_v2 q=8 | TBD | TBD |
| phase6_v2 native_q | TBD | TBD |

> q-level sweep 進行中（tmux: `q_sweep`），結果在 `eval_output/metrics/phase6_v2_q*/metrics.txt`

---

## 注意事項

- **不要動 `meanaudio/model/networks.py` 裡的 MeanAudio 類別**（Stage 2 架構），只改 FluxAudio
- `run_*.sh` 已在 `.gitignore`，臨時腳本可用這個命名規則避免 commit
- `exps/`、`eval_output/`、`*.pth`、`*.flac` 都在 `.gitignore`，不會進 git
- 資料路徑：`/mnt/HDD/kojiek/phase4_jamendo_data/`

### Backward Compatibility（pre-Phase6 checkpoints）

Phase6 以前的 checkpoint 沒有 `q_embed.weight`。`load_weights()` 會自動偵測並將 q_embed **歸零**（不影響 global_c），並印出 WARNING。phase4_v2 / phase6_v1 等舊 checkpoint 可正常使用。
