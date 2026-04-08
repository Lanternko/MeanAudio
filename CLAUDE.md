# MeanAudio — Claude Code 上手指南

## 專案概覽

**MeanAudio** 是一個文字驅動的音訊生成系統，採用兩階段訓練架構：
- **Stage 1**：FluxAudio（Flow Matching，單向 ODE）
- **Stage 2**：MeanAudio（Mean Flow，更快推理）

目前進行到 **Phase 8 V4**：驗證 Qwen2-Audio 作為 captioning model 的泛化性。

| Phase（內部） | 對外名稱 | 核心改動 | 狀態 |
|--------------|---------|---------|------|
| Phase 4 V2 | `JamendoFull-BestConsensus-NoQ` | 基礎 MeanFlow 兩階段訓練 | ✅ Baseline（歷史參考） |
| Phase 5 V1 | `JamendoHalf-BestConsensus-NoQ-HardFilter` | 117K 硬過濾 | ✅ 完成，退步（資料量 -53%） |
| Phase 5 V2 | `JamendoHalf-BestConsensus-NoQ-Random` | 117K 隨機抽樣 | ✅ 完成，≈ V1（量是問題） |
| Phase 6 V1 | `JamendoFull-BestConsensus-MeanSim-Q-S2Only` | q 只在 Stage 2 | ✅ 完成，效果受限 |
| Phase 6 V2 | `JamendoFull-BestConsensus-MeanSim-Q` | q Stage 1+2 | ✅ 完成 |
| Phase 7 V1 | `JamendoFull-Random-MeanSim-Q` | Caption 隨機選一（seed=42） | ✅ 完成，**目前最佳** |
| Phase 7 V2 | `JamendoFull-CLAPBest-MeanSim-Q` | Caption 取 CLAP 最高 | ✅ 完成，劣於 V1 |
| Phase 7 V3 | `JamendoFull-WorstConsensus-MeanSim-Q` | Caption 取最低共識 | ✅ 完成，≈ V1 |
| Phase 8 | `JamendoFull-Random-NoQ` | 無 q conditioning（消融） | ✅ 完成，q embedding 有獨立貢獻 |
| Phase 8 V2 | `JamendoFull-Random-AudioboxPQ-Q` | q 信號改用 Audiobox PQ | ✅ 完成，劣於 Phase 7 V1 |
| Phase 8 V3 | `JamendoFull-Random-CLAP-Q` | q 信號改用 audio-text CLAP sim | ✅ 完成，全面退步（信號語義錯誤） |
| Phase 8 V4 | `JamendoFull-Qwen2Audio-MeanSim-Q` | Caption 換用 Qwen2-Audio-7B | 🔄 前處理中（caption 生成中） |

> Phase 編號作內部追蹤用；對外報告和論文使用描述性名稱（`資料集-Caption策略-Q信號`）。

**下一個方向（Phase 9 候選，教授討論 2026-04-04）**：

優先順序如下：

1. **CLAP score as q signal**（第一優先，「相對 minor」，先確認有沒有效）
   - 設定：JamendoFull + random caption + audio-text CLAP sim 作為 q 信號
   - 與 Phase 7 V1 完全相同，只換 q 信號來源（單一變量）
   - 需對 251K clip 預算 audio-text CLAP sim → 重訓
   - ⚠️ data leakage：若用 CLAP 過濾/標記，evaluation 改用 Audiobox AES

2. **換 captioning model**（第二優先）
   - 目的：證明 random + q conditioning 對任何 captioning model 都有效（泛化性）
   - 具體做法待定

3. **多 captioning model 綜合**（第三優先，待前兩個有結果再討論）
   - 多個 model 各生成 caption，交給 LLM 綜合成完整 prompt
   - 或單一 model 跑 5 次再綜合

> **mean_similarity vs CLAP score as q 的根本差異**：
> - mean_similarity：5 個 caption 的 text-text 相似度平均 → 衡量 caption 群一致性
> - CLAP score：audio-text CLAP 相似度 → 衡量 caption 有多準確描述音訊

---

## 文件導覽

| 何時查閱 | 文件 |
|---------|------|
| 快速查目前狀態、指令、架構說明 | 本檔案（`CLAUDE.md`） |
| 查某個 Phase 的完整數據、結論、對比表 | `docs/experiments/Phase4_to_Phase8_Complete_Summary.md` |
| 查文獻對本研究的啟示（Audiobox、Resonate、PE-AV） | `docs/literature/Literature_Insights.md` |
| 查 checkpoint / log / TSV / NPZ 的實際路徑 | `~/research/README.md` |
| 查累積實驗數字（含舊版中間結果） | `EXPERIMENT_LOG.md` |

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

> **長時間任務一律用 tmux**（預估超過 5 分鐘的 job 都用 `tmux new-session -d -s <name>`，不用背景 Bash task）
> **連續任務一律用 `&&` 串接**（不要分段執行再等使用者回來確認，讓整條 pipeline 在 tmux 裡自動跑完）
> **修改主 repo 檔案時，路徑必須是 `~/MeanAudio/`**，不是 worktree（`~/.claude/worktrees/...`）

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

### 客觀評估（CLAP + AES，與 baseline 比較用）

> **主要指標**：CLAP ↑、CE ↑、PQ ↑（FAD 僅供歷史參考，不再作為主要指標）

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

**Step 2：計算 metrics（CLAP + AES）**
```bash
python ~/research/meanaudio_eval/phase4_eval.py \
    --gen_dir eval_output/EXPNAME_jamendo/audio \
    --exp_name EXPNAME \
    --num_samples 2048 \
    2>&1 | tee ~/logs/EXPNAME_metrics.log
```

> `--fad` flag 預設關閉，AES（CE/CU/PC/PQ）預設開啟。
> 結果存在 `eval_output/metrics/EXPNAME/metrics.txt`。

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

### 目前最佳數字（Jamendo test set，n=2048）

> 主要指標：CLAP ↑、CE ↑、PQ ↑。完整實驗記錄見 `EXPERIMENT_LOG.md`。

| Phase | Caption 策略 | q | CLAP ↑ | CE ↑ | PQ ↑ |
|-------|------------|---|--------|------|------|
| phase4_v2 | best-consensus | — | 0.1929 | 5.905 | 6.620 |
| phase6_v2 | best-consensus | q=6 | 0.1979 | 6.175 | 6.859 |
| phase6_v2 | best-consensus | q=9 | **0.2139** | 6.109 | 6.821 |
| **phase7_v1** | **random** | **q=6** | 0.1980 | **6.276** | 6.936 |
| **phase7_v1** | **random** | **q=9** | 0.1984 | 6.254 | **6.939** |
| phase7_v2 | CLAP-best | q=6 | 0.1943 | 6.230 | 6.938 |

**→ Phase 7 V1 (random caption) 是目前最佳模型，詳細分析見 `EXPERIMENT_LOG.md`。**

---

## 注意事項

- **不要動 `meanaudio/model/networks.py` 裡的 MeanAudio 類別**（Stage 2 架構），只改 FluxAudio
- `run_*.sh` 已在 `.gitignore`，臨時腳本可用這個命名規則避免 commit
- `exps/`、`eval_output/`、`*.pth`、`*.flac` 都在 `.gitignore`，不會進 git
- 資料路徑：`/mnt/HDD/kojiek/phase4_jamendo_data/`

### Backward Compatibility（pre-Phase6 checkpoints）

Phase6 以前的 checkpoint 沒有 `q_embed.weight`。`load_weights()` 會自動偵測並將 q_embed **歸零**（不影響 global_c），並印出 WARNING。phase4_v2 / phase6_v1 等舊 checkpoint 可正常使用。

---

## 教授討論紀錄

### 2026-03-27 — 資料品質、評估指標、Phase 7 方向

**資料品質過濾的效果（尚未定論）**

兩種假說：
- **假說 A（大數法則）**：資料夠多，爛的 caption 被平均掉，過濾不重要
- **假說 B（方法問題）**：我們的過濾方式本身不夠好，才沒看到效果

教授直覺傾向假說 B：「如果 caption 是爛的，怎麼可能訓出好的模型？」→ 需要實驗驗證。

**評估指標：聽感 > CLAP**

> 「CLAP 很好啦，通常也可以相信，只是 CLAP score 到底代表了什麼，我個人覺得還是耳聽為憑。你聽起來覺得有變好就是有變好。這些其實都是輔助。」

→ 主觀評估不能省，metrics 是輔助工具。

**新的資料過濾方向：Caption-Audio CLAP 相似度**

- 不用 cross-model consistency（現行做法）
- 改算 **caption ↔ audio CLAP 相似度**，高相似度 = 好 caption
- 預先計算存起來，訓練時直接讀，成本可攤平

**⚠️ Data Leakage 原則**

> 「如果訓練資料過濾用了 CLAP，evaluation 就不能用 CLAP。」

| 過濾方法 | 可用 eval | 不可用 eval |
|----------|-----------|------------|
| Caption-audio CLAP 過濾 | FAD、Meta Aesthetics | ❌ CLAP |
| 不用 CLAP 過濾 | CLAP + FAD | — |

→ Phase 7 若採用 CLAP 過濾資料，evaluation 需改用 FAD 或 Meta Audiobox Aesthetics。

---

## Meta Audiobox Aesthetics 指標

### 安裝
```bash
pip install audiobox_aesthetics   # CC-BY 4.0，無需申請，自動下載權重
```

### 四個子指標
| 指標 | 名稱 | 物理意義 |
|------|------|---------|
| **CE** | Content Enjoyment | 主觀聽感、情感影響、藝術性、整體喜好 |
| CU | Content Usefulness | 內容是否符合使用情境 |
| PC | Production Complexity | 製作複雜度 |
| **PQ** | Production Quality | 技術品質：清晰度、保真度、無雜訊失真 |

### 與人類 MOS 的相關係數（文獻，PAM-music，utterance-level）
| 指標 | ↔ 人類 OVL | ↔ 人類真實標註 | 備註 |
|------|-----------|--------------|------|
| **CE** | **0.528** | **0.661** | 單樣本層級 |
| **PQ** | 0.464 | 0.587 | 單樣本層級 |

成對偏好預測準確率（From Aesthetics to Human Preferences）：
- CE、CU：> 60%（顯著高於盲猜 50%）
- PQ（保真度偏好）：59.1%

→ **CE 與人類主觀評分相關性最強**，是評估「音樂品質提升」的最佳指標。

### 已採用此指標的論文（學術引用依據）
- **LeVo (2025)**：多偏好對齊歌曲生成，評估 Suno-V4.5、Mureka-O1、YuE
- **ACE-Step (2025)**：音樂生成基礎模型，Table 1 全面採用四指標
- **SongBloom (2025)**、**MIDI-SAG (2025)**：客觀評估全面採用
- **SMART**：直接用 CE 作為 RL reward 微調符號音樂生成
- **AudioMOS Challenge 2025（Track 2）**：以四指標作為官方評測框架

### 對 MeanAudio 研究的意義
- **CE** → 回答「quality conditioning 是否讓音樂更好聽、更有藝術性」
- **PQ** → 回答「是否降低了低品質訓練資料帶來的技術瑕疵（雜訊、失真）」
- 最強論述：CE 和 PQ **同時提升** = q_embed 帶來全方位感知品質升級；只有 PQ 升而 CE 不動 = 只學會「清理背景雜訊」
- 學術寫作建議：**四個指標全部列出**（如 LeVo、ACE-Step 做法），以 CE 為主軸論述
- ⚠️ 若用 CLAP 過濾訓練資料，evaluation 改用 Audiobox Aesthetics（避免 data leakage）
