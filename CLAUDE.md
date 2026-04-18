# MeanAudio — Claude Code 上手指南

## 專案概覽

**MeanAudio** 是文字驅動音訊生成系統，兩階段訓練：
- **Stage 1**：FluxAudio（Flow Matching，單向 ODE）
- **Stage 2**：MeanAudio（Mean Flow，更快推理）

**目前重心**：Phase 9.5 — Qwen2.5-Omni-3B 5 task captions，跨 captioner 驗證 diversity hypothesis。

**目前最佳**：Phase 7 V1 (`JamendoFull-Random-MeanSim-Q`) — Jamendo + MusicCaps benchmark 雙料最佳。

**當前排程**（教授 2026-04-18 指示）：Lane A (MusicCaps benchmark ✅) → Lane B (P9 V1/V2 訓練) → Lane C (Qwen resume)。

> Phase 編號作內部追蹤，對外報告/論文用描述性名稱（`資料集-Caption策略-Q信號`）。

---

## 文件導覽

| 何時查閱 | 文件 |
|---------|------|
| Phase 狀態、實驗進度、NPZ/captioning 檔案狀態 | `docs/experiments/phase_status.md` |
| Phase 9/9.5 實驗設計、multi_cap 機制、LP-MusicCaps 真相 | `docs/experiments/phase9_design.md` |
| 完整實驗數字（Jamendo + MusicCaps） | `docs/experiments/best_results.md` |
| 教授討論紀錄（Lane A/B/C、data leakage） | `docs/meetings/` |
| Meta Audiobox Aesthetics 指標細節 | `docs/metrics/audiobox_aesthetics.md` |
| 五首固定主觀 prompt + 下載指令 | `docs/eval/subjective_prompts.md` |
| Phase 4→8 完整對比表（歷史） | `docs/experiments/Phase4_to_Phase8_Complete_Summary.md` |
| 文獻啟示（Audiobox、Resonate、PE-AV） | `docs/literature/Literature_Insights.md` |
| 累積實驗數字（含舊版） | `EXPERIMENT_LOG.md` |

---

## 主動更新規則（Claude 必讀）

> CLAUDE.md 與 `~/.claude/projects/-home-kojiek-MeanAudio/memory/` 必須在下列事件發生時主動更新，不等使用者提醒。

**必須更新 CLAUDE.md / docs：**
1. Phase 狀態轉換 → `docs/experiments/phase_status.md`
2. 新實驗設計確認 → `docs/experiments/phase9_design.md` 或新 doc
3. 環境限制發現 → 本檔「環境」段
4. 最佳數字更新 → `docs/experiments/best_results.md`
5. 教授新方向 → 新增 `docs/meetings/YYYY-MM-DD_*.md`
6. 關鍵檔案/指令改名 → 本檔對應段

**必須寫入 memory：**
1. 使用者糾正做法 → `feedback_*.md`
2. 非直覺選擇被確認 → `feedback_*.md`
3. 長期研究原則（diversity hypothesis、data leakage 等）→ `feedback_*.md`
4. 硬體/環境特殊性 → `project_*.md` 或 `reference_*.md`
5. 使用者身份偏好（首次知道）→ `user_*.md`

**流程**：做決策 → 立即更新 → 更新 `MEMORY.md` index → 繼續任務。

---

## 環境

```bash
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
```

---

## 目錄結構

```
MeanAudio/
├── meanaudio/
│   ├── model/{networks.py, mean_flow.py, flow_matching.py}   # 模型、loss、ODE solver
│   ├── runner_meanflow.py                                     # 訓練迴圈
│   ├── data/{extracted_audio.py, data_setup.py}               # Dataset / DataLoader
│   └── eval_utils.py                                          # generate_mf / generate_fm
├── train.py / eval.py / infer.py                              # 入口
├── train_pipeline.sh                                          # S1 → migrate → S2 → eval
├── set_training_stage.py                                      # 切 Stage 1/2（patch runner）
├── migrate_stage1_to_stage2_ckpt.py                           # S1→S2 checkpoint 轉換
├── config/ sets/ scripts/ archive/
└── docs/                                                      # 詳細文檔（見上方導覽）
```

資料路徑：`/mnt/HDD/kojiek/phase4_jamendo_data/`

---

## 實驗前 Checklist

> **違反任何一項都可能燒掉數小時 GPU 時間。**

1. **腳本推 GitHub**：`~/research` 下所有會用到的腳本先 `git add && commit && push`
2. **Caption 多樣性 sanity check**（唯一率 < 90% 停止訓練）：
   ```bash
   cd ~/research/meanaudio_training && python sanity_check_50.py
   ```
3. **Eval TSV 確認**（必須明確傳 `--tsv`，不依賴 hardcode default）：
   - MusicCaps (ISMIR benchmark)：`/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv`
   - Jamendo 歷史比較：`phase4_test.tsv`（90,063 筆）

---

## 訓練流程

> 長時間任務一律用 **tmux**（超過 5 分鐘的 job 都用 `tmux new-session -d -s <name>`）
> 連續任務一律用 **`&&`** 串接，不要分段等使用者回來
> 修改主 repo 檔案時路徑必須是 **`~/MeanAudio/`**，不是 worktree

```bash
tmux new -s phaseX
cd ~/MeanAudio && source ~/venvs/dac/bin/activate && bash train_pipeline.sh
```

`train_pipeline.sh` 只需改參數區塊：`EXP_PREFIX` / `S1_ITERATIONS` / `S2_ITERATIONS` / `LEARNING_RATE`。

---

## 關鍵架構：Quality Conditioning

**q_embed**：`nn.Embedding(11, hidden_dim)` — idx 0~9 = 品質等級，idx 10 = null token。

- `FluxAudio.predict_flow(q=None)` → `q=None` 填 null token
- `FluxLoss.loss(q=...)` → conditional pass 用 q、unconditional 用 null
- `set_training_stage.py --stage {1,2}` → patch runner 切 FluxAudio / MeanAudio

**Checkpoint S1→S2 遷移**（`migrate_stage1_to_stage2_ckpt.py`）：
1. `t_embed → r_embed`（S2 新增）
2. **保留 S1 已訓練的 q_embed**（Phase 6 V2+）
3. 清除 optimizer / scheduler state

**Backward compat**：pre-Phase 6 checkpoint 無 `q_embed.weight` → `load_weights()` 自動歸零 + WARNING。

---

## Eval

主要指標：**CLAP ↑、CE ↑、PQ ↑**（FAD 僅歷史參考）。

### ⚠️ q 旗標選擇規則（混用會啟用未訓練 embedding 污染結果）

| 訓練時 q conditioning | Phase 例 | Eval 旗標 |
|---|---|---|
| **true**（q 0~9 訓練） | 6 V2、7 V1/V2、9 V2、9.5 V2 | `--quality_level N` |
| **false**（永遠 null token） | 8、9 V1、9.5 V1 | `--no_q` |
| **pre-Phase 6**（無 q_embed 層） | 4 V2、5 V1/V2 | 兩者等價 |

詳見 `memory/reference_eval_q_flag_rule.md`。踩坑：2026-04-17 Phase 8 誤用 `--quality_level 9` → CLAP 0.1907（污染），`--no_q` 正確 0.1851。

### 指令

```bash
# 生成音訊
python eval.py --variant meanaudio_s \
    --model_path exps/EXP/EXP_ema_final.pth \
    --output eval_output/EXP_jamendo/audio \
    --tsv <TSV> --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --full_precision \
    {--quality_level N | --no_q}

# 計算 metrics（AES 預設開啟，--fad 預設關閉）
python ~/research/meanaudio_eval/phase4_eval.py \
    --gen_dir eval_output/EXP_jamendo/audio \
    --exp_name EXP --num_samples 2048
```

結果：`eval_output/metrics/EXP/metrics.txt`。完整數字見 `docs/experiments/best_results.md`。

主觀評估五首 prompt 見 `docs/eval/subjective_prompts.md`（25 steps + cfg 3.5）。

---

## NEVER

- **不要動 `meanaudio/model/networks.py` 裡的 MeanAudio 類別**（Stage 2 架構），只改 FluxAudio
- **不要在 worktree（`~/.claude/worktrees/...`）改主 repo 檔案**，必須是 `~/MeanAudio/`
- **不要混用 q 旗標**（見上方規則）
- **不要 commit 臨時腳本**：`run_*.sh` 已在 `.gitignore`，臨時腳本照此命名
- **不要用 CLAP 過濾訓練資料同時用 CLAP eval**（data leakage — 教授 2026-03-27 原則）

---

## Git

- Remote: `https://github.com/Lanternko/MeanAudio.git`
- Branch: 直接在 `main`，每個 Phase 結束後 tag
- Identity: `lanternko <jerry86012@gmail.com>`（token 已存在 credential store）
- Commit 格式：`phaseN_vX: 簡短描述` + bullet 改動
