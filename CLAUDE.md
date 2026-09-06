# MeanAudio — Claude Code 上手指南

## 專案概覽

**MeanAudio** 是文字驅動音訊生成系統，兩階段訓練：
- **Stage 1**：FluxAudio（Flow Matching，單向 ODE）
- **Stage 2**：MeanAudio（Mean Flow，更快推理）

**目前重心**（2026-07-20）：
1. **P8 legacy repro 完成** — MusicCaps CLAP **0.1684**（`--quality_level 9` + NoMask；對照歷史 q=9 條件 0.1907，delta −0.022 在 audit ±0.03 內，量級符合 S1-effective-q-training penalty ~0.02）。首次 pipeline eval 誤用 `--no_q` 得 0.0134（Q-trained 模型 + q=10=uncond 記號 → unconditional 模式），已修：pipeline 從 `USE_Q_CONDITIONING` 推導 eval 旗標、audit 目標改 0.1907、無效產物存 `*_noq_invalid`。詳見 forensics addendum（`docs/experiments/history/phase8/phase8_baseline_forensics_2026_07_17.md`）+ memory `project_legacy_repro_noq_eval_trap_2026_07_18.md`
2. **✅ `phase8_catalog_matched_noq` clean full 完成** — S1/S2 NoQ + NoMask + `--no_q`，MusicCaps **CLAP 0.1888**（CE 5.7252 / CU 6.4241 / PC 4.8893 / PQ 6.4174）；5,521/5,521 audio、final contract audit PASSED。訓練中唯一一次 AMP grad NaN 已恢復，ckpt/EMA 無 corruption；舊 Grok job `019f798b2408` 已取消。
3. **🟢 Phase8 clean S2-only Q 因果對照 RUNNING**（tmux `p8_s2_q_ablation`，2026-07-20 12:19 起）
   - 共用上述 clean-NoQ S1 400k；依序跑 `phase8_catalog_matched_s2_realq` → `phase8_catalog_matched_s2_shuffledq`，各自 S2 追加 200k。
   - Real-Q 用原始 per-row `q_level`；Shuffled-Q 只用 seed 424242 打亂 Q（資料順序、id/caption、NPZ/cache、Q histogram 不變）。MusicCaps q9 primary、q6 secondary。
   - 目的：Real-Q 必須同時超過 NoQ 0.1888 與 Shuffled-Q 才支持 Q information 有貢獻；q9 ≥0.1998 才達歷史最佳門檻。
   - Grok durable 5m watcher job `019f7dc10ba5`；任何停止須取得 Codex SOL `stop_authorized=true`，Grok 不可自行停止或修改 live run。Handoff：`docs/experiments/archive/ops/phase8_s2_q_ablation_grok_handoff_2026_07_20.md`。
4. **Phase 9 multi-cap clean rebuild**（P0，disk-blocked）— 需 ~413G，HDD 僅 ~314G free；tooling 已修好，等空間後再 encode
5. **Music Flamingo caption ablations** A1–A6 已完成（見 `docs/experiments/results/phase8/music_flamingo_ablation_todo.md`）
6. **Qwen collapse diagnostic** EXP-A~H 完成；**ISMIR 2026 paper 487** reviews 已歸檔

**目前最佳（paper primary）**：`LP-Rnd-Q`（Phase 7 V1，`JamendoFull-Random-MeanSim-Q`）— Jamendo + MusicCaps 跨 benchmark 最穩。歷史 10-exp 定稿見 `docs/experiments/results/benchmarks/ten_exp_full_benchmark.md`。

**4-token paper-facing 命名**（2026-05-08 統一）：`{Caption}-{Sel}-{Q}` 預設 MC eval；非預設加括號 `(JMQ)` 等。Phase ID → 4-token 速查表見 `docs/experiments/phase_status.md` 頂端。Music Flamingo 實驗沿用 pipeline/exp 名（如 `mfshort100k_direct`），尚未併入 4-token 表。

> Phase 編號作內部追蹤；對外報告/論文一律用 4-token 命名（範例：`LP-Rnd-Q`、`Qwen-Rnd-NoQ (JMQ)`）。

---

## 文件導覽

| 何時查閱 | 文件 |
|---------|------|
| Phase 狀態、實驗進度、P9 cache 失效、P8 clean retrain | `docs/experiments/phase_status.md` |
| 完整實驗數字（Jamendo + MusicCaps 10-exp + P8V4） | `docs/experiments/best_results.md` |
| Music Flamingo caption ablations（A1–A6 結果） | `docs/experiments/results/phase8/music_flamingo_ablation_todo.md` |
| **MF 全覆蓋線**（038/039 queue job、判讀規則、early-kill gate） | `docs/experiments/mf_full_coverage_line.md` |
| Qwen collapse 10-model audit | `docs/experiments/history/phase8/qwen_collapse_audit_10model.md` |
| Qwen collapse root-cause EXP-A~H | `docs/experiments/history/phase8/qwen_collapse_root_cause_2026_05_08.md` |
| Qwen single-cap rerun 總結 | `docs/experiments/history/phase8/qwen_rerun_summary.md` |
| Phase 9/9.5 實驗設計、multi_cap 機制 | `docs/experiments/history/phase9/phase9_design.md` |
| Phase 9.5 完整總結（歷史；multi-cap 結果已失效） | `docs/experiments/history/phase9/phase9_5_summary.md` |
| 訓練 / Eval 時間估算 | `docs/experiments/training_time_estimates.md` |
| ISMIR 2026 reviews + correctness plan | `docs/reviews/ismir2026-487-promptcc/` |
| 教授討論紀錄（Lane A/B/C、data leakage） | `docs/meetings/` |
| Meta Audiobox Aesthetics 指標細節 | `docs/metrics/audiobox_aesthetics.md` |
| 五首固定主觀 prompt + 下載指令 | `docs/eval/subjective_prompts.md` |
| Phase 4→8 完整對比表（歷史） | `docs/experiments/history/phase4-phase8/Phase4_to_Phase8_Complete_Summary.md` |
| 文獻啟示（Audiobox、Resonate、PE-AV） | `docs/literature/Literature_Insights.md` |
| **Negative prompting / prompt engineering 文獻定位**（QA-MDT、NAG/VSF、APG、Open Prompt Challenge） | `docs/literature/negative_prompting_and_prompt_engineering_2026_09_04.md` |
| 早期累積實驗數字（→ 改查 `best_results.md`） | `EXPERIMENT_LOG.md` |

---

## 主動更新規則（Claude 必讀）

> CLAUDE.md 與 `~/.claude/projects/-home-kojiek-MeanAudio/memory/` 必須在下列事件發生時主動更新，不等使用者提醒。

**必須更新 CLAUDE.md / docs：**
1. Phase 狀態轉換 → `docs/experiments/phase_status.md`
2. 新實驗設計確認 → 對應 `docs/experiments/*.md` 或新 doc
3. 環境限制發現 → 本檔「環境」段
4. 最佳數字更新 → `docs/experiments/best_results.md`
5. 教授新方向 → 新增 `docs/meetings/YYYY-MM-DD_*.md`
6. 關鍵檔案/指令改名 → 本檔對應段
7. Ablation / pipeline 完成 → 對應 todo doc + `scripts/README.md` 必要時更新

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

Music Flamingo captioning 另用 `~/venvs/music_flamingo/`（見 memory `reference_music_flamingo_setup.md`）。

---

## 目錄結構

完整 repo 地圖見 `STRUCTURE.md`；scripts/ 子目錄索引見 `scripts/README.md`。

```
MeanAudio/
├── meanaudio/                       # 套件：model/、data/、runner_*.py、eval_utils.py
├── train.py / eval.py / infer.py    # 入口
├── train_pipeline.sh                # CANONICAL S1 → migrate → S2 → eval
├── set_training_stage.py            # 切 Stage 1/2（patch runner）
├── migrate_stage1_to_stage2_ckpt.py # S1→S2 ckpt 轉換（吃 ckpt_last.pth，不是 ema_final.pth）
├── scripts/                         # 所有 helper（training_pipelines/, eval/, preprocess/, analysis/, legacy/, runs/）
├── config/ sets/ data/ training/    # configs、latent stats、symlinks、訓練工具
├── av-benchmark -> .external/av-benchmark
├── .archive/                        # 隱藏歷史/次要資料
├── .external/                       # 隱藏外部 checkout：av-benchmark/
├── .side_projects/                  # 非 MeanAudio 主線的 side projects
└── docs/                            # experiments/ meetings/ eval/ metrics/ literature/ reviews/
```

資料路徑：`/mnt/HDD/kojiek/phase4_jamendo_data/`（NPZ + TSV）；舊 eval audio：`/mnt/HDD/kojiek/MeanAudio_eval_output_OLD/`（2026-05-16 reorg 搬出 repo root）

**實驗變體 pipeline** 一律在 `scripts/training_pipelines/`，啟動：`bash scripts/training_pipelines/<name>.sh`（每個都自帶 `cd "$WORK_DIR"`）。

**近期架構變更**：`text_attention_mask` 已接入 joint attention + mean pooling（commit `a148aaf`），T5 padding 不再污染 text conditioning。

---

## 實驗前 Checklist

> **違反任何一項都可能燒掉數小時 GPU 時間。**

1. **腳本推 GitHub**：`~/research` 下所有會用到的腳本先 `git add && commit && push`
2. **Caption 多樣性 sanity check**（唯一率 < 90% 停止訓練）：
   ```bash
   cd ~/research/meanaudio_training && python sanity_check_50.py
   ```
3. **Multi-cap / NPZ pairing audit**（若涉及 multi-caption cache）：
   - 必須走 `npz_cache_train.txt` mapping，**禁止** row-index → `i.npz`
   - 用新版 `gen_multicap_npz.py --gt-cache ...` + `validate_multicap_npz.py`（v2 manifest / mean-std exact equality）
   - 舊 Phase 9 cache **不可**重訓或引用為 method result
4. **Eval TSV 確認**（必須明確傳 `--tsv`，不依賴 hardcode default）：
   - **預設：MusicCaps**（ISMIR 黃金標準，2026-04-19 定為主要 benchmark）：`/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv`（5,527 筆，~11 min eval）
     - 理由：ISMIR benchmark 發表用；無 data leakage（訓練 Jamendo、eval MusicCaps）；16x 比 Jamendo 快
   - **次要：Jamendo** 歷史比較：`phase4_test.tsv`（90,063 筆，~3.1 hr eval）— 只在需要跟 Phase 4-8 舊數字對照時才跑
   - **快速 sanity**：`eval.py` **無** `--num_samples` 參數；要做 2048 筆小 subset sanity 須先 `head -n 2049 <TSV> > <TSV>_2048.tsv` 切檔再傳 `--tsv`。`phase4_eval.py` 的 `--num_samples` 只控制 metric 計算樣本數（FAD），不影響生成數量
5. **Caption-source 換訓練時**：必須有 same-distribution eval；跨 style 結果只能寫 generalization，不是 conditioning test

---

## 實驗中/後 Monitoring（launch ≠ done）

> **啟動實驗不等於完成 — 不 monitor 就等於沒跑（可能悄悄掛掉、stall、或產生 NaN/全 0 音檔而你不知道）。**

**每次啟動 tmux 訓練或 eval job 後，必須主動排定 monitoring：**

1. **第一次 check（啟動後 1~2 分鐘內）**：確認 job 真的在跑
   - `tmux ls` 看 session 還在
   - `nvidia-smi` GPU 有吃到（memory > 1 GB、util > 0%）
   - 讀 log 開頭：無 import error / OOM / checkpoint load error
   - 有 progress bar（`it/s` 合理，不是卡住）

2. **定期 check（用 `ScheduleWakeup` 或 `/loop`，間隔依 job 長度）**：
   - **短 job（<30 min）**：每 5~10 min check 一次
   - **中 job（30 min ~ 2 hr）**：每 20~30 min check（用 `ScheduleWakeup delaySeconds=1200~1800`）
   - **長 job（>2 hr）**：每 30~60 min，關鍵轉折點（Stage 切換、eval 開始）加 check

3. **每次 check 要看的東西**：
   - tmux session 還活著（`tmux ls`）
   - GPU 還在跑（util > 0%，memory 沒降到閒置）
   - log tail 有新行（沒 stall）
   - 無 exception / NaN / OOM traceback
   - 預期的階段轉換有發生（e.g. gen 完換 metrics、Q 切換到下一個）

4. **結果 sanity check（每個階段完成時）**：
   - Gen 完：`ls <output>/audio | wc -l` 接近預期數、抽一個 `soxi` 看長度/取樣率正常、檔案 size > 0
   - Metrics 完：`cat metrics.txt` 無 NaN、數字在合理範圍（CLAP 0.05~0.25、CE 5~8、PQ 5~8）
   - 發現異常 → 先懷疑 bug（見 `memory/feedback_suspect_bug_before_explaining.md`）

5. **禁止「啟動後就當完成」** — 沒排 monitoring 等於沒做這份工作。

---

## GPU idle backlog policy

> **GPU 不該 idle；但只有已定義、可恢復、可插隊的實驗，才能在 idle 時自動接手跑。**

4 個 guardrail（詳見 `memory/feedback_gpu_idle_backlog_policy_2026_04_21.md`）：

1. 只能自動開**已排隊、已定義目的**的實驗 — 不能因 GPU 空就臨時發明題目
2. 必須**可恢復** — checkpoint 有、resume 驗證過、save interval 合理
3. 優先級分類：
   - **P0**：短 probe / sanity / bug verification — 隨時可插隊
   - **P1**：關鍵 control run（例 P8 bug-free retrain、clean multi-cap rebuild）
   - **P2**：探索型長實驗 — 沒更明確 backlog 時才跑
4. 啟動時**留紀錄**（跑什麼、為什麼現在跑、checkpoint 點、被插隊時的停機點），不默默開

**流程**：自動啟動前先逐項檢查這 4 個 guardrail，檢查通過才動。

---

## 訓練流程

> 長時間任務一律用 **tmux**（超過 5 分鐘的 job 都用 `tmux new-session -d -s <name>`）
> 連續任務一律用 **`&&`** 串接，不要分段等使用者回來
> 修改主 repo 檔案時路徑必須是 **`~/MeanAudio/`**，不是 worktree
> shell 外層也要 `set -eo pipefail`（見 `feedback_pipefail_silent_crash_2026_04_22.md`）

```bash
tmux new -s phaseX
cd ~/MeanAudio && source ~/venvs/dac/bin/activate && bash train_pipeline.sh
```

`train_pipeline.sh` 只需改參數區塊：`EXP_PREFIX` / `S1_ITERATIONS` / `S2_ITERATIONS` / `LEARNING_RATE`。

**S1 必須** `model=fluxaudio_s`（不是 `meanaudio_s`）— 見 `feedback_pipeline_s1_uses_fluxaudio_s.md`。

---

## 關鍵架構：Quality Conditioning

**q_embed**：`nn.Embedding(11, hidden_dim)` — idx 0~9 = 品質等級，idx 10 = null token。

- `FluxAudio.predict_flow(q=None)` → `q=None` 填 null token（**必須是 10**；歷史 bug 曾填 9）
- `FluxLoss.loss(q=...)` → conditional pass 用 q、unconditional 用 null
- `set_training_stage.py --stage {1,2}` → patch runner 切 FluxAudio / MeanAudio

**Checkpoint S1→S2 遷移**（`migrate_stage1_to_stage2_ckpt.py`）：
1. `t_embed → r_embed`（S2 新增）
2. **保留 S1 已訓練的 q_embed**（Phase 6 V2+）
3. 清除 optimizer / scheduler state
4. **輸入必須是 `ckpt_last.pth`**，不是 `ema_final.pth`（後者缺 `it`/optimizer keys）

**Backward compat**：pre-Phase 6 checkpoint 無 `q_embed.weight` → `load_weights()` 自動歸零 + WARNING。

**已修結構性 bug（必須知道）**：
1. `networks.py` MeanAudio `q=None` 曾填 9 → 已改 10
2. `runner_meanflow.py` `text_f_undrop` 別名 → 已 `.clone()`
3. `runner_flowmatching.py` 曾完全不傳 q → 已修 6 處（歷史 Phase 6–8 +Q = half-Q）

---

## Eval

主要指標：**CLAP ↑、CE ↑、PQ ↑**（FAD 僅歷史參考；PE-AV 作 fine-grained retrieval）。

### ⚠️ q 旗標選擇規則（混用會啟用未訓練 embedding 污染結果）

| 訓練時 q conditioning | Phase 例 | Eval 旗標 |
|---|---|---|
| **true**（q 0~9 訓練） | 6 V2、7 V1/V2、9 V2、9.5 V2 | `--quality_level N` |
| **false**（永遠 null token） | 8、9 V1、9.5 V1、MF NoQ 系列 | `--no_q` |
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

主觀評估五首 prompt 見 `docs/eval/subjective_prompts.md`（25 steps + **cfg 0.5**）。**不要用 cfg ≥ 2.0** — 在非 null Q + 高能量 prompt 會觸發波形飽和（crest < 2.0，2026-04-21 於 subjective_ab v3 踩坑，mc18_abl_A–J 證實，york135 指出）。

`infer.py` **沒有 `--no_q`** — NoQ 模型用 `--quality_level 10`（null token workaround）。

---

## NEVER

- **不要動 `meanaudio/model/networks.py` 裡的 MeanAudio 類別**（Stage 2 架構），只改 FluxAudio
- **不要在 worktree（`~/.claude/worktrees/...`）改主 repo 檔案**，必須是 `~/MeanAudio/`
- **不要混用 q 旗標**（見上方規則）
- **不要 commit 臨時腳本**：`run_*.sh` 已在 `.gitignore`，臨時腳本照此命名
- **不要用 CLAP 過濾訓練資料同時用 CLAP eval**（data leakage — 教授 2026-03-27 原則）
- **不要把 Phase 9/9.5 multi-cap 歷史數字當 method result**（2026-07-16 pairing 全量錯配）
- **不要對 Qwen-trained 模型只用 LP/MusicCaps prompt 就下「沒學到」結論**（需 same-distribution eval）

---

## Git

- Remote: `https://github.com/Lanternko/MeanAudio.git`
- Branch: 直接在 `main`，每個 Phase 結束後 tag
- Identity: `lanternko <jerry86012@gmail.com>`（token 已存在 credential store）
- Commit 格式：`phaseN_vX: 簡短描述` + bullet 改動
