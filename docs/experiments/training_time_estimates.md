# MeanAudio 訓練 / Eval 時間參考

> 所有數字皆自 Phase 9 V1 實測訓練 log 抽出（2026-04-18~19）。
> 硬體：RTX 5090 (33.67 GB VRAM, sm_120)、BS=8、num_workers=4。

---

## 單階段時間

| 階段 | 指令 / 位置 | iter 數 | iter 速度 | 總時間 |
|------|------------|---------|----------|--------|
| **Stage 1 (FluxAudio)** | `train.py model=fluxaudio_s` | 400,000 | 0.111 s/iter | **~12.3 小時** |
| **Stage 1→2 Migrate** | `migrate_stage1_to_stage2_ckpt.py` | — | — | **~1-2 分鐘** |
| **Stage 2 (MeanAudio)** | `train.py model=meanaudio_s` | 200,000 | 0.121 s/iter | **~6.7 小時** |
| **Eval 生成** | `eval.py --num_steps 1`，phase4_test.tsv (90,063 筆) | — | 8~12 it/s | **~2.5-3 小時** |
| **Eval metrics** | `phase4_eval.py --num_samples 2048` | — | CLAP ~60 it/s | **~25-40 分鐘** |

### 完整 pipeline（`train_pipeline_*.sh` 一條龍）

- **從頭（S1 + migrate + S2 + eval）**：~22 小時
- **只重跑 S2 + eval**（沿用既有 S1 ckpt）：**~10 小時**
- **只重跑 eval**（沿用既有 S2 EMA）：~3.5 小時

### 主觀評估（5 首固定 prompt）

- 25 steps + cfg 3.5，單 GPU 約 **~3-5 分鐘**（整組 5 首）
- 指令見 `docs/eval/subjective_prompts.md`

---

## 訓練時間估算公式

### 自訂 iter 數

```
S1_hours = S1_iters × 0.111 / 3600
S2_hours = S2_iters × 0.121 / 3600
eval_hours = 3   # 固定開銷（跟模型無關，只跟 TSV 大小有關）
total = S1_hours + S2_hours + eval_hours + 0.03   # migrate
```

### 常見組合

| 配置 | S1 | S2 | Total |
|------|-----|-----|-------|
| 標準（400k + 200k）| 12.3h | 6.7h | ~22h |
| 快速 smoke（100k + 50k）| 3.1h | 1.7h | ~8h |
| 只 S2（200k）| — | 6.7h | ~10h |

---

## 調大 batch size / accumulation 的影響

- 從 log 看，**data loader 僅 0.009~0.010 s**（pin_memory + num_workers=4 足夠），**GPU bound** 無 stall
- S2 比 S1 慢 ~10%（0.121 vs 0.111）是 MeanFlow JVP `create_graph=True` 造成
- 若 OOM 或想跑更長，把 `BATCH_SIZE=8, ACCUM_STEPS=1` 改成 `BATCH_SIZE=4, ACCUM_STEPS=2`，等效 BS 不變但記憶體減半；但 iter 數須翻倍，總時間約 1.5× 因為梯度 accumulate 沒省

---

## 2026-04-19 Bug-fix 驗證排程

背景：`networks.py:526/558` `q=None→9` bug + `runner_meanflow.py:238/268` undrop 別名 bug 兩條都只影響 **Stage 2**。S1 (FluxAudio) 本身沒 bug。

**最小驗證流程**（沿用既有 S1 ckpt，只重跑 S2）：

| 步驟 | 內容 | 時間 |
|------|------|------|
| 0 | 修 `networks.py:526, 558` 9→10 + `runner_meanflow.py:238-239, 268-269` 加 `.clone()` | 幾分鐘 |
| 1a | 從現有 Phase 8 S1 ckpt re-migrate 到新 S2 ckpt | 2 分鐘 |
| 1b | Phase 8 S2 重訓練 + eval | ~10 小時 |
| 2a | 從現有 P9 V1 S1 ckpt re-migrate | 2 分鐘 |
| 2b | P9 V1 S2 重訓練 + eval | ~10 小時 |
| 3（選配）| P9.5 V1 同上 | ~10 小時 |

**總計**：Phase 8 + P9 V1 的驗證約 **20 小時**（連跑一天）；加 P9.5 V1 約 **30 小時**（1.5 天）。

### 若 bug fix 後結果

| Phase 8 新 CLAP | 判讀 |
|----------------|------|
| > 0.185 | Bug fix 本身有貢獻，確認 mismatch 真的在害 |
| ≈ 0.185 | Bug 對 static-cap 影響有限，但可能仍放大 V1 崩盤 |
| < 0.185 | 出乎意料，需檢查 migrate / fix 流程是否正確 |

| P9 V1 新 CLAP | 判讀 |
|---------------|------|
| ≥ 0.185 | 原本「true random + no Q」實驗設計**成立**，bug 是唯一元兇 |
| 0.05~0.15 | Bug 是主因但 true random 仍有額外損失，需討論架構 |
| < 0.05 | Bug fix 幾乎沒影響，york135 的「unconditional fallback」假設成立 |

---

## 歷史 Phase 的訓練時間回顧

以下皆 S1=400k + S2=200k 標準配置：

| Phase | 訓練總時（S1+S2+eval） | 備註 |
|-------|----------------------|------|
| Phase 4 V2 | ~22h | baseline |
| Phase 6 V2 | ~22h | + Q signal（不影響時間）|
| Phase 7 V1 | ~22h | best so far |
| Phase 8 | ~22h | — |
| Phase 9 V1 | ~22h（崩盤那次） | multi_cap NPZ 讀取速度略慢但影響 < 5% |

> multi_cap NPZ 每個檔案是 single-cap 的 ~5× 大小，但 `np.load` 加 pin_memory + persistent workers 不會成為瓶頸（data_timer 仍 0.009~0.010s）。

---

## 其他時間參考

### 前處理（一次性）

| 任務 | 時間 |
|------|------|
| multi_cap NPZ 生成（251,599 clips × 5 caps）| ~6 小時（T5+CLAP encoding，batch_size=64）|
| Qwen2.5-Omni captioning（251,599 × 1 slot）| ~20 小時/slot（batch=32, tokens=60）|
| Qwen 5 slot 全部 | ~4 天（見 `reference_qwen_omni_captioning_profile.md`）|

### 驗證 / Sanity check

| 任務 | 時間 |
|------|------|
| Caption diversity sanity check（50 NPZ 抽樣）| ~30 秒 |
| multi_cap NPZ deep validation（200 抽樣）| ~2-3 分鐘 |
| Migrate S1→S2 + verify | ~2 分鐘 |
