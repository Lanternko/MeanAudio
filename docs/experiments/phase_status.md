# Phase 狀態總表

> Phase 編號作內部追蹤用；對外報告和論文使用描述性名稱（`資料集-Caption策略-Q信號`）。

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
| ~~Phase 8 V4 (舊)~~ | `JamendoFull-Qwen2Audio-MeanSim-Q` | Caption 換用 Qwen2-Audio-7B | ❌ 廢棄（僅1 cap/clip，不支援 true random）→ V4 名額重用給 PromptConsistency 實驗 |
| **Phase 8 V4** | `JamendoFull-Random-PromptConsistency-NoQ` | mean_similarity raw float 寫進 caption prefix `[consistency=X.XX]`，訊號全走 text encoder（QA-MDT 風格），不依賴 q_embed。eval inference prefix 固定 `0.90`（in-support，median–p90 區間） | ✅ **完成 2026-04-27**。**Natural-ref CLAP**：MusicCaps 0.0571 / Jamendo seed42 0.0591（vs P8 NoQ baseline 0.1851 / 0.1986），AES 僅小跌（PC 甚至略升）。⚠️ 這是 **natural-ref**（metric tsv 用原始未 prefix caption）— 只能說 cross-format alignment 弱，不等於 prompt-following 失敗。Prompt-following（prefixed-ref CLAP）+ dual-ref backfill 排在 priority queue（Codex Round 2 P1 驅動）。Working hypotheses（行為類似 P9 multi-cap 但 mechanism 未證）見下方分析段 |
| **Phase 8 V4 Q** | `JamendoFull-Random-PromptConsistency-S2OnlyQ` | 同 P8 V4（保留 `[consistency=X.XX]` prefix），但 S2 開 `use_q_conditioning=true`（複用 P8 V4 S1 NoQ ckpt）。測試問題：text prefix + q_embed 雙 pathway 並存能否救回 semantic CLAP？ | ✅ **完成 2026-04-27**（dual-ref）。**MusicCaps**：q=6 prefixed_ref **0.0626** / natural_ref 0.0562；q=9 prefixed_ref 0.0598 / natural_ref 0.0539。**Jamendo seed42**：q=6 prefixed_ref 0.0450 / natural_ref 0.0447；q=9 prefixed_ref 0.0417 / natural_ref 0.0411。**讀數**：(1) prefixed_ref 一致高於 natural_ref（MC ~+11%、JM ~+1%）→ 模型有在 follow prompt format；(2) 加 Q pathway 沒救回 CLAP（vs P7 V1 ~0.20 仍差 3-4×）；(3) MC q=6 > q=9（P7 V1 是 flat，這裡不同）。⚠️ caveat：S1 沒看過 Q → S2-only Q regime（half-Q 等級劣化已知）|
| Phase 9 V1 (buggy) | `JamendoFull-TrueRandom-NoQ` | LP-MusicCaps 5 caps 動態採樣，無 Q | ❌ 廢棄（帶 q=9→10 bug、undrop 別名 bug，Jamendo CLAP 0.0260 崩盤）|
| **Phase 9 V1 bugfix** | 同上（修 bug 後）| 修 networks.py q=10 + runner_meanflow.py undrop clone | ✅ 完成 2026-04-20。MusicCaps CLAP 0.0650（2.5x 修前），AES 四項超 Phase 8，但 CLAP 遠不及 static random。跨 test set 一致（非 overfit），殘差尚未被單一機制定位 |
| Phase 9 V2 (half Q) | `JamendoFull-TrueRandom-MeanSim-Q` | 同 V1 + Q=pairwise MeanSim of 5 caps | ❌ 廢棄於 iter 31k（發現 runner_flowmatching.py 沒讀 q；artifact 保留為 `phase9_v2_s1noq_s2q_partial_*`）|
| **Phase 9 V2 bugfix** | 同上（真 Q end-to-end） | 額外修 runner_flowmatching.py 6 處傳 q | ✅ 完成 2026-04-21。MusicCaps **q=9** CLAP 0.0403 < V1。**需注意 confound**：(a) multi_cap 本身、(b) full Q vs half Q、(c) q=9 不是訓練分布眾數 — 三變量未拆開。假說：aggregate-q 與 random-1/5 mismatch（未證）|
| Phase 9.5 V1 | `JamendoFull-QwenOmni-TrueRandom-NoQ` | Qwen2.5-Omni-3B 5 task caps | ⏸️ **Captioning 暫停 2026-04-26**（Slot 0 ✅ 251,599、Slot 1 ~67K 後停；ETA 期限內跑不完，讓位給 P8 V4 PromptConsistency 實驗。jsonl 不丟，可後續 `--resume`） |
| Phase 9.5 V2 | `JamendoFull-QwenOmni-TrueRandom-MeanSim-Q` | 同上 + Q=pairwise MeanSim of 5 task caps | ⏸️ 同上（共用 captions） |

## Phase 9 NPZ 前處理狀態（2026-04-18）

- `gen_multicap_npz.py` 已跑完，iter 6243 崩潰原因為 `~/phase9_multicap_npz/990.npz` 和 `1218.npz` 缺 `text_features_c`
- 已透過 `gen_multicap_npz.py --resume` 重新生成，251,599/251,599 齊全
- `train_pipeline_phase9_v1.sh` 已加上 pre-flight 驗證

## Phase 9 V1/V2 bugfix 核心發現（2026-04-20/21）

**三個結構性 bug（Codex 抓到兩個關鍵）**：
1. `networks.py:526/558` MeanAudio q=None 填 9（應為 10 null token）→ `use_q_conditioning=False` 實驗 train/eval mismatch。Codex 2026-04-19 發現。
2. `runner_meanflow.py:238-239/268-269` `text_f_undrop = text_f` 是別名不是 clone → in-place null mask 污染 CFG target。Claude 2026-04-19 獨立發現。
3. `runner_flowmatching.py` 完全沒讀 q_level、沒傳 q 到 FluxAudio → 所有 Phase 6+「+Q」實驗 S1 都沒訓 q_embed[0-9]。Codex 2026-04-20 發現。已修 6 處（L224/252/262/285/307-309/414-416）。

**實測觀察（需 control 佐證）**：
- multi_cap + NoQ (V1)：CLAP 0.0650 < static random NoQ (Phase 8) 0.1851。AES 超 Phase 8，跨 test set 一致（非 overfit）
- multi_cap + Q E2E (V2, q=9)：CLAP 0.0403 < V1。**但只測了 q=9，未 q sweep**

**Codex 2026-04-21 警告的 confound**：
- V2 比歷史 Phase 7 V1 (0.1975) 差，但那是 half-Q；V2 是真 full Q → 混了 (a) multi_cap 效應、(b) full Q vs half Q、(c) q=9 vs 最適 q 三個變量
- 不能直接下「multi_cap 本質性不適合」的定論

**已完成（2026-04-21）**：
1. **P9 V2 q sweep**：q=6/7/8/9 皆 flat（CLAP 0.0402–0.0417）→ P9 V2 failure 不能歸因於 q=9 選錯
2. **P7 V1 q-sweep**（既有 checkpoint）：q=0/3/6/9 顯示 support-set gating — q=0/3 OOD 區 CLAP ≤ 0.045，q=6/9 in-support 區 CLAP ~0.197 等價。Q 表現為 coarse regime marker，非 ordinal quality controller

**已完成（2026-04-22）**：
3. **Phase 7 V1 full-Q control rerun**（~36 hr wall clock）：乾淨 implementation (S1 q-passing fix + S2 text_f_undrop clone fix) 下的 full-Q E2E。全 5 eval 一致低於歷史 P7 V1 best ~8-12% CLAP：
   - Jamendo q=6: 0.1816 vs 0.1980（−8.3%）
   - Jamendo q=9: 0.1799 vs 0.1984（−9.3%）
   - Jamendo native_q: 0.1801 vs 0.1977（−8.9%）
   - MusicCaps q=9: **0.1748 vs 0.1975（−11.5%）**
   - MusicCaps q=6: 0.1759（歷史無直接對照）
   - support-set gating 行為 replicate（q=6/9/native_q 內部差 ≤ 0.002）

**這個 control 的活躍 implementation 差異只有 2 個**（非 3 個）：
- S1 runner_flowmatching q-passing fix（S1 現在真訓 q_embed[0-9]）
- S2 runner_meanflow `text_f_undrop.clone()` fix（CFG target 不再被污染）
- `networks.py q=None→10` fix 不活躍（train+eval 都用顯式 q_level）

### Puzzle — apparent tension with historical P6 V1 vs V2

Historical P6 V2 outperformed P6 V1, but this should not be interpreted as evidence that Stage 1 successfully trained q embeddings. At that time, the runner_flowmatching q-passing bug was still present, so P6 V2 tested the presence of a q_embed layer in the Stage 1 architecture, not effective Stage 1 q learning. The current P7 full-Q rerun shares the former but differs in two active respects: Stage 1 q embeddings are now actually trained, and the Stage 2 text_f_undrop alias bug is fixed. Therefore, the current drop should not be summarized as "full-Q is harmful"; it remains compatible with at least two unresolved contributors: effective Stage 1 q training, the Stage 2 clone fix, or their interaction.

### 已被 falsify 的 strong version

「P9 V2 的差可以完全歸因於 multi-cap」不成立。P9 V2 gap 至少包含：
1. **Clean-implementation penalty** ~0.02 CLAP（相對 historical half-Q baseline 的觀察，attribution 未分離在 S1 q training vs S2 clone fix）
2. 一個 P9-specific residual（~0.13 CLAP，行為上與 multi-cap 強相關但未證因果）

### ✅ 已解決（2026-04-24 ablation chain 完整後）

- ~~「clone fix 造成 drop」~~ → **已 falsify**：s2only 5/5 eval ≈ historical，clone fix 非主因
- ~~「pseudo-EMA bootstrap 膨脹 ema_final」~~ → **已 falsify**：兩實驗 EMA gap 一致（~13-14%），結構性現象
- **現在可以寫**：`The primary remaining contributor is Stage 1 effective q training.`

### 仍不能寫的 strong claims

- 「full-Q 本身有代價」/ `S1 q training 本質上有害` — 有代價是觀察，mechanism 未證
- 「S1 q-training 造成 drop」（mechanism claim）— 只能說 primary remaining contributor，不能說 causation
- 「multi-cap 本質不適合 MeanAudio」— P9-specific residual 仍未有 mechanism proof

### Confound 記錄

- **A. gt_cache / TSV alignment**：✅ 已驗證 — 歷史與 rerun 都用 `npz_cache_train.txt` (MD5 `1e1641f0...`) + `~/research/meanaudio_training/npz`，相同。
- **B. Pseudo EMA bootstrap**：僅適用 Clean S2 only ablation，不適用 finished full-Q control rerun（後者是從零訓 S1）。
- **C. Eval pipeline 版本**：歷史 (Mar 2026) 與 rerun (Apr 2026) 都用當前 eval 流程 + num_samples=2048 metric。顯式驗證 TODO（若有疑問可跑歷史 ckpt eval 驗證是否還得 0.1984）。

### ✅ 完成（2026-04-23）

4. **Clean S2 only ablation** (`phase7_v1_s2only_ablation`, tmux `p7v1_s2only`)：用歷史 P7 V1 S1 weights (wrapped into load-compatible pseudo training-state ckpt) + 只重訓 S2 with clone fix + 5 eval（Jamendo q=6/q=9/native_q + MusicCaps q=6/q=9）。

   **EMA final 結果（5/5 eval 完成）**：
   | Eval | Historical | fullq_control | s2only | s2only Δ vs hist |
   |---|---|---|---|---|
   | Jamendo q=6 CLAP | 0.1980 | 0.1816 | **0.2008** | +0.0028 (+1.4%) |
   | Jamendo q=9 CLAP | 0.1984 | 0.1799 | **0.1993** | +0.0009 (+0.5%) |
   | Jamendo native_q CLAP | 0.1977 | 0.1801 | **0.1995** | +0.0018 (+0.9%) |
   | MusicCaps q=6 CLAP | — | 0.1759 | **0.1981** | — |
   | MusicCaps q=9 CLAP | 0.1975 | 0.1748 | **0.1951** | −0.0024 (−1.2%) |

   **Attribution（5/5 一致訊號）**：s2only ≈ historical across all evals；fullq_control 持續低 ~8-12%。
   **結論**：The Stage 2 text_f_undrop clone fix is not the main driver of the ~8-12% CLAP drop in fullq_control. The primary remaining contributor is Stage 1 effective q training (enabled by the runner_flowmatching q-passing fix).

   - **Pseudo-EMA bootstrap confound**：兩條 ema_models (sigma 0.05 / 0.1) 都從同一份 `_ema_final.pth` 起跑，**非歷史真實雙軌跡**，是 load-compatible approximation，**不是 semantic equivalent**。
   - **Last.pth insurance ✅ 完成（2026-04-24）**：
     | | ema_final q=9 | last.pth q=9 | EMA gap |
     |---|---|---|---|
     | s2only ablation | 0.1993 | 0.1757 | +13.4% |
     | fullq_control | 0.1799 | 0.1575 | +14.2% |
     兩者 EMA-vs-online gap 一致（~13-14%）→ **pseudo-EMA bootstrap confound 排除**，gap 為 S2 訓練結構性現象。

5. **最終 drop 拆解（2026-04-24 定稿）**：
   - **General penalty ~0.02 CLAP**：主要來自 S1 effective q training（runner_flowmatching q-passing fix 啟用後）
   - **P9-specific residual ~0.13 CLAP**：行為上與 multi-cap 強相關，機制未證
   - **S2 clone fix**：不是 fullq_control drop 的主因（已 falsify）
   - **Pseudo-EMA bootstrap**：不影響 ema_final 比較結論（已 falsify）

## Phase 9 caption responsiveness — behavior-level 診斷（2026-04-21）

**Behavior-level association**（非 causal / 非 mechanistic claim）：在目前實作下，**single-cap 訓練組**保有明顯 prompt steering；**multi-cap 訓練組**的 same-seed prompt steering 大幅衰弱。

**方法**：
- 固定 cfg=0.5, num_steps=1（benchmark-matching）
- 4 個 A/B prompt pair（樂器、人聲、鼓、編制密度），每 pair 同 seed 對打
- 3 seeds × 2 prompts × 4 pairs = 24 檔/model
- 量 `(A-B L2) / (noise L2)` ratio — **noise L2 是同 prompt 不同 seed 的 L2 baseline**
- Probe battery 375-state grid（6 ckpts S1+S2、5 seeds、5 timesteps、10 prompt pairs、4 metrics）determinism check d=0 通過
- P9 V2 q=9 sanity 與 q=8 差距 |Δ| ≤ 0.006，q sweep 結論穩定

**結果 — 4 模型 2x2 分組（A/B same-seed L2 / noise L2）**：

| Model | 01 instr | 02 vocals | 03 drums | 04 density |
|---|---:|---:|---:|---:|
| **P7 V1** (Q, single-cap) | 1.457 | 1.071 | 1.702 | 0.884 |
| **P8** (NoQ, single-cap) | 1.121 | 0.950 | 1.723 | 0.913 |
| **P9 V1** (NoQ, multi-cap) | 0.075 | 0.025 | 0.068 | 0.147 |
| **P9 V2** (Q=9, multi-cap) | 0.015 | 0.012 | 0.021 | 0.056 |

- ratio > 1 ⇒ prompt 效應 > noise 效應（single-cap 組）
- ratio < 0.2 ⇒ noise 主導、prompt 微弱（multi-cap 組）
- **Q 與架構都不是區分因素**；single-cap vs multi-cap 是行為分界線

**可說**：
- Same-seed prompt steering weakens strongly in multi-cap runs（behavior-level association）
- P9 不是完全不看 caption；prompt effect 已經弱到遠小於 noise effect
- P9 V1 殘留最弱反應維度：density（0.147）與 instrument/drums（~0.07），vocals 最弱（0.025）
- P9 V2 在所有維度比 V1 更弱（0.01-0.06）
- Probe battery 一致：P9 a/c ratio 0.001-0.015 vs P7 0.10-0.21（差 20-200x）；P9 S1→S2 ratio 再跌 4-6x，P7 沒跌

**不能說**：
- ❌ multi-cap "導致" conditioning 失敗（correlation, not causation；data 混合比例、lr 等 confound 未控制）
- ❌ text_cond_proj 梯度被毒、weight 崩壞等 mechanism
- ❌ P9 "unconditional generation"（殘留 ratio 非 0）
- ❌ P9 "完全不看 caption"

**Artifacts**（`eval_output/probe_subjective_v2/`）：
- `p7v1/`、`p8/`、`p9v1/`、`p9v2/`（q=8）、`p9v2_q9/`（sanity）各 24 wav
- `probe_battery_results.json` 3450 條 records

## Phase 9.5 Qwen captioning 狀態（2026-04-25/26 active）

**Lane C 啟動 2026-04-25 18:06** — `tmux qwen_omni` 跑 `gen_qwen25omni_captions.py --slot all --resume`，GPU 99% / 24 GB（獨佔）。

| Slot | Task framing | 狀態 |
|------|-------------|------|
| 0 | NaturalProse | ✅ **251,599 / 251,599** done (4/26 ~03:47 estimate) |
| 1 | Summary | 🔄 進行中 (~67K / 251K @ 00:58 4/26) |
| 2 | Writing | ⏸️ 等 slot 1 |
| 3 | Paraphrase | ⏸️ 等 slot 2 |
| 4 | Attribute | ⏸️ 等 slot 3 |

- 模型：`Qwen/Qwen2.5-Omni-3B`（Thinker-only，SDPA attention）
- 輸入：`phase7_v1_train.tsv` (251,599 clips) + audio root `/home/hsiehyian/dataset/segments_no_vocals`
- 輸出：`/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions_slot{0..4}.jsonl`
- Throughput: ~216 captions/min sustained (varies 144–280 based on audio difficulty)
- ETA: slot 1 末 ~15:00 4/26、slot 2 ~07:00 4/27、slot 3 ~23:00 4/27、slot 4 ~15:00 4/28（約 2.5 天）

### n=11 早期 diversity sample（slot 0 vs slot 1，2026-04-25 21:34）

| 評估 | 數量 | 解讀 |
|------|------|------|
| ✅ Consistent (multi-task 真產生 valid 多角度) | 6/11 = 55% | 同 genre / 同 instruments / 同 mood，差別只是 verbosity 或 task focus |
| ⚠️ Mild contradiction / ambiguous | 3/11 = 27% | 同 genre 但 energy 描述偏移（e.g. soft vs vibrant reggae） |
| ❌ Clear contradiction (hallucination) | 2/11 = 18% | 互斥屬性（acoustic vs electric guitar、somber slow vs feel-good upbeat） |

**對 P9.5 訓練的意涵**（Day 4 全 100 筆完整檢查再定論）：
- 82% at-least-同義 → multi-cap 有真正的 diversity 信號（hypothesis 仍成立）
- 18% hallucination 給訓練信號加噪聲，但**不是垃圾資料**（語意仍在，只是樂器/情緒 misjudge）
- **觀察點**：mean_sim 信號可能被「captioner stability」而非「audio difficulty」污染 → 與 `project_mean_sim_interpretation_hypothesis.md` 反向假設可能對應，P9.5 訓完後值得分析 q 分布 vs audio 特徵

詳細設計見 `phase9_design.md`，Lane A/B/C 排程見 `../meetings/2026-04-18_lane_abc_and_lpmc.md`。

## Phase 8 V4 結果與分析（2026-04-27 完成）

### 數字

| Benchmark | CLAP ↑ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ |
|-----------|--------|------|------|------|------|
| MusicCaps n=5521 | **0.0571** | 5.7708 | 6.5038 | 5.2716 | 6.4163 |
| Jamendo seed42 n=2048 | **0.0591** | 5.7342 | 6.4665 | 5.2631 | 6.3741 |

### 與 P8 NoQ baseline 比較

| 指標 | P8 V4（PromptConsistency-NoQ） | P8（Random-NoQ） | Δ |
|------|-------------------------------|-----------------|---|
| MusicCaps CLAP | 0.0571 | 0.1851 | **−67%** |
| Jamendo CLAP | 0.0591 | 0.1986 | **−70%** |
| MusicCaps CE | 5.771 | 5.913 | −2.4% |
| MusicCaps PQ | 6.416 | 6.544 | −2.0% |
| MusicCaps PC | **5.272** | 4.983 | **+5.8%** ↑ |

### 診斷：P9 multi-cap 模式重現

**Pattern (behavior-level observation)**：CLAP（natural-ref）大跌（~70%）但 AES 僅小跌（CE/PQ −2-3%），PC 反而上升。這與 P9 V1 NoQ 的「好聽但不貼 prompt」模式相似（P9 V1 MusicCaps natural-ref CLAP 0.0650、AES 超 P8）。

**⚠️ Codex P1 2026-04-27 caveat**：上述 CLAP 0.0571 / 0.0591 是 **natural-ref**（metric tsv = 原始未 prefix caption；generation tsv = 帶 `[consistency=0.90]` prefix）。**這只是 cross-format alignment，不是 prompt-following metric**。要真正測「模型有沒有跟著 prompt 走」必須補 prefixed-ref pass（generation 與 metric 同 TSV）。dual-ref backfill 排在 2026-04-27 priority queue #1.5。

**Working hypotheses（需 embedding/probe evidence 才能升級為 mechanism claim）**：
- H1: `[consistency=X.XX]` 前綴占據 T5 token sequence 前幾個位置，可能影響 text embedding 主方向（**未測 embedding norm/方向變化**）
- H2: 模型可能學到「consistency 數值 → 音質/風格」捷徑而 underweight 語義 caption（**未做 attention/probe 驗證**）
- H3: 行為層觀察（CLAP↓ + AES≈持平 + PC↑）與 multi-cap collapse 模式相似（**僅相似，未證共享機制**）

**對 QA-MDT 類方法的觀察**（design suggestion，非 finding）：
- QA-MDT 原始設計用離散 quality token（`[high]`/`[medium]`/`[low]`），semantic caption 完整保留
- P8 V4 用 raw float prefix `[consistency=0.83]`，T5 encode 成連續向量
- 一個可能的設計改進：quality token 接 special-vocab embedding 或走 separate q_embed pathway（與 P7 V1 同層）

### 暫定結論（pending dual-ref result + ablation chain）

**已證（behavior-level）**：在 natural-ref CLAP 上，P8 V4 NoQ 的 prompt prefix 設計顯著低於 P8 NoQ baseline。AES 維持/PC 微升。

**尚不能證**：
- ❌ 「prefix 主導 T5」（mechanism claim — 需 embedding 變化證據）
- ❌ 「shortcut learning」（需 probe / attention evidence）
- ❌ 「複製 multi-cap text conditioning collapse」（行為相似 ≠ 同機制）

**待 dual-ref + 後續 ablation 補完**：
- ~~prefixed-ref CLAP（P8 V4 Q）~~ ✅ **完成 2026-04-27 14:25**：MC q=6 prefixed=0.0626 / natural=0.0562；q=9 prefixed=0.0598 / natural=0.0539。**Δ ~+11% on MC**, **~+1% on JM**。Prefixed_ref 高於 natural_ref → 模型有 follow prompt format，只是與 natural caption 對齊度低
- prefixed-ref backfill for P8 V4 NoQ p=0.90 baseline — 跑中（priority queue #1.5）
- prefixed-ref + natural-ref for P8 V4 NoQ p=1.00 — 跑中（priority queue #2）
- P7 V1 q=0..9 q-sweep on MusicCaps — 跑中（priority queue #3）

### Dual-ref 結果讀解（P8 V4 Q，2026-04-27）

|  | MC q=6 | MC q=9 | JM q=6 | JM q=9 |
|---|---|---|---|---|
| prefixed_ref CLAP | **0.0626** | 0.0598 | 0.0450 | 0.0417 |
| natural_ref CLAP  | 0.0562 | 0.0539 | 0.0447 | 0.0411 |
| Δ (P>N) | +0.0064 (+11%) | +0.0059 (+11%) | +0.0003 (+1%) | +0.0006 (+1%) |
| vs P7 V1 baseline | × 3.2 較低 | × 3.3 較低 | × 4.4 較低 | × 4.7 較低 |

**可說（observation）**：
- Prefixed_ref 一致高於 natural_ref，**MC 上幅度顯著（~11%）**、JM 上接近 noise（~1%）
- 即使用 prefixed_ref（對 P8 V4 Q 較有利的 metric），CLAP 仍遠低於 P7 V1（~0.20）→ **加 q_embed pathway 無法補回 prefix 設計造成的 CLAP 缺口**
- q=6 略高於 q=9（與 P7 V1 在 in-support 區的 flat 行為不同；可能 S2-only Q regime 的副作用）

**不能說**：
- ❌ 「prefix 沒影響語意」— prefixed_ref 仍遠低於 P7 V1 baseline，只是 prefixed 比 natural 多 ~11%
- ❌ 「q_embed pathway 無效」— 這次 S2-only Q + 跟 prefix 共用 text encoder 是混在一起的（confound：S2-only regime + prefix 同存）
- ❌ MC/JM 差異反映「prefix vs no-prefix」— JM 也用 prefix，差異更可能是 cross-domain vs in-domain

### 後續 ablation：prefix value sweep（2026-04-27 排隊中）

**P8 V4 NoQ + `[consistency=1.00]` eval**（用既有 ckpt，不重訓）：
- 動機：0.90 是 in-support p90，是否 push 到 1.00 邊界值會更好/更差？
- caveat：1.00 是 OOD-edge（PM 2026-04-26 警告原則），這是 paper-completeness ablation
- TSV：`~/eval_tsvs_p100/phase8_v4_{musiccaps_test,jamendo_seed42_2048}_p100.tsv`
- Pipeline：`~/MeanAudio/eval_p8v4_noq_p100.sh`
- 排隊：等 P8 V4 Q (S2 + 4-eval) 跑完才插隊（避免 GPU 衝突）
- ETA：P8 V4 Q ~14:15 完 → p100 eval ~14:35 完

### 平行進行：P8 V4 + Q 變體（2026-04-27）

`phase8_v4_q_stage2_200000`（tmux `p8v4q`）：複用 P8 V4 S1（NoQ ckpt），S2 開 `use_q_conditioning=true`。測試假說：text prefix + q_embed 雙 pathway 並存能否救回 semantic CLAP？S2 from scratch q_embed → S2-only Q regime，已知 half-Q 等級劣化。Eval q=6 + q=9 × MusicCaps + Jamendo seed42 (4 跑)。S2 速度 0.110 s/iter, ETA 13:15 訓練完、~14:15 全 eval 完。
