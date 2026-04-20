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
| Phase 8 V4 | `JamendoFull-Qwen2Audio-MeanSim-Q` | Caption 換用 Qwen2-Audio-7B | ❌ 廢棄（僅1 cap/clip，不支援 true random） |
| Phase 9 V1 (buggy) | `JamendoFull-TrueRandom-NoQ` | LP-MusicCaps 5 caps 動態採樣，無 Q | ❌ 廢棄（帶 q=9→10 bug、undrop 別名 bug，Jamendo CLAP 0.0260 崩盤）|
| **Phase 9 V1 bugfix** | 同上（修 bug 後）| 修 networks.py q=10 + runner_meanflow.py undrop clone | ✅ 完成 2026-04-20。MusicCaps CLAP 0.0650（2.5x 修前），AES 四項超 Phase 8，但 CLAP 遠不及 static random → **multi_cap 呈 unconditional drift**（跨 test set 一致）|
| Phase 9 V2 (half Q) | `JamendoFull-TrueRandom-MeanSim-Q` | 同 V1 + Q=pairwise MeanSim of 5 caps | ❌ 廢棄於 iter 31k（發現 runner_flowmatching.py 沒讀 q；artifact 保留為 `phase9_v2_s1noq_s2q_partial_*`）|
| **Phase 9 V2 bugfix** | 同上（真 Q end-to-end） | 額外修 runner_flowmatching.py 6 處傳 q | ✅ 完成 2026-04-21。MusicCaps **q=9** CLAP 0.0403 < V1。**需注意 confound**：(a) multi_cap 本身、(b) full Q vs half Q、(c) q=9 不是訓練分布眾數 — 三變量未拆開。假說：aggregate-q 與 random-1/5 mismatch（未證）|
| Phase 9.5 V1 | `JamendoFull-QwenOmni-TrueRandom-NoQ` | Qwen2.5-Omni-3B 5 task caps | ⏸️ 暫緩：待 P9 V2 q sweep + Phase 7 V1 full-Q control 完成才決定（避免在 multi_cap 路線未完全釐清前擴展）|
| Phase 9.5 V2 | `JamendoFull-QwenOmni-TrueRandom-MeanSim-Q` | 同上 + Q=pairwise MeanSim of 5 task caps | ⏸️ 暫緩：同上 |

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

**下一步（最小實驗清單）**：
1. **P9 V2 q sweep** (~2 hr)：`--quality_level 6/7/8`，用既有 S2 EMA，只重跑 eval
2. **Phase 7 V1 full-Q control** (~20 hr)：S1+S2 真 Q end-to-end + static random
3. 以上完成才能分離 multi_cap 與 full Q 的獨立效應

## Phase 9.5 Qwen captioning 狀態

- Slot 0 captioning 暫停於 232,681/251,599（92.5%），Lane C 時恢復
- 模型：`Qwen/Qwen2.5-Omni-3B`（Thinker-only，SDPA attention）
- 輸入：`phase7_v1_train.tsv` (251,599 clips) + audio root `/home/hsiehyian/dataset/segments_no_vocals`
- 輸出：`/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions_slot{0..4}.jsonl` → merge 後 `phase9_omni_captions.jsonl`

詳細設計見 `phase9_design.md`，Lane A/B/C 排程見 `../meetings/2026-04-18_lane_abc_and_lpmc.md`。
