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
| Phase 9 V1 | `JamendoFull-TrueRandom-NoQ` | LP-MusicCaps 5 caps 動態採樣，無 Q | ⏳ 待啟動（2026-04-18 修復完成）|
| Phase 9 V2 | `JamendoFull-TrueRandom-MeanSim-Q` | 同上 + Q=pairwise MeanSim of 5 caps | ⏳ 待啟動（跟隨 V1，共用同一 NPZ 目錄）|
| Phase 9.5 V1 | `JamendoFull-QwenOmni-TrueRandom-NoQ` | Qwen2.5-Omni-3B 5 task caps (Writing/Summary/Paraphrase/Attribute/NaturalProse)，動態採樣，無 Q | ⏸️ Captioning 暫停於 slot 0 = 232,681/251,599（92.5%）。2026-04-18 教授指示：先完成 Lane A → Lane B (P9 V1/V2) → Lane C (Qwen resume)|
| Phase 9.5 V2 | `JamendoFull-QwenOmni-TrueRandom-MeanSim-Q` | 同上 + Q=pairwise MeanSim of 5 task caps | ⏳ 待 Qwen captions 完成後開訓 |

## Phase 9 NPZ 前處理狀態（2026-04-18）

- `gen_multicap_npz.py` 已跑完，iter 6243 崩潰原因為 `~/phase9_multicap_npz/990.npz` 和 `1218.npz` 缺 `text_features_c`
- 已透過 `gen_multicap_npz.py --resume` 重新生成，251,599/251,599 齊全
- `train_pipeline_phase9_v1.sh` 已加上 pre-flight 驗證

## Phase 9.5 Qwen captioning 狀態

- Slot 0 captioning 暫停於 232,681/251,599（92.5%），Lane C 時恢復
- 模型：`Qwen/Qwen2.5-Omni-3B`（Thinker-only，SDPA attention）
- 輸入：`phase7_v1_train.tsv` (251,599 clips) + audio root `/home/hsiehyian/dataset/segments_no_vocals`
- 輸出：`/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions_slot{0..4}.jsonl` → merge 後 `phase9_omni_captions.jsonl`

詳細設計見 `phase9_design.md`，Lane A/B/C 排程見 `../meetings/2026-04-18_lane_abc_and_lpmc.md`。
