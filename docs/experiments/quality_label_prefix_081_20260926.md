# 081 真品質標籤前綴重訓（QA-MDT 式；2026-09-26）

## 問題

negprompt 線到目前的結論是：fidelity8 負向 prompt 的 PQ 增益（≈ +0.83，lvl30）來自「負向槽裡的保真度領域文字」，
但模型從來沒被告知哪些訓練音訊真的品質差——Qwen caption 幾乎不談錄音品質。
075 用**程式化缺陷**（加雜訊、削波…）配「點名缺陷」的 caption 去教，結果增益反而變小。

081 換成 QA-MDT（arXiv 2405.15863）的做法：**不造假缺陷**，用語料**本身**的品質分數，
把最差 20% 的列 caption 前面加 `Low quality recording.`、最好 20% 加 `High quality recording.`，其餘 60% 原樣。

**要回答的**：模型在訓練時看過「真的比較差的音訊 ↔ Low quality recording.」這個配對後，
推論時把 `Low quality recording.` 放進負向槽（或把 `High quality recording.` 放進正向 prompt），PQ 增益是否比沒看過標籤的 control 大？

## 設定

- **語料**：`slot0clean_nmv2matched`（251,596 列，= 066/070/071 nmv2pair control 的原封輸入：同 audio NPZ、同 cache list、同列序）。
- **標籤**：Audiobox Aesthetics PQ，對**模型實際訓練的窗口**打分（30 s wav peak-normalize 到 0.95 後取前 10 s @16 kHz，即 075 builder 的 `load_window()`，sha 綁定）。靜音窗（peak < 1e-6 或 RMS < −45 dBFS）與打分失敗的列不分級、caption 不動。
- **分級**：PQ 最低 20% → `Low quality recording. <caption>`；最高 20% → `High quality recording. <caption>`；中間 60% caption 逐位元等於 control。
- **Overlay**：只有 40% 前綴列重編 T5/CLAP（`(1,77,1024)`，寫 HDD `/mnt/HDD/kojiek/quality_label_081/overlay_new/`，分片 `int(stem)//1000`）；NVMe 上是 flat symlink farm（前綴列 → 新 overlay、其餘 → `text_overlays/slot0clean`）。
- **Recipe**：與 control 完全相同（S1 `fluxaudio_s` 100k + S2 `meanaudio_s` 50k、NoQ、NoMask、LR 1e-4、BS 8、`cap_index_fixed=0`、`require_text_overlay=true`），只換 train TSV 與 overlay。
- **Seed**：14159265 / 16180339 / 27182818（與 control 逐 seed 配對）。
- **為什麼用 prefix 文字不用 q_embed**：QA-MDT 的做法就是文字前綴；q_embed 線已收（Q 在乾淨 code 下無淨貢獻）。這裡問的是「負向槽的文字若在訓練時被綁到真實劣質音訊，效果會不會變大」，必須走文字。

## Eval 格（每個 seed；MusicCaps 5521、MeanFlow 25、seed 42、fp32、NoMask、`--no_q`；CLAP 一律對原始 caption、batch 1）

| 格 | 正向 prompt | 負向 | arm | control |
|---|---|---|---|---|
| cfg0 | caption | — | 新跑 | 已有 |
| cfg3_neg | caption | fidelity8 | 新跑 | 已有 |
| cfg3_lqneg | caption | `Low quality recording.` | 新跑 | 新跑 |
| hqpos cfg0 | `High quality recording. ` + caption | — | 新跑 | 新跑 |
| hqpos cfg3_lqneg | `High quality recording. ` + caption | `Low quality recording.` | 新跑 | 新跑 |

每格評完做 −30 LUFS 對齊重評（`level_match_rescore.py`），然後**刪音檔**（REPORT、per_clip、lvl30 per_clip 保留；音檔可決定性重生）。
新 wrapper `scripts/eval/mc_mf25_negvariant_eval.sh` 只換負向文字，其餘旗標與 `mc_mf25_eval.sh` 的 CFG3+neg 格相同——已驗證以 fidelity8 餵它時音檔與 stock cfg3_neg 逐樣本相同（8/8）。

## 端點（預先登記）

PQ 一律用 lvl30、逐 clip 配對；3 seed 先逐 clip 平均再 clip bootstrap 10000 次（seed 20260926）。
門檻 = 2× 訓練 seed 雜訊底線（`reference_training_seed_pq_noise_floor.md`）。

- **E1（主）**：[arm lqneg − arm cfg0] − [ctrl lqneg − ctrl cfg0] ≥ **0.19**、CI 下界 > 0、3 個 seed 各自 > 0 → 「真標籤讓短負向 prompt 變有效」成立。
- **E2**：arm lqneg − arm cfg3_neg。≥ 0 表示訓練過的標籤可取代 fidelity8。
- **E3**：[arm hqpos cfg0 − arm cfg0] − [ctrl 同差]，門檻 **0.155**（CFG0 底線 2×）。正向槽 HQ 標籤是否變有效（正向槽不對稱先例：未訓練時 ≈ 0）。
- **E4**：hqpos＋lqneg 的同型差中差；以及 arm hqpos lqneg − arm cfg3_neg。
- **E5（安全）**：arm vs control 的 stock 兩格，未對齊 CLAP 非劣性（CFG0 −0.004、CFG3+neg −0.0158）；PQ 同列。
- **報告規則**：每個 PQ 贏面都要同列 CLAP；未對齊 PQ 與 lvl30 方向不一致 → 報成響度效應；靜音數 > control 2 倍要標記。
- 分析：`scripts/analysis/quality_label_081_analysis.py` → `docs/experiments/results/quality_label_prefix_081_summary.json`（083 結束時自動跑）。

## 預先登記的限制

1. **標籤洩漏**：分級用 AES PQ、主端點也是 AES PQ。模型可能只是學會「PQ 評分器喜歡的東西」。CLAP 是唯一獨立讀數；沒有 FAD（目前不可用）。**任何正面結果對外前都要先過五首固定主觀 prompt 試聽。**
2. **先驗是負的**：075 點名缺陷讓 negprompt 增益縮小（3 seed 反向成立）。081 與 075 的差別是「真實分布內的劣質」vs「程式化缺陷」，但先驗不看好。
3. **MusicCaps caption 本身常寫 "low quality"**（如 "The audio quality is poor"）。arm 在 cfg0 下可能因此對這類 eval prompt 生成更差的音訊——E5 與 CLAP 會量到，但解讀要分開。
4. **T5 截斷**：加前綴後超過 77 token 的列比例上升（smoke 400 列：44% → 56%），截掉的是 caption 尾端。這是前綴法本身的代價，不另修。
5. **冷 HDD overlay 讀取**：40% 前綴列的 overlay 在 HDD（exFAT），隨機讀未量測。訓練 it/s 要和 control 比，明顯變慢要記錄（不影響正確性）。
6. **磁碟**：開跑時 NVMe 約 27 GB、HDD 約 103 GB。overlay 約 32 GB 寫 HDD；每個 seed 的 S1 / S2 run（瘦身後各約 5 GB）train 完就搬 HDD；Step 0 要求 NVMe ≥ 20 GB（S1 已完成則 13 GB）。
7. 單一語料、quarter 預算、單一生成 seed；結論不外推到 full budget。

## 流程與資源

- Queue：p2 `081_quality_label_quarter_s14159265.sh` → `082_…_s16180339.sh` → `083_…_s27182818.sh`（每個 >30 min，走 queue）。
- 第一個 seed 多做：AES 打分全語料（scandir 順序讀 wav；exFAT 隨機讀每檔 ~2 s、順序 ~8 ms）＋ 重編 ~100k overlay ＋ verify。後兩個 seed 的 builder `all` 只 verify。
- 每 seed：S1 100k + S2 50k（約 7–8 h）＋ 8 格 eval × 5521（arm 5 格、control 3 格，約 1.5 h）＋ lvl30 重評。三個 seed 合計約 30 h 以上。
- 每個 seed 的 queue evidence = 該 seed arm 的 S2 ema_final ＋ arm stock cfg0 REPORT。

## Smoke 驗證（2026-09-26，400 列）

- builder `score → build → verify` 全過（分級 80/239/80，1 列靜音）；前綴列 overlay 與原 overlay 不同、中段列逐位元相同。
- 真正的訓練 `ExtractedAudio`（`require_text_overlay`、`cap_index_fixed=0`、caption sha 綁定）400/400 列讀取通過。
- action 的 verify 區塊、eval 區塊（8 列、5 格＋5 個 lvl30、音檔刪除、重跑全 SKIP）、`archive_dir`／`thin_ema`／`drop_shadows` 在假目錄上逐一驗證。
- `mc_mf25_negvariant_eval.sh`：fidelity8 → 與 stock cfg3_neg 逐樣本相同；lqneg＋HQ gen TSV 的 log 顯示正確的正／負 prompt。
- 分析腳本在現有 control 格上：ctrl fidelity8 增益 lvl30 +0.837 [+0.818, +0.855]（與 075 control 一致）。

## 檔案

- Builder：`scripts/preprocess/build_quality_label_081_arm_inputs.py`；HQ eval TSV：`scripts/preprocess/build_hq_prefix_musiccaps_tsv.py`
- Action：`scripts/training_pipelines/quality_label_081_action.sh`；per-seed：`quality_label_081_s{14159265,16180339,27182818}.sh`
- Eval：`scripts/eval/mc_mf25_negvariant_eval.sh`
- 分析：`scripts/analysis/quality_label_081_analysis.py`
- Contracts：`docs/experiments/harn/quality_label_081/`
- 輸入：`~/exps_nvme/quality_label_081/{arm_inputs,overlay_farm,musiccaps_test_hqprefix.tsv}`；HDD `/mnt/HDD/kojiek/quality_label_081/`
