# 106：086 短負向措辭組套到 081 arm（2026-10-06）

## 問題

081 的 E1（`Low quality recording.` 負向差中差 +1.061 PQ）有兩個成分混在一起：

- **arm 學到標籤字**：arm 訓練時，PQ 最差的 20% caption 加上了這串前綴。
- **control 對這個措辭特別遲鈍**：086 發現，在 A（081 control，nmv2pair quarter s14159265）上，`Low quality recording.` 是最弱的負向。它只有 +0.041 lvl30，比無關文字低 0.48；同一個 A 用 `low quality, noisy` 是 +0.59，用 fidelity8 是 +0.83。

所以 E1 有一部分可能只是 control 吃了措辭虧。本 probe 在 arm 上跑 086 的同一組措辭，讓每個措辭都有乾淨的差中差。

## 設計

- **C**：081 arm `phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s14159265_stage2_50000`（與 A 同訓練 seed）。
- **格**：cfg0、none（stored null）、lqrec、lqnoisy、lq、irrel、fid8。文字與 086 逐字相同。
- **重現錨點**：A cfg0 重生一次，與 086 的 `A__cfg0.json` 比對。只記錄、不擋：|ΔPQ| ≤ 0.005，因為 AES 會隨 batch 組成漂約 1e-3。
- **A 的逐 clip 讀數**：直接讀 086 的 cells JSON。子集、雜訊、評分器、pinned sha 全部相同。
- **協定**：MusicCaps subset1024、MeanFlow 25、seed 42、fp32、NoMask、`--no_q`、CFG 3（cfg0 格為 0）。評分用 eval_metrics（CLAP batch 1），再做 −30 LUFS 對齊讀數。bootstrap 10000 次，seed 20261006。
- **腳本**：`scripts/eval/shortneg_arm081_probe_20261006.py`。guest 抄自 086。
- **規模**：8 格 × 1024 ≈ 50 分鐘，所以走 p2 queue 106。

## 預登錄讀法（主讀 PQ lvl30）

1. **逐措辭差中差** DiD(k) = [C k − C cfg0] − [A k − A cfg0]。
2. **標籤專一性** = DiD(lqrec) − DiD(k)，其中 k ∈ {lqnoisy, lq, irrel, fid8}；等價於 C(lqrec − k) − A(lqrec − k)。
   - 若 **lqrec − lqnoisy ≥ 0.19 且 CI 下界 > 0**，判定為「標籤訓練對這串字有專一效果」：arm 對自己的標籤字反應多過對一般短負向的反應。
   - 若 **DiD(lqnoisy) 與 DiD(fid8) 也 ≥ 0.19**，判定為「arm 對負向普遍更敏感」：標籤訓練放大的是整個負向槽，不只是那串字。
   - 兩者可以同時成立。
3. **乾淨的 E1 估計** = C lqrec 增益 − A 的最佳短措辭（lqnoisy）增益。它是 E1 扣掉 control 措辭虧之後的下界讀法。
4. **CLAP 是唯一獨立讀數**，原因是標籤與端點都是 AES PQ。

## 限制

- 只有一個訓練 seed、一個生成 seed，而且是子集，沒有 FAD。081 的 FAD 已經顯示 PQ 增益有 FAD 代價（p2 105）。
- 門檻 0.19 沿用 081（3 seed 全量 CFG3+neg 底線的 2×）。這裡是單 seed、子集，所以只當方向讀。
