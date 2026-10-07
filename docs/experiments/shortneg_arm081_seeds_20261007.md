# 107：106 短負向措辭組補另外兩個訓練 seed（2026-10-07）

## 問題

106 只用了一個訓練 seed（s14159265）。它的兩個判讀條件都成立：
- 標籤專一：lqrec − lqnoisy 差中差 +0.44。
- 整槽放大：DiD(lqnoisy) +0.54、DiD(fid8) +0.44。

乾淨 E1 是 +0.43 lvl30。但 106 的 CI 只反映 clip 抽樣，081 的 seed 間差異遠大於這個寬度（例如 075 lab 臂增益在 seed 間從 0.62 掉到 0.07）。本 job 補 s16180339、s27182818，讓每個讀數都有 3 seed。

## 設計

- **checkpoint**：每個 seed 都用 A = `..._slot0clean_nmv2pair_noq_quarter_<seed>_stage2_50000` 與 C = `..._slot0clean_qlabel_noq_quarter_<seed>_stage2_50000`，即 081 的 control 與 arm。
- **格**：A、C 各 7 格（cfg0、none、lqrec、lqnoisy、lq、irrel、fid8），2 seed × 2 ckpt × 7 = 28 格。文字與 086／106 逐字相同。
- **s14159265**：直接讀 086 的 A cells 和 106 的 C cells，不重生。106 已驗證 A cfg0 重生逐 clip 完全一致。
- **協定**：與 106 相同。MusicCaps subset1024、MeanFlow 25、生成 seed 42、fp32、NoMask、`--no_q`、CFG 3（cfg0 格為 0）。評分用 eval_metrics（CLAP batch 1），再做 −30 LUFS 對齊。bootstrap 10000 次，seed 20261007。
- **腳本**：`scripts/eval/shortneg_arm081_seeds_20261007.py`。guest 抄自 106。
- **規模**：28 格，約 3 小時，走 p2 queue 107。
- **路徑驗證**：拿 s14159265 的資料假扮另外兩個 seed 跑分析，逐位重現 106 的數字。

## 預登錄讀法（主讀 PQ lvl30，3 seed 平均）

1. **標籤專一**：3 seed 平均 lqrec − lqnoisy 差中差 ≥ 0.19，而且 3 個 seed 都 > 0。
2. **整槽放大**：DiD(lqnoisy) 與 DiD(fid8) 的 3 seed 平均都 ≥ 0.19，而且各自 3 個 seed 都 > 0。
3. **乾淨 E1** = C 的 lqrec 增益 − A 的 lqnoisy 增益。報 3 seed 平均、逐 seed 值，以及占原始 DiD(lqrec) 的比例。
4. **CLAP** 是唯一獨立讀數，原因是標籤與端點都是 AES PQ。
5. **探索（106 預登錄以外的觀察）**：`low quality` 在 arm 上失效（DiD(lq) < 0）是否 3 seed 都成立；stored null 的 CFG3 懲罰在 arm 上是否消失。

如果某條在 3 seed 平均過門檻，但有 seed 翻號，就報「不一致」，不判定成立。

## 限制

- 只有一個生成 seed、subset1024，沒有 FAD。081 的 FAD 已未過（105）。
- 3 個 seed 只能看方向與符號一致性，seed 間 SD 只是粗估。
