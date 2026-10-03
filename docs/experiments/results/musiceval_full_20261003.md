# MusicEval 完整 PQ / CE / CU / MEva 比較（2026-10-03）

2,748 首 / 31 系統。MEva 固定 pooled-small-f03；AES 沿用凍結的完整逐曲結果。
此為訓練資料來源內的描述性比較，非 MEva 官方 held-out 重現，也非外部泛化驗證。

| 指標 | Pearson [95% CI] | Spearman [95% CI] | 系統內 Pearson | 同 prompt 勝負一致率 |
|---|---|---|---|---|
| PQ | 0.625 [0.600, 0.650] | 0.656 [0.632, 0.679] | 0.432 | 75.5% |
| CE | 0.660 [0.634, 0.684] | 0.683 [0.657, 0.706] | 0.484 | 77.0% |
| CU | 0.660 [0.634, 0.685] | 0.692 [0.668, 0.715] | 0.485 | 76.9% |
| MEva | 0.862 [0.849, 0.875] | 0.865 [0.852, 0.877] | 0.708 | 86.8% |

Pearson：分數升降相關；Spearman：排序相關，均非正確率。
系統內：兩邊扣除各系統平均後的 Pearson。配對只用共用 100 prompts × 25 系統的 2,500 首；真人平手排除，模型平手算半分。
CI 為 prompt cluster bootstrap 2,000 次；MEva−三個 AES 的差異 CI 有 Bonferroni 同時校正。

視窗例外：既有 AES 對 S013_P013（349秒）評前90秒；MEva 評完整音檔。JSON 另附排除此音檔的2,747首敏感度分析。
MEva 可能見過本資料的訓練部分，因此高相關不可推論 PromptCC 或 PAM 的外部泛化。AES 維持使用。

逐曲結果：`/home/kojiek/MeanAudio/runtime/musiceval_full_20261003/per_clip.tsv`；系統平均：`/home/kojiek/MeanAudio/runtime/musiceval_full_20261003/per_system.tsv`；完整JSON：`/home/kojiek/MeanAudio/docs/experiments/results/musiceval_full_20261003.json`。
