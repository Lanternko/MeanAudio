# 指標 × 介入覆蓋補完結果（161–177，2026-11-03 重判）

> 預登錄：`docs/experiments/metric_coverage_completion_20261027.md`。資料來源：`/home/kojiek/eval_output_nvme/metric_coverage_20261027/coverage_summary.json`（2026-11-03 以修正後的 judge 重判；m = 0.10 S、通過 80%）。舊版（有 bug）備份在同目錄 `_prefix_judge_bug_20261103/`。門檻與判定規則沒改。

## 結論

- 22 指標 × 6 協定 = 132 格：130 格有判定，**沒有任何一格是 pass**。判定分布：{'fail': 60, 'ood_only': 46, 'partial': 24, 'n/a': 2}。
- `ood_only` 是預先登錄的 OOD 標記（SingMOS 在器樂、SongEval 在 10 秒器樂、floor 是合成底線），不能當模型缺點，也不能當通過。
- 第一版矩陣有 judge 讀檔 bug：`_single_path` 只讀第一個 shard，PAM／A1／A3a／CMI-RM／SingMOS 的 126b、126c、134 都只用了部分單元，rec 欄是 0 單元；`existing_126b` 沒帶 MIR 四欄。已修，結論方向沒翻（0 pass 不變），舊版 11 n/a／4 inconclusive 變成 6 n/a／0 inconclusive。
- 剩下 2 格 `n/a`：MERT／MuQ-embed × floor，預登錄本來就是 N/A。MIR 四項 × rec 已於 2026-11-03 用 `mir_rec_backfill.py` 補跑（32,807 clip、4,316 s、0 err），四格皆 fail（tie 通過率 0.27–0.74、lose 通過率 0.00–0.37）。

## 判定矩陣（126a | 126b | rec | floor | 126c | 134）

| # | 指標 | 126a | 126b | rec | floor | 126c | 134 |
|---:|---|---|---|---|---|---|---|
| 1 | AES PQ | fail | fail | fail | ood_only | fail | fail |
| 2 | AES CE | fail | partial | fail | ood_only | fail | fail |
| 3 | AES CU | fail | partial | fail | ood_only | fail | fail |
| 4 | AES PC | partial | partial | partial | ood_only | partial | partial |
| 5 | PAM | fail | ood_only | fail | ood_only | ood_only | partial |
| 6 | MEva | fail | fail | fail | ood_only | fail | fail |
| 7 | MuQ-Eval A1 | fail | ood_only | fail | ood_only | fail | fail |
| 8 | MuQ-Eval A3a | fail | ood_only | fail | ood_only | fail | fail |
| 9 | CMI-RM | fail | ood_only | fail | ood_only | partial | fail |
| 10 | SongEval coh | ood_only | ood_only | fail | ood_only | fail | ood_only |
| 11 | SongEval mem | ood_only | ood_only | fail | ood_only | fail | ood_only |
| 12 | SongEval nat | ood_only | ood_only | fail | ood_only | fail | ood_only |
| 13 | SongEval cla | ood_only | ood_only | fail | ood_only | fail | ood_only |
| 14 | SongEval mus | ood_only | ood_only | fail | ood_only | fail | ood_only |
| 15 | SingMOS | ood_only | ood_only | fail | ood_only | ood_only | ood_only |
| 16 | MERT-ref | partial | ood_only | partial | n/a | partial | partial |
| 17 | MuQ-embed-ref | fail | ood_only | partial | n/a | partial | partial |
| 18 | CLAP | fail | fail | fail | ood_only | fail | fail |
| 19 | MIR pulse_clarity | fail | partial | fail | ood_only | partial | fail |
| 20 | MIR ibi_cv | fail | partial | fail | ood_only | fail | fail |
| 21 | MIR key_cnn_conf | fail | partial | fail | ood_only | partial | fail |
| 22 | MIR chroma_entropy | partial | partial | fail | ood_only | partial | fail |

## 已知問題

- job 177 的 `coverage_matrix.md`／`.html` 渲染有誤：只畫 4 欄、總格數寫 88。請讀 summary 或本表。
- 這次修的是 `judge_coverage.py`（讀檔 bug）；它被 161–177 的 contract sha 綁住，已完成 job 的 contract 我沒有改，所以 sha 與 contract 不再相符，需要的話另做重新綁定。
- 各 job 的逐 clip 輸出沒逐一抽驗。
