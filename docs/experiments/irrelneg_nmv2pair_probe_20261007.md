# 107：無關文字負向 × nmv2pair control（3 seed 全量）

2026-10-07 登錄。只做推論，不重訓。

## 問題

084 Stage C 把 NegMF N100 的增益（E1 +0.948 lvl30 PQ）拆成兩部分：

- 約 38%：極性相反的文字（rev100，`high quality recording, clean, professional, pristine, hi-fi`）也拿得到。
- 約 62%：只有缺陷描述文字拿得到。

文件原本把 38% 寫成「領域詞彙部分」。但 086 在 s14159265 control 上量到，無關文字（`a photograph of a cat, a spreadsheet, printed text`）放在推論期負向就有 +0.519。這個數字比同一個 checkpoint 上 reversed 文字的 G_rev（+0.459）還高。

如果任意非空文字都拿得到這 38%，它就是「非空文字層」，不是「fidelity 領域詞彙層」。

086 只有 1 個 seed，而且是在 subset1024 上量的；reversed 用的是全量。兩者不能直接相減，所以要補這一格。

## 設計

- **Checkpoint**：nmv2pair control quarter S2 50k ema_final，3 seed（14159265／16180339／27182818）。這和 Stage C 讀 G_rev 用的是同一組 checkpoint。
- **新格**：每個 seed 跑一格 CFG3，負向文字是上述無關文字（tag `irrel`）。
- **協定**：與 revneg 格逐項相同。
  - 生成：`mc_mf25_negvariant_eval.sh`，MusicCaps 5521、MeanFlow 25 步、seed 42、fp32、NoMask、`--no_q`、CLAP batch 1。
  - 評分：FAD（2048，`musiccaps_reference`），以及 −30 LUFS 重評分。
  - 音檔：評完就刪。
- **沿用的既有格**（同一批 checkpoint 全量，不重跑）：
  - cfg0 與 cfg3_neg（fidelity8）
  - cfg3_revneg（084 Stage C）
  - cfg3_lqneg（081，`Low quality recording.`）
- **增益**：G_k = 格 k − cfg0，逐 clip 計算，k ∈ {neg, revneg, lqneg, irrel}。

## 主要比較與判讀（預登錄）

主要比較：D = G_rev − G_irr，PQ lvl30，3 seed 合併。CI 用 10,000 次 clip bootstrap，seed 20261007。門檻 0.10。

| 結果 | 判讀 | 文件改法 |
|---|---|---|
| D ≥ +0.10 且 CI 下界 > 0 | reversed 文字有超出任意文字的效果 | 38% 有一部分是領域詞彙，保留「詞彙」說法並寫明大小 |
| D ≤ −0.10 且 CI 上界 < 0 | reversed 文字低於任意文字 | 38% 是非空文字層；reversed 的高品質描述反而扣分 |
| 其他 | 兩者相當 | 38% 是非空文字層 |

另外報告以下讀數，不作判讀依據：

- 各 seed 的 D 及其正負號
- 排除靜音 clip 後的 D
- R_irr = G_irr / G_neg
- CE／CU／PC／CLAP 增益
- FAD、LUFS、crest、靜音數

### 預測

- **D**：落在「相當」或「reversed 較低」，D ≤ 0。依據是 086 s14 子集的 irrel +0.519，高於 G_rev s14 全量的 +0.459。
- **G_irr**：3 seed 合併約 +0.4～+0.6；R_irr 約 0.5～0.7。
- **未訓練的推論**：既然 R_train ≈ R_inf（0.38 vs 0.31），訓練期的 irrel arm 應該落在 rev100 附近或更高。這一點本 probe 只是預測，不會驗證。

## 限制

- 只量推論期，沒有訓練期 irrel arm。對 NegMF 的推論靠 R_train ≈ R_inf 的經驗對應。
- 只有一種無關文字，086 的同一句。換別的無關文字可能不同。
- 用 lvl30 對齊響度。靜音逃逸會拉低 PQ（084 已確認），所以另報排除靜音後的結果。

## 為什麼走 queue

3 格全量生成，加上評分、FAD 和 lvl30，預估 0.8～1.2 GPU 小時，超過 30 分鐘，依規走 p2 queue（`107_irrelneg_nmv2pair_probe.sh`）。

## 檔案

- Action：`scripts/eval/irrelneg_nmv2pair_probe_20261007.py`（`--preflight`／`--validate-only`）
- Guest：`scripts/experiment_harness/irrelneg_nmv2pair_probe_20261007_guest.py`
- Contract：`docs/experiments/irrelneg_nmv2pair_probe_20261007_contract.json`
- Summary：`~/nvme_experiment_artifacts/meanaudio/irrelneg_nmv2pair_probe_20261007/summary.json`
- 格：`~/eval_output_nvme/<ctrl>_mc_mf25_cfg3_irrel{,_lvl30,_fad}/`
