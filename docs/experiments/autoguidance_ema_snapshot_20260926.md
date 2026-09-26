# 080 Autoguidance：用同一個 run 較早的 EMA 快照當負向分支（2026-09-26）

## 問題

075 的後續 probe 排除了兩種解釋 fidelity8 負向 prompt 為何有效的說法：
- **推力大小**：lab 的 ‖A−B‖ 反而最大，增益卻最小（分支幾何 probe）。
- **推離程式化缺陷方向**：A−B 與五類缺陷方向的 cos 都接近 0（缺陷方向投影）。

剩下的候選說法是「負向分支提供一個**較差的同類預測**，CFG 把差距外插出去」。
Autoguidance（Karras et al., arXiv 2406.02507）正好把這件事做成方法：負向分支不用文字，
而是換成**同一模型較弱的版本**、吃**同一條文字**：

```
pure : u = A + (w−1)(A − A_bad)
combo: u = [3A − 2B_fidelity8] + (w−1)(A − A_bad)     （= stock CFG3+neg 再加 AG 項）
```

A = 最終 checkpoint（ema_final）的條件預測；A_bad = 同一個 run 較早 EMA 快照的條件預測，
用它**自己的** text projection 處理同一份原始文字特徵。

**要回答的**：不帶任何文字的模型式負向分支，能否在 PQ 上達到文字負向 prompt 的增益？兩者能否疊加？

## 設定

- **Good model**：`slot0clean_nmv2pair` quarter（075 的 control），3 個訓練 seed（14159265 / 27182818 / 16180339），`stage2_50000` 的 ema_final。
- **Bad 階梯**（EMA σ_rel 0.05 = default_output_sigma 的快照）：

| 名稱 | 快照 | 網路 | 地位 |
|---|---|---|---|
| s2_110k | `stage2_50000/ema_ckpts/0.110000.pt` | MeanAudio（同目標） | primary |
| s2_130k | `stage2_50000/ema_ckpts/0.130000.pt` | MeanAudio | primary |
| s1_100k | `stage1_100000/ema_ckpts/0.100000.pt` | FluxAudio（瞬時速度、無 r） | exploratory |
| s1_30k | `stage1_100000/ema_ckpts/0.30000.pt` | FluxAudio | exploratory |

  S1 快照預測的是瞬時速度不是平均速度，A−A_bad 同時帶著「目標不同」的差，所以只作探索，結果不可寫成 autoguidance 本身的效果。
- **w** ∈ {1.5, 2.0, 3.0}；combo 固定 w=2。
- **生成**：標準 MusicCaps TSV 前 N 列、stock eval 旗標（MeanFlow 25 步、seed 42、fp32、NoMask、`--no_q`）。eval.py 整個 run 只 seed 一次 RNG，前 N 列的 noise 與全量 stock run 相同（d2_075_segment_cfg 的 gate 已逐樣本驗證）→ 每個 clip 都能跟同 checkpoint 的 stock cfg0 / cfg3_neg 配對，不必在 run 內重生參考。
- **實作**：monkeypatch `MeanAudio.ode_wrapper` 與 `preprocess_conditions`（附掛原始文字特徵），eval.py 以 runpy 原樣執行；networks.py / eval.py 不改。`load_weights` 是 strict=False，所以 bad 模型改用顯式 key 集合比對（排除重算 buffer），缺或多一個 key 就失敗。

## 階段

1. **Replication gate**（256 列，s14159265）：把 ema_final 自己當 bad（A−A_bad ≡ 0），w=2。pure 必須逐樣本等於 stock cfg0、combo 必須逐樣本等於 stock cfg3_neg，容許不一致 0。8 列 smoke 已過（0/8、0/8）。
2. **Pilot**（1024 列，s14159265）：4 bad × 3 w 的 pure 共 12 格 + 2 格 combo（s2_110k、s1_100k）。
3. **Stage B（必跑）**：選出的 pure 與 combo 各在另外 2 個 seed 跑 1024 列。
4. **Stage C（有閘）**：選出的格在 3 個 seed 上 lvl30 dPQ 的 CI 下界都 > 0 且 CLAP 不低於 floor，才在 s14159265 跑 5521 全量。

## 評分與判讀

- 每格：`eval_metrics.py`（CLAP batch 1 + AES + level）＋ `level_match_rescore.py`（−30 LUFS 對齊後重評；對齊音檔評完即刪）。
- 與 stock cfg0、stock cfg3_neg 做逐 clip 配對差，未對齊與 lvl30 兩種，bootstrap 10000 次（seed 20260926）95% CI。同一批 clip 上的文字負向增益 G_neg（cfg3_neg − cfg0）一併報，作比較尺。
- 逐步幾何診斷：‖A−A_bad‖/‖A‖ 與 cos(A, A_bad)，分 t 三段。
- **選格規則**：pure 取「對 cfg0 的 lvl30 dPQ 最大、且 dCLAP ≥ −0.005」；combo 取「對 cfg3_neg 的 lvl30 dPQ 最大、且 dCLAP ≥ −0.005」。沒有格過 CLAP floor 時仍取 dPQ 最大者，讓 null 在多 seed 上被量到而不是被假設。
- **主端點**：選出的 pure 格對 stock cfg0 的 lvl30 dPQ，與同批 clip 的 G_neg（lvl30）比。
- **判讀**：
  - AG ≥ G_neg：在這個模型上，模型式負向分支可取代文字負向 prompt。
  - 0 < AG < G_neg：部分取代。
  - AG 的 CI 跨 0 或 ≤ 0：「在這個設定下沒幫助」，不可寫成「autoguidance 在音訊無效」。
  - combo 對 cfg3_neg 的 lvl30 dPQ CI 下界 > 0：兩者可疊加。
  - 未對齊與 lvl30 方向不一致 → 報成響度效應，不是品質效應（063／065 的形狀）。
  - 靜音數 > stock 的 2 倍要標記（slot0nm 靜音模式先例）。

## 預先登記的限制

- 單一語料、單一 checkpoint 家族（nmv2pair quarter）、單一生成 seed 42；pilot 只在一個訓練 seed 上選格。
- Karras 的原版建議 bad model 是**更小且訓練更短**的模型；這裡只用「訓練更短」這一個維度（同尺寸 EMA 快照），是最省事的變體。null 不能外推到原版。
- quarter run 的 S2 只有 50k iter，110k / 130k 快照與 final（150k）距離很近，差異可能太小（smoke 裡 ‖A−A_bad‖/‖A‖ ≈ 0.13–0.17，cos ≈ 0.99）。
- CLAP／AES 不是人耳判斷；正面結果要先過五首固定主觀 prompt 才能對外。
- MeanFlow 25 步屬少步數區間，結論不外推到多步 flow matching。

## 修正紀錄

- **2026-09-26 第一次上座失敗**：gate 已過（0/256），pilot 第一格 1023 首 ≠ 1024。MusicCaps TSV 有 5 條 caption 在引號內跨兩行（第一條在第 463 行），073 harness 的 `G.generate(limit=)` 按實體行切，1025 行只有 1023 筆。改由 action 的 `write_subset_tsv` 按 record 切（引號奇偶判斷）再交給 `G.generate`；已驗證 8／256／1024／5521 列逐筆等於全量 TSV 的前 N 筆，256 列與 gate 用的檔逐位元相同 → gate 結果仍有效。contract 只更新 action／preflight sha，設計與判讀不變。073 harness 綁在歷史 contract 上不改；該 bug 只影響 limit > 461 的呼叫。
- **2026-09-26 第二次上座失敗**：pilot 第三格（s2_110k、w=3.0）生成了 1 首全靜音 clip，`eval_metrics` 的 lufs 欄是空字串，`contrasts` 的 `float('')` 崩在 AES 都已評完之後。改成空字串讀為 NaN（該欄本來就用 `nanmean`）；AES／CLAP 欄不受影響。已用該格既有輸出離線重算驗證（lvl30 dPQ −0.32、靜音 19 vs stock 10）。前兩格 JSON 保留，第三格依續跑規則整格重生。設計與判讀不變。

## 資源

- GPU：約 3–4.5 小時（gate 2×256 列、pilot 14×1024、Stage B 4×1024，Stage C 最多 2×5521）。
- 儲存：約 6 GB（對齊音檔評完即刪）。開跑時 NVMe 只剩約 31 GB，preflight 要求 ≥ 20 GB、執行中 < 10 GB 硬停。
- 走 p2 queue（`080_autoguidance_ema_snapshot.sh`），因為預期 > 30 分鐘。

## 檔案

- Action：`scripts/eval/autoguidance_ema_snapshot_20260926.py`（`--preflight`／`--validate-only`）
- Guest：`scripts/experiment_harness/autoguidance_ema_snapshot_20260926_guest.py`（從 073 複製，只改 progress）
- Contract：`docs/experiments/autoguidance_ema_snapshot_20260926_contract.json`
- 輸出：`~/nvme_experiment_artifacts/meanaudio/autoguidance_ema_snapshot_20260926/{cells/,summary.json}`
- 結果文件：`docs/experiments/results/autoguidance_ema_snapshot_20260926_results.md`（收線時寫）
