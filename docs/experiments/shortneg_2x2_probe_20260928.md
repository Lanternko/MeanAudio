# 086 短負向 prompt 2×2 probe（checkpoint × 措辭）（2026-09-28）

## 問題

081 的 control（slot0clean_nmv2pair quarter s14159265）在 CFG3 下用負向 prompt `Low quality recording.` 時，PQ 幾乎不動（lvl30 −0.007，raw +0.018，全量 5521）。
09-03 消融的結果不同。在 c2p0_slot0（`phase8_qwen_caption10s_multisent_noq_full_stage2_200000`）、subset1024、CFG3 下：

- `low quality, noisy` 拿到 +0.971 PQ（raw，同子集 cfg0 6.582 → 7.553），約是 fidelity8 +1.067 的 91%。
- 無關文字 +0.357。
- `music` +0.250。
- 不給負向文字（stored null）≈ 0。

兩組結果之間同時有兩個變數不同：checkpoint 和措辭。子集和 lvl30 也不一樣。
不拆開的話，081 的 E1 差中差仍然成立，因為 control 是配對基準。但「未訓練的模型對短標籤沒反應」不能寫成一般結論。

## 設計

| cell | A = nmv2pair quarter s14159265 | B = c2p0_slot0 full 200k | 負向（CFG 3，cfg0 除外） |
|---|---|---|---|
| cfg0 | ✓ | ✓ | CFG 0，只跑 conditional 分支 |
| none | ✓ | | stored null（不傳 `--negative_prompt`） |
| lqrec | ✓ | ✓ | `Low quality recording.`（081 措辭） |
| lqnoisy | ✓ | ✓ | `low quality, noisy`（09-03 fidelity_short） |
| lq | ✓ | ✓ | `low quality`（拆出 `noisy` 的作用） |
| irrel | ✓ | | `a photograph of a cat, a spreadsheet, printed text` |
| fid8 | ✓ | | fidelity8 |

生成協定：
- 資料：MusicCaps subset1024。沿用 09-03 的同一個檔案 `negprompt_ablation/musiccaps_subset1024.tsv`（seed 20260830），sha 已釘。
- 取樣：MeanFlow 25 步、seed 42、fp32、NoMask、`--no_q`。旗標與 09-03 matrix 相同。

評分：
- `eval_metrics.py`（CLAP batch 1）＋ −30 LUFS 對齊重評（`level_match_rescore.py`）。
- 兩種讀數都寫進 cell JSON 之後，才刪除音檔。

## 讀法（事先寫定）

所有增益都是「同 checkpoint、同 clip、對自己 cfg0」的逐 clip 配對差，bootstrap 10000 次。

- **checkpoint 效應**：比較同一措辭在 B 與 A 上的增益差（`<key>__gain_B_minus_A`）。如果 lqrec 在 B 上也 ≫ 0，那麼 control 的 0 就是 A 這個 checkpoint 特有的。
- **措辭效應**：在 A 上比較 lqnoisy 和 lq 相對 lqrec 的增益。如果 A 上 lqnoisy ≫ lqrec ≈ 0，差別就在措辭（`noisy`，或大小寫、句號、`recording`）。
- **交互**：`<key>_vs_lqrec__B_minus_A`。
- **「任何文字」層**：在 A 上看 none、irrel、fid8。如果 A 的 irrel 也 ≈ 0，而 B 在 09-03 有 +0.357，就表示 A 整體對非 fidelity 文字比較鈍。
- raw 與 lvl30 反號時，照 063/065 的規則報成響度效應。

## 09-03 錨點（只記錄，不阻擋）

B cfg0 與 B lqnoisy 的 raw PQ 平均要落在 6.5822／7.5534 的 ±0.01 內，同時記錄逐 clip 相關係數。09-03 的 AES 用 batch 32，與逐檔評分約有 1e-3 的漂移。
不過的話，只代表舊數字無法用今天的程式碼重現。probe 內部的交叉比較仍然有效，因為 11 格都在同一條件下生成和評分。

## 為什麼走 queue、為什麼插隊

- 預估約 60～70 分鐘（11 格 × 1024 列，參考 080 每格約 6.4 分）。超過 30 分鐘，所以走 queue，編號 `002_`。
- 使用者 2026-09-28 指定：在 084 跑完後、085 之前跑。
- 本 job 只做推論，不動任何訓練產物。

## 限制

- 單一生成 seed，只用一個訓練 seed（A = s14159265）。A 與 B 同時差了預算（quarter vs full 200k）和語料（slot0clean_nmv2 vs multisent），所以 checkpoint 效應無法再拆到這兩者。
- 子集 1024 列，不是全量 5521。
- 不跑 FAD。

輸出：`/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_2x2_probe_20260928/summary.json`
腳本：`scripts/eval/shortneg_2x2_probe_20260928.py`、`scripts/experiment_harness/shortneg_2x2_probe_20260928_guest.py`
