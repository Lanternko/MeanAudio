# 正向 prompt 槽位消融：negprompt 的增益需要負向槽，不只是詞彙

日期：2026-09-04 ｜ arm：`c2p0_slot0`（`phase8_qwen_caption10s_multisent_noq_full_stage2_200000`）
腳本：`scripts/eval/posprompt_ablation_matrix.py`（import `negprompt_ablation_matrix`，共用同一份
1024-row MusicCaps subset seed 20260830、MeanFlow 25 steps、NoMask、seed 42、CLAP batch 32）

## 動機

2026-08-31 的消融定論是「增益來自 fidelity 領域詞彙**不是**缺陷極性」。但那 43 個 cell
**全部**只動 `--negative_prompt`；正向 prompt 在每一個 cell 裡都是 TSV caption 逐字照抄
（`eval.py` 當時也沒有修改正向 prompt 的參數）。所以「詞彙是活性成分」這個結論，
從來只在「文字作為 CFG 參考點」的槽位裡被測過。

本消融加入 `eval.py --prompt_suffix`（只改餵給 generator 的字串，TSV 不動，
CLAP 的 text side 仍是原始 caption），並讓 `hifi` 與既有 `NEGATIVES['reversed']`
**逐字元相同**，使 slot 成為唯一變數。

## 結果（配對於同 arm 同 subset，n=1024）

同一串 `high quality recording, clean, professional, pristine, hi-fi`：

| | CLAP | PQ | CE |
|---|---|---|---|
| **cfg 1.5，放負向槽** | **+0.0048** (t=+4.6) | **+0.3688** (t=+28.7) | +0.3498 (t=+19.9) |
| cfg 1.5，放正向槽 | −0.0030 (t=−3.0) | −0.0143 (t=−1.3) | −0.0736 (t=−4.3) |
| **cfg 3.0，放負向槽** | **+0.0053** (t=+3.2) | **+0.7253** (t=+34.7) | +0.6157 (t=+22.5) |
| cfg 3.0，放正向槽 | −0.0032 (t=−2.9) | −0.0122 (t=−1.0) | −0.1189 (t=−6.4) |

正向槽的內容控制（cfg 3.0）：

| | CLAP | PQ |
|---|---|---|
| `music` 放正向槽 vs 無 suffix | −0.0024 (t=−3.0) | −0.0228 (t=−3.0) |
| `hifi` vs `music`（皆正向槽） | −0.0008 (t=−0.6) | +0.0106 (t=+0.8) |
| `music` 放**負向**槽 vs 無 | −0.0007 (t=−0.5) | **+0.2528** (t=+17.5) |

疊加測試（cfg 3.0）：正向 `hifi` 加到目前最佳的負向 `fidelity` 之上，
CLAP −0.0009 (t=−1.1)、PQ −0.0245 (t=−2.5) — 無增益，方向微負。

## Observation 層

1. 槽位不對稱是完全的：同一串字在負向槽 PQ +0.37～+0.73（t≈+29～+35），
   在正向槽 PQ 落在 0 附近（t=−1.0～−1.3），CLAP 小幅為負。
2. 正向槽對**內容不敏感**：`hifi` vs `music` 全指標無差（CLAP t=−0.6、PQ t=+0.8）。
   兩者都略低於「不加 suffix」。
3. 負向槽對內容**敏感**：`music`（+0.25 PQ）< `hifi`（+0.73 PQ）< `fidelity`（PQ 7.6495）。
4. 正向 suffix 無法疊加到已最佳的負向配置上。

## 推論層（supports / weakens）

- 這 weakens「增益來自 fidelity 詞彙本身」的最寬版本讀法。詞彙從正向槽投遞時
  什麼也沒發生；同一批詞從負向槽投遞才有效。更貼合資料的說法是：
  **增益需要「有內容的文字位於負向槽」這個機制，詞彙選擇是在該機制內部調變強度。**
- 與既有「極性不被遵守」的發現一致並延伸：往「high quality」推**開**（負向槽）有效，
  往「high quality」**靠近**（正向槽）無效。兩個槽位都顯示語義方向不是作用軸。
- 正向 suffix 的小幅負向 CLAP 與 T5 特徵 mean-pooling 的稀釋預期方向一致，
  但本實驗未測 pooling 本身，不能宣稱機制。

## 不能這樣寫

- ❌「正向 prompt engineering 無效」— 只測了「逗號後接短 suffix」一種形式，
  未測 prefix、未測改寫整段 caption、未測 caption 長度量級的正向文字。
- ❌ 跨 arm 推廣 — 只跑了 `c2p0_slot0`，未跑 `fulltrack`。
- ❌ 把 PQ 增益讀成音質增益而不看 loudness — 負向槽各 cell 的 rms/crest 已存於 JSON，
  仍受 2026-09-03 loudness confound 分析的既有限制約束。

## 產物

`/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation/` 下 4 個新 cell：
`c2p0_slot0__cfg{3.0,1.5}__none__POShifi`、`__cfg3.0__none__POSneutral`、
`__cfg3.0__fidelity__POShifi`。既有 43 個 cell 未被修改。
