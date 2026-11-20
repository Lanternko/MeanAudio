# 180：Music Arena 拿掉 8 kHz 以上頻段後，MuQ 跟人類投票的一致率會不會變高（2026-11-05）

設計登錄：2026-11-05，**評分前登錄**。直接跑（不走 queue；預估 < 30 min，腳本掛 `notify_on_exit`）。

## 0. 為什麼做

- **已知的偏誤**：10-28 與 11-03 發現，寬頻真實音樂做 4–6 kHz 低通後，MuQ-Eval A1 MI 會上升約 0.4，人類 MUSHRA 卻從 98 掉到 25。換句話說，MuQ 會對 8–12 kHz 的內容扣分。
- **york135 的假說**（2026-11-04 Discord）：寬頻的大模型（SUNO 一類）可能因此被 MuQ 冤枉。拿掉 8 kHz 以上的頻段後，分數應該會升回去，算是幫它們翻案。
- **反方觀察**（`scratchpad/bw.py`，Arena 15 系統 × 12 段）：在系統之間，越寬頻的系統 MI 越高。lyria／riffusion 是 44.1k、MI 4.4–4.8；musicgen 是 32k、MI 3.2–3.4。所以 MuQ 並沒有在系統層級懲罰寬頻。不過頻寬跟品質是混在一起的，系統之間的比較不能證明段內的偏誤不存在。
- 這次要回答的問題：用人類投票當地面真值。如果 8 kHz 以上的偏誤真的讓 MuQ 誤判 Arena 對戰，把每一段都限制在 8 kHz 以下再評分，一致率應該上升。

## 1. 資料

- **Music Arena**：r2 的 `arena` manifest，3,998 段、2,000 場 battle，其中 1,412 場是決定票（A／B），15 個系統。
- **輸入**：用 r2 `native_raw` 格已解碼的 wav（`derived/arena/index.tsv` 的 `files`），也就是 r2 主格當時評分的同一份音檔。
- **人類標籤**：`analyze_r2.arena_pairs()`。主判準只用決定票；TIE 與 BOTH_BAD 只做描述。

## 2. 條件（每段 3 種 × MuQ-Eval A1、A3a）

| 條件 | 做法 | 用途 |
|---|---|---|
| `F0` | 原樣（MuQ 內部重採樣到 24 kHz） | 基準；應與 r2 已存的分數一致 |
| `sr16k` | soxr HQ 重採樣到 16 kHz | 等同 11-03 的 F1；Nyquist 8 kHz |
| `lp8k` | 原取樣率做 Butterworth 8 階 8 kHz `sosfiltfilt` | 跟 `sr16k` 對照，拆出重採樣本身的影響 |

- 不另做 12 kHz：MuQ 輸入是 24 kHz，12 kHz 以上本來就看不到。
- 4–8 kHz 的偏誤這次不碰。問題只問 8 kHz 以上，也就是寬頻系統特有的那一段。

## 3. 指標

- **主讀數**：A1 MI 的 **forced 一致率**。逐場比較兩段分數，看高分的一方是否就是人類選的一方；分數完全相同算 0.5。
- **次讀數**：
  - r2 的 `agree_m`（m = 0.10 S，S 在各條件內各自算）。
  - A1 TA、A3a MI／TA 的一致率。
  - 系統排名對人類 BT 的 Kendall τ。
  - 各系統平均 ΔMI，以及它與該系統 8 kHz 以上能量占比的 Spearman。
- **寬頻落差子集（描述）**：兩方 8 kHz 以上能量占比相差 ≥ 6 dB 的對戰。這組最有機會看出翻案。能量用每段前 30 s 的 STFT 估計。

## 4. 預登錄判準

主判準：`lp8k` 與 `F0` 的 A1 MI forced 一致率差 Δ。以 battle 為單位做成對 bootstrap（2,000 次，seed 20261105），取 95% CI。

- **翻案成立**：Δ ≥ +0.02，且 CI 下界 > 0。
- **反向**（8 kHz 以上帶有有效的品質資訊）：CI 上界 < 0。
- **其他情況**：判「沒有可測影響」。

**附帶條件**：
- `sr16k` 必須同號。如果 `lp8k` 與 `sr16k` 異號，結論改寫成「依處理方式而定」。
- A3a MI 分開報，不進主判準。

**預期**（評分前寫下）：
- 我預期判「沒有可測影響」或反向，理由是系統之間寬頻與高分正相關。
- 各系統 ΔMI 在寬頻系統上會比較大（44.1k 系統 +0.1～+0.3，musicgen ≈ 0）。但寬頻系統原本就多半勝出，再加分也很少翻轉對戰結果。

**不可寫**：
- Arena 每場只有 1 票，人類 BT 天花板是 0.742。
- 這次只測 8 kHz 以上；不代表 4–8 kHz 的偏誤不存在。
- 結果只適用於 Arena 這 15 個系統。

## 5. 執行與輸出

- **腳本**：`scripts/eval/bench_r2/muq_arena_bandlimit_180.py`（評分＋分析）。
- **輸出**：`~/nvme_experiment_artifacts/meanaudio/muq_arena_bandlimit_180/`（`scores.tsv`、`band.tsv`、`summary.json`）。
- **結果檔**：`results/muq_arena_bandlimit_180_results.md`、`results/muq_arena_bandlimit_180_summary.json`。
- **預估時間**：3,998 段 × 3 條件 × 2 模型，GPU 約 20–30 分鐘。
