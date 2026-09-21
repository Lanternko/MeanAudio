# 072 預註冊：淺層向下 limiting 階梯（−18 dB 到原音量，MusicCaps 5521, MF25, CFG3, fidelity8）

2026-09-22 設計，補 064 與 065 之間的空白。結果寫在
`docs/experiments/results/shallow_limit_ladder_aes_20260922_results.md`。
腳本 `scripts/eval/shallow_limit_ladder_aes_20260922.py`；產物
`/home/kojiek/nvme_experiment_artifacts/meanaudio/shallow_limit_ladder_aes_20260922/`。

## 問題

到目前為止量過 limiter 的只有兩條線，中間空著：

| 實驗 | 方向 | 範圍 |
|---|---|---|
| 064 | 往上推 LUFS | 原音量 → +6.8 LU |
| 065 | 往下壓 LUFS | −18 dB → −84 dB |

**−18 dB 到原音量之間沒有任何 limiter 資料**，而那正是實際輸出所在的區間（生成音檔
−14～−30 LUFS）。畫成曲線時 limiter 那條線在原音量附近是斷的，接不回純量曲線。

另外 065 的 `analyze()` 其實對四個 AES 軸＋CLAP 都算了 `limiter_minus_scalar`，
但結果文件只報了 PQ 與 CLAP；CU / CE / PC 的向下 limiting 從未被讀出來。

使用者（2026-09-22）：四個軸一起補，接成連續曲線。

## 設計

065 的機制原封不動，只換兩個清單：

| CFG | 值 | 意義 |
|---|---|---|
| `ladder_db` | `[0, 3, 6, 9, 12, 15, 18]` | 純量階 `z0`、`m3`…`m18`，每個 D arm 的等響度對照 |
| `limit_db` | `[3, 6, 9, 12, 15, 18]` | 向下 limited arm `D3`…`D18`，L\* 與同名 m arm 對齊 |

沿用的部分：x42-dpl（true peak、ceiling −1 dBTP、release 50 ms）、pre-gain 二分搜尋
（`tolerance_lu` 0.05）、F/Q 雙模式、`D<g>m` 處理雙胞胎（同一條 limited 波形調回原片段 L\*）、
051 hash 驗證過的 baseline FLAC、AES batch 16、CLAP 逐檔、bootstrap seed 20260918 / 10,000 次。

每片段 33 個評分條件（7 純量階 × 2 模式 ＋ 6 D arm × 2 模式 ＋ 6 個 `D<g>m` ＋ 靜音），
**5,521 × 33 = 182,193 筆**。音訊 transient：渲染 → 記 content sha256 → 評分 → 刪。

`D18` 是刻意重測的：065 已經有它，所以本次每一片段的 D18 分數都是一個**逐片段**的
複製閘門，比對平均值嚴格得多。

## 端點

**Primary — 處理代價如何從零長出來。**
`processing_at_source_level` = `D<g>m − z0`（F 模式、PQ），g ∈ {3, 6, 9, 12, 15, 18}。
兩邊都在原片段響度，差的只有 limiter 壓過沒壓過，所以這是乾淨的處理代價，
且在 g → 0 時必須趨近 0。065 在 g ≥ 36 量到它飽和在約 −0.77 PQ；本實驗問的是
**它從 0 爬到 −0.77 的形狀**，以及在多小的 GR 下就已經可測。

**Secondary — 讓公開曲線接得起來。**
`limiter_minus_scalar_F` 與 `_Q`，四個 AES 軸＋CLAP，g ∈ {3…18}。這是等 L\* 下
limiter 對純量的差，也就是圖上粉線與藍線的垂直距離。

> ⚠️ 預先寫明的混淆：等 L\* 的 `D−m` 是**兩個都已劣化的 arm 的差**。在深處
> （065 的 g ≥ 60）純量 arm 自己也逼近底線，差值因此縮小，所以 `D−m` 隨 g 的形狀
> **不是**處理代價的形狀。處理代價只讀 primary 的 `D<g>m − z0`。

**Covariates**：每個 arm 的 L\*、LUFS、peak、crest、zero_fraction，以及每個 D arm 的
`gr_max_db`（平均最大增益衰減）與 `effective_ceiling_dbfs`。

## 閘門

| 閘門 | 判準 | 來源 |
|---|---|---|
| 063 複製（`z0.F`、`m18.F`、`z0.Q`） | 逐片段 max \|diff\| ≤ 5e-3 | 065 既有（容忍度放寬，見下） |
| 純量 arm 的 L\* 共變 | max 誤差 ≤ 0.01 LU | 065 既有 |
| limiter 目標命中率 | ≥ 99%（±0.05 LU） | 065 既有 |
| **065 D18 交叉複製（音訊層）** | `D18.F/Q`、`m18.F/Q`、`D18m.F` 的 `content_sha256` 逐片段**完全相同**，容忍 0 筆 | 本實驗新增 |
| **065 D18 交叉複製（分數層）** | 同五個條件 × 五個指標，逐片段 max \|diff\| ≤ 5e-3（AES）／1e-3（CLAP） | 本實驗新增 |

任一閘門未過 → 新的階不可讀，先當本次執行的 bug 處理。

### 為什麼分數層不能用 1e-4（2026-09-22 smoke test 發現）

065 的 `score_all` 把 **AES 一個條件對「當前這批還沒完成的 clip」一次呼叫**（batch 16），
所以 batch 的組成取決於還剩多少 clip 要跑，而 padding 會隨組成改變分數。

6 片段的 smoke test 對 065 實測：**所有 `content_sha256` 完全相同**（plan 的 pre-gain 二分
搜尋與渲染都是決定性的），但 46 個條件的 AES 分數差到 9.3e-4；CLAP 差 0（CLAP 已經是逐檔，
見 `reference_clap_batch_size_sensitivity`）。

含意：
- **全量從頭跑**會重現 065/063 的分組，閘門會是 0.0（065 當初對 063 就是 0.0）。
- **被搶佔後續跑**時 `todo` 變短、batch 邊界位移，分數會漂約 1e-3，1e-4 的閘門會**假性失敗**。
  這與 `reference_eval_py_topup_rng_nondeterministic` 是同型的坑（那邊是 RNG，這邊是 batch 組成）。

所以改成兩層：**音訊逐位元零容忍**（這才是真正能抓到「arm 做錯、limiter 用錯」的閘門），
**分數給 5e-3 容忍**（比本實驗要讀的任何效果小兩個數量級；`D18 − m18` 是 −0.571 PQ）。
`replication_tolerance` 同步放寬到 5e-3，理由相同。

> 這也是一個可外用的觀察：**AES 分數對 batch 組成敏感**（~1e-3），與 CLAP 對 batch size 敏感
> （+0.004～+0.025）同類但小兩個數量級。跨 run 比較 AES 絕對值時，batch 組成要一致。

## 不適用、不可讀的輸出

腳本沿用 065 的 `analyze()`，所以 `summary.json` 仍會寫出兩個在這個範圍沒有意義的區塊，
**預先登記為不可讀**：

- `floors`：評分器底線在 −96 dB 以下、格式底線在 −42 dB 以下，階梯根本沒走到。
- `shape_float`：階梯截在 −18 dB，argmax 必然落在邊緣階。頂點位置由 063/065 定案
  （PQ/CU −21 dB、CE/PC −6 dB、CLAP 0 dB），本實驗不重問。

## 判定規則（事前寫定）

| 觀察 | 寫法 |
|---|---|
| `D3` 的處理代價 CI 含 0，且代價隨 g 單調變深 | 處理代價由 GR 驅動、從零連續長出；065 的 −0.77 是這條曲線的飽和端 |
| `D3` 的處理代價已明顯非零（\|mean\| > 0.02 PQ 且 CI 不跨零） | 即使很輕的 limiting 也要付 PQ；報出最小可測 GR |
| 代價非單調 | 未預期，只能寫成 exploratory，並回報 GR 與代價的散布 |
| 四個軸方向不一致 | 照 064／065 已知型態報（PC 偏好 limiting），不做合併宣稱 |

CI 跨零是**證據不足，不是等效**。逐點區間，不做多重比較校正、不自動晉升。

## 限制

單一 checkpoint、單一 generation seed（沿用 051 baseline 音檔）、單一 limiter
（x42-dpl, release 50 ms）。064b 已證明**處理代價的大小依 limiter 實作差 12 倍**，
所以本實驗的絕對代價數字只對 dpl 成立；可外推的是方向與形狀。
AES 無響度正規化且在 16 kHz 截頻，絕對值不可跨論文比較。AES 與 CLAP 都不是人類判斷。
本實驗若中途被搶佔後續跑，AES 分數會帶約 1e-3 的 batch 組成漂移（見上），結果仍可讀，
但 `summary.json` 與一次跑完的版本不會逐位元相同。

## 佇列

`p2/pending/072_shallow_limit_ladder_aes.sh`，contract
`docs/experiments/shallow_limit_ladder_aes_20260922_contract.json`，
eval-only guest `scripts/experiment_harness/shallow_limit_ladder_aes_20260922_guest.py`
（063 的 guest 加上 flat `items/*.json` 的 progress glob 修正）。
P2、可搶佔：每片段的評分是原子的 per-clip JSON，重啟只補缺的片段。
