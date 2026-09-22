# 074 預註冊：微幅向下 limiting 階梯（原音量 −3 dB 以內，MusicCaps 5521, MF25, CFG3, fidelity8）

2026-09-22 設計，把 limiter 曲線從 −3 dB 接到原音量。結果寫在
`docs/experiments/results/micro_limit_ladder_aes_20260922_results.md`。
腳本 `scripts/eval/micro_limit_ladder_aes_20260922.py`；產物
`/home/kojiek/nvme_experiment_artifacts/meanaudio/micro_limit_ladder_aes_20260922/`。

## 問題

072 把 limiter 資料補到 −3 dB，但**那條線還是接不到原音量**，而且 072 預註冊要回答的
「處理代價如何從零長出來」只答了一半：

| 實驗 | L\* 範圍 | 最小 GR |
|---|---|---|
| 064 | 0 → +6.8 LU | — |
| 072 | −3 → −18 dB | **8.40 dB** |
| 065 | −18 → −84 dB | 43 dB |

limiter 只削峰，所以要讓 integrated L\* 只降 3 dB 就已經需要 8.40 dB 的 GR，
而那裡代價已經是 −0.2365 PQ（飽和值的 31%）。**GR 在 0～8.4 dB 之間完全沒有資料**，
畫圖時 z0 到 −3 dB 那段只能用虛線推（g → 0 時 D 必然回到 z0），形狀不可讀。

使用者（2026-09-22）：連起來。

## 設計

065／072 的機制原封不動，只換兩個清單：

| CFG | 值 | 意義 |
|---|---|---|
| `ladder_db` | `[0, 0.5, 1, 1.5, 2, 2.5, 3, 18]` | 純量階 `z0`、`m0.5`…`m3`，每個 D arm 的等響度對照；`m18` 只為了 065 既有的 063 複製閘門 |
| `limit_db` | `[0.5, 1, 1.5, 2, 2.5, 3]` | 向下 limited arm `D0.5`…`D3`，L\* 與同名 m arm 對齊 |

**階距訂在 L\* 軸上、不訂在 GR 軸上**，理由有二：(1) 圖上那個看得見的缺口就是
L\* 的 0～−3 dB，訂在 L\* 才真的把線接起來；(2) 等響度雙胞胎（`m<g>`）與
level／processing 的精確拆解都是靠 L\* 對齊，改成瞄準 GR 會拆不開。
GR 因此是被動取樣的 —— 24 片段 pilot 量到 GR 平均 **3.17 → 8.69 dB**，正好蓋住空白。

沿用：x42-dpl（true peak、ceiling −1 dBTP、release 50 ms）、pre-gain 二分搜尋
（`tolerance_lu` 0.05）、F/Q 雙模式、`D<g>m` 處理雙胞胎、051 hash 驗證過的 baseline FLAC、
AES batch 16、CLAP 逐檔、bootstrap seed 20260918 / 10,000 次。

每片段 35 個評分條件（8 純量階 × 2 模式 ＋ 6 D arm × 2 模式 ＋ 6 個 `D<g>m` ＋ 靜音），
**5,521 × 35 = 193,235 筆**。音訊 transient：渲染 → 記 content sha256 → 評分 → 刪。

### 可達性（事前檢查過）

pre-gain 0 時 limiter 的 ceiling 是 −1 dBTP，這批音檔的峰值在它之下，所以 G → 0 時
L\* 降幅 → 0，(0, 3] 的目標都被 [0, hi] 包住。若某片段原始峰值已超過 −1 dBTP，
最小的幾階就不可達 —— 這種情況**報成 miss 不做夾擠**，且命中率閘門**逐階**評估
（新增 `per_rung_hit`）。24 片段 pilot：六階命中率都是 1.0，最大 L\* 誤差 0.050 LU。

## 端點

**Primary — 處理代價在小 GR 的形狀。**
`processing_at_source_level` = `D<g>m − z0`（F 模式、PQ），g ∈ {0.5, 1, 1.5, 2, 2.5, 3}。
兩邊都在原片段響度，差的只有 limiter 壓過沒壓過。072 在 GR 8.40 dB 量到 −0.2365；
本實驗問的是 GR 3.2～8.7 dB 這段，也就是**代價是否真的從零連續長出**，以及
最小可測 GR 在哪。

**Secondary — 把公開曲線接起來。**
`limiter_minus_scalar_F` 與 `_Q`，四個 AES 軸＋CLAP，g ∈ {0.5…3}。

> ⚠️ 沿用 072 寫明的混淆：等 L\* 的 `D−m` 是兩個都已劣化的 arm 的差。
> 在本實驗這個淺區純量 arm 幾乎沒劣化（m0.5 只衰減 0.5 dB），混淆比 072／065 小，
> 但**處理代價仍然只讀 primary**。

**Covariates**：每個 arm 的 L\*、LUFS、peak、crest、zero_fraction；每個 D arm 的
`gr_max_db`、`effective_ceiling_dbfs`、逐階命中率。

## 閘門

| 閘門 | 判準 | 來源 |
|---|---|---|
| 063 複製（`z0.F`、`m18.F`、`z0.Q`） | 逐片段 max \|diff\| ≤ 5e-3 | 065 既有 |
| 純量 arm 的 L\* 共變 | max 誤差 ≤ 0.01 LU | 065 既有 |
| limiter 目標命中率（整體） | ≥ 99%（±0.05 LU） | 065 既有 |
| **逐階命中率** | 每一階分開報；某階 < 99% 則該階不可讀，其餘階仍可讀 | 本實驗新增 |
| **072 D3 交叉複製（音訊層）** | `D3.F/Q`、`m3.F/Q`、`D3m.F` 的 `content_sha256` 逐片段完全相同，容忍 0 筆 | 本實驗新增 |
| **072 D3 交叉複製（分數層）** | 同五個條件 × 五個指標，逐片段 max \|diff\| ≤ 5e-3（AES）／1e-3（CLAP） | 本實驗新增 |

分數層的容忍度為什麼不能是 1e-4，見 072 預註冊「為什麼分數層不能用 1e-4」與
memory `reference_aes_batch_composition_sensitivity`：AES 一個條件對「當前還沒完成的
clip」一次呼叫，batch 組成隨剩餘量位移，續跑會漂約 1e-3。音訊層零容忍才是真正抓
「arm 做錯、limiter 用錯」的閘門 —— pilot 已驗證 24 片段的 D3 plan 對 072 逐欄位相同。

## 不適用、不可讀的輸出

沿用 065 的 `analyze()`，以下兩個區塊**預先登記為不可讀**：

- `floors`：評分器底線 −96 dB 以下、格式底線 −42 dB 以下，本階梯只走到 −3。
- `shape_float`：階梯截在 −3 dB（外加一個為閘門而存在的 −18），argmax 必然落在邊緣階。
  頂點位置由 063/065 定案，本實驗不重問。

## 判定規則（事前寫定）

| 觀察 | 寫法 |
|---|---|
| `D0.5` 的代價 CI 含 0，且代價隨 GR 單調變深 | 處理代價由 GR 驅動、從零連續長出；可寫「GR ≤ X dB 測不到代價」並報 X |
| `D0.5` 的代價已明顯非零（\|mean\| > 0.02 PQ 且 CI 不跨零） | 連 GR 3 dB 都要付 PQ；免費區間（若存在）在 GR 3 dB 以下，本實驗仍未量到 |
| 代價對 GR 非單調 | 未預期，只能寫成 exploratory，並回報 GR 與代價的散布 |
| 四個軸方向不一致 | 照 064／065／072 已知型態報（PC 偏好 limiting），不做合併宣稱 |

CI 跨零是**證據不足，不是等效**。逐點區間，不做多重比較校正、不自動晉升。

## 限制

單一 checkpoint、單一 generation seed（沿用 051 baseline 音檔）、單一 limiter
（x42-dpl, release 50 ms）。064b 已證明處理代價大小依 limiter 實作差 12 倍，
絕對數字只對 dpl 成立。即使本實驗全部 CI 含 0，**也只能說「GR 3.2 dB 以下未量」**，
不能說「存在免費區間」—— 0 到 3.2 dB 的 GR 一樣造不出來（L\* 目標小於 0.5 dB 時
二分搜尋的 0.05 LU 容忍度會吃掉整個訊號）。
AES 無響度正規化且在 16 kHz 截頻，絕對值不可跨論文比較。AES 與 CLAP 都不是人類判斷。

## 佇列

`p2/pending/074_micro_limit_ladder_aes.sh`，contract
`docs/experiments/micro_limit_ladder_aes_20260922_contract.json`，
eval-only guest `scripts/experiment_harness/micro_limit_ladder_aes_20260922_guest.py`
（072 的 guest，只換 out root 與 document_kind）。
P2、可搶佔：每片段的評分是原子的 per-clip JSON，重啟只補缺的片段。
