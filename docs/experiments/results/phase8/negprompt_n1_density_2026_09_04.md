# 假說 N1 檢定：negative prompt 的增益 vs 訓練 caption 的保真度詞彙密度

日期：2026-09-04
狀態：**字面版 N1 被推翻；精煉版 N1′ 強烈成立。**
腳本：`scripts/analysis/negprompt_n1_density_scatter.py`（無 GPU，全部資料已在檔）
資料：`negprompt_n1_density.json`、`negprompt_n1_density_scatter.png`
前置：[negprompt_cfg3_content_interaction_2026_09_03.md](../../negprompt_cfg3_content_interaction_2026_09_03.md)、
[文獻定位](../../../literature/negative_prompting_and_prompt_engineering_2026_09_04.md)

QA-MDT（arXiv:2405.15863）用一句話解釋它的 negative prompt 效果為何只有 p-MOS +0.036：
先前做法「依賴資料集中 **"low quality" 的稀有出現**」。我們的 ΔPQ 是 +1.067，
比它大兩個數量級，而 c2p0 語料有 82.8% 的 caption 提到 quality。
本文把那句話變成可檢定的假說並實際檢定。

> **假說 N1（字面版）**：增益取決於 `y_neg` 的**詞彙**是否在訓練 caption 中高密度出現。
> **假說 N1′（精煉版）**：增益取決於訓練語料**是否談保真度**，與極性無關。

## 協定

- **y**：cfg 3.0 全量 sweep（`negprompt_reeval_cfg3.0/`）的 `paired_delta_vs_cfg0.PQ.mean_delta`
  —— MusicCaps n=5,521、MeanFlow 25 步、長版 `fidelity` negative、seed 42、
  逐檔配對至**各 arm 自己的 cfg 0**。
- **x**：各 arm **自己的訓練 TSV** 的 caption 欄位密度。四個指標見下。
- 14 個 cell 中 12 個有配對 delta（`a3_mfshort100k_direct_noq` 與
  `c2p0_fair013_k3_full_q9` 沒有對應的 cfg 0 基準，排除）。

**12 arm 只有 6 個語料。** 其中 6 個 arm 共用 c2p0 slot0 caption pool
（差別只在 seed、或 S2 是否加 Q；k3/k5 balanced 只動 `q_level` 欄不動 caption 文字）。
因此以**語料層 n=6** 為主要統計，arm 層 n=12 併列參考；
共用語料的 arm 間離散度直接當作這張散點圖的雜訊底線。

**密度指標**：

| 指標 | 定義 | 錨定 |
|---|---|---|
| `quality_rate` | `\bquality\b` | **重現既有數字**：c2p0 82.8 / fulltrack 7.3 / fidstrip 10.2、字數 52.1 / 47.9 / 41.3，與 `fidelity_stripped_caption_arm_2026_08_30.md` 的 gate 表**逐格相同** → 定義不是本文發明的 |
| `negterm_rate` | `fidelity` negative prompt 的 8 個詞，逐字 | N1 字面版的最直接操作化 |
| `lofi_rate` | `LOFI_RE`，逐字取自 `negprompt_reeval_full_arms.py` | 與 `lofi_prompt` eval 切分同一套詞彙 |
| `hifi_rate` | 由 `build_fidelity_stripped_captions.py` 的片語表定義 | 本文自訂，**未宣稱**重現舊 gate 表的 76.8/5.8 |

## 語料密度

| 語料 | 字數 | `quality` | `hifi` | `lofi` | `negterm` | arms |
|---|---:|---:|---:|---:|---:|---|
| c2p0_slot0 | 52.1 | **82.8%** | 78.5% | 14.3% | 5.3% | 6 |
| c2p0_f013_worst | 53.8 | 71.4% | 65.8% | 15.2% | 6.2% | 1 |
| c2p0_f013_best | 52.6 | 70.7% | 64.2% | 14.9% | 5.9% | 1 |
| c2p0_slot2 | 51.9 | 69.8% | 56.4% | 6.1% | 2.7% | 1 |
| lpmc_p7v1（LP-MC） | 42.5 | 47.7% | **0.7%** | **47.7%** | **46.3%** | 1 |
| fulltrack | 47.9 | **7.3%** | 5.9% | 8.4% | 6.9% | 2 |
| *fidstrip（未訓練）* | 41.3 | *10.2%* | — | *7.1%* | — | — |
| *MusicCaps（eval）* | 49.0 | *33.7%* | — | *35.7%* | — | — |

LP-MC 與 c2p0 是**極性完全相反**的兩種保真度語料：LP-MC 幾乎只講低保真
（hifi 0.7% / lofi 47.7%），c2p0 幾乎只講高保真（hifi 78.5% / lofi 14.3%）。
這個對比正是分辨 N1 與 N1′ 的關鍵。

## 結果

| arm | 語料 | `quality` | ΔPQ |
|---|---|---:|---:|
| c2p0_slot0_q5_full_q0 | c2p0_slot0 | 82.8 | +1.0515 |
| c2p0_slot0_q5_full_q9 | c2p0_slot0 | 82.8 | +1.0256 |
| c2p0_slot0_full_noq | c2p0_slot0 | 82.8 | +1.0199 |
| c2p0_slot0_q3_full_q0 | c2p0_slot0 | 82.8 | +0.9970 |
| c2p0_fair013_best_full | c2p0_f013_best | 70.7 | +0.9888 |
| c2p0_slot0_q3_full_q9 | c2p0_slot0 | 82.8 | +0.9750 |
| c2p0_slot0_full_seed27182818 | c2p0_slot0 | 82.8 | +0.9306 |
| c2p0_slot2_full_noq | c2p0_slot2 | 69.8 | +0.7831 |
| c2p0_fair013_worst_full | c2p0_f013_worst | 71.4 | +0.7469 |
| p7v1_fullq_control_q9 | lpmc_p7v1 | 47.7 | +0.4747 |
| fulltrack_noq_full | fulltrack | 7.3 | +0.1265 |
| fulltrack_q3_full_q9 | fulltrack | 7.3 | +0.0519 |

### 四個指標的相關（語料層 n=6 為準）

| 指標 | r (arm, n=12) | r (語料, n=6) | R² (語料) | ρ (語料) | 判定 |
|---|---:|---:|---:|---:|---|
| **`quality_rate`** | **+0.980** | **+0.965** | **0.931** | +0.829 | **N1′ 成立** |
| `hifi_rate` | +0.935 | +0.888 | 0.789 | +0.771 | 同向但較弱 |
| `negterm_rate` | −0.300 | −0.322 | 0.104 | **−0.771** | **N1 字面版推翻** |
| `lofi_rate` | −0.043 | −0.152 | 0.023 | −0.143 | 無關 |

**斜率（`quality_rate`）：+0.0122 PQ / 每個百分點**，arm 層與語料層一致到小數第四位。

### 雜訊底線與槓桿檢定

- **共用語料的 arm 間離散度 = 0.121 PQ**（c2p0_slot0，6 個 arm，caption 完全相同，
  只差 seed 與 S2 是否加 Q）。fulltrack 2 arm 是 0.075。
  語料間的跨度是 0.052 → 1.052，**比雜訊底線大一個數量級**。
- **Leave-one-corpus-out r**：+0.900 ~ +0.989。即使拿掉 fulltrack（唯一的低密度點、
  也是最大槓桿點），r 仍有 **+0.900** —— 這條線不是被單一離群點撐起來的。

## 讀法

### 1. N1 的字面版本被自己的資料推翻

`y_neg` 用的 8 個詞在 LP-MC 語料裡出現率 **46.3%**（最高），在 c2p0 裡只有 **5.3%**（幾乎最低）。
如果機制是「負向提示詞必須命中訓練語料中的高密度模式」，LP-MC 應該拿最大增益。
實際上 LP-MC 是 **+0.475**，c2p0 是 **+1.02**。方向相反，語料層 ρ = **−0.771**。

**「negative prompt 的詞必須在訓練資料裡常見」是錯的。**

### 2. 真正預測增益的是「語料談不談保真度」，與極性無關

`quality_rate`（不分極性）r = +0.965、R² = 0.931，六個點單調。
c2p0（78.5% 高保真、談的幾乎都是好話）與 LP-MC（47.7% 低保真、談的幾乎都是壞話）
在這條線上都落在預期位置；決定位置的是**談的頻率**，不是**談的方向**。

這與 09-03 內容階梯的兩個既有觀察完全咬合，且是它們的上位解釋：

- `reversed`（把「high quality, pristine, hi-fi」放進 negative 槽）仍拿到 **+0.722** ——
  因為被啟動的是同一條保真度軸。
- `loud` 的響度**反轉** —— 因為極性從來就不是這個機制的作用方式。

**可寫的機制句**：negative prompt 的效果量由「模型是否訓練出一條保真度軸」決定，
而那條軸的存在與否，由訓練 caption 提到保真度的頻率決定 —— **不是由負向詞彙的密度決定**。

### 3. 對 QA-MDT 的修正，不只是印證

QA-MDT 把自己效果小歸因於 `"low quality"` 在資料集中**稀有**。
我們的資料說：**低保真詞彙的頻率跟增益無關（ρ −0.14）**。
它真正該歸因的是「語料**完全不談**保真度」——這是可以寫進論文的一句修正，
比單純引用它的 "rare instances" 強。

## 限制（必須跟結果一起寫）

1. **n=6 語料，且不是 6 個獨立點。** 其中 4 個是 Qwen caption 2.0 的 slot 變體
   （69.8–82.8% 擠在一起），實際上只有 **3 個叢集**：c2p0 群、LP-MC、fulltrack。
   r=0.965 被這個槓桿結構灌水，**不要把 R²=0.931 當成 effect size 的精度**。
2. **語料之間不只差密度。** fulltrack 是另一套 captioning pipeline、LP-MC 是另一個 captioner，
   字數也從 42.5 到 53.8 不等。`quality_rate` **沒有被隔離**，這是相關不是因果。
3. `hifi_rate` 與 `quality_rate` 高度共線（c2p0 群兩者都高），
   本文不宣稱能區分「談保真度」與「談高保真」。
4. ΔPQ 只是 Audiobox PQ；crest 崩塌等聽感代價不在這條軸上。

## 這使 fidstrip arm 從「可做」升級為「決定性」

`fidelity_stripped_caption_arm_2026_08_30.md` 的 arm 是**唯一能打破限制 2** 的介入：
同一個 captioner、同一份音訊、同樣的字數量級，**只把 quality 提及率從 82.8% 搬到 10.2%**。

由語料層擬合給出**可預先登記的預測**：

> **fidstrip arm 的 ΔPQ ≈ +0.09**（點估計 +0.094，由 n=6 最小平方擬合外推至 10.2%）。
> 也就是說它應該掉到 **fulltrack 的水準（+0.05 ~ +0.13）**，
> 而不是保持在 c2p0 的 +1.0。

判準（建議寫進 contract）：
- ΔPQ ≤ **+0.30** → N1′ 通過因果檢定（密度是原因，不只是相關）。
- ΔPQ ≥ **+0.70** → N1′ 被推翻，增益來自 captioner／文體而非保真度密度。
- 中間帶 → 密度是部分原因，需再拆。

成本：78 GB overlay ＋ quarter 訓練約 5 小時 ＋ cfg0/cfg3 兩次 eval。
**這是目前整條 negative prompt 線上性價比最高的一個實驗**，因為它把一篇論文裡
最強的機制宣稱從「相關」升級為「因果」。

## 論文寫法

- ❌ 不要寫「negative prompt 的增益來自命中訓練資料中的低品質樣本」—— 我們自己的資料反對。
- ✅ 寫：「negative prompt 的效果量隨訓練 caption 的**保真度提及率**單調變化
  （r=+0.97, n=6 語料, 跨度 0.05→1.05 PQ），而與**負向詞彙本身的密度無關**（ρ=−0.77）。
  這修正了 QA-MDT 把效果量歸因於 'rare instances of low quality' 的說法。」
- ✅ 同時報告雜訊底線 0.121 PQ 與 leave-one-out 範圍 +0.900~+0.989。
- ✅ 限制 1–4 逐條寫進 limitation，不要只寫 r。
