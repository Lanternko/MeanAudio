# Negative-prompt 內容 × CFG 交互作用（cfg 3.0）

日期：2026-09-03
狀態：**內容階梯完成（9 格全在檔）。** 13-arm 全量 cfg 3.0 sweep 執行中。
資料：`/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation/*.json`
腳本：`scripts/eval/negprompt_ablation_matrix.py`（Q3/Q4 區塊）、
`scripts/analysis/negprompt_loudness_covariate.py`
前置：[negprompt_cfg_content_ablation_2026_08_31.md](negprompt_cfg_content_ablation_2026_08_31.md)

前一份 36 格矩陣的內容掃描只在 cfg 1.5 做，留下未解問題 #2（內容 × cfg 交互作用未測）與
#3（短版在高 cfg 是否仍有 loudness confound 未測）。本文在 PQ 最佳點 cfg 3.0 補完整個內容階梯，
並用 per-clip 資料對 loudness confound 首次定價。

> **本文推翻了自己的期中版本。** 2026-09-03 早上的期中版把 `silence` / `fidelity_short`
> 判為 loudness confound、把長版 `fidelity` 判為「唯一乾淨」。那些判定建立在**沒有被量化過的
> 前提**上（響度位移了，而 Audiobox PQ 非 level-invariant，所以增益不可採信）。
> 定價之後前提不成立，判定隨之作廢，詳見第 2 節。

## 協定與 provenance

MusicCaps 1,024 筆 seeded 子集（seed 20260830）、MeanFlow 25 步、NoMask、seed 42、full precision、
CLAP batch 32。arm 為 `c2p0_slot0`（`phase8_qwen_caption10s_multisent_noq_full_stage2_200000`，NoQ）。
每格逐檔配對至同 arm 的 cfg 0 格（PQ 6.5822 / CLAP 0.2194 / RMS −17.54 dB / crest_mean 6.171）。

`none` 與 `fidelity` 來自 2026-08-31；`fidelity_short` / `silence` / `reversed` 來自 09-02；
`irrelevant` / `neutral` / `loud` / `fidelity_loud` 來自 09-03。已核對：checkpoint mtime 2026-08-12、
subset TSV mtime 2026-08-30，皆早於全部三次執行且期間未被覆寫；9 格的 exp_id / subset / seed /
n=1024 完全一致，三天的間隔不構成 confound。

## cfg 3.0 內容階梯（完整）

ΔPQ 為逐檔配對均值；ΔRMS / Δ重心為對 cfg 0 基準的位移。

| negative | 內容 | ΔPQ | ΔRMS | crest_min | Δ重心 | frac+ |
|---|---|---:|---:|---:|---:|---:|
| `none` | （純 CFG） | −0.003 | +0.37 | 2.17 | +224 | 0.52 |
| `neutral` | music | +0.250 | −0.08 | 1.99 | +360 | 0.68 |
| `irrelevant` | 貓照片、試算表、印刷文字 | +0.357 | −1.46 | 2.41 | +141 | 0.72 |
| `loud` | loud, clipping, saturated… | +0.515 | **+1.10** | 1.95 | **+3** | 0.79 |
| `silence` | silence, empty track | +0.663 | +4.77 | 1.76 | +706 | 0.80 |
| `reversed` | high quality, pristine, hi-fi | +0.722 | +1.57 | 2.51 | +360 | 0.88 |
| `fidelity_short` | low quality, noisy | +0.971 | +1.67 | 1.90 | +208 | 0.91 |
| `fidelity_loud` | fidelity 全套 ＋ loud 全套 | +1.033 | −0.67 | 2.46 | +95 | 0.94 |
| **`fidelity`** | 8 詞全套 | **+1.067** | −0.50 | 2.75 | +72 | 0.95 |

絕對值：

| negative | PQ | CLAP | CE | CU | PC | clean−lofi gap |
|---|---:|---:|---:|---:|---:|---:|
| cfg0 基準 | 6.5822 | 0.2194 | 6.3107 | 6.7046 | 5.1439 | +0.335 |
| `none` | 6.5790 | 0.2392 | 6.2729 | 6.6763 | 4.7738 | +0.361 |
| `neutral` | 6.8318 | 0.2385 | 6.3383 | 7.0143 | 4.5097 | +0.301 |
| `irrelevant` | 6.9390 | 0.2357 | 6.5532 | 7.0833 | 4.4287 | +0.331 |
| `loud` | 7.0974 | 0.2559 | 6.9655 | 7.1613 | 5.2831 | +0.280 |
| `silence` | 7.2450 | 0.2478 | 7.0596 | 7.3735 | 5.0651 | +0.257 |
| `reversed` | 7.3043 | 0.2445 | 6.8886 | 7.4458 | 5.0180 | +0.278 |
| `fidelity_short` | 7.5534 | 0.2612 | 7.2807 | 7.5788 | 5.2401 | +0.218 |
| `fidelity_loud` | 7.6155 | 0.2599 | 7.2721 | 7.6486 | 5.1266 | +0.192 |
| **`fidelity`** | **7.6495** | 0.2598 | 7.2894 | **7.6807** | 5.1592 | **+0.185** |

## 1. `loud`：極性完全沒有被遵守，方向是反的

`loud` 的內容是 `loud, clipping, over-compressed, saturated, blown out`，放在 **negative** 槽位，
語意上應該把輸出推離「大聲／壓縮／飽和」。它同時是操作檢定（manipulation check）：
如果文字能控制響度，這格必須變**安靜**。

實測 **ΔRMS = +1.10 dB —— 變大聲了**，crest_min 掉到 1.95。方向與語意要求相反。

這與 `reversed` 是同一個現象的第二次獨立出現：把語意上該推離高品質的詞放進 negative 槽位，
拿到的是 +0.722 的 PQ **增益**。兩格合起來的讀法是：

> **negative 槽位的文字內容會影響輸出，但不是按照它的語意極性影響。**
> 「在 prompt 裡加入不要變大聲」在這個模型上不但無效，還會讓輸出更大聲。

附帶一個乾淨的旁證：`loud` 的重心位移是全表最小的 **+3 Hz**（其他格 +72 ~ +706 Hz）。
響度動了 1.10 dB 而頻譜形狀幾乎沒動 —— 該詞彙確實碰到了響度這個維度，只是符號相反。

## 2. Loudness confound 判定作廢：定價後只值 ≲0.15 PQ，且方向多半相反

期中版把 `silence`（+4.77 dB）與 `fidelity_short`（+1.67 dB）判為 confound，
理由是「響度位移大 + PQ 非 level-invariant」。這個理由從未被量化。現在量化了。

### 2a. Cell 層級回歸（n=37）：無法定價，但也沒支持 confound

`scripts/analysis/negprompt_loudness_covariate.py`。單變量 ΔRMS 對 ΔPQ：
r=+0.208、R²=0.043、p=0.22（Spearman +0.119）—— 跨格幾乎沒有關係。

決定性的是純 CFG 那 12 格，唯一「響度會動、槽位裡沒有文字」的子設計：
`fulltrack` cfg4.5 拿到 ΔRMS +0.55 dB、crest_min 1.68，ΔPQ 是 **−0.017**；
`c2p0_slot0` cfg4.5 是 +0.25 dB / 1.85 / **−0.050**。斜率 −0.033（p=0.49）。
**純 CFG 把 confound 的全部特徵都做出來了，一分 PQ 都沒收到。**

但這層分析有兩個硬限制，所以它只能存疑不能定案：純 CFG 格只跨到 +0.55 dB，
而 corr(ΔRMS, Δcrest) = **−0.938**，兩條 confound 軸近乎共線，雙變量係數
（0.53 / 1.09，p<0.001）坐在退化方向上，據以算出的 per-cell 殘差不可單獨採信。

### 2b. Per-clip 配對（n=1024）：定價完成

`per_clip_signal` 讓同樣的問題變成 cell 內配對比較。四格兩兩相配、逐 clip 取
Δ響度與 ΔPQ：

| 配對 | 平均 ΔRMS | slope(ΔPQ/dB) | r | slope(crest) | 聯合 R² |
|---|---:|---:|---:|---:|---:|
| neutral→irrelevant | −1.44 | +0.0064 | +0.033 | −0.0113 | 0.006 |
| neutral→loud | +1.73 | **−0.0262** | −0.233 | +0.0278 | 0.058 |
| neutral→fidelity_loud | −0.02 | −0.0176 | −0.148 | +0.0177 | 0.023 |
| irrelevant→loud | **+3.16** | **−0.0280** | −0.281 | +0.0217 | 0.079 |
| irrelevant→fidelity_loud | +1.42 | −0.0168 | −0.157 | +0.0115 | 0.025 |
| loud→fidelity_loud | −1.74 | −0.0343 | −0.123 | +0.0828 | 0.036 |

六組配對的響度斜率有五組是**負的**，量級 −0.017 ~ −0.034 PQ/dB。
`irrelevant→loud` 這組平均跨了 **+3.16 dB**，已覆蓋 `silence` +4.77 dB 的大部分範圍。
兩條軸聯合起來最多只解釋 **7.9%** 的逐 clip PQ 變異。

拿最有利於 confound 假說的斜率外推到全表最大的響度位移：
**+4.77 dB × 0.03 ≈ 0.14 PQ，而且符號多半是負的。** `silence` 實得 +0.663。
crest 那條軸同向：crest 係數為正（+0.011 ~ +0.083），而 `silence` 的 crest **下降**
（6.171 → 4.235），照這個斜率應該**扣分**。

**結論：loudness / crest confound 解釋不了 0.5–1.07 級別的增益，作廢期中版的兩個
confound 標記。** 未解問題 #3 的答案是「confound 存在但幾乎不值錢」，
不是期中版寫的「隨 cfg 放大」。

保留的限制：這個定價來自 4 格、單一 arm、單一子集；crest 1.76 這種波形是否另有
PQ 以外的聽感代價，本節不涉及 —— 那是主觀評估的事，不是 PQ 的事。

## 3. 三層拆解在 PQ 最佳點重算完成

`irrelevant` 與 `neutral` 補上後，cfg 3.0 的拆解可以收：

| 層 | 對照 | ΔPQ | 佔 1.067 |
|---|---|---:|---:|
| guidance 本身 | `none` | −0.003 | **0%** |
| 槽位有任何文字 | `irrelevant` − `none` | +0.360 | 34% |
| fidelity 領域詞彙 | `reversed` − `irrelevant` | +0.365 | 34% |
| 極性正確 | `fidelity` − `reversed` | +0.345 | 32% |

**純 CFG 在 PQ 最佳點的貢獻是零。** 這比 cfg 1.5 的 +0.016 更強，也與前一份
「guidance 是語意對齊旋鈕不是品質旋鈕」一致 —— CLAP 仍隨 cfg 上升（0.2194 → 0.2392）。

`neutral`（music）+0.250 **低於** `irrelevant`（貓照片）+0.357，兩者差 0.107，
高於訓練 seed 雜訊底線 0.052 但只有兩倍。「任何文字」這層不宜再細分。

### 極性的地位（相對期中版上修）

期中版寫「目前的資料不支持任何極性方向的宣稱」。補上 `irrelevant` 之後這句要放寬：
`fidelity`(1.067) > `reversed`(0.722) 差 **0.345**，遠高於 seed 雜訊 0.052，
所以**極性確實有貢獻**，約佔三分之一。

但仍不能寫成乾淨的極性效果：`reversed` 與 `fidelity` 的 T5 masked-mean cosine 是 **0.814**，
兩者在 embedding 空間大致同方向而非反方向，所以這 0.345 也可以是「embedding 距離」
而非「語意極性」。而 `loud` 的響度反轉（第 1 節）直接顯示極性在這個模型上不被遵守。
**能寫的是「fidelity 領域詞彙與極性各貢獻約三分之一」，不能寫「negative prompt 依語意極性運作」。**

## 4. `fidelity_loud`：加抑制詞沒有用

工程問題是「能不能既拿到 `fidelity` 的增益又避開 crest 崩塌」。答案是不能，但也沒有變差：

- ΔPQ +1.033 vs `fidelity` +1.067 —— 差 0.034，**低於 seed 雜訊底線 0.052**，視為同級。
- ΔRMS −0.67 vs −0.50 —— 確實略安靜，但 `fidelity` 本來就已經比基準安靜。
- crest_min 2.46 vs 2.75 —— **反而更差**。

加了 5 個抑制詞，PQ 不動、crest 沒救回來。結合第 1 節（`loud` 讓輸出變大聲），
一致的讀法是這些詞彙沒有按語意作用。**主結果維持長版 `fidelity`，不採用 `fidelity_loud`。**

## 5. Prompt-fidelity 代價：增益最大者最不服從

clean−lofi 的 PQ 差距（越小＝越不服從「低保真」指令），基準 +0.335：

| negative | gap @1.5 | gap @3.0 |
|---|---:|---:|
| `none` | +0.351 | +0.361 |
| `irrelevant` | — | +0.331 |
| `neutral` | — | +0.301 |
| `loud` | — | +0.280 |
| `reversed` | +0.325 | +0.278 |
| `silence` | +0.266 | +0.257 |
| `fidelity_short` | +0.273 | +0.218 |
| `fidelity_loud` | — | +0.192 |
| **`fidelity`** | +0.311 | **+0.185** |

gap 與 ΔPQ 幾乎單調反向：**增益最大的設定同時最不服從低保真指令。**
長版 `fidelity` 的代價從 cfg 1.5 的幾乎為零（0.311 vs 0.335）擴大到 cfg 3.0 的
gap 縮小 45%。這是真實取捨。前一份提出的條件式協定（negative 只施於乾淨提示詞）
在 cfg 3.0 保留 +0.655（61%），仍是建議的主結果形式。

## 待辦

1. 13-arm 全量 cfg 3.0 sweep（`negprompt_reeval_full_arms.py --cfg=3.0` → `negprompt_reeval_cfg3.0/`）
   執行中，2026-09-03 07:29 起，13 arm × 5,521 筆。檢驗 cfg 1.5 的排序反轉是否為該 cfg 特例。
2. 主觀確認 crest 1.76–1.95 那幾格（`silence` / `loud` / `fidelity_short`）**聽起來**如何。
   第 2 節只證明 PQ 沒有被響度買通，沒有證明那些波形沒問題；PQ 也可能單純對飽和不敏感。
3. 極性若要真正檢驗，須在 embedding 空間操作（例如餵 `−v`），改文字達不到 —— 第 1 節的
   `loud` 反轉是這一點的直接證據。

## 限制

1. 單一 arm（`c2p0_slot0`）、單一 seeded 子集、單一評估集；未做 paired bootstrap CI。
2. Per-clip 定價只有 4 格有資料（09-03 之後跑的格才存 `per_clip_signal`）。
3. CLAP 為 batch 32，不可與歷史逐檔數字做 exact 比較。
4. 逐 clip 配對的斜率是跨 cell 的關聯，不是干預：若某個機制同時推高響度與品質，
   斜率會把它吸收掉。方向上這讓「增益不是響度」的結論偏保守，是安全的一側。

## 附記：pause 機制的 bug（2026-09-03）

本文的 cfg 3.0 補跑需要搶佔 P2 的 035_slot3_full。`lib_scheduler.py pause-p2` 寫出 pause request
之後，`train.py` 正常存完 checkpoint，卻在 `_write_pause_ack` 因 **`train.py` 未 import `time`**
拋 `NameError`，job 以 rc=1 而非 rc=75 結束，`classify_exit` 因此把一次可續訓的 pause 記成 failed。
協作式 pause 從未成功過。修復見 `ebe09a2`。035 的 `ckpt_last.pth`（it 249,565）在崩潰前已寫出，
訓練進度無損，但腳本需從 `p2/failed/` 手動搬回 `p2/pending/` 才會續訓。
