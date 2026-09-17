# Meta Audiobox Aesthetics 指標

## 安裝

```bash
pip install audiobox_aesthetics   # CC-BY 4.0，無需申請，自動下載權重
```

## 四個子指標

| 指標 | 名稱 | 物理意義 |
|------|------|---------|
| **CE** | Content Enjoyment | 主觀聽感、情感影響、藝術性、整體喜好 |
| CU | Content Usefulness | 內容是否符合使用情境 |
| PC | Production Complexity | 製作複雜度 |
| **PQ** | Production Quality | 技術品質：清晰度、保真度、無雜訊失真 |

## 與人類 MOS 的相關係數（文獻，PAM-music，utterance-level）

| 指標 | ↔ 人類 OVL | ↔ 人類真實標註 | 備註 |
|------|-----------|--------------|------|
| **CE** | **0.528** | **0.661** | 單樣本層級 |
| **PQ** | 0.464 | 0.587 | 單樣本層級 |

成對偏好預測準確率（From Aesthetics to Human Preferences）：
- CE、CU：> 60%（顯著高於盲猜 50%）
- PQ（保真度偏好）：59.1%

→ **CE 與人類主觀評分相關性最強**，是評估「音樂品質提升」的最佳指標。

## 已採用此指標的論文（學術引用依據）

- **LeVo (2025)**：多偏好對齊歌曲生成，評估 Suno-V4.5、Mureka-O1、YuE
- **ACE-Step (2025)**：音樂生成基礎模型，Table 1 全面採用四指標
- **SongBloom (2025)**、**MIDI-SAG (2025)**：客觀評估全面採用
- **SMART**：直接用 CE 作為 RL reward 微調符號音樂生成
- **AudioMOS Challenge 2025（Track 2）**：以四指標作為官方評測框架

## 對 MeanAudio 研究的意義

- **CE** → 回答「quality conditioning 是否讓音樂更好聽、更有藝術性」
- **PQ** → 回答「是否降低了低品質訓練資料帶來的技術瑕疵（雜訊、失真）」
- **最強論述**：CE 和 PQ 同時提升 = q_embed 帶來全方位感知品質升級；只有 PQ 升而 CE 不動 = 只學會「清理背景雜訊」
- **學術寫作建議**：四個指標全部列出（如 LeVo、ACE-Step 做法），以 CE 為主軸論述
- ⚠️ 若用 CLAP 過濾訓練資料，evaluation 改用 Audiobox Aesthetics（避免 data leakage，見 `../meetings/2026-03-27_filtering_and_metrics.md`）

---

## 已知限制（2026-09-18 於 source code 驗證）

以下 1–3 點讀自本機安裝的官方實作 `audiobox_aesthetics==0.0.4`
（`~/venvs/dac/lib/python3.12/site-packages/audiobox_aesthetics/`），**不是文獻轉述**。

### 1. 完全沒有響度正規化 → 分數不是 gain-invariant

推論前的前處理只有 resample 到 16 kHz + 轉 mono（`infer.py:149` `audio_resample_mono`），
**沒有任何響度或峰值正規化**。WavLM 的 waveform layer_norm 也是關的：checkpoint config
寫死 `"normalize": False`（`model/aes.py:56`），所以 `model/aes.py:156` 的
`layer_norm(wav, wav.shape)` 分支不會執行。唯一的 normalize 是 `normalize_embed`，
作用在 pooling 之後的 embedding 上，救不了輸入尺度。

→ **原始絕對振幅直接餵進 WavLM 的 conv feature extractor，模型在架構上就不是 gain-invariant。**

實測（051 `loudness_aes_cfg3_20260911`）：同一個檔案衰減 6 dB，**PQ +0.097**，CI 不跨零。
內容除尺度外逐位元相同，分數卻變了 —— 這是指標缺陷，不是音訊變好。

**必須遵守**：
- 任何 arm 間的邊際比較，響度必須先鎖住，否則差異會被響度污染到 seed 雜訊量級
- AES 絕對值**不可跨論文比較**，除非對方明確交代做了響度正規化（依此實作，預設是沒有）

### 2. 16 kHz 重取樣 → 8 kHz 以上完全不參與評分

`infer.py:85` `sample_rate: int = 16000  # const`。我們的輸入是 48 kHz（見
`memory/reference_clap_scoring_input_contract.md`），等於高頻三分之二在評分前就被丟掉。
用一個聽不到 8 kHz 以上的模型判斷 music production quality，是 PQ 這個指標的結構性弱點。

### 3. 感受野 10 秒

`infer.py:51` window_size = hop_size = 10 s。SMART（arXiv 2504.16839）指出這無法涵蓋
樂句重複與發展等長於 10 秒的結構。我們的 eval clip 剛好是 10 秒，**每首只有一個窗，這點對我們無害**；
但若改評估更長的音訊，長結構不會被看到。

### 4. crest 不是可刷的旋鈕（與響度相反）

061（`docs/experiments/results/crest_intervention_cfg3_20260916_results.md`）在固定 LUFS 下
主動移動 crest：拉高 crest 讓 PQ **下降**（up −0.024 / upmax −0.030），
且與音樂結構脫鉤的 `rand`（crest 更高）掉最多（−0.139）。
所以真正要防的低階旋鈕是**響度**，crest 是被冤枉的那一個。

## 文獻批評（他人指出的缺點）

| 論文 | 批評點 |
|---|---|
| SMART（arXiv 2504.16839） | 拿 CE 當 RL reward 直接示範 **reward hacking**：省掉 KL penalty 後分數上升但 diversity 崩潰、輸出高度重複。另指出 10 秒感受野問題 |
| SongEval（arXiv 2505.10793） | 立場是 AES 維度有限，因此另建人類標註 benchmark；報告 **PC 與各人類美學維度相關性最低** |
| Survey on Eval Metrics for Music Generation（arXiv 2509.00051） | 客觀指標普遍缺可解釋性與明確門檻、與人類感知錯位；跨文化偏誤偏袒西方音樂 |
| Limits of Reference-Free Speech Quality Metrics（arXiv 2609.13150） | 語音不是音樂，但方法論最貼近：指標在**乾淨**音訊上退化到亂猜（0.50–0.53，乾淨樣本上「片段長度」就跟最好的指標打平）；當 reward 會被 hack（SCOREQ 推到滿分但 held-out 從 4.51 崩到 1.23）；短時響度線索在 −23 LUFS 正規化後仍存活（d=0.23） |
| 原始論文（arXiv 2502.05139） | 未揭露音樂語料組成、標註者文化背景與音樂訓練 |

⚠️ 截至 2026-09-18 **沒找到任何論文對 AES 做「同一份音訊、只改增益／crest」的介入測試** ——
051/061 是在補這個空白，不是重複他人工作。
