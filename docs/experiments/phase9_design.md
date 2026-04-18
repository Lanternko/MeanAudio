# Phase 9 / 9.5 實驗設計

## 核心問題

Phase 7 V1 的提升，是「random caption 本身更好」還是「Q 信號帶來的額外資訊」？

Phase 7 V1 的 random 是每個 clip 在 TSV 生成時 seed=42 選定，**整個訓練過程每個 clip 只看到一個 caption**（固定 random，非 epoch-to-epoch random）。Phase 9 改為每個 epoch、每個 clip 從 5 個 caption 中隨機抽一個（動態採樣 via `random.randint` in `__getitem__`），才是**真正的 true random**。

## 四組對比

| 實驗 | Caption 來源 | Q 信號 |
|-----|-------------|-------|
| **Phase 9 V1** | LP-MusicCaps 5 caps | 無 Q |
| **Phase 9 V2** | LP-MusicCaps 5 caps | pairwise MeanSim of 5 captions |
| **Phase 9.5 V1** | Qwen2.5-Omni 5 task caps | 無 Q |
| **Phase 9.5 V2** | Qwen2.5-Omni 5 task caps | pairwise MeanSim of 5 task captions |

## Phase 9.5 為主結果（2026-04-17 確認）

LP-MusicCaps 的 Jamendo caption 是透過第三方模型輸出，審稿人可能質疑「不是真實 human caption」。用自家跑的 Qwen2.5-Omni 多面向 caption 能同時證明：
1. diversity hypothesis 跨 captioner 成立
2. 不依賴第三方資料集

### ⚠️ LP-MusicCaps Jamendo caption 的真實生成方式

**這是 2026-04-18 york135 澄清的重要事實**（見 `../meetings/2026-04-18_lane_abc_and_lpmc.md`）：

我們手上的 `results_20260119_043407.jsonl`（251K Jamendo × 5 caps）**不是** LP-MusicCaps 論文原始 pipeline（tag → GPT-3.5 → pseudo-caption）的產物。它是 wei-jaw 用 LP-MusicCaps **captioning MODEL** 本身，在 Jamendo 音訊上跑 **5 個不同 seed** 產生的。

因此：
- 5 caps 間 diversity 來自 **seed-sampled decoding noise**
- LP-MusicCaps 論文的 4-task 設計**從未套用於 Jamendo**
- Phase 9.5 Qwen 的 5-task 設計是**我們自己的新貢獻**，不是 LP 複刻

### Phase 9.5 Qwen 5-task prompt 設計

原先設計為 5 個 aspect-focused prompt（整體/情緒/樂器/節奏/場景），但 1/5 機率抽到 partial caption，且 mean_similarity q 信號語義變成「aspect 差異」而非 captioning confidence。2026-04-17 修正為**每個 caption 都 comprehensive**，diversity 來自 task framing：

| Slot | Qwen task prompt 方向 |
|------|-------------------|
| 0 | Writing：詳細一句話綜合描述 |
| 1 | Summary：壓縮綜合短句 |
| 2 | Paraphrase：豐富詞彙改寫（同樣綜合）|
| 3 | Attribute Prediction：屬性為主的描述 |
| 4 | Natural Prose：中性自然敘述 |

每個 caption 都含樂器 + 情緒 + 節奏 + 風格，pairwise MeanSim 反映「跨 task 的 Qwen captioning 一致性」。

→ 每個 slot 跑一次獨立 pass（`--slot 0..4`），最後 `--merge`。

## multi_cap 機制（`extracted_audio.py` commit d94db11）

```python
# ExtractedAudio(multi_cap=True) → __getitem__ 每次隨機選一個 caption
if self.multi_cap:
    cap_idx = random.randint(0, n_caps - 1)
    text_features   = torch.from_numpy(np_data['text_features'][cap_idx])    # [77, 1024]
    text_features_c = torch.from_numpy(np_data['text_features_c'][cap_idx])  # [512]
```

Config 加 `multi_cap: true` 即可啟用，向下相容（舊 NPZ 不受影響）。

## mean_similarity vs CLAP score 作為 Q 的根本差異

- **mean_similarity**：5 個 caption 的 text-text 相似度平均 → 衡量 caption 群**一致性**
- **CLAP score**：audio-text CLAP 相似度 → 衡量 caption 有多**準確**描述音訊

這兩種 diversity 機制（seed-noise vs task-framing）的對比本身就是有意思的研究問題：哪個產生的 MeanSim 更能當品質 q 信號。
