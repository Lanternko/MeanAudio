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
