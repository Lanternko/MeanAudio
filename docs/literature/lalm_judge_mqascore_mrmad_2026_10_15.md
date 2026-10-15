# LALM 當評審的文獻定位：MQAScore 與 MRMAD（2026-10-15）

背景：p2 124 的兩個 LALM 評審（Music Flamingo、Qwen2.5-Omni-3B）用 1–5 分評「品質」，都沒過噪音階梯門檻（`../experiments/results/lalm_quality_judge_124_results.md`）。
這份文件整理兩篇文獻：它們怎麼操作 LALM，以及對我們下一步的意義。

## MQAScore（ICASSP 2027 投稿，Lee, Wang, Chen, Yang；NTU／Taiwan AILabs）

題目：Fine-grained text-music alignment evaluation via large audio language model。專案頁 https://lonian6.github.io/MQAScore.github.io/。

**範圍**
- 評文字與音樂的語意對齊，分五個維度：genre、instrument、mood/theme、vocal、usage（取自 CAL500）。
- **明確排除音質**：「We exclude acoustic qualities, which reflect audio fidelity rather than semantic alignment」。
- 和聲、節奏等音樂學概念也排除，理由是 LALM 難以偵測（引 MRMAD、MuChoMusic）。

**操作方式**
- 一題只問一個原子概念，用二元問題，例如 `Does this music belong to the genre {t}? Answer Yes or No.`
- 分數 S = p(Yes) / (p(Yes) + p(No))，一次 forward、不抽樣。沿用 AQAScore。
- 逐概念問一致勝過整句 caption 問：概念被同句其他概念稀釋，近義替換幾乎不動分數，會讓 R_S/I 虛高。

**可靠度檢查**
- genre／instrument：用 AUC（正例 vs 缺席概念）與 within-clip win rate。
- 其餘三維：用 Sensitivity-Invariance Ratio。
  - R_S/I = z-MAD⁻ / z-MAD⁺，分子是換成語意最遠概念 t⁻ 的分數變動，分母是換成近義概念 t⁺ 的分數變動。
  - ≫ 1 代表評審跟著音訊走；≈ 1 代表只跟著字面走。

**結果**
- 評審：Qwen3-Omni-30B-A3B、Audio Flamingo Next、Music Flamingo。Qwen3-Omni 整體最好，Music Flamingo 多數維度最弱。
- LALM 在 mood／vocal／usage 明顯勝過 CLAP／MuQ-MuLan。
- 與專家成對標註的一致率接近人與人之間的一致率（Table 2）。

## MRMAD（arXiv 2608.22236，EMNLP 2026）

題目：A Multi-Round Multi-Audio Benchmark for Evaluating Acoustic Degradation Perception in LALMs。

**設計**
- 涵蓋語音、音樂、環境聲。
- 九種劣化：ambient noise、low-pass、band-pass、MP3、reverberation、echo、neural vocoder artifacts、DSP denoising artifacts、packet loss。每種分 audible／medium／strong 三級。
- 沒有白噪音與削波。

**任務**
- 都是多輪、多音訊的選擇題：
  - 劣化類型辨識（四選一，機率 25%）。
  - 兩段嚴重度比較（50%）。
  - 三段排序（25%）。
- 18 個 LALM。

**音樂部分的結果（Table 4）**

| 模型 | 類型辨識 | 兩段比較 | 三段排序 |
|---|---|---|---|
| Qwen2.5-Omni | 30.5 | **49.6** | 33.3 |
| Qwen3-Omni-Instruct | 42.8 | 52.3 | 27.3 |
| Qwen3-Omni-Thinking | 39.8 | 61.5 | 34.5 |
| Audio Flamingo 3 | 27.0 | 48.8 | 28.5 |
| Audio Flamingo Next | 28.0 | 47.9 | 31.0 |

- 結論：多數開源模型只略高於機率，人類明顯較好。
- Music Flamingo 沒被評。
- 我只讀了前半部 HTML，逐劣化類型的數字在圖 3，未取得。

## 對我們的意義

1. **124 的失敗與 MRMAD 一致。**
   - Qwen2.5-Omni 在音樂的兩段嚴重度比較是 49.6%，等於擲硬幣。我們 pair 任務的 p_clean_better 也是 0.50–0.52。
   - Audio Flamingo 系列同樣接近機率。
   - 124 的 explain 評語（Flamingo 對 SNR 10 說「沒有背景噪音」）屬於同一型態：聽得出內容，聽不出劣化。
2. **MQAScore 刻意不碰音質**，所以它不能直接支持或否定我們的 1–5 分品質 prompt。但它的操作格式（二元問題、P(Yes) 正規化、AUC 與成對替換比值）可以直接搬到劣化**偵測**上。
3. **下一步**：用 MQAScore 格式做一個劣化偵測 probe（p2 125，`../experiments/lalm_degradation_perception_125_20261015.md`）。**2026-10-15 擱置，未執行。**
   - 單段、二元、一題一個缺陷，比 MRMAD 的多輪選擇題容易。
   - 如果連這個都做不到，LALM 品質評分就沒有感知基礎。
4. **換評審**：Qwen3-Omni-30B-A3B 在兩篇都是最好的開源選項。
   - bf16 約 60 GB，放不進 5090 的 32 GB，要量化，量化本身會是新變因。
   - MRMAD 顯示它在音樂劣化上也只比機率好一點（兩段比較 52–62%），預期提升有限。
