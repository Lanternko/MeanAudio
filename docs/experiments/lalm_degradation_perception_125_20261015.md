# p2 125：LALM 劣化偵測 probe（MQAScore 格式）——預先登錄設計

狀態：2026-10-15 設計登錄，尚未實作／排入。

## 為什麼做

p2 124 的兩個 LALM 評審用 1–5 分評品質，都沒過噪音階梯門檻（`results/lalm_quality_judge_124_results.md`）。
explain 評語顯示的是**感知**失敗：
- Flamingo 對 SNR 10 白噪音的生成音訊說「沒有背景噪音」。
- Qwen 把純白噪音描述成「樂器平衡的混音」。

MRMAD（`../literature/lalm_judge_mqascore_mrmad_2026_10_15.md`）也報告 Qwen2.5-Omni 的音樂劣化兩段比較只有 49.6%。

這個 probe 回答一個更基本的問題：**單段、二元、一題只問一個缺陷時，評審能不能偵測出劣化？**

- 能 → 124 的失敗在「把感知轉成品質分數」這一步，換評分設計才有意義。
- 不能 → LALM 品質評分沒有感知基礎，這條線對這兩個評審收線。

## 刺激（全部沿用 124 的檔案，不新生成）

| 組 | 內容 | n |
|---|---|---|
| noise | 4 個來源（ours CFG3＋neg、MusicCaps 真實、MusicGen-medium、MusicLDM）× clean／SNR 40／30／20／10／0 × 100，另加純白噪音 100 | 2,500 |
| midi | 123 的 full30 刺激，150 excerpt × 2 SoundFont：`base__clean` 與 floor 的 `white_snr20`、`white_snr0`、`pure_white`、`hardclip_30db`、`bitcrush_3bit`、`clicks_50ps` | 2,100 |

合計 4,600 段。路徑取自 124 的 `items.tsv`（sha 已凍結）。

## 問題（MQAScore 格式）

- 分數 S = p(Yes) / (p(Yes) + p(No))，取 assistant 回答第一個 token，一次 forward、batch 1、不抽樣。
- 同時記錄 Yes／No 兩個 token 的機率質量。
- 訊息格式與 124 相同：Flamingo 文字在前，Qwen 音訊在前，無 system prompt。

| id | 問題 | 方向 |
|---|---|---|
| noise | Is there audible background hiss or white noise in this recording? Answer Yes or No. | 有缺陷 → Yes |
| noise_rev | Is this recording free of background hiss and white noise? Answer Yes or No. | 有缺陷 → No（分析時取 1 − S） |
| distort | Is there audible distortion or clipping in this recording? Answer Yes or No. | 有缺陷 → Yes |
| music | Does this recording contain music? Answer Yes or No. | 純白噪音 → No |

- noise_rev 是極性對照：如果評審只是偏好答 Yes（yes-bias），noise 與 noise_rev 會互相矛盾。
- 評審：Music Flamingo、Qwen2.5-Omni-3B，與 124 相同版本、相同 venv。
- 4,600 段 × 4 題 × 2 評審 = 36,800 次 forward。

## 讀數與預先登錄判讀

所有 AUC 都是 Mann-Whitney、unpaired，正例是有缺陷的版本，負例是同來源的 clean。CI 用 bootstrap 2,000 次，以 excerpt／prompt 為 cluster，seed 20261015。

### 主判準（每個評審各自判）

**P1：noise 偵測**
- noise 問題在 SNR 10 對 clean 的 AUC，四個 noise 來源**都** ≥ 0.80，且 midi 的 `white_snr20` 對 clean 的 AUC ≥ 0.80。
- SNR 10 在 AES PQ 上已扣約 1.9 分（表 D，四個來源都在 −1.9 左右），推定人耳明顯可聞。

**P2：極性一致**
- 在 noise 組全部 2,500 段上，S_noise 與 1 − S_noise_rev 的 Spearman ≥ 0.5。
- 且 noise_rev 單獨在 SNR 10 的 AUC ≥ 0.70。

**P3：純噪音不是音樂**
- music 問題對純白噪音（noise 組 100 段＋midi `pure_white` 300 段）的平均 S < 0.5。
- 且低於每一個 clean 來源的平均。

**判讀**
- P1 與 P2 都過 →「單段二元格式下偵測得到白噪音」。124 的失敗歸到評分轉換，可以考慮重設計品質評分（另立新設計）。
- P1 不過 →「在 SNR 10 偵測不到白噪音」。這個評審的 LALM 品質評分線收線。
- P1 過但 P2 不過 →「偵測訊號與答 Yes 的傾向無法分開」，視同不過。
- P3 只描述，不影響主判讀。

### 次要讀數（只描述）

- **偵測門檻**：各來源 AUC ≥ 0.75 的最高 SNR。
- **單調性**：各來源平均 S_noise 是否隨 SNR 單調。
- **distort 問題**：在 `hardclip_30db`、`bitcrush_3bit` 對 clean 的 AUC，以及它對白噪音是否也有反應（專一性）。
- **clicks**：`clicks_50ps` 在 noise 與 distort 兩題的反應。
- **參考偵測器**：同一批對比用 AES PQ 的 AUC（負號，PQ 越低越像有缺陷）。
  - noise 組取 `~/eval_output_nvme/ttm_noise_ladder_20261012/lvl30/scores/ladder_lvl30/per_clip.tsv`。
  - midi 取 `~/eval_output_nvme/midi_realpiece_control_123/full30/*/aes.json`。
  - 這是「一個與人類評分相關 0.84 的自動指標能做到多少」的參照，不是正確答案。
- **與 124 對照**：生成音訊（ours、MusicLDM）在 124 加噪音反而加分。這裡看 S_noise 對這兩個來源的 clean 是否已經偏高，即評審是否把擴散系統的原始音色聽成「有噪音」。

### 限制

- 沒有人類評分：「可聞」是依 SNR 與 AES 的既有數字推定，不是聽測結果。
- 只測白噪音、削波、bitcrush、click，不涵蓋 MRMAD 的殘響、壓縮、vocoder 瑕疵等。
- 評審只有兩個，且與 124 相同。Qwen3-Omni-30B 要量化才放得進 32 GB，不在這次範圍。

## 實作與資源

- 新 driver `scripts/eval/lalm_degradation_probe.py`：載入評審、計分方式與 Qwen 的 16 kHz 重採樣都沿用 `lalm_quality_judge.py`。只換問題與兩個 token 的正規化。
- 輸出 `~/eval_output_nvme/lalm_degradation_probe_125/`，續跑以（評審, 問題, 組）為粒度，逐列續寫。
- 時間估計：124 的評分任務約每段 0.1–0.2 s，36,800 次約 1.5–2.5 h，走 p2 queue＋contract。
- 刺激依賴：`~/eval_output_nvme/midi_realpiece_123/audio/`、`~/eval_output_nvme/ttm_noise_ladder_20261012/lvl30/audio/`，125 完成前不可刪。
