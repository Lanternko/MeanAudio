# 真實 MusicCaps 參考音檔的標準指標（2026-09-30）

用途：回答「生成的 AES PQ 比真實錄音還高」到底是多少。

- 音訊：`/mnt/HDD/kojiek/musiccaps_reference`（FAD 參考集，16 kHz、10 s，5,131/5,521 首；檔名經 symlink farm 改成 `.flac` 餵 `eval_metrics.py`，libsndfile 依內容判格式）
- 輸出：`~/eval_output_nvme/musiccaps_reference_real/`（`per_clip.tsv`）；直接跑（~35 min，HDD 讀取慢），有 notify trap

## 全量

| | PQ | CE | CU | PC | CLAP | LUFS |
|---|---|---|---|---|---|---|
| 真實 MusicCaps（n = 5131） | **6.90** | 6.19 | 6.66 | 5.38 | **0.299** | −18.6 |

- AES-natural 那 522 首（有人類分數）的 AES PQ 是 5.45（與 2026-09-29 逐位一致），其餘 4,609 首 7.06 → Meta 挑的評估子集刻意偏向低品質/分散，**522 首的 5.45 不能代表 MusicCaps 整體**。

## 同一 prompt 逐首配對（seed 14159265 的各 arm，n = 5131）

| arm | PQ | ΔPQ vs 真實 | PQ 勝過真實的 prompt 比例 | ΔCE | ΔCLAP（勝率） | ΔLUFS |
|---|---|---|---|---|---|---|
| nmv2pair control CFG0 | 6.43 | −0.47 | 32% | −0.17 | −0.100（23%） | −0.3 |
| control CFG3+neg | 7.25 | +0.35 | 59% | +0.53 | −0.075（28%） | +2.3 |
| 084 N100 CFG0 | 7.46 | +0.56 | 64% | +0.98 | −0.094（23%） | +0.6 |
| 081 HQ 正向＋LQ 負向 | **8.00** | **+1.10** | **87%** | +0.91 | −0.070（30%） | −0.9 |

在有人類分數的 522 首上：真實錄音人類 PQ 5.68、AES 5.45；081 的 AES PQ 7.94。

## 判讀

1. 未加 guidance 的 CFG0 模型 AES 仍低於真實錄音；**超車完全發生在 PQ 導向的推論/訓練手法之後**（負向 prompt、081 品質標籤、084 負向蒸餾）。這正是 Goodhart 的形狀：我們優化的手段就是在推 AES。
2. 081 在 87% 的 prompt 上 AES PQ 高過同一首真實錄音，響度還比較小聲（−0.9 LU），不是響度假象；但使用者主觀試聽生成音訊明顯不如真實錄音 → 在這個區間 AES PQ 已與感知脫鉤。
3. CLAP 仍把真實錄音排在所有 arm 之上（真實勝 70–77% 的 prompt）——CLAP 量的是文字對齊，不是品質，但至少這個軸沒有被推過天花板。
4. 限制：參考集是 16 kHz 重下載，跟 MeanAudio 同頻寬（沒有 16k 偏誤問題）；但 YouTube 原始錄音本身品質分散（MusicCaps caption 常寫 "low quality recording"），「真實」不等於「高品質」。
