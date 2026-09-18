# 065 預註冊：把音量降到底線——純量階梯 ＋ limiter 向下壓（MusicCaps 5521, MF25, CFG3, fidelity8）

2026-09-18 設計，承接 063（`gain_ladder_aes_20260918.md`）與 064（`limiter_loudness_aes_20260918.md`）。
結果寫在 `docs/experiments/results/floor_ladder_aes_20260918_results.md`。
腳本：`scripts/eval/floor_ladder_aes_20260918.py`；產物：
`/home/kojiek/nvme_experiment_artifacts/meanaudio/floor_ladder_aes_20260918/`。

## 問題

使用者（2026-09-18）：聲音一直往下降，不會碰到 dBFS 的下限嗎？→ 那就設計更低音量的實驗，一路降到極限；
然後再用 limiter 的方式壓得更低。

063 只量到 −18 dB，並把 PQ 讀成「越小聲越高（單調）」。**32 片段 pilot 推翻了這個讀法的適用範圍**：
PQ 在 −18～−24 dB 附近到頂，之後一路下降，約 −108 dB 等於靜音分數。所以 063 的「單調」只在 −18～0 dB 成立。

## 三種底線（設計要把它們拆開）

| 底線 | 機制 | pilot 位置（32 片段） |
|---|---|---|
| **評分器底線** | AES：WavLM 第一層卷積無 bias，後接逐通道 GroupNorm（eps 1e-5）→ 本質上對音量不變，**音量效應全來自 eps**；變異數遠小於 eps 時特徵趨近 0，分數收斂到全零輸入的值。CLAP：log-mel 的 amin 1e-10 | PQ/CE/PC 約 −108 dB、CLAP 約 −120 dB 收斂到靜音分數 |
| **格式底線** | 交付用 PCM_16（eval.py 的 `sf.write`）。**libsndfile 在這裡用 floor 量化**：+0.6 LSB → 0、−0.6 LSB → −1 | PQ 從 −36 dB 開始可測、CLAP −72 dB |
| **量表底線** | BS.1770 的 −70 LUFS 絕對門檻 | m18 最安靜片段 −69.4 LUFS，更深就量不到 |

量表底線的解法：**L\***＝把訊號放大 200 dB 再量 LUFS、減回 200（等於把絕對門檻移到 −270）。
絕對門檻是 BS.1770 唯一不隨音量等比移動的步驟，所以純量 arm 的 L\* 精確等於來源 L\* − 衰減量（smoke 誤差 6e-14 LU）。
原本 shift 80 dB 在 m96 仍會擋掉來源 −54 LUFS 以下的區塊（smoke 抓到 0.047 LU 誤差），已改 200。

## 設計

**每個音量都評兩次，同一條波形：**

| 模式 | AES 讀到 | CLAP 讀到 | 意義 |
|---|---|---|---|
| `F` | float32 WAV | float 48 kHz 重取樣，**全程不量化**（`get_audio_embedding_from_data`, use_tensor） | 純音量效應 |
| `Q` | PCM_16（soundfile，同 eval.py） | canonical filelist 路徑 | 音量 ＋ 格式底線 |

`Q − F` 扣掉 z0 的 `Q − F` 就是格式底線的損傷（z0 的 CLAP 本身就有 Q−F 差，見下方「附帶發現」）。

### Arms（62 條件 × 5,521 片段 = 342,302 筆評分）

| arm | 做法 |
|---|---|
| `z0`, `m18`…`m120` | 純量衰減 0, 18, 21, 24, 27, 30, 33, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 108, 120 dB；各 F/Q |
| `D18`…`D84` | **limiter 向下壓**：064 的主 limiter x42-dpl（true-peak, −1 dBTP），輸入先乘前級增益 G、輸出再除以 G ＝ 在 −1−G dBFS 的 ceiling 做 limiting（已驗證 dpl 對音量不變，相對誤差 5e-8）。G 以二分搜尋，使 L\* 恰好下降 g dB（±0.05 LU）。g ∈ {18, 24, 36, 48, 60, 72, 84}；各 F/Q |
| `D*m` | 同一條 limited 波形用純量調回來源 L\*（064 的響度對齊雙胞胎），F |
| `sil` | 全零（評分器底線的參考值） |

`D<g>` 與 `m<g>` 的 L\* 逐片段相同，所以：

| 對照 | 意義 |
|---|---|
| `D<g>.Q − m<g>.Q` | **limiter vs 純量，同響度、交付格式**（limiter 系列的 primary） |
| `D<g>.F − m<g>.F` | 同上但無格式底線＝limiter 在這個音量的處理代價 |
| `(Q−F)` 分別對 D 與 m | limiter 是否真的讓內容離格式底線較遠 |
| `D<g>m.F − z0.F` | limiter 處理代價（在原音量） |

## 閘門（先過才讀結論）

1. `z0.F` 的 AES、`m18.F` 的 AES、`z0.Q` 的 AES＋CLAP 逐片段等於 063（|diff| ≤ 1e-4；同樣本、同 AES 批次分組）
2. 純量 arm 的 L\* − (來源 L\* − g) ≤ 0.01 LU
3. D arm 的目標命中率 ≥ 99%

smoke（前 32 片段）五道全過，前三道誤差 0.0。

## 指標與分析

AES 四軸（batch 16，同 063）＋ CLAP（逐檔）。bootstrap 95% CI（seed 20260918、10,000 次、over clips）。

- **Primary A（形狀）**：F 模式各指標在哪一階到頂（argmax，bootstrap 峰值位置分布）＋相鄰階配對差
- **Primary B（底線位置）**：評分器底線＝最淺的一階，其後所有階與 `sil` 差 ≤ 容忍值；
  格式底線起點＝最淺的一階，其 (Q−F)−(Q−F)\_z0 的 CI 不含 0 且 |mean| ≥ 容忍值。容忍值 AES 0.01、CLAP 0.002
- **Primary C（limiter）**：各 g 的 `D.Q − m.Q`，以及 limiter／純量各自的格式損傷

## 預期（看過 32 片段 pilot 與 smoke，不是盲測）

- PQ（F）峰值在 −18～−24 dB；CU 更深（smoke −33）；CE、PC、CLAP 在 z0 就最高，一路往下
- 評分器底線：AES 約 −108 dB、CLAP 約 −120 dB
- 格式底線：PQ 最早（−36 dB 起可測，−0.01 級），CLAP 約 −72 dB
- limiter：同響度下 PQ 在每個深度都比純量差（smoke：D18 −0.61、D48 −0.14），格式損傷只小一點（D36 −0.006 vs −0.012）
  → **「用 limiter 壓更低來躲底線」預期不划算**：處理代價遠大於省下的格式損傷，直到 −84 dB 兩者都已貼底
- `Q` 模式在 m96 以下的 AES 不是靜音分數（smoke PQ 7.23 vs 靜音 6.73）：floor 量化把負值變成 −1 LSB，
  AES 讀到的是一個 1-bit 的殘影；CLAP 的 filelist 路徑在 48 kHz 再截斷一次，所以看到全零

## 附帶發現（pilot，32 片段）

- canonical CLAP（filelist，48 kHz 截斷式 int16）在 **z0** 就比純 float 低約 0.005～0.008。
  推測是 16 kHz 來源重取樣到 48 kHz 後，8～24 kHz 原本是空的，截斷雜訊落進去改變了 log-mel 的 amin 區。
  所有 arm 共用同一路徑時是常數偏移，不影響 arm 間比較，但 **CLAP 絕對值帶這個偏移**
- AES 對音量的敏感性完全來自 GroupNorm 的 eps：這是 051/063 響度效應的機制來源

## 限制

單一 checkpoint、單一 generation seed，沿用 051 baseline 音檔。單一 limiter（x42-dpl，release 50 ms）。
−36 dB 以下的音量在實際使用中不會出現；這個實驗量的是**評分器與格式的行為**，不是聽感。
AES 與 CLAP 都不是人類判斷。設計參考了 pilot/smoke 的前 32 片段，這 32 片段也在全量 5,521 之內。

## 排程

GPU 被 064（x42-dpl 主跑）和串在它後面的 064b 佔用。065 並行的話 AES 每 16 檔要 2.2 s（被搶），
全量約 20 小時；因此用程序 gate 排在兩者之後（tmux `floor065`），單獨跑預估 3～4 小時。
無 resume checkpoint：每片段一個 JSON，重啟只補缺的片段（批次分組會變，閘門 1 可能因此失敗 → 需從頭跑）。
