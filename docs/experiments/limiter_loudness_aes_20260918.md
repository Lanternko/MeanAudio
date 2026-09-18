# 064 預註冊：用 limiter 提升 LUFS 後的 AES / CLAP（MusicCaps 5521, MF25, CFG3, fidelity8）

2026-09-18 設計，承接 063（`gain_ladder_aes_20260918.md`）。結果寫在
`docs/experiments/results/limiter_loudness_aes_20260918_results.md`。
腳本：`scripts/eval/limiter_loudness_aes_20260918.py`；產物：
`/home/kojiek/nvme_experiment_artifacts/meanaudio/limiter_loudness_aes_20260918/`。

## 問題

063 只能在 4.7% 有 headroom 的片段上測「比原檔更大聲」，因為生成音檔 peak 中位數 −0.89 dBFS，
純量增益再往上就削波。使用者（2026-09-18）：dBFS 有上限、超過會削波 → 改用 **limiter** 提升 LUFS
（大聲的片段加得少、小聲的加得多），不削波，然後測 AES。

## 設計：每個 limiter arm 都有一個「響度對齊雙胞胎」

limiter 會改變波形（crest 下降），這正是 061 踩過的「處理劣化蓋掉訊號」結構。
所以每個 limited arm `X` 都配一個 `Xm`：**同一條 limited 波形，用純量調回原片段的 integrated LUFS**。

| 對照 | 意義 |
|---|---|
| `X − z0` | 總效果：「用 limiter 變大聲」這件事 |
| `X − Xm` | **純響度效果**：同一波形，只差純量（063 的延伸，但到原檔以上） |
| `Xm − z0` | **limiter 處理效果**：同響度，只差壓縮／crest |

兩部分逐片段精確相加等於總效果。

## Arms（15 × 5,521 = 82,815 筆評分條件）

| arm | 做法 |
|---|---|
| `z0` | 原檔（與 063 `z0` 同一條件，當重現閘門） |
| `L3` `L6` `L9` `L12` | 固定前級增益 +3/+6/+9/+12 dB → limiter |
| `T16` `T14` `T12` | 逐片段增益朝 integrated −16/−14/−12 LUFS → limiter，迭代到 limited 後落在目標 ±0.1 LU（前級上限 +24 dB；比目標大聲的片段會被**衰減**） |
| `L*m` `T*m` | 上述各 arm 的響度對齊雙胞胎（±0.05 LU） |

**Limiter**：lookahead 5 ms、release 50 ms、ceiling **−1 dBTP**。偵測走 4× 過取樣的 true-peak 包絡
（只看 sample peak 時 inter-sample peak 會到 +0.5 dBTP，而 CLAP 會重取樣到 48 kHz）。
增益曲線 = 前向 min filter（lookahead 窗）→ 指數 release → 同長度 box average；
每個樣本的增益 ≤ 該樣本所需增益，所以 ceiling 在構造上不會被超過。
斷言：limited arm sample peak ≤ −1 dBFS 且 true peak ≤ −0.5 dBTP；雙胞胎 sample peak < 0 dBFS。
float32 WAV、16 kHz、暫存（評分完即刪，記錄 content sha256）。

## 指標與閘門

AES 四軸（batch 16，同 063）＋ CLAP（逐檔，見 `reference_clap_batch_size_sensitivity`）。
**CLAP 在 16 kHz 保留音檔上評分，絕對值不可與 48 kHz 表格比，只讀差值**（同 063）。

**重現閘門**：`z0` 的五個分數逐片段必須等於 063 `z0`（|diff| ≤ 1e-4）。不過就是本次有 bug，不讀結論。

**Primary**：`T14`（業界串流標準 −14 LUFS）的 PQ 分解 total / level / processing，bootstrap 95% CI
（seed 20260918、10,000 次、over clips）。
**Secondary**：所有 arm 的 CE/CU/PC/CLAP 分解；L 系列的劑量反應；T 系列依原片段響度
（中位數切半）分成「小聲半」（加很多）／「大聲半」（加很少或被衰減）；每 LU 的純響度斜率。

## 預期（寫下來，事後對照）

- level 部分：依 063，PQ/CU 應隨響度下降、CLAP 上升；CE/PC 在 −22 LUFS 附近已過頂點，應下降。
- processing 部分：未知。這是本實驗真正新增的量——limiter 的 crest 壓縮在**同響度**下是否被 AES 懲罰。
- 若 total 的 PQ 為負：「用 limiter 變大聲」對 PQ 無法當作免費提升。

## 限制

單一 checkpoint、單一 generation seed，沿用 051 baseline 音檔。單一 limiter 設定（5/50 ms），
不代表所有 limiter／mastering chain。T 系列對極安靜片段可能碰到 +24 dB 上限而未達目標，
命中率會回報。AES 與 CLAP 都不是人類判斷。

## 064b 穩健性（2026-09-18 追加，使用者要求確認實作是否算真正的 limiter）

腳本 `scripts/eval/limiter_robustness_aes_20260918.py`，產物 `.../limiter_robustness_aes_20260918/`。
同一設計換成三個開源 limiter，只跑 L6 與 T14（各含響度對齊雙胞胎），外加 ffmpeg loudnorm：

| limiter | 來源 | 偵測 |
|---|---|---|
| `dpl` | x42 dpl.lv2 的 Peaklim（Fons Adriaensen DPL），vendored 於 `scripts/eval/third_party/x42_dpl`，編成離線 CLI | true-peak 模式（在 16 kHz 實測仍可到 +0.44 dBTP，濾波器為 44.1/48k 設計） |
| `alimiter` | ffmpeg 6.1.1，`level=0:latency=1`（預設值會把輸出拉回 0 dBFS 並位移 attack 時間） | sample-peak |
| `hyrax` | Matchering 2.0.6 的 brickwall limiter（只 vendor limiter 本體） | sample-peak |
| `loudnorm` | ffmpeg EBU R128，I=−14、TP=−1、LRA=50 | AGC + true-peak limiter（**不是**純 limiter；會把太大聲的片段往下拉） |

判讀：三個 limiter 與 064 自製版的 processing 部分若同號，064 的結論可寫；不同號則只能寫成「取決於 limiter 實作」。
`z0` 必須與 064 逐片段相同。
