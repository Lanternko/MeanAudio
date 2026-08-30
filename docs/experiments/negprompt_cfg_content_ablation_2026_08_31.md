# CFG × negative-prompt 消融矩陣（36 格）

日期：2026-08-31
資料：`/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation/*.json`（每格含逐檔分數）
腳本：`scripts/eval/negprompt_ablation_matrix.py`、`scripts/analysis/negprompt_ablation_summary.py`

## 協定

MusicCaps 1,024 筆 seeded 子集（seed 20260830）、MeanFlow 25 步、NoMask、seed 42、full precision、
CLAP batch 32（與 `novocal_reeval` / `negprompt_reeval` 一致；**不可與逐檔 CLAP 數字比較**）。
每格逐檔配對至同 arm 的 cfg 0 格。

子集代表性：`c2p0_slot0` cfg 0 在子集上 PQ 6.5822 / CLAP 0.2194，全量 5,521 為 6.5793 / 0.2201。

## 本矩陣更正的三項先前結論

### 1. 「約七成增益來自 generic CFG sharpening」— 比例是反的

先前的拆解拿 `T5('')` 當「無 negative」基準，而那不是訓練用的 null
（cosine −0.158，見 `project_negative_prompt_empty_string_trap.md`）。用正確的 stored-null 基準重算：

| 來源 | ΔPQ | 佔比 |
|---|---:|---:|
| guidance 本身（對 stored null，cfg 1.5） | +0.016 | 2% |
| 槽位含離題內容（`irrelevant` − `none`） | +0.167 | 25% |
| **文字指向音訊缺陷（`fidelity` − `irrelevant`）** | **+0.497** | **73%** |

純 CFG 對 PQ 在 cfg 0→3.0 全程不動（6.579~6.598），cfg 4.5 反而低於基準 0.05。
它只推 CLAP（單調 +0.020），是純粹的語意對齊旋鈕。

### 2. 「短版 negative 一樣好且詞彙更乾淨」— 短版有 loudness confound

cfg 1.5、對照 cfg 0 基準（RMS −17.54 dB、重心 1551 Hz）：

| negative | ΔPQ | ΔRMS | Δ重心 | crest_min | 判定 |
|---|---:|---:|---:|---:|---|
| `fidelity`（8 詞） | +0.680 | **+0.15 dB** | −52 Hz | 3.29 | **乾淨** |
| `fidelity_short`（low quality, noisy） | +0.685 | +1.70 dB | +66 Hz | 2.39 | loudness confound |
| `silence` | +0.525 | +2.85 dB | +465 Hz | 2.16 | loudness confound |
| `reversed`（high quality…） | +0.385 | +0.72 dB | +150 Hz | 2.93 | 邊緣 |
| `irrelevant` | +0.183 | −0.42 dB | +28 Hz | 2.79 | 乾淨 |
| `neutral`（music） | +0.112 | −0.06 dB | +131 Hz | 2.58 | 乾淨 |
| `none` | +0.016 | +0.21 dB | +70 Hz | 2.43 | 乾淨 |

Audiobox PQ 在實務上不是 level-invariant。**只有長版 `fidelity` 同時做到大幅 PQ 增益與零 loudness 變化**，
它在 cfg 2.0/3.0/4.5 更是變得比基準安靜（−17.77 / −18.04 / −17.91 dB）。
短版與 `silence` 的分數不能採信為品質增益。

排除 confounded 的兩格後，語意階梯仍成立：`fidelity` 0.680 ≫ `irrelevant` 0.183 > `neutral` 0.112 > `none` 0.016。

### 3. 「cfg 1.5 是合適的操作點」— 低估了 0.39

`c2p0_slot0` ＋`fidelity`：

| cfg | PQ | CLAP | crest_min | RMS |
|---|---:|---:|---:|---:|
| 1.5 | 7.2624 | 0.2499 | 3.29 | −17.39 |
| 2.0 | 7.4986 | 0.2575 | 2.67 | −17.77 |
| **3.0** | **7.6495** | 0.2598 | 2.75 | −18.04 |
| 4.5 | 7.6071 | **0.2627** | 2.90 | −17.91 |

PQ 在 cfg 3.0 見頂（+1.067 vs 基準），CLAP 到 4.5 仍升。全程無飽和。

## negative 文字是波形的穩定劑，不是風險來源

同為 cfg 4.5、`c2p0_slot0`：純 CFG crest_min **1.85**（失真）、＋`fidelity` crest_min **2.90**（健康）。
純 CFG 隨 cfg 上升單調變亮（重心 1551→1860）、變壓縮（crest 2.93→1.85）；
加入實質 negative 文字後兩者都回穩。推測：對「模型自己的 null」做外插是無內容方向的無差別放大，
對具體文字做外插則仍留在合理的音訊流形上。此為推測，未證。

## Prompt-fidelity 代價（metric-gaming 疑慮的精確形式）

`clean_prompt` 與 `lofi_prompt` 的 PQ 差距（越小＝越不服從低保真指令）：

| cell | gap |
|---|---:|
| cfg 0 基準 | +0.335 |
| cfg 1.5 ＋fidelity | +0.311 |
| cfg 2.0 ＋fidelity | +0.232 |
| cfg 3.0 ＋fidelity | **+0.185** |

**gap 縮小 45%** — negative prompt 確實讓模型較不服從「低保真」這類指令。這是真實代價，不是假設。

**條件式協定**（negative 只施於乾淨提示詞，低保真提示詞維持 cfg 0；由逐檔資料零成本合成）：

| cell | ΔPQ（全套） | ΔPQ（條件式） | 保留 |
|---|---:|---:|---:|
| cfg 1.5 ＋fidelity | +0.680 | +0.434 | 64% |
| cfg 3.0 ＋fidelity | +1.067 | **+0.655** | 61% |

條件式版本仍有 +0.655，且不與任何提示詞衝突。**建議論文採用條件式為主結果。**

## Arm 依賴性

| @ cfg 1.5 | c2p0_slot0 | fulltrack |
|---|---:|---:|
| 純 CFG ΔPQ | +0.016 | +0.028 |
| ＋`fidelity` ΔPQ | **+0.680** | +0.088 |
| `reversed` ΔPQ | **+0.385** | **−0.141** |
| 最佳 cfg（＋fidelity） | ≥3.0（4.5 仍安全） | **2.0** |
| cfg 4.5 ＋fidelity crest | 2.90 | **1.21**（嚴重失真）|
| cfg 4.5 ＋fidelity CLAP | 0.2627 | 0.1783（低於自身基準）|

fulltrack 在 canonical cfg 0 的優勢在任何有 guidance 的協定下都反轉，且可用 cfg 範圍窄得多。
`reversed` 在兩 arm 上符號相反，是目前唯一直接的極性訊號。

## 未解 / 設計缺陷

1. **極性未被有效檢驗。** `reversed` 與 `fidelity` 的 T5 masked-mean cosine 為 **0.814** —— 兩者大致同方向，
   不是反方向。T5 對反義詞區辨力弱，因此任何「反向 prompt」都可能只是同方向的弱化版。
   要真正測極性須在 embedding 空間直接操作（例如餵 `−v`），非改文字可達。
2. 內容只在 cfg 1.5 掃描，內容 × cfg 的交互作用未測。
3. 短版在高 cfg 是否仍有 loudness confound 未測。
4. 單一 seeded 子集、單一評估集；未做 paired bootstrap CI。
5. CLAP 為 batch 32，不可與歷史逐檔數字做 exact 比較。
