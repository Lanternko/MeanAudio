# caption10s multisent NoQ quarter — 設計與判準（2026-08-09 啟動前登記）

> Caption 來源、track/segment/10s-window 粒度，以及後續 AES 2×2 控制設計，統一見
> [`caption_provenance_granularity_and_aes_controls.md`](../../caption_provenance_granularity_and_aes_controls.md)。

## 目的

測試 **caption 風格（多句 + production 描述）能否提升 AES**，同時 CLAP 不下降。

這是 `caption10s` onesent 的 fair-compare twin：相同 id set（251,599）、相同 first-10s crop、
相同 Qwen2.5-Omni-3B captioner、相同 NPZ audio features、相同 NoQ quarter 訓練超參
（S1 100k + S2 50k, lr 1e-4, batch 8, seed 14159265）、相同 MusicCaps MF1 CFG0.5 eval。
**唯一差異是 caption 風格**：multisent（2–5 句、max_new_tokens 160）vs onesent（單句、max 80 + first_sentence）。

## 對照基準

| 實驗 | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| onesent quarter（直接對照） | 0.1734 | 5.7702 | 6.4237 | 5.0037 | 6.3559 |
| onesent full（參考） | 0.1927 | 5.7088 | 6.4092 | 4.9280 | 6.3793 |
| 舊 upstream track-level Qwen NoQ full（僅參考；非 LP-MC、非 10s matched） | 0.1735 | — | — | — | — |

## 機制假設

multisent caption 幾乎必定描述 production quality，onesent 幾乎不描述：

| | 提及 production/quality 詞彙 | 平均命中詞數 | 平均字數 |
|---|---|---|---|
| multisent | 88.7% | 2.77 | 51.9 |
| onesent | 1.4% | 0.01 | 19.1 |

（n=20,000 隨機抽樣，詞表：production quality / recording fidelity / mix / mastered / reverb /
clarity / polished / balanced / distortion / professional 等。）

AES 的 CE/PQ 正是量測這個維度，因此存在明確作用路徑。

## 判準（結果產出前登記，不得事後調整）

| 結論 | 條件 |
|---|---|
| **成功** | CE 或 PQ 提升 ≥ +0.10，且 CLAP ≥ 0.1684（掉幅 ≤ 0.005） |
| **Trade-off** | AES 明顯升但 CLAP 掉 > 0.005 → 需人工裁決，並考慮 full-scale 複驗 |
| **Null** | AES 位移落在 ±0.06 內 → 視為 caption style 對 AES 無效 |
| **失敗** | AES 沒升且 CLAP 掉 |

±0.06 null 帶的依據：onesent 從 quarter 到 full（3× iterations）AES 僅位移 0.015–0.076，
小於此幅度無法歸因於 caption style。

歷史動態範圍佐證 AES 有訊號：MusicCaps 11 個實驗 CE 5.02–6.12、CU 5.99–6.97、
PC 4.53–5.27、PQ 5.92–6.83；`best_results.md` 記錄 caption 策略單獨貢獻在 no-Q 情境下
達 CE +0.454 / PQ +0.311 —— 比訓練量效應大一個數量級。

## 啟動前的已知風險

小樣本 caption-audio alignment（n=1024 paired，同 id 同 audio）：

| | CLAP mean | median |
|---|---|---|
| onesent | 0.2088 | 0.2162 |
| multisent | 0.2012 | 0.2056 |
| Δ | −0.0077 | −0.0091 |

frac_positive 46.0%，bootstrap CI95 [−0.0136, −0.0017] 不含 0 → 小幅下降統計上可靠。
可能是 CLAP 對長文字的稀釋偏差（multisent 51.9 字 vs onesent 19.1 字），也可能是
語義密度真的被 production 描述稀釋；無 ground truth 無法分離。報告：
`outputs/caption10s_pipeline/multisent_vs_onesent_clap_n1024.json`。

## 語料 provenance

| 項目 | SHA-256 |
|---|---|
| corpus | `ab6687142a4ec67c5ab45539268c2bd6f82ae9c332a9c62ba7ed7e242ea94433` |
| train TSV | `eaa35ada59a598f3e86b7d3b37409636cb74d7e5bbfa76a981b3aff83197e90e` |
| chain script | `ef49cf8682646390ebc670866cf01cdc6f77099b17b78b99bc583e29d297e4db` |
| 原始備份（0444） | `432af03b1728d46cb2f90f60150a0200c733ecaa1cf4dfc8e99749a83450cd7f` |
| round-2 前備份（0444） | `1cd26125d5c4728e6c7b63e530dcd1fbfd382031e6d5935e60e119f691151133` |

語料曾因 Qwen-Omni no-EOS bug 汙染（見 memory `reference_qwen_omni_generate_no_eos_trap.md`），
已用 first-entity-line 結構性修復 102,008 列 + EOS 重生 473 列（399 + 74），
擴充 gate 全量掃描 0 缺陷。

## 啟動

2026-08-09 22:32:47，`tmux ms_quarter`，經 `scripts/launch_caption10s_multisent_quarter.sh`。

---

## 結果（2026-08-10 07:12:15 完成）

MusicCaps n=5,521、MeanFlow 1 step、CFG 0.5、`--no_q`。

| 指標 | onesent quarter | **multisent quarter** | Δ | 判定 |
|---|---|---|---|---|
| CLAP | 0.1734 | **0.1916** | **+0.0182** | 不僅沒掉，反而升 |
| CE | 5.7702 | **5.9628** | **+0.1926** | ✅ 超過 +0.10 門檻，遠離 ±0.06 null 帶 |
| CU | 6.4237 | 6.4747 | +0.0510 | 落在 null 帶內 |
| PC | 5.0037 | 5.0493 | +0.0456 | 落在 null 帶內 |
| PQ | 6.3559 | 6.3911 | +0.0352 | 落在 null 帶內 |

**判定：成功**（依啟動前登記的判準：CE +0.1926 ≥ +0.10，且 CLAP 0.1916 ≥ 0.1684）。

model sha256 `e9d22dd545e9b6214a02f72b63a6c1afa287996b463130a5f6816948fc0eadc9`，
報告 `~/logs/phase8_qwen_caption10s_multisent_noq_quarter_FINAL_METRICS.json`。

### 值得注意的三點

1. **CE 是唯一明確位移的 AES 指標**。CU/PC/PQ 的 +0.035~+0.051 都落在事前定義的 ±0.06 null 帶內，
   依自訂規則不可宣稱為效果。所以結論是「multisent 提升 CE」，不是「提升 AES 全項」。

2. **caption 層級的 CLAP alignment 沒有預測到下游 CLAP**。啟動前 n=1024 paired 檢查顯示
   multisent alignment 比 onesent 低 0.0077（CI95 不含 0），據此預期的 trade-off（AES 升、CLAP 降）
   **沒有發生** —— 下游 CLAP 反而 +0.0182。caption-audio 相似度不是下游條件化品質的可靠代理指標。

3. **訓練效率**：multisent quarter（150k 累積 iterations）的 CLAP 0.1916 幾乎追平
   onesent **full**（600k iterations）的 0.1927，而 CE 5.9628 比 onesent full 的 5.7088 高 +0.254。
   用 1/4 訓練量達到同等 CLAP 並超越 CE。

### 限制

- 單次 run，無 seed 重複。±0.06 null 帶是由「quarter→full 訓練量差異」推得，**不是 seed variance 估計**，
  因此 CE +0.19 雖遠高於訓練量效應，仍無法形式上排除 seed 變異。
- AES 量測音訊美學而非文字遵循度；CE 上升可能來自「生成更乾淨的音訊」，也可能來自
  「生成更保守、變化更少的音訊」。要區分需跑 `docs/eval/subjective_prompts.md` 五首主觀樣本。

### 建議後續

1. Full-scale multisent（S1 400k + S2 200k）複驗，對照 onesent full（CLAP 0.1927 / CE 5.7088）
2. 主觀樣本聽測，確認 CE 上升是音質改善而非多樣性塌縮
3. 若要主張 seed-robust，需第二個 seed 的 quarter run
