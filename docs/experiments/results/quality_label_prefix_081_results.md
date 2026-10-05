# 081–083 真品質標籤前綴重訓 結果（2026-09-28）

設計見 `../quality_label_prefix_081_20260926.md`。原始數字：`quality_label_prefix_081_summary.json`（`scripts/analysis/quality_label_081_analysis.py`，083 結尾自動跑，2026-09-28 09:51 本地時間；3 seed × 10 格全齊，`cells_missing` 為空）。

## 一句話

**真品質標籤讓短標籤 prompt 從無效變有效，E1 與 E3 兩個預登錄端點都以數倍門檻通過，三個 seed 同號。**
`Low quality recording.` 放負向槽：lvl30 PQ 差中差 **+1.061**（門檻 0.19）。
`High quality recording.` 放正向槽且**不開 CFG**：**+0.824**（門檻 0.155）。
兩者並用時，arm 比同模型的 fidelity8 高 **+0.37 PQ**，比 control 的 fidelity8 高 0.56。
CLAP 在所有標籤格都不降反升，這是唯一獨立於標籤的讀數。
代價：arm 不加前綴時 CFG0 PQ −0.15。

**標籤洩漏未排除**：分級與端點都是 AES PQ。對外前必須先做五首固定 prompt 試聽，並補 FAD（音檔已刪，要重生）。

## 每格絕對分數（3 seed 平均；5521 首；靜音為 3 seed 加總）

| 模型 | 格 | PQ raw | PQ lvl30 | 各 seed PQ lvl30 | CLAP | CE | CU | PC | LUFS | crest | 靜音 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | cfg0 | 6.475 | 6.700 | 6.65 / 6.73 / 6.72 | 0.1988 | 6.100 | 6.642 | 5.087 | −19.0 | 6.54 | 130 |
| control | cfg3_lqneg | 6.493 | 6.694 | 6.71 / 6.64 / 6.74 | 0.2124 | 5.994 | 6.718 | 4.428 | −17.8 | 6.28 | **340** |
| control | hqpos cfg0 | 6.474 | 6.697 | 6.65 / 6.72 / 6.72 | 0.1974 | 6.113 | 6.646 | 5.073 | −19.1 | 6.60 | 132 |
| control | hqpos cfg3_lqneg | 6.496 | 6.695 | 6.71 / 6.64 / 6.73 | 0.2110 | 6.008 | 6.729 | 4.400 | −17.9 | 6.35 | **342** |
| control | cfg3_neg（fidelity8） | 7.358 | 7.537 | 7.48 / 7.58 / 7.55 | 0.2289 | 6.830 | 7.462 | 4.698 | −17.6 | 6.41 | 231 |
| arm | cfg0 | 6.324 | 6.548 | 6.53 / 6.56 / 6.56 | 0.2002 | 6.034 | 6.494 | 5.171 | −18.7 | 6.45 | 128 |
| arm | cfg3_lqneg | 7.408 | 7.602 | 7.54 / 7.57 / 7.69 | 0.2336 | 6.897 | 7.482 | 4.855 | −16.3 | 5.80 | 149 |
| arm | hqpos cfg0 | 7.179 | 7.369 | 7.41 / 7.27 / 7.42 | 0.2122 | 6.851 | 7.306 | 5.234 | −19.5 | 7.00 | 98 |
| arm | **hqpos cfg3_lqneg** | **7.968** | **8.097** | 8.09 / 8.00 / 8.20 | 0.2293 | **7.233** | **7.937** | 4.775 | −17.9 | 6.77 | 114 |
| arm | cfg3_neg（fidelity8） | 7.551 | 7.732 | 7.79 / 7.68 / 7.73 | 0.2321 | 7.056 | 7.609 | 4.933 | −17.4 | 6.11 | 117 |

粗體靜音 = 超過同模型 cfg0 的 2 倍（報告規則的標記）。只有 control 的兩個 lqneg 格觸發：未訓練的模型被要求「不要低品質」時，部分 clip 以靜音逃逸，與 D1 的觀察一致。arm 沒有任何格觸發。

## 預登錄端點（lvl30 PQ；逐 clip 配對、3 seed 先逐 clip 平均、clip bootstrap 10000 次）

| 端點 | 定義 | ΔPQ lvl30 | 95% CI | 各 seed | ΔPQ raw | ΔCLAP | 判定 |
|---|---|---|---|---|---|---|---|
| **E1（主）** | [arm lqneg − arm cfg0] − [ctrl 同差] | **+1.061** | [+1.044, +1.078] | +0.953 / +1.109 / +1.120 | +1.067 | +0.0198 | **過**（≥ 0.19、CI > 0、3 seed 皆 > 0） |
| E1a | arm lqneg − arm cfg0 | +1.054 | [+1.037, +1.071] | +1.011 / +1.017 / +1.135 | +1.085 | +0.0334 | — |
| E1b | ctrl lqneg − ctrl cfg0 | −0.007 | [−0.021, +0.007] | +0.057 / −0.092 / +0.015 | +0.018 | +0.0136 | raw 與 lvl30 反號 → 報成響度效應，實質為 0 |
| E2 | arm lqneg − arm cfg3_neg | −0.130 | [−0.136, −0.123] | −0.247 / −0.107 / −0.035 | −0.143 | +0.0016 | 未能取代 fidelity8 |
| E2c（參照） | ctrl lqneg − ctrl cfg3_neg | −0.844 | [−0.860, −0.827] | — | −0.864 | −0.0165 | — |
| **E3** | [arm hqpos cfg0 − arm cfg0] − [ctrl 同差] | **+0.824** | [+0.811, +0.838] | +0.887 / +0.722 / +0.864 | +0.857 | +0.0134 | **過**（≥ 0.155） |
| E4 | hqpos＋lqneg 的差中差 | +1.555 | [+1.533, +1.577] | +1.502 / +1.528 / +1.635 | +1.624 | +0.0170 | — |
| E4b | arm hqpos lqneg − arm cfg3_neg | **+0.366** | [+0.355, +0.376] | +0.309 / +0.315 / +0.473 | +0.416 | −0.0027 | 勝過 fidelity8 |
| E5 CFG0 | arm − ctrl（stock cfg0） | −0.152 | [−0.171, −0.134] | −0.119 / −0.173 / −0.166 | −0.152 | +0.0014 | CLAP 非劣性過（界 −0.004） |
| E5 CFG3+neg | arm − ctrl（stock cfg3_neg） | +0.195 | [+0.178, +0.212] | +0.311 / +0.097 / +0.177 | +0.194 | +0.0031 | CLAP 非劣性過（界 −0.0158） |
| 參照 | arm fidelity8 增益 | +1.184 | [+1.166, +1.202] | — | +1.228 | +0.0318 | — |
| 參照 | ctrl fidelity8 增益 | +0.837 | [+0.818, +0.855] | — | +0.882 | +0.0301 | 與 075 control 一致 |

除 E1b 外，raw 與 lvl30 都同號，所以 PQ 的贏面不是響度造成的。
CI 只反映 clip 抽樣；seed 間差距（例如 E1 的 0.95～1.12、E4b 的 0.31～0.47）遠大於 CI 寬度。判定以「3 seed 各自同號且都遠超門檻」為準，不以 CI 為準。

## 讀法

1. **control 對標籤字的 PQ 無反應**：LQ、HQ、兩者並用都停在 6.69～6.70。正向槽 ≈ 0 與 `project_posprompt_slot_asymmetry` 一致。
   **負向槽的 ≈ 0 則與 09-03 消融矛盾，尚未解釋**：當時在 c2p0_slot0（`phase8_qwen_caption10s_multisent_noq_full_stage2_200000`，subset1024，cfg3）上，`low quality, noisy` 拿 +0.971，連 `music` 也有 +0.250、無關文字 +0.357（「任何文字」層約佔 fidelity 的 1/3）；這裡 control 的 `Low quality recording.` 只有 +0.018 raw／−0.007 lvl30。
   **086 已拆開**（`shortneg_2x2_probe_20260928_results.md`）：主因是措辭。`Low quality recording.` 在兩個 checkpoint 上都是最弱的；在 control 上只有 +0.04 lvl30，比無關文字（+0.52）還低。同一 control 用 `low quality, noisy` 有 +0.59。checkpoint 是次要效應（B 比 A 多 +0.27～+0.38）。
   因此 E1 的差中差定義不變，但「未訓練的模型對短標籤沒反應」只對這個措辭成立。E1 的幅度有很大一部分來自 control 恰好落在最弱的措辭上。
2. **arm 學會了標籤**。正向 HQ 不開 CFG 就 +0.82。這是第一個在 **CFG0** 拿到可測 PQ 的介入：075、080、073 都做不到，084 NegMF 也是在追這件事。
3. **fidelity8 在 arm 上也變強**（+1.18 vs +0.84）。fidelity8 字串本身以 `low quality recording` 開頭，吃到了訓練過的標籤。所以 E2 小輸不代表標籤弱；arm 的所有負向用法都比 control 好。
4. **無前綴的代價**：arm stock CFG0 −0.15 PQ，3 seed 同號、CLAP 不變。合理解釋是中間 60% 的無前綴列讓「無標籤」被學成「中等品質」。實用上只要一律加 HQ 前綴，這個代價就被覆蓋（7.37 vs control 6.70）。
5. **CLAP 不降反升**：E1 +0.020、E3 +0.013、E4 +0.017。E4b 對 fidelity8 −0.0027，在 CFG3+neg 的非劣性界 −0.0158 內。
6. **PC 下降**：arm HQ＋LQ 4.78，arm cfg0 5.17。這與 fidelity8 同向（負向 prompt 的通用副作用）。arm hqpos cfg0 的 PC 5.23 反而不降。

## 限制

1. **標籤洩漏（預登錄第 1 條）**：分級用 AES PQ，端點也是 AES PQ。模型可能學到的是「AES PQ 評分器偏好的聲音」，而不是人耳感知的品質。CLAP 同向是間接支持但不是證明。**未過主觀試聽前不可對外。**
2. **沒有 FAD**：每格評完即刪音檔（設計如此）。要補 FAD 必須重生，並明確傳 `--ref_dir /mnt/HDD/kojiek/musiccaps_reference`。negprompt 曾讓 FAD 變差（`project_negprompt_hurts_fad`），這裡不能假設沒有同樣問題。
3. **T5 截斷**：前綴 5 token。前綴列超過 77 token 的比例從 46.7% 升到 54.3%，caption 保留比例 92.0% → 89.7%（全量 100,564 列實測；smoke 400 列是 44% → 56%）。截掉的是尾端，前綴本身不會被截。MusicCaps 加 HQ 前綴後超長比例 36.0% → 41.8%。這是對 arm 不利的代價，但 CLAP 沒有因此落後。
4. **MusicCaps caption 自帶品質描述**（例如 "The file is of poor audio-quality."）。arm 在 cfg0 下的 −0.15 可能有一部分來自這類 prompt 被「讀懂」了。本輪沒拆，要拆可以按 caption 是否含品質字分組讀 per_clip。
5. 單一語料、quarter 預算、單一生成 seed；不外推到 full budget。

## 下一步（未排隊）

- 主觀試聽：五首固定 prompt（`docs/eval/subjective_prompts.md`）。比較 arm hqpos＋lqneg vs arm fidelity8 vs control fidelity8，seed 14159265。`infer.py` 用 `--quality_level 10`。
- 補 FAD：**已排 p2 105（2026-10-06）**，重生 arm 全部 15 格；control 五格 FAD 已由 084／104 算好（`scripts/eval/quality_label_081_fad_backfill.py`）。
- MusicCaps 品質字子集拆讀（限制 4）。
- 若試聽與 FAD 都過：考慮 full budget 單 seed，以及「HQ 前綴＋fidelity8」這一格（本輪沒跑）。

## Checkpoint

`~/exps_nvme/phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s{14159265,16180339,27182818}_stage2_50000/`（S2 ema_final）。
前綴 overlay 在 `/mnt/HDD/kojiek/quality_label_081/overlay_new/`（exFAT 實佔約 99 GB）。重生音檔不需要它，只有續訓或重訓才需要。
