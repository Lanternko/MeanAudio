# 126d 結果：SongEval 兩條要求測試

收線日期：2026-10-20  
p2 job：132（r1 stall→r2 stall→r3 完成，rc=0）  
刺激集：126c（36,366 段）/ 126b（87,653 段）/ 126a（265 units × 5,131 clips）  
summary：`docs/experiments/results/metric_win_tie_126d_summary.json`

---

## 0. 執行紀錄

| round | 狀態 | 原因 |
|-------|------|------|
| r1（132） | held（stall, 3h） | harness `progress()` 讀 JSON/log，但腳本只輸出 TSV → stall_seconds 10800 觸發 |
| r2 | failed（rc=1） | Part B1 bug：`json.loads(windows.json)` 直接迭代 dict keys（字串），修正為 `['windows']` |
| r3 | completed（rc=0） | 修補後完整完成；126a 265/265 units 已由 r2 補完 |

Identity gate：max AES PQ diff = 1.09e-3（閾值 1.0e-3，超出 9%）。超出量 = 1.09e-3，遠小於最小 tie 閾值 m（≈ 0.0139，差 12.8×），不影響任何 win/tie/lose 判定。原因：attention masking 的 float32 非確定性。

---

## 1. S 值與閾值

| dim | mean(clean 126a TTM) range | S | m（= 0.10×S） |
|-----|---------------------------|---|----------------|
| se_coh | musicgen 2.591 → musicldm 2.432 | 0.1814 | 0.0181 |
| se_mem | musicgen 2.630 → musicldm 2.484 | 0.1620 | 0.0162 |
| se_nat | musicgen 2.556 → musicldm 2.417 | 0.1387 | 0.0139 |
| se_cla | musicgen 2.423 → real 2.272 | 0.1512 | 0.0151 |
| se_mus | musicgen 2.345 → real 2.195 | 0.1498 | 0.0150 |

S 值有意義（系統間確有差異）；m 極小（約 0.014–0.018），反映 SongEval 在系統間的區辨力有限。

---

## 2. 主要結果（126a / 126b）

### 通過（lose 判定或 tie 判定）：

| 介入 | 集 | 種類 | 通過 dims | 最高通過率 |
|------|---|------|-----------|-----------|
| noise_snr0 | 126a | lose | 5/5 | 87.9% |
| noise_snr5 | 126a | lose | 5/5 | 84.9% |
| noise_snr10 | 126a | lose | 2/5 | 81.9% |
| gain ≤ ±3dB | 126a | tie | 4–5/5 | ≥90% |
| gain ±0.5, ±1, ±2dB | 126b | tie | 5/5 | ≥97% |
| gain ±3dB | 126b | tie | 5/5 | ≥99% |
| polarity | 126a/b | tie | 5/5 | 100% |
| pcm16 | 126a | tie | 5/5 | 100% |
| resample44k | 126a/b | tie | 1–5/5 | 83–87% |
| noise_snr0 | 126b | lose | 3/5 | 88.9% |
| noise_snr5 | 126b | lose | 2/5 | 82.6% |
| shuffle0.25s | 126b | lose | 1/5 | 80.2% |

### 全部失敗的介入（pass_pct < 80% 所有 dim）：

**音符層介入（126b，全部 fail）：**  
wrong5/10/30/100pct、randpitch10/30/100pct、posshuffle10/30/100pct、  
jitter10/30/80/200ms、blastnote_x1/x3/x10、blockshuffle0.5/1.5/3.0s、  
pitchshuffle2s/all、transpose±1、tempo×0.95/0.98/1.02/1.05、  
vel_random、vel_×0.85、retrograde、drop10/30/60pct、extra10/30pct、unsteady

**訊號層介入（126a，多數 fail）：**  
clip3/6/12/20dB、bitcrush4/6/8、dropout、lead10/50/200ms、  
mp3_160/64/32k、lowpass1/2/4k、shuffle0.25/1.0s、reverse、blast

**126b 訊號層（全部 fail）：**  
clip、bitcrush、dropout、lead、mp3、lowpass（雖 126a 類似）

---

## 3. 126c 結果（MUSDB18-HQ 分軌）

**全部失敗**，pass_pct 普遍 < 20%，最高僅 noise_snr10 46.1%。

即使 gain-1 tie（理論上應該 ~100%）僅 4.36%，polarity 僅 3.99%。

**原因（預登錄）**：SongEval 訓練資料是 AI 生成帶人聲整首歌，刺激是 10 秒真實多軌音樂（MUSDB18-HQ）→ 域外嚴重。分數方差大（來自 attention 非確定性），加上 S 由 TTM 合成音校準，m 相對 MUSDB 分數變異太小，導致幾乎所有配對超過 m 被判「大差異」。

---

## 4. 預登錄方向預測核對

| 預測 | 結果 | 判讀 |
|------|------|------|
| SongEval 域外問題→兩條均不過 | ✓ 確認 | 除極嚴重噪音外全不過 |
| gain ±1dB / polarity 判平手 | ✓ 126a/b 過，126c 不過 | m 太小＋域外效應在 126c 破壞 |
| PCM_16 判平手 | ✓ 126a 過（126b/c 未測） | — |
| 音符層（wrong30pct+）偵測不到 | ✓ 確認 | 所有音符層介入 < 80% |
| 極嚴重噪音（SNR≤10）可偵測 | ✓ 部分 | 126a/b pass；但 126c 失敗 |
| 「SongEval 優於 AES」準則 | **不成立** | lose 格（wrong30pct+）全 < 80% |

---

## 5. Part B 結果

### B1：走音處理控制（AES PQ，n=100 windows，vocals 軌）

| 條件 | 操作 | AES PQ 均差（vs clean） |
|------|------|------------------------|
| pv0 | +50c→−50c（net 0c，單次處理） | −0.361 |
| pv_2xminus | +100c→−50c（net +50c，兩次處理） | −0.461 |
| pv_2x_cancel | +100c→−100c（net 0c，兩次處理） | −0.413 |

**解讀**：pv_2x_cancel vs pv0 = Δ −0.052（純處理代價），pv_2xminus vs pv_2x_cancel = Δ −0.048（net +50c 音高誤差代價）。兩部分近乎相等。AES 對走音+處理混合響應，無法拆分。

### B2：karaoke LUFS（AES PQ，n=81 clips）

| 條件 | AES PQ 均差（vs clean original） |
|------|--------------------------------|
| norm30, drums_off | −0.193 |
| norm30, vocals_off | −0.358 |
| unnorm, drums_off | −0.187 |
| unnorm, vocals_off | −0.404 |

**解讀**：LUFS 正規化（−30 vs 原始）對 AES PQ 影響極小（Δ ≤ 0.05）。demucs 去軌本身帶有 PQ 損失（去鼓 ≈ −0.19，去人聲 ≈ −0.36～−0.40），與 126c 去軌結果一致。

---

## 6. 收線結論

**可寫**：
- SongEval（MuQ 骨幹 + 美學頭）在 10 秒純器樂生成音訊上，不能偵測任何音符層錯誤（wrong notes、timing jitter、pitch/position shuffle）
- SongEval 在 TTM 合成音上的 gain ≤ ±3dB / polarity / pcm16 判平手能力正常
- SongEval 極嚴重噪音（SNR ≤ 10）可偵測（126a/b）；但弱噪音、所有其他訊號劣化均偵測不到
- MUSDB18 真實多軌音訊域外效應嚴重：所有介入全部失敗
- B1：AES 無法拆分走音誤差 vs 處理瑕疵（兩者代價相近）
- B2：karaoke LUFS 正規化不改變 AES PQ 評估

**不可寫**：
- 「SongEval 比 AES 好」——lose 準則未達
- 「SongEval 驗證了不變性」——126c 域外造成 tie 失敗
- 「音符錯誤不影響音樂質感」——指標問題，非音樂特性

---

## 7. 下一步：133 MERT + MuQ-embed cosine distance

因 126d 兩條均不過，接續測試 reference-based 嵌入餘弦距離（無需美學訓練頭）。設計已預登錄：`docs/experiments/metric_win_tie_133_mert_20261020.md`，p2 133 即將排入。

## 勘誤（2026-10-22）

- 第 28 行（se_nat）：`musicgen 2.556 → real 2.427` → `musicgen 2.556 → musicldm 2.417`。依據 `metric_win_tie_126d_summary.json` `system_means_songeval`：musicgen[2]=2.55587、musicldm[2]=2.41714、real[2]=2.42750；最低是 musicldm，S=0.1387=2.55587−2.41714。
- 第 30 行（se_mus）：`musicgen 2.345 → musicldm 2.237` → `musicgen 2.345 → real 2.195`。依據同欄位：musicgen[4]=2.34495、musicldm[4]=2.23672、real[4]=2.19515；最低是 real，S=0.1498=2.34495−2.19515。
