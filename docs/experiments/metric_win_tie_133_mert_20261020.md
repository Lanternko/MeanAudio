# 133：MERT + MuQ-embed 余弦距離在 126/126b/126c 刺激集兩條要求測試

設計登錄日期：2026-10-20（基於 132 結果確認後最終化）
設計登錄人：Claude Code（派工）
GPU 任務：p2 133（`133_metric_win_tie_mert.sh`）
收線依據：`docs/experiments/results/metric_win_tie_133_summary.json`

---

## 0. 背景與動機

126d（p2 132）在 126c/126b/126a 三個刺激集上以 SongEval（MuQ 骨幹 + 美學訓練頭）測試兩條要求（偵測明顯差距 + 判平相近差距）。預期结果：SongEval 因域外問題（訓練在含人聲整首歌，刺激是 10 秒純器樂）會和 AES/MEva/CLAP 一樣兩條均不過。

**開放問題**：如果去掉美學訓練頭，直接用音樂 SSL 骨幹的 embedding 餘弦相似度（參考式指標），是否能偵測音符層錯誤（AES 完全偵測不到）？

**備選指標（最終由 132 結果決定）**：
- **MERT-v1-95M**（m-a-p/MERT-v1-95M）：HuBERT 架構，在音樂上預訓練，95M 參數，768 維，對音高/節奏有明確學習目標
- **MuQ-embed**（OpenMuQ/MuQ-large-msd-iter）：SongEval 使用的同一 MuQ backbone，1024 維，不加美學頭

**初步校準（設計登錄前，dalla_turca_w00 單一 clip，見 explore notes）**：
| 條件 | MERT dist | 評語 |
|------|-----------|------|
| gain ±1dB | 0.000 | 完全不變 ✓ |
| polarity | 0.002 | 幾乎不變 ✓ |
| tempo ±2% | 0.003 | 極小 ✓ |
| wrong5pct | 0.003 | 極小（不確定） |
| jitter 10ms | 0.006 | 小 |
| wrong10pct | 0.006 | 小但可測 |
| blockshuffle 0.5s | 0.009 | 中 |
| wrong30pct | 0.014 | 中 ✓ |
| pitchshuffle_all | 0.027 | 大 ✓ |
| wrong100pct | 0.059 | 大 ✓ |
| randpitch100pct | 0.060 | 大 ✓ |
| posshuffle100pct | 0.055 | 大 ✓ |
| **transpose±1** | **0.058** | **很大（等於 wrong100pct！） ✗** |

關鍵觀察：MERT 對音高移調極敏感（transpose±1 距離 ≈ wrong100pct），可能在 transpose 的 tie 要求上失敗。

---

## 1. 判定協定

### 判定方式（reference-based，不同於 SongEval 的 no-reference）

對每個 (case, base) 配對：
- **dist(case, base) = 1 - cosim(embed(case), embed(base))**（L2 正規化後的 embedding）
- **d = -dist**（高值 = 相似 = 更好）
- **等價判定**：case 可偵測為「較差」iff dist > m；可判平手 iff dist ≤ m
- 本質上：cosim 閾值 = 1 - m

### 兩種 embedding（對 126b+126c 兩者都算）

1. **mert_cosim**：MERT-v1-95M 平均池化 768 維，余弦相似度
2. **muq_cosim**：MuQ-large 平均池化 1024 維，余弦相似度（只在 126b/126c；126a 時間不夠）

### S 定義（預登錄）

與 126d SongEval S 的 parallel 定義：
- **S_mert** = range(μ_s for s in 5 TTM systems)
  - μ_s = mean(cosim(embed(clean_clip), μ_sys) for clean_clip in system s)
  - μ_sys = L2-norm(mean embed of 5,131 clean clips for system s)
  - 「clean mean cosim」在系統間的差異是對訊號多樣性的自然校準
- **S_muq**：同上但使用 MuQ embedding

### 通過門檻

- m = 0.10 × S（同 126d）
- 通過率 ≥ 80%（同 126d）

---

## 2. 預登錄方向預測（評分前）

### 2A. 應偵測差距（lose）的介入

| 介入 | 預測通過的 dim | 理由 |
|---|---|---|
| wrong30pct（≥30% 音符錯誤） | mert_cosim、muq_cosim | 初步校準 dist≈0.014 >> polarity dist≈0.002 |
| wrong100pct | 兩者 | dist≈0.060 |
| randpitch100pct | 兩者 | dist≈0.060 |
| posshuffle100pct | 兩者 | dist≈0.055 |
| blockshuffle0.5s | mert_cosim（soft） | dist≈0.009，依 m 而定 |
| drums_lag≥100ms（126c） | 可能 mert | 節奏破壞 |
| vocals_detune100c（126c） | 可能 mert | 音高偏差 |

### 2B. 應判平手（tie）的介入

| 介入 | 預期 | 
|---|---|
| gain ±1dB | 兩者均通過（dist=0）|
| polarity | 兩者均通過（dist≈0.002，m 應 > 0.002）|
| PCM_16 | 兩者均通過 |
| tempo ±2% | 可能通過（dist≈0.003）|
| transpose±1 | **預測 FAIL**（dist≈0.058，與 wrong100pct 相同！）|

### 2C. 判定「MERT 優於 SongEval」準則（預登錄）

**MERT 優於 SongEval**（需同時）：
1. mert_cosim 在 126b lose 格（wrong30pct+/randpitch/posshuffle）≥ 1 格通過 ≥ 80%
2. mert_cosim 在 gain ±1dB / polarity 通過率 ≥ 80%

**MERT 不優於 SongEval**：
- 所有 lose 格均 < 80%，OR
- 任何 hard-tie 格（gain/polarity）< 80%

**補充問題（預登錄）**：
- transpose±1 的 mert_cosim 通過率（預測 <50%，因 MERT 對音高敏感）
- muq_cosim vs mert_cosim 差異（同 backbone 不同架構的比較）

---

## 3. 刺激集規格

### Part A-1：126c（已有，36,366 FLACs）
- 讀 `OUT_126C/stimuli/items.json`，取 path + base 欄位
- 計算 cosim(embed(case), embed(base)) for each pair
- 輸出：`{OUT_133}/126c_mert_cosim.tsv`、`{OUT_133}/126c_muq_cosim.tsv`

### Part A-2：126b（已有，87,653 FLACs）
- 同上
- 輸出：`{OUT_133}/126b_mert_cosim.tsv`、`{OUT_133}/126b_muq_cosim.tsv`

### Part A-3：126a（53 × 5 × 5,131 = 1,334,060 clips，on-the-fly）
- **MERT only**（MuQ 太慢，兩者加起來 >7h）
- 預先計算 5 個 TTM 系統 clean embedding 均值 μ_sys（每系統 5,131 clips）
- 對每個 (vname, system) unit：載入 R115 → transform → 24kHz → MERT embed → cosim(e, μ_sys)
- 輸出：`{OUT_133}/126a/{vname}/{system}/mert_cosim.tsv`

### Part A-4：S 計算
- 從 Part A-3 clean mean cosim 計算 S_mert、S_muq
- 存入 `{OUT_133}/s_values_mert.json`

---

## 4. 磁碟與時間預算

**磁碟**：
- 126c/126b cosim TSV：~10 MB（每個 key 兩個 float）
- 126a cosim TSV：~50 MB（1.33M clips × 1 float × 8 bytes）
- 臨時音訊：一次一個 unit，≤ 256 MB，評分後刪除
- 峰值：< 400 MB extra

**時間（RTX 5090，MERT@98 clips/s）**：
- Clean mean 計算（126a, 5 系統, 5131 clips each）：~2,600s ≈ 43 min
- 126c：~6 min（MERT + MuQ）
- 126b：~18 min（MERT + MuQ，兩者 in same batch → ~50 clips/s）
- 126a MERT：~3.8h（1.33M / 98）
- Analysis：~15 min
- **總計：~5.2h（< 7h cap）**

**輸出目錄**：`/home/kojiek/eval_output_nvme/metric_win_tie_133_20261020/`

---

## 5. MERT 安裝細節

**模型**：m-a-p/MERT-v1-95M（94.37M 參數，768 dim，24kHz input）
**已下載**：revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`（361 MB，~15 min download）
**venv**：`~/venvs/dac/`（transformers 4.57.3 已在）
**吞吐量**：~98.6 clips/s（bs=16，24kHz 10s clips，RTX 5090）

---

## 6. 停止規則

- MERT S_mert < 1e-4（系統均值相差極小）：不報告 lose 判定
- 任何 unit 失敗：停止並回報
- NVMe free < 5 GB：停止

---

## 7. 結果判讀規則（預登錄）

- **可寫**：MERT cosim ≥ 80% pass 在某介入 → 通過偵測要求
- **可寫**：MERT pass transpose±1 tie → 額外 surprise（預測為失敗）
- **不可寫**：「MERT 驗證了不變性」（未設計不變性測試集）
- **不可寫**：「MERT 因為訓練在音樂上所以應該比 SongEval 好」——這是假說，結果才是判決

---

## 8. 相關文件

- `docs/experiments/metric_win_tie_126d_songeval_20261020.md` — 上輪 SongEval 設計
- `docs/experiments/results/metric_win_tie_126d_results.md` — 132 結果
- `scripts/eval/metric_win_tie_133.py` — 主驅動腳本
- `docs/experiments/metric_win_tie_133_20261020_contract.json` — 合約
