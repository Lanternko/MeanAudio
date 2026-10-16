# 評分不變性修補 × 論文結論重判（128）

> **狀態**：2026-10-17 設計（未登錄 queue、腳本與 contract 未寫）。使用者要求「根據 126 結果設計下一個 24 h 實驗」，沿用 126 的預設（m＝0.10 S、通過 80%、不再逐項詢問）。
> 來源：126a／126b summary（`results/metric_win_tie_126{a,b}_summary.json`，2026-10-17 01:18 推上 `codex/meva-promptcc-095` 的 `9d9f44b`）。126 的結果文件尚未寫；本文引用的 126 數字是直接從 summary JSON 讀出來的。

## 1. 為什麼是這個實驗

126 證實了兩件事：

1. **AES 對不相干的改動判不出平手。** 改音量 −3 dB，PQ 平手率只剩 65–90%；開頭補 10 ms 靜音，PQ 平手率只有 49–69%，CLAP 只有 33–38%；極性反轉時 PQ 78–91%、PC 54–66%。
2. **AES 對音樂內容錯誤是盲的。** 錯音 100%，PQ 只有 15% 判變差；一顆音符爆音，PQ 反而有 80% 判變好；−30 LUFS 的 1 kHz 正弦波，PQ 高過 34% 的真實／MIDI 音樂。

第 1 點直接威脅論文 TTM 表 B（lvl30，`ttm_quality_comparison_plan_20261007.md`）的結論。拿 126 的 m 對照：

| 表 B 結論（lvl30，同 prompt 配對） | Δ | 126 的 m（0.10 S） | Δ ／ m |
|---|---|---|---|
| ours CFG3+neg − MusicGen，PQ | +0.155 | 0.092 | 1.7 |
| ours CFG3+neg − MusicGen，CLAP | +0.0105 | 0.0104 | **1.0** |
| ours CFG0 − MusicLDM，PQ | −0.014 | 0.092 | 0.2 |
| NegMF N100 − control，CFG0 PQ（120–122） | +0.956 | 0.092 | 10 |

CLAP 的優勢**剛好等於** 126 定的平手容許量，而 CLAP 只要開頭差 10 ms 靜音就有六成配對判不成平手。
所以下一步不是再找新的缺陷，而是回答：**把評分器最便宜的不變性修好以後，我們已經寫下的結論還成立嗎？**

第 2 點（內容盲）無法靠輸入正規化修補，128 不嘗試修；只在 §5 用 CPU 檢查 PQ 優勢是否集中在「節奏弱、像持續音」的輸出上，因為 126b 顯示這類訊號會被 AES 高估。

## 2. 問題與預登錄預測

- **Q1**：輸入端正規化（響度、時間平移、極性）能不能讓 AES／CLAP 通過 126 的平手要求，同時不損失對訊號劣化的偵測力？
  - 預測：響度正規化讓增益格「依構造」通過；平移集成能把 lead 10–200 ms 的 PQ 平手率拉到 ≥ 80%；兩者合用不會讓 noise／clip 階梯的偵測門檻退步超過一階。
- **Q2**：在通過的評分方案下，表 B 與 NegMF 的結論是「仍成立」「縮成平手」還是「減弱」？
  - 預測：PQ（ours CFG3+neg vs MusicGen）仍成立；**CLAP（同組）縮成平手**；ours CFG0 vs MusicLDM 的 PQ 仍平手；NegMF E1 仍成立。
- **Q3**：126b 的「一顆音符爆音 PQ 反而變好 80%」是不是響度造成的？
  - 預測：響度正規化後，blastnote_x1 的 PQ 變好率降到 50% 以下。若沒降，答反的原因就不是響度，要另記。
- **Q4**（CPU）：ours 對 MusicGen 的 PQ 優勢，是否集中在節奏弱的 prompt 上？
  - 預測：無方向性預測，只登錄判讀規則（§5）。

## 3. 評分方案（只改送進評分器的輸入，不改被評的音訊）

被比較的兩段音訊都經過同一個方案，再算 Δ。

| 方案 | 做法 | 每段評分次數 |
|---|---|---|
| **C0 raw** | 現行做法（126a／115 的 lvl30 輸入原樣送進評分器） | 1 |
| **C1 響度正規化** | 每段先以 integrated LUFS（pyloudnorm，與 `eval_metrics.py` 同設定）正規化到 −30 LUFS，峰值超過 0 dBFS 的段落照實記錄、不 limiting | 1 |
| **C2 平移集成** | 4 個版本取平均：開頭補 0／40／80／120 ms 零、尾端截齊 10 s；極性依序 ＋／−／＋／− | 4 |
| **C3 = C1＋C2** | 先 C1、再 C2 | 4 |

- AES 四軸跑全部方案；CLAP（batch 1，對原 caption）跑 C0 與 C2（C1 對 CLAP 的響度影響另由 C3 讀，避免 CLAP batch 1 的時間翻倍）。
- MEva 不納入通過判定（126 中 MEva 在 −0.5 dB 就只有 6–8% 平手，m 對它太細），只在 C0 與 C3 報描述數字。
- **m 固定用 126a 的 S**（五系統 clean 平均的全距，C0 量），不隨方案重算；另報各方案自己的 S 當敏感度。

## 4. 子實驗與 GPU 預算（合計約 21 h）

吞吐估計來自 126a：AES＋MEva＋CLAP b1＋VGGish 一起約 20 段／s（265 unit × 5,131 段、每 unit 約 260 s）。128 不跑 VGGish，MEva 只跑兩個方案，實際應更快；**開跑前用 50 段 smoke 量吞吐**，依 §4.1 的規則決定子集大小。

### 4.1 128a：方案校準（約 7 h）

- **TTM 部分**：126a 的 5 個系統（real、ours CFG3+neg／CFG0 s14159265、MusicGen、MusicLDM），以 rng(20261017) 從 5,131 個共用 clip 抽 N_a 段，五系統同一批 ID。
  - N_a 取 {1000, 600, 400} 中，按 smoke 吞吐能在 7 h 內跑完的最大值；寫進 summary。
- **變體**（34 個，含 clean，全部沿用 126a 的 `transform()` 與 noise seed）：
  - 該平手（16）：clean_shuf、gain −1／−3／−6／−10／−20／+3／+6、polarity、pcm16、resample44k、lead 10／50／200 ms、mp3 160k／64k。
  - 該分出好壞（17）：noise SNR 40／30／20／10、clip 3／6／12 dB、bitcrush 8／6、clicks 1／5 每秒、dropout 20 ms ×3／×10、blast 80 ms ×1／×3、shuffle 0.25 s、reverse。
- **MIDI 部分**（約 0.3 h）：126b GeneralUser 的 402 段 clean × 11 個變體（blastnote ×1／×3／×10、vel ×0.85、gain −3／−10、lead 10 ms、polarity、wrong100pct、posshuffle100pct）。126b 刺激若已刪，用 `metric_win_tie_126b_stimuli.py` 以原 seed 重建這 11 個變體，並以 clean 段 C0 PQ 與 126b per-clip 逐位比對當身分閘。
- 規模（N_a＝1000）：TTM 1000 × 5 × 34 × 10 次評分 ≈ 1.7M 次 AES、0.85M 次 CLAP。

**判定（每個方案、每個指標 PQ／CE／CU／CLAP 各一次）**：

- **平手要求**：集合 T＝{gain ±1／±3／−6／+6（排除 clip 子集）、polarity、lead 10／50／200、pcm16、resample44k}，五個系統都要平手率 ≥ 80%。mp3 與 gain ≤ −10 只報不判（聽得出差異或已接近底線）。
- **偵測保留**：noise、clip、bitcrush、clicks、dropout 五條階梯，各自的偵測門檻（第一個答對率 ≥ 80% 的幅度）不比 C0 退步超過一階；在門檻幅度上，答反率不比 C0 高 5 個百分點以上。
- 方案**通過**＝兩項都過。**C\*** 定為通過方案中評分次數最少的；同次數時選 C1 優先於 C2。若沒有方案通過，C\*＝C3，並在結果寫「輸入正規化修不好這個指標的不變性」，128b 改成只當敏感度分析。
- Q3（blastnote）：報 C0 與 C1 的 PQ 變好率；依 §2 的預測判讀。

### 4.2 128b：表 B 重判（約 7 h）

- 系統：ours CFG3+neg × 3 seed（114／116／117）、ours CFG0 × 3 seed、MusicGen-medium、MusicLDM、MusicLDM＋fidelity8、真實 MusicCaps，共 10 格，各 5,521 段（真實 5,131）。
- lvl30 音訊用 115 的同一個函式從 raw 現算，不落地。
- 方案：C0（只對 200 段重算，當身分閘：PQ 與 115／118 per-clip 差 ≤ 1e-4，否則停止）、C1、C2、C3，以及 C\*（若不在前述之中）。

**判定（全部在 C\* 下，同 prompt 配對）**：

| 結論 | 仍成立 | 縮成平手 | 減弱 |
|---|---|---|---|
| K1 PQ：ours CFG3+neg − MusicGen | 3 seed 同號，每個 seed \|Δ\| > m，且每個 seed 的逐對「較好」率 > 「較差」率 | 3 seed 平均 \|Δ\| ≤ m | 其他 |
| K2 CLAP：ours CFG3+neg − MusicGen | 同上（m_CLAP＝0.0104） | 同上 | 其他 |
| K3 PQ：ours CFG0 − MusicLDM | 同上 | 同上 | 其他 |

- 另報：每格的 C2 集成內 SD（同一段四個平移版本的分數 SD）當「單段量測雜訊」，以及 \|Δ\| 超過兩倍合併量測雜訊的 prompt 比例。
- 另報：十格系統的 PQ／CLAP 排名在 C0 與 C\* 之間的 Kendall τ。
- **表 B 不在 128 內改寫**。結果出來後，依判定在表 B 加一欄「C\* 下的判定」，原數字保留。

### 4.3 128c：NegMF 重判（約 4.5 h，可被砍）

- 120–122 的 arm CFG0 × 3 seed 與對應 control CFG0 × 3 seed，共 6 格 × 5,521 段，方案同 128b。
- K4（NegMF E1）：判定規則同 K1。
- **音檔可能已刪**（120–122 收線後 mf25 音檔保護已解除）。若缺：用原旗標重生（約 11 min／格）＋112 的 per_clip peak／LUFS 身分閘，生成時間算在本段預算內。
- 若 128a＋128b 實際超過 17 h，128c 整段順延，不壓縮 128b。

### 4.4 時間總表

| 段 | 內容 | 估計 |
|---|---|---|
| smoke | 50 段 × 全部方案量吞吐、身分閘 | 0.3 h |
| 128a | 方案校準（TTM＋MIDI） | 7 h |
| 128b | 表 B 重判（10 格） | 7 h |
| 128c | NegMF 重判（6 格，含可能的重生） | 4.5 h |
| 分析 | summary JSON | 0.2 h |
| 緩衝 | — | 2 h |
| **合計** | | **約 21 h** |

順序：smoke → 128a → 128b → 128c。128b 需要 128a 先定出 C\*，所以不能並行；走 queue（p2，三個 job 串接）。

## 5. 128d：PQ 優勢的節奏分層（CPU，不佔 GPU）

用 115／118 已有的 raw MIR（`mir_per_clip.tsv`）。

- **分層依據用 prompt 端，不用輸出端**：以真實 MusicCaps 參考錄音的 pulse_clarity 把 prompt 分成三等分（約 1,710 個／層；真實缺的 390 個 prompt 不分層）。以輸出自己的 MIR 分層會變成依結果選樣，只當描述報。
- 讀數：每層的 ΔPQ（ours CFG3+neg − MusicGen，3 seed 平均、配對）。
- **預登錄判讀**：若最高 pulse 層的 ΔPQ 小於整體 ΔPQ 的 50%，寫「PQ 優勢集中在節奏弱的 prompt」；若三層都在整體的 50–150% 之間，寫「PQ 優勢與節奏無關」；其他情況只描述。
- 描述性：各系統「像底線」的輸出比例（pulse_clarity < 0.10 且 chroma_entropy 落在 126b 正弦／方波底線的範圍內），以及這些段落的平均 PQ。

## 6. 輸出與保留

- `results/metric_invariance_128a_summary.json`、`..._128b_summary.json`、`..._128c_summary.json`、`..._128d_summary.json`。
- 只存逐段分數；轉換後的音訊不落地（同 126a 的做法，noise seed＝crc32(system/variant/id)，可重生）。
- 128c 若重生音檔，評完照 112 的規則刪除。
- 磁碟：除 128c 的暫存音檔（6 格約 3 GB）外不新增大檔；開跑前照慣例 `df -h` NVMe。

## 7. 限制

- 沒有人類評分。「平手應該平手」只用在聽覺上確定無差的介入（響度、平移、極性、格式）；128 的結論只回答「評分器是否自洽」，不回答「人是否聽得出來」。人類聽測 pilot（`human_listen_real_vs_ours_20261015.md`）仍在等聽者。
- C1 讓增益格依構造通過，它的資訊只在偵測保留、blastnote 和 128b 的結論變化上。
- 外部系統只有一個 run，沒有 seed 變異；K1–K3 的「3 seed 同號」只對 ours 端有意義。
- 內容盲（錯音、節奏錯誤）不在 128 的修補範圍；128d 只是間接檢查。

## 8. 開跑前待辦（主機上做；這個設計是在沒有資料與 GPU 的 cloud session 寫的）

1. 寫 `scripts/eval/metric_invariance_128.py`（`--part a|b|c|d`），`transform()` 與 noise seed 直接 import 126a 的，不複製。
2. 寫 contract（128a／128b／128c 各一個，綁腳本 sha），排 p2 128 → 129 → 130。
3. smoke：50 段 × 全部方案，確認 C0 與 126a／115 逐段一致，並量吞吐、定 N_a。
4. 確認 115／118 的 raw 音檔與 126b 刺激是否還在；不在的照 §4.1／§4.3 處理。
