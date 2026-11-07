# 178：不給原版，MERT 特徵能不能判斷錯音（MIDI 分布內）

設計登錄：2026-11-03，**評分前登錄**。GPU 任務：p2 178。

## 0. 為什麼做

133 的 MERT-ref 是改版對同段原版的對稱距離。「改版較差」的方向是我們指定原版給的，所以它沒證明 MERT 分得出好壞，也不能用在沒有原版的生成音樂上（2026-11-03 更正，見 `results/metric_win_tie_133_results.md`）。

這次問的是更窄、也更必要的前一步：**只給一段音訊、不給原版**，凍結的 MERT 特徵加一個小判斷頭，能不能在 MIDI 鋼琴上判斷有沒有錯音？

- 分布內（同一種 MIDI 渲染）先過，才談 OOD（AI 生成音樂），那是下一份設計。
- 已知反例：SongEval（MuQ 骨幹＋以人類美學評分訓練的頭）在 126d 對所有音符層介入全部失敗。所以這次的頭**直接用錯音標籤訓練**，不用美學分數。

## 1. 刺激

- **來源**：MAESTRO v3 MIDI（`/mnt/seagate/kojiek_data/maestro_v3_midi/maestro-v3.0.0`）。音符讀取（含延音踏板）沿用 `scripts/eval/midi_realpiece_stimuli.py` 的 `read_notes`。
- **切分**：照 MAESTRO 官方 split，官方保證同一首作品不跨 split。
  - train：800 段。
  - val：150 段。
  - test：250 段。
  - 每段取自不同演奏，10 s 窗、音符數 ≥ 20，以固定 seed 選窗。
- **渲染**：
  - tinysoundfont 以 44.1 kHz 渲染，soxr 重採樣到 **16 kHz** 單聲道，存 PCM_24 FLAC。用 16 kHz 是為了對齊之後的 OOD 目標（MeanAudio 輸出）。
  - **音色**：train／val 只用 `generaluser`。test 用 `generaluser` 和沒看過的 `fluidr3` 兩種。
- **響度**：每段正規化到 −23 LUFS（錯音版與原版各自正規化），避免響度差變成線索。

### 版本（每段 12 個）

| 標籤 | 版本 | 說明 |
|---|---|---|
| ok | `clean` | 原樣 |
| ok | `transpose`（±1 半音，隨機） | 單段聽不出，不算錯 |
| ok | `tempo`（±3%，隨機） | 不算錯 |
| ok | `gain_m6`（−6 dB，正規化之前套） | 正規化後應無差，控制組 |
| wrong | `wrong_{1,5,10,30}_{in,out}` | 共 8 個，見下 |

錯音的定義：

- **調性估計**：用整段 MIDI 的音級直方圖對 Krumhansl–Kessler 24 個調性模板，取最相關的一個。
- **out（調外）**：被選中的音移到最近的**非**音階音，距離 ≤ 2 半音。
- **in（調內）**：被選中的音移到最近的**其他**音階音，距離 ≤ 2 半音。原音本來不在音階內時改選下一個音。
- **錯音比例**：
  - `1` 是一個音，選中間 2 s 以外、最靠中央的那個音。
  - `5/10/30` 是隨機抽音符的比例。
- 只改音高，其他不動：時間、力度、音色都不變。

總量約 1,200 段 × 12 版本，再加 test 的第二種音色 250 × 12，約 17,400 段（約 8 GB）。

## 2. 特徵與判斷頭

- **MERT-v1-95M**（pin 133 的 revision）：
  - 16 kHz 重採樣到 24 kHz。
  - 取 **13 層 hidden state**，各自做時間平均，存成 (13, 768) 的特徵。
  - **用哪一層由 val 決定**，test 只看一次。
- **對照組**（必跑）：
  1. **MERT 隨機權重**：同架構，`from_config` 隨機初始化，seed 0，同一套流程。
  2. **AES 四軸**（PQ／CE／CU／PC）：當 4 維特徵。
  3. **音訊規則基線**（不用 MIDI）：
     - `chroma_cqt` 的調外能量比（對最佳 KK 模板的 7 音之外的 chroma 能量占比）。
     - madmom `key_cnn_conf`。
     - 兩者各自當分數，也合在一起當 2 維特徵。
  4. **符號 oracle**（只作天花板參考，不算「贏過對照組」的對象）：直接從 MIDI 算調外音比例。
- **判斷頭**：
  - 標準化後做 L2 logistic regression。C 在 val 上從 {1e-3…1e2} 選。
  - MLP（1 層 256、early stopping on val）當次要。
  - 訓練用全部 ok 對 wrong 版本。

## 3. 評分（test 只跑一次）

- **T1 單段偵測**：對每個錯音強度×類型，算 AUC。正例是該 wrong 版本，負例是全部 4 個 ok 版本。95% CI 對 excerpt 做 bootstrap（2,000 次）。
- **T2 兩段挑原版**：同一段的 (clean, wrong_x) 配對，判斷頭給的 wrong 機率較低者當原版，報正確率。
- **T3 不算錯的不能誤判**：transpose／tempo／gain 版本被判為 wrong 的比例，對比 clean 的誤報率。

## 4. 預登錄判準

- **主判準（MERT 分布內有判斷能力）**：test 上 `generaluser`、`wrong_10` 的 AUC 要過兩條。in／out 合併算，各自也要報。
  1. AUC ≥ 0.80。
  2. 比最強的非 oracle 對照組高 ≥ 0.05，差值的 bootstrap CI 下界 > 0。
- **附帶條件**：
  - T3 的 transpose 與 tempo 誤報率，各自 ≤ clean 誤報率 + 0.10。
  - 不過的話，結論改寫成「判得出改動，判不出錯」。
- **分開報、不進主判準**：
  - in 與 out 各自的 AUC。
  - 如果只有 out 過、in 接近 0.5，結論是「學到調性，不是錯音」。
  - `wrong_1`／`5`／`30`。
  - `fluidr3` 未見音色。
  - T2 正確率。
  - 最佳層。
- **不可寫**：
  - 過了也只能寫「MIDI 鋼琴分布內」。
  - 不可寫成能判斷 AI 生成音樂的錯音；OOD 另立設計。

## 5. 預期（評分前寫下）

- out 會比 in 容易，規則基線在 out 上會很強。
- MERT 的關鍵是 in 能不能明顯高於規則基線。
- 中間層（約 4–8 層）應比最後一層好；133 用的是最後一層。
- 隨機權重 MERT 應接近 0.5～0.6。

## 6. 時間與執行

- **預估**：
  - 渲染約 30–40 分（CPU 平行）。
  - MERT 13 層特徵加隨機權重 MERT 約 30 分。
  - AES 約 15 分，規則基線約 15 分（CPU）。
  - 判斷頭訓練加評分 < 10 分。
  - **GPU 總計約 1–1.5 h**，走 p2 queue（> 30 min）。
- **輸出**：`~/nvme_experiment_artifacts/meanaudio/wrongnote_probe_178/`。
- **結果檔**：`results/wrongnote_probe_mert_178_results.md`、`results/wrongnote_probe_mert_178_summary.json`。

## 7. 收線（2026-11-03，Lanternko 定）

主判準不過（AUC 0.640、比規則只高 0.045），**MERT 錯音守門員線收線**，否證結果寫進指標分析。與 126d（SongEval）、133（MERT-ref）一致：現有神經指標對音符層錯誤沒感覺。

- 不做成對差值版：要有原版，對生成音樂無用，且 133 已顯示 ≤10% 錯音無感。
- 重啟條件：守門員改走「轉譜＋規則」等非 SSL-embedding 路線時，另寫新設計，不沿用本線。
