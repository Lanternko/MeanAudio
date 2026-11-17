# 197 攻擊 × 指標兩張大表（2026-11-17 預登錄）

來源：2026-11-17 會議（`docs/meetings/2026-11-17_attack_matrix_and_fix.md`）第 1、2、3、4、6 點。目的不是找新攻擊，而是把已做過與還沒做的處理放進**同一包資料、同一套判定**，排成兩張表：

- **表 1**：任何音檔都能做的處理（MusicCaps 真實錄音＋TTM 生成音檔）。
- **表 2**：MIDI 合成音樂（音符層）。

欄固定四個：**MuQ A1（MI）、AES PQ、AES CE、CMI-RM quality**。CLAP、PAM、A3a、CU、PC 照常存檔，只進附表。MEva 已判定不穩定，不跑；FAD 不跑（逐段配對判定用不到）。

eval-only，不訓練、不用梯度、不在測試集選配方。所有 cell 的參數都在本檔寫死，**沒有 dev 選擇步驟**。

## 1. 資料

### 表 1 的底（同 189／190／193／196）

9 個 MusicCaps 底 × 同一份 1,000 個 Part B id，鎖 −30 LUFS：`real`、`ma_cfg0_s{14159265,16180339,27182818}`、`ma_cfg3neg_s{同三 seed}`、`musicgen`、`musicldm`。底的波形由 `muq_overlay_matrix_189.py` 的載入函式取得，不重生。

- **主表**用 6 個能聽的底（LISTEN）：`real`、`ma_cfg3neg_s*` × 3、`musicgen`、`musicldm`。
- CFG0 三個底進附表。
- 不跑 MUSDB（表 1 要「同一包」，MUSDB 是另一包；189／190／193 的 MUSDB 讀數仍可在附註引用）。

### 表 2 的素材

- **A 區（真實曲）**：195 Part B 的 MAESTRO 150 段 × 2 SoundFont（generaluser、fluidr3），16 kHz、−30 LUFS，沿用 `ce_weakness_195.py` 的 `b_ids`／合成路徑。
- **B 區（合成和弦進行）**：全新。12 主音 × 大／小調 × 2 八度 × 4 拆法 × 2 SoundFont ＝ 384 段，10 秒、120 BPM、鋼琴（GM program 0）、力度固定 80、−30 LUFS。
  - 進行：大調 I–V–vi–IV，小調 i–VI–III–VII；每和弦 2 拍（1 秒），共 10 個和弦（進行重複 2.5 次）。
  - 八度：`o4`＝根音落在 C4–B4；`o5`＝整段高八度。
  - 拆法（每拍兩個八分音符的格子上）：
    - `block`：三個音同時彈，每拍一次。
    - `up`：do-mi-sol（根-三-五，八分音符上行，循環）。
    - `down`：sol-mi-do。
    - `rep`：do-do-do（只重複根音，八分音符）。

## 2. Cell

逐段配對差 d ＝ 處理後 − 同段 `base`。`base` 在 197 內重評（不借 193 的分數，避免 AES 的 batch 組成漂移）。

### 表 1：197 新評的 cell（9 底 × 1,000 id，tier 2）

除了 `gain_*` 以外，所有處理後都重新鎖回 −30 LUFS（音量不是混淆因子）。

| cell | 一句話配方 | 預期方向 |
|---|---|---|
| `base` | 原檔 | — |
| `gain_p6` | 音量 +6 dB（不重鎖響度；float 存檔不削波） | 不變 |
| `gain_m6` | 音量 −6 dB | 不變 |
| `sil05` | 前面加 0.5 秒靜音，尾端裁掉 0.5 秒維持長度 | 不變 |
| `rb_id` | rubberband 原速原調走一遍（處理底） | 不變 |
| `ps_p1`／`ps_m1` | 移調 ±1 半音（rubberband，長度不變） | 不變或變差 |
| `ps_p2`／`ps_m2` | 移調 ±2 半音 | 不變或變差 |
| `ts_090`／`ts_110` | 變速 0.9／1.1 倍（rubberband，音高不變；長度隨之改變、不補不裁） | 不變或變差 |
| `kick05_r12` | 每 0.5 秒打一下合成大鼓（55 Hz 指數衰減正弦＋起音），比音樂小 12 LU | 變差 |
| `kickrand_r12` | 同樣數量的大鼓，時間隨機（每段固定種子）——對照「規律」 | 變差 |
| `click05_r12` | 每 0.5 秒一個 1 ms 脈衝，小 12 LU | 變差 |
| `sq220_r18` | 疊一個持續的 220 Hz 方波，小 18 LU | 變差 |
| `nburst05_r12` | 每 0.5 秒一段 50 ms 白噪音，小 12 LU（非樂音） | 變差 |
| `pno0` | 疊同響度、調內的鋼琴琶音（196 的 `pno0`，改用 1,000 id） | 變差（蓋掉原曲、CLAP 應降） |

「r 小 N LU」的定義同 189：疊入聲的整段 LUFS ＝ 底 − N，相加後再鎖 −30 LUFS。

### 表 1：由既有逐段分數重整的列（不重評）

同 9 底、同 1,000 id、同評分器、同 −30 LUFS，直接讀 `~/eval_output_nvme/<exp>/per_clip/`，對各自實驗內的 `base` 配對：

| 列 | 來源 | 預期方向 |
|---|---|---|
| 低通 6 kHz／4 kHz／2 kHz | 193 `lp6k`、`lp4k`、`lp2k` | 不變或變差 |
| high pass 200 Hz | 193 `hp200` | 不變或變差 |
| 250 Hz ＋10 dB | 193 | 不變或變差 |
| 中段加白噪 SNR 10 | 190 | 變差 |
| 頭尾（MuQ 視窗外）加白噪 SNR 10 | 190 `side_snr10` | 變差 |
| htdemucs 分離再相加 | 190 F5 | 不變 |
| 疊 1 kHz 正弦 r18 | 189 `sine1k_r18` | 變差 |
| 疊合成循環音 r18 | 189 `osynth_r18` | 變差 |

196 的 `pno0` 只有 800 個 test id，所以在 197 重評（上表）。

### 表 2 A 區（MAESTRO 150 × 2 SoundFont，tier 2，CMI-RM 用空 prompt）

全部在 197 重評（含 195 已做過的），確保同一輪、同一 batch 流程。

| cell | 配方 | 預期方向 |
|---|---|---|
| `base` | 原 MIDI | — |
| `wrong_10`／`wrong_30`／`wrong_100` | 該比例的音符換成隨機音高（±1–6 半音） | 變差 |
| `extra_harm_30`／`extra_rand_30` | 多 30% 音符（和聲內／隨機音高） | harm：無法判斷；rand：變差 |
| `jitter_50` | 每個 onset 加常態抖動（σ 50 ms，截在 ±150 ms） | 變差 |
| `drop_30` | 隨機拿掉 30% 音符 | 變差 |
| `tr_p1`／`tr_m1` | MIDI 移調 ±1 半音（重新合成，無訊號處理） | 不變 |
| `oct_p1` | 整段高八度 | 無法判斷（報讀數） |
| `tempo_090`／`tempo_110` | MIDI 變速（重新合成） | 不變 |
| 換 SoundFont | generaluser `base` 對 fluidr3 `base`（不另設 cell） | 無法判斷 |

移調／變速在 MIDI 層沒有處理雜訊，所以預期是「不變」；與表 1 的 rubberband 版對讀，就能把「內容偏好」和「處理代價」分開。

### 表 2 B 區（384 段，tier 2，CMI-RM 用空 prompt）

沒有 `base`；比較都在段與段之間配對：

- **拆法**：同（主音、調式、八度、SoundFont）下，`up`／`down`／`rep` 各自對 `block`，以及 `up` 對 `rep`、`up` 對 `down`。n＝96 對。
- **八度（調還是音高）**：同（主音、調式、拆法、SoundFont）下 `o5` − `o4`。n＝192 對。
- **調式**：同（主音、八度、拆法、SoundFont）下 小調 − 大調。n＝192 對（進行不同，列為「無法判斷」，只報讀數）。
- **主音**：每個主音的平均（對 12 主音總平均的差），大小調分開畫。附 CE 對主音的全距與「同主音跨八度相關」：若 CE 認的是調，o4／o5 兩條 12 主音曲線應該同形（Pearson r 高）；若認的是絕對音高，o5 曲線會整條平移或變形。

## 3. 評分

沿用 190／193 的評分路徑（`muq_train_exploit_183`／`muq_attractor_probe_188`／`muq_overlay_matrix_189` 的評分器）：

| tier | 評分器 | 用在 |
|---|---|---|
| 2 | MuQ A1／A3a（MI、TA）、AES PQ／CE／CU／PC、CLAP、PAM、CMI-RM（表 1 用 caption；表 2 用空 prompt） | 所有 cell |

CLAP 逐檔。表 2 沒有 caption：CLAP 欄留空。

## 4. 判定（表 1、表 2 共用）

每個（處理 × 指標 × 底）一格：

1. d 的平均、逐段 SD、變高的段數比例、n。
2. bootstrap 5,000 次（種子 20261117）的 95% CI；表 2 A 區以 MAESTRO 演奏為 cluster。
3. 記號：CI 不跨零 `*`；CI 整段在平手帶 m 之外 `**`。m：A1 0.055、PQ 0.0914、CE 0.119、CMI-RM quality 0.0657（沿用 181／195／194）。
4. 對照預期方向給判讀：
   - 預期「變差」：`**` 且為負 →「抓得到」；否則「沒反應」；`**` 且為正 →「**方向錯**」。
   - 預期「不變」：CI 整段在 ±m 之內 →「不變（對）」；`**` 負 →「過敏」；`**` 正 →「**方向錯**」；其餘 →「未定」。
   - 預期「不變或變差」：`**` 正 →「**方向錯**」；其餘 →「可接受」（並註明是不變還是變差）。
   - 「無法判斷」：只報數字與記號。
5. **主表每格**寫：6 個能聽的底的平均 d（各底等權）＋同號且 `**` 的底數 k/6＋判讀。逐底數字進附表。
6. 表 2 不分底；A 區兩個 SoundFont 分開報、再報合併。

主要輸出是矩陣本身，沒有單一的成功／失敗端點。預先寫下的讀法只有兩條：

- **R1（方向錯的格數）**：每個指標在「不變」「不變或變差」列裡有幾格判「方向錯」。
- **R2（盲點格數）**：每個指標在「變差」列裡有幾格判「沒反應」。

依既有結果的預測（先登錄，之後對照）：A1 在低通 6 kHz 與循環音方向錯、視窗外加噪沒反應；CE 在 `pno0` 方向錯、`gain_m6` 過敏（−6 dB 降 0.12～0.21）；PQ 在 `gain_m6` 略升；CMI-RM 在低通方向錯（192 Part 0）。移調、變速、規律大鼓、方波沒有預測。

## 5. 產物

- `~/eval_output_nvme/attack_metric_matrix_197/`：`per_clip/*.tsv`（欄位同 193）、`summary.json`、`REPORT.json`。音檔評完即刪（只留每個 cell 各底前 5 段在 `examples/`，供簡報試聽）。
- `scripts/analysis/attack_metric_matrix_tables.py`（無 GPU）：讀 197＋189／190／193 的 per_clip，輸出 `docs/experiments/results/attack_metric_matrix_197_table1.md`、`_table2.md`、`_summary.json`。可在 197 跑完前先用既有列試跑。

## 6. 成本與開跑條件

- 表 1：15 cell × 9 底 × 1,000 段，tier 2 約 200 s／千段，加上 rubberband 與 I/O，估 9–10 h。
- 表 2：A 區 13 cell（含 `base`）× 300 段、B 區 384 段，估 0.5 h（含 fluidsynth 合成）。
- 合計估 10–11 h，hard cap 16 h。smoke（每底 8 段）外推約 7.5 h，含暖機，實際以 `timing.json` 為準。走 p2 queue（197），可續跑（per_clip 檔存在即跳過該 cell）。
- 開跑前：NVMe ≥ 20 GB、MemAvailable ≥ 25 GB、GPU 空 20 GiB。

## 7. 限制

- 沒有人類評分。「方向錯」是依訊號處理常識定的（教授 11-17 第 6 點），不是聽測結果。
- 表 1 的重整列與 197 新評列出自不同 job；同底同 id 同評分器，但 AES 的 batch 組成漂移約 1e-3，遠小於 m。
- rubberband 的處理底（`rb_id`）若本身就超過 m，移調／變速列改對 `rb_id` 配對再報一次。
- B 區只有鋼琴、兩個 SoundFont、兩種進行；結論限於此範圍。
- `ts_*` 改變長度，MuQ 只取中心 10 秒、CMI-RM 取前 30 秒，視窗內容與 `base` 不完全相同。

## 8. 實作偏差（開跑前登錄，2026-11-17）

- **D1** A 區是 12 個介入＋`base` 共 13 cell（初稿 §6 寫 14 是筆誤）。
- **D2** 重整列「中段加白噪 SNR 10」＝190 的 `ctr_snr10`；它與 `side_snr10` 都對 190 的 20 秒延伸版 `ext` 配對，`demucs` 對 `base` 配對。
- **D5** driver 只寫逐段分數與各 cell 平均；bootstrap、顯著性記號、判讀全在 `scripts/analysis/attack_metric_matrix_tables.py`。
- **D6** `jitter_50` 用常態 σ 50 ms（截 ±3σ），不是均勻 ±50 ms；音符層操作的亂數種子 20261117。
- **D7** 主表每格的 CI：逐 id 重抽，統計量是 6 個能聽的底的等權平均 d；k/6 是各底自己的 CI 判定。
- **D8** B 區 bootstrap 以（主音, 調式）為 cluster。
- **D9** 表 2 沒有 caption：CLAP 記 NaN，`CMI_q` 欄放空 prompt 的分數。
- **D10** MIDI 讀回檢查用 `~/karaoke-jp` venv 的 pretty_midi；合成用 fluidsynth。
- **D13** 189／190／193 的列只重整既有逐段分數，不重評。
- **D14（2026-11-17，評分開始前）**：preflight 的 GPU 空顯存門檻由 20 GiB 降為 17 GiB。理由：同一組評分器、同 chunk 64 在 195／196 的實測峰值是 torch reserved 14.5 GiB、nvidia-smi 約 15.6 GiB；GPU 上另有一支占 13.9 GiB 的訓練，20 GiB 門檻讓 197 無限等待。只改等待條件，不改評分設定（chunk、評分器、刺激）；改動時 197 尚未產生任何評分輸出。
