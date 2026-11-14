# 190：MuQ A1 的簡單後製攻擊——看不見的頻帶、看不見的時段、安靜段補噪、分離再相加（預登錄設計，2026-11-12）

狀態：設計登錄；driver 由 sonnet 子代理實作，smoke 通過後排入 p2 190（contract `muq_postproc_attack_190_contract.json`）。
授權：使用者 2026-11-12「Queue 190 post-processing」。上游設計：`muq_ce_cmi_next_stage_20261111.md` 的 190 段；評分底座沿用 189（`muq_overlay_matrix_189.py`）。
分工：規格與審查由 App 端負責，driver／guest 由 sonnet 子代理依本文件實作。

## 0. 為什麼要做

189 說明疊加循環音能把 A1 推高，但需要「另外加一個聲音」，而且 CE 的降幅不比 1 kHz 正弦多，專一性只在 2/9 個底成立。教授（2026-11-10）要的是越簡單、越好解釋的後製手法越好。190 測四種不加新樂音的後製：

1. **F1 8–12 kHz 補噪音**：MuQ 在 24 kHz 下看得到 8–12 kHz，AES 內部是 16 kHz 看不到。問：MuQ 會不會因為「高頻變滿」而加分，同時 AES 完全看不到？
2. **F2 視窗外劣化**：MuQ-Eval 官方只取中間 10 秒。把劣化放在 10 秒以外，MuQ 的輸入逐位元不變。問：其他評分器（看全長的）會掉多少？這是「看不到」型的盲點，不是騙高分。
3. **F4 安靜段補噪（dither）**：把安靜段填上低電平噪音。問：MuQ 會不會把「不安靜」當成加分？
4. **F5 htdemucs 分離再相加**：126c 發現它讓 AES PQ +0.16。問：MuQ 是否也被分離殘差推高？

**F3 反相成分（左右聲道反相）不跑 GPU**：程式稽核確認所有評分器都先把聲道取平均降成單聲道（`newmetrics_common.load_mono`、MuQ-Eval `AudioProcessor`、CMI-RM、PAM 都是 `mean`；AES／CLAP 在我們的 pipeline 只吃單聲道陣列），反相成分在進入任何模型前就被抵銷為 0。結論直接寫「所有評分器依構造看不到，耳機／立體聲播放時人聽得到」，不需要實驗。

## 1. 固定協定

- 底與 189 相同：9 個 MusicCaps 底（`real`、`ma_cfg0_s{3 seed}`、`ma_cfg3neg_s{3 seed}`、`musicgen`、`musicldm`）共用 Part B 1,000 個 prompt id，加 `musdb` 約 551 窗（語料外複製，以曲為 cluster）。底一律先載入成 16 kHz、10 秒、−30 LUFS（189 的 `load_base`）。
- 刺激即時產生、不落地；只寫逐段分數表。MuQ 只評分，不訓練、不引導。
- 每個刺激的隨機噪音以 `(base, clip id, cell)` 決定的種子產生，可重現。
- CLAP 逐檔（batch 1）。主端點 A1 MI，m = 0.055；CE 平手帶 0.10 × S_CE（S_CE 1.1905）；PAM／CMI-RM／SongEval 平手帶沿用 189 的定義。

### F1 高頻補噪（24 kHz 刺激）

| cell | 內容 |
|---|---|
| `up24` | 底以 `resample_poly(3, 2)` 升到 24 kHz（參考，8 kHz 以上幾乎為空） |
| `hf_snr{30,20,10}` | `up24` + 8–12 kHz 帶通高斯噪音（Butterworth 8 階 sos），噪音功率 = 訊號功率 / 10^(SNR/10) |
| `mf_snr{30,20,10}` | 對照：同樣算法、同樣功率，頻帶換成 2–4 kHz |

混好後整段正規化回 −30 LUFS（在 24 kHz 下量）。評分器以 24 kHz 原生輸入：MuQ 直接中心裁 10 s（不再重採樣）、SongEval 直接吃、AES `sample_rate=24000`、PAM／CMI-RM `sr=24000`、CLAP 24k→48k。`up24` 對原 16 kHz 底的差只當描述（重採樣本身的效果）。

### F2 視窗外劣化（20 秒刺激，16 kHz）

| cell | 內容 |
|---|---|
| `ext` | 20 s = clip[5 s:] ⧺ clip ⧺ clip[:5 s]（中間 10 s 就是原 clip；參考） |
| `side_snr{20,10,0}` | `ext` 的頭尾各 5 s 加寬頻白噪（SNR 對該 5 s 段的訊號功率），中間 10 s 不動 |
| `ctr_snr10` | 對照：同樣的噪音只加在中間 10 s，頭尾不動 |

**不做響度再正規化**（再正規化會改到中間 10 s 的振幅，破壞「MuQ 輸入不變」的構造）；超過 ±1 的樣本數記錄並報告。MuQ 用既有中心裁切（`to24` 取中間 240,000 點），所以 `side_*` 的 MuQ 輸入應與 `ext` 逐位元相同——**driver 必須驗證 side 格 ΔA1、ΔA3a 全為 0（|d| < 1e-6），否則判 harness 錯誤**。CLAP（laion_clap 長於 10 s 時隨機截取，不具決定性）在 F2 不評分；AES／PAM（7 s 分塊平均）／CMI-RM（≤30 s）／SongEval 評全長。

### F4 安靜段補噪（16 kHz，10 秒）

- 以 50 ms 不重疊 frame 算 RMS；安靜 frame = RMS < 該 clip frame RMS 中位數 − 20 dB。
- `qd_{-40,-30,-20}`：在安靜 frame 加白噪，噪音 RMS = clip 整段 RMS × 10^(L/20)，L ∈ {−40, −30, −20} dB；每段噪音前後 5 ms 升餘弦淡入淡出。
- `ld_{-40,-30,-20}`：對照：同樣電平、同樣總 frame 數的噪音，加在 RMS 最高的 frame 上。
- 混好後正規化回 −30 LUFS。
- **分析子集**：安靜 frame 合計 ≥ 0.5 s 的 clip，逐底報筆數；子集 < 100 筆的底不進主端點計數（寫明）。全量的分數照算（不足 0.5 s 的 clip 改變很小，只當描述）。

### F5 分離再相加（16 kHz，10 秒）

- `demucs`：底以 `resample_poly` 升到 44.1 kHz、單聲道複製成雙聲道、htdemucs（`htdemucs`，shifts=0、split 預設）分成 4 軌、相加、取平均回單聲道、降回 16 kHz、正規化回 −30 LUFS。對照只有底本身（沒有同類對照聲）。

### 評分層級（沿用 189 的 tier）

| tier | 評分器 | 用在 |
|---|---|---|
| 1 | MuQ A1／A3a（MI、TA）、AES PQ／CE／CU／PC、CLAP | 所有 cell |
| 2 | ＋ PAM、CMI-RM（caption；musdb 用 empty） | 所有 cell |
| 3 | ＋ CMI-RM empty、SongEval 五維 | 中間劑量：`hf_snr20`、`mf_snr20`、`side_snr10`、`ctr_snr10`、`qd_-30`、`ld_-30`、`demucs`，與參考 `up24`、`ext`、底 |

MEva 與 FAD 不跑（190 的問題是 MuQ 與其他評分器的方向是否分岔，PAM／CMI-RM 當獨立守門已足夠；FAD 是分布指標不能逐段配對）。底本身（16 kHz）的分數若 189 的 cache 有完整欄位就直接重用（要驗 id 與欄位齊），否則重評。

## 2. 預登錄判讀

逐段配對差 d = 處理後 − 同一段參考（F1 對 `up24`、F2 對 `ext`、F4／F5 對底）。bootstrap 5,000 次（種子 20261112）；musdb 以曲為 cluster。

1. **加分型（F1、F4、F5）單格判定（v3）**：ΔA1 CI 下界 < m → `no_effect`；否則 ΔCE CI 上界 < 0 → `success`，不然 `not_success_CE_not_down`。另報 ΔA1 CI 上界 ≤ −m → `A1_down`。
2. **扣同族對照（F1、F4）**：同底同劑量 treatment − control 的配對差：A1 多漲 ≥ m（CI 下界）且 CE 多降（CI 上界 < 0）→ `success_specific`；只過一半 → `A1_specific_CE_nonspecific`／`CE_specific_A1_nonspecific`；都不過 → `nonspecific`。F5 沒有同族對照，最高只能判 `success`。
3. **盲點型（F2，以及 F1 若 A1 不動）**：|ΔA1| 的 CI 落在 ±m 內（F2 依構造為 0），且 PAM、CMI-RM caption、AES PQ、AES CE 四個中至少兩個的 CI 上界 < −平手帶 → `blind_spot`；都不動 → `nobody_sees`。另報 `ctr_snr10` 相對 `side_snr10`：MuQ 在中間看到噪音時掉多少（對照 MuQ 的敏感度本身沒問題）。
4. **主端點（各族分開、預登錄中間劑量）**：
   - F1 `hf_snr20`、F4 `qd_-30`、F5 `demucs`：9 個 MusicCaps 底中判 `success_specific`（F5 用 `success`）的個數；≥ 7 且 musdb 的 ΔA1、ΔCE 同號 → **普遍成立**；3–6 → **依底而定**；≤ 2 → **不成立**。
   - F2 `side_snr10`：9 個底中 `blind_spot` 的個數，同樣的三段門檻。
5. **劑量**：每族每底報 ΔA1 過 m 的最小劑量。
6. **獨立守門**：每個加分型成立格，PAM 與 CMI-RM caption 各自報 CI 是否 < 0、是否超出平手帶；兩者都不動 → 寫「獨立評分器看不到」。
7. A3a 跟漲不算失敗（只描述）；AES PQ 不當變差的代理（188/189：循環音讓 PQ 漲）。
8. 預先寫下的預期：F2 MuQ 不動、其他評分器掉 → 盲點；F1 AES 不動（16 kHz）、MuQ 方向未知；F4、F5 方向未知。任何「騙高分」只在第 2 條（扣對照）成立時才寫。

## 3. 成本與停機

20 個 cell × 10 個底。smoke 實測後換算：評分約 13.0 h（CMI-RM 兩種模式占最多）＋ htdemucs 約 1.2 h，合計約 14 h（原估 5–7 h 偏低；沒有刪 tier 或 cell，以免改動預登錄設計）。smoke 顯存峰值 torch reserved 17.8 GiB，driver 等 GPU 空出 20 GiB 才開始。每格分數表寫完即落地，中斷後可從已完成的格續跑。開跑前 NVMe ≥ 20 GB、MemAvailable ≥ 25 GB（oomd，見 memory `reference_oomd_kills_p2_host.md`）。

## 3b. 實作偏差（smoke 後記錄，評 test 前）

- F1：24 kHz 下 12 kHz 是 Nyquist，帶通無效 → `hf` 改成 8 kHz 高通（Butterworth 8 階），實質相同；`mf` 維持 2–4 kHz 帶通。
- F2：噪音蓋 [0, 79,000) 與 [L−79,000, L)（L = 319,200），比滿 5 s 少 1,000 點，留給重採樣 kernel 的餘裕；smoke 驗證 side 格的 MuQ 輸入張量與 `ext` 逐位元相同、ΔMuQ 全 0。
- F5：demucs 在 `karaoke-jp-align` venv 的子行程跑（dac venv 沒有 demucs）；htdemucs shifts=0、split=True、overlap 0.25；四軌在正規化域相加後只反正規化一次。
- 底的分數重評，不重用 189 cache。cell 名用 `qd_m30` 形式取代 `qd_-30`。
- musdb 同號規則與 189 相同：A1 CI 下界 > 0 且 CE CI 上界 < 0；F2 的 musdb 以 `blind_spot` 計。
- smoke 讀數：`up24` 對 16 kHz 底 ΔA1 −0.06～+0.01、ΔPQ −0.09～−0.17（n = 8–16，只當重採樣效果的描述）；F4 安靜子集在 smoke 只有 0–3/16 筆，全量要看 `f4_quiet_frames.json`。

## 4. 不做的事

- 不改 189 的 driver／guest 與其 contract 綁定的檔案（`metric_win_tie_126d.py`、`muq_lowpass_probe_20261028.py`）；190 用 import 重用。
- 不在測試集上用 MuQ 挑選任何參數（劑量是預先定好的網格）。
- 不做人類評分；結論只寫評分器之間的方向。
