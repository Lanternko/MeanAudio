# 193：MuQ A1 的濾波階梯——低通往下壓、high pass、單一頻段 ±10 dB（預登錄設計，2026-11-13）

狀態：設計登錄；driver 由 sonnet 子代理實作，smoke 通過後排入 p2 193（contract `muq_filter_ladder_193_contract.json`）。
授權：使用者 2026-11-13「排低通／EQ 階梯」。上游方向：`docs/meetings/2026-11-10_muq_attack_simple_explainable.md` 第 1–3、5、6 點。評分底座沿用 189／190（`muq_overlay_matrix_189.py`、`muq_postproc_attack_190.py`）。
編號：191／192 已被 `muq_ce_cmi_next_stage_20261111.md` 預留給 CE／CMI-RM 弱點線，本線用 193。

## 0. 為什麼要做

已知：事後 6 kHz 低通讓 A1 +0.15（179 對 CFG3＋fid8、182 對疊加格），4 kHz 在 179 只有 +0.01。教授 11-10 要三件事：

1. **能不能刷得更高**：截止頻率往下壓（5k／4k／3k／2k／1k）、或做一句話講得完的 EQ（某頻段 +10 dB／−10 dB、high pass）。
2. **要在能聽的音檔上證明**：「好音檔壓掉高頻，分數反而升」。所以主讀數放在真實錄音、MeanAudio CFG3＋neg、MusicGen、MusicLDM 這些底上，CFG0 只當描述。
3. **同一套手法也打 AES PQ**：讀數裡同時報 PQ 是否被推高（CE 降）。

這是「研究者不給梯度、反覆試幾個簡單後製、挑分數最高的」情境。挑選只在 dev 子集做，test 子集只用來確認。

## 1. 固定協定

- 底同 190：9 個 MusicCaps 底（`real`、`ma_cfg0_s{3 seed}`、`ma_cfg3neg_s{3 seed}`、`musicgen`、`musicldm`）共用 Part B 1,000 個 prompt id，加 `musdb` 約 551 窗（以曲為 cluster）。底一律 16 kHz、10 秒、−30 LUFS（189 的 `load_base`）。
- **能聽的底**（主讀數）：`real`、`ma_cfg3neg_s*`（3）、`musicgen`、`musicldm` = 6 個。CFG0 三個底只當描述。
- 濾波在 16 kHz 底上做（Nyquist 8 kHz）；做完正規化回 −30 LUFS（`lvl_sr`），所以響度已鎖。
- 刺激即時產生、不落地；只寫逐段分數表。MuQ 只評分。
- 主端點 A1 MI，m = 0.055；CE 平手帶 0.10 × S_CE（S_CE 1.1905）；PQ 的 m_PQ = 0.10 × S_PQ，S_PQ 的定義與 S_CE 相同（5 個 TTM 系統 clean 平均的全距，從 189 module 取；沒有就用同法算並寫進 summary）。
- dev／test：Part B 1,000 id 以固定種子 20261113 抽 200 個當 dev、800 個當 test；musdb 以曲為單位抽約 20% 的曲當 dev。挑選規則只讀 dev，結果寫進 `selection.json` 後才讀 test 的彙總。

### 格子

| 族 | cell | 內容 |
|---|---|---|
| 參考 | `base` | 底本身 |
| LP | `lp7k`、`lp6k`、`lp5k`、`lp4k`、`lp3k`、`lp2k`、`lp1k` | Butterworth 8 階 `sosfiltfilt` 低通（同 182 的 `lp`） |
| HP | `hp100`、`hp200`、`hp500` | Butterworth 4 階 `sosfiltfilt` 高通 |
| EQ | `eq{125,250,500,1k,2k,4k}_{p10,m10}` | RBJ peaking biquad，中心頻率 f0，Q = √2（約一個八度），增益 ±10 dB；用 `sosfilt`（單向，確保增益就是 ±10 dB，不是 filtfilt 的兩倍） |

共 23 格（含 base）。

### 評分層級（同 190）

| tier | 評分器 | 用在 |
|---|---|---|
| 1 | MuQ A1／A3a（MI、TA）、AES PQ／CE／CU／PC、CLAP | 所有 cell |
| 2 | ＋ PAM、CMI-RM（caption；musdb 用 empty） | 所有 cell |
| 3 | ＋ CMI-RM empty、SongEval 五維 | `base`、`lp6k`、`lp4k`、`lp2k`、`hp200` |

不跑 MEva、FAD（同 190 理由）。

## 2. 預登錄判讀

逐段配對差 d = 處理後 − 同段 `base`。bootstrap 5,000 次（種子 20261113）；musdb 以曲為 cluster。所有判定都在 **test** 上做。

1. **單格判定（v3）**：ΔA1 CI 下界 < m → `no_effect`；否則 ΔCE CI 上界 < 0 → `success`，不然 `not_success_CE_not_down`。另報 ΔA1 CI 上界 ≤ −m → `A1_down`。
2. **挑選（只看 dev）**：
   - LP 族：9 個 MusicCaps 底 dev 平均 ΔA1 最大的截止頻率 → `lp_star`。
   - 一句話族（HP ∪ EQ，共 15 格）：9 個底 dev 平均 ΔA1 最大、且 dev 平均 ΔCE < 0 的格 → `oneliner_star`；沒有格滿足 ΔCE < 0 就取平均 ΔA1 最大者並註明。
   - 同時記下 dev 上 ΔA1 前三名（描述用）。
3. **主端點**：
   - **E1（能聽的底）**：`lp_star` 與 `oneliner_star` 各自在 6 個能聽的底上判 `success` 的個數；≥ 5 → **成立**；2–4 → **依底而定**；≤ 1 → **不成立**。musdb 的 ΔA1 CI 下界 > 0 且 ΔCE CI 上界 < 0 另報（語料外複製）。
   - **E2（刷得比 6 kHz 高）**：若 `lp_star` ≠ `lp6k`，在 6 個能聽的底上算 `lp_star − lp6k` 的配對差，CI 下界 > 0 的底數 ≥ 4 → **比 6 kHz 更高**；否則 → **6 kHz 附近已是頂**。`lp_star` = `lp6k` 時直接寫 6 kHz 是頂。
4. **劑量曲線**：每底報 LP 階梯的 ΔA1、ΔCE、ΔPQ 曲線（7 點），以及 ΔA1 過 m 的最高截止頻率。
5. **獨立守門**：每個 `success` 格，PAM 與 CMI-RM caption 各自報 CI 是否 < 0、是否超出平手帶；兩者都不動 → 寫「獨立評分器看不到」。
6. **PQ 讀數（次要）**：每格每底報 ΔPQ CI 下界 ≥ m_PQ 且 ΔCE CI 上界 < 0 → `pq_success`；計數 6 個能聽的底。教授第 6 點的「同一手法打 PQ」只看這一條。
7. A3a 跟漲不算失敗（只描述，報轉移比 ΔA3a／ΔA1）。

### 預先寫下的預期

- LP：A1 在 6–5 kHz 附近最高，4 kHz 以下回落（179 的 4k ≈ +0.01）；CE 隨截止頻率下降單調降。
- HP：方向未知；EQ：+10 dB 在 125／250 Hz 可能加分（188 的吸引子集中在 200–400 Hz、真實曲 A1 與 200–500 Hz 正相關），4 kHz −10 dB 可能類似低通。
- AES PQ：低通可能不降甚至升（AES 內部 16 kHz、偏好較暗的聲音的證據有限），這是 PQ 讀數要看的。

## 3. 成本與停機

23 格 × 10 底。以 190 的實測速率（tier 2 約 200 s／1,000 段、tier 3 約 300 s）估：18 格 tier 2 × 約 32 min ＋ 5 格 tier 3 × 約 48 min ≈ 13.6 h（原估）；smoke 實測換算約 5.3 h，見 §3b。顯存峰值同 190（約 18 GiB reserved），driver 等 GPU 空出 20 GiB 才開始。每格分數表寫完即落地，中斷後可從已完成的格續跑。開跑前 NVMe ≥ 20 GB、MemAvailable ≥ 25 GB。

## 4. 不做的事

- 不改 189／190 driver 與其 contract 綁定的檔案；193 用 import 重用。
- 不在 test 上用 MuQ 挑選任何參數。
- 不做人類評分；結論只寫評分器之間的方向。「人聽起來沒有變好」只能引用教授的聽感，不當證據。
- 183 L 臂 CFG3＋neg 的 +0.026 不重評；193 用事後濾波直接在 CFG3＋neg 底上回答同一個問題。

## 3b. 實作偏差（smoke 後記錄，評 test 前）

- EQ 的 Q = √2 頻寬較寬：f0 處 ±10.0 dB，但兩個八度外仍有 ±0.26～0.41 dB；LP／HP 用 `sosfiltfilt`，標稱 fc 處是 −6 dB（lp6k 的 −3 dB 點 5.90 kHz、hp200 是 223 Hz）。每格實測響應寫在 `filter_response.json`。
- m_PQ 用 189 harness 的 `S['PQ_lvl30']` = 0.914 → m_PQ 0.0914；由底算出的 S 另存 `S_tie_from_bases`。
- musdb dev 曲以種子 20261114 排序後取約 20%。`selection.json` 寫入後凍結，`validate` 檢查它不晚於 `summary.json`。
- A3a 轉移比在 ΔA1 ≤ 0.01 時記 `None`。
- smoke（16 id、real／ma_cfg3neg_s14159265／musicgen／musdb 8 窗）只驗流程：real lp6k ΔA1 +0.08、lp2k −0.35；選出 lp6k、eq125_p10。時間實測換算約 5.3 h（190 實際評分 8.1 h 含 24 kHz 格與 demucs），顯存本身約 13 GiB。
