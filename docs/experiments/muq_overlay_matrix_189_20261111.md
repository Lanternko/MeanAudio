# 189：MuQ A1 疊加攻擊的全矩陣——換配方、換底、換音量後還在不在（預登錄設計，2026-11-11）

狀態：設計登錄；driver 已實作、smoke 通過，2026-11-11 排入 p2 189（contract `muq_overlay_matrix_189_contract.json`）。
授權：使用者 2026-11-11「排 189」。上游設計：`muq_ce_cmi_next_stage_20261111.md` 的 189 段。
分工：規格與審查由 App 端負責，driver／guest 由 sonnet 子代理依本文件實作。

## 0. 為什麼要做

188／188b 的結論是：把小聲的合成循環音疊進音樂，MuQ A1 上升、AES CE 下降，而且扣掉同音量的對照聲（1 kHz 正弦、另一段音樂）後仍在（20 格中 13 格專一成立）。但證據只有：

- 一個循環音配方（dev 200 筆選出的 `P3_sawlp_const`）；
- 兩個底（MusicCaps 真實 500 筆、一個 MeanAudio checkpoint 的 CFG0）；
- 兩種對照聲，沒有「聽起來幾乎一樣但 A1 不該漲」的對照；
- 獨立守門（PAM、CMI-RM caption）只評了 64 筆。

本實驗問一題：**小聲疊入循環音讓 A1 專一上升，換配方、換底、換音量之後還在不在？**

## 1. 固定協定

- 全部刺激 16 kHz、10 秒、混音後正規化到 −30 LUFS（沿用 188 的 `mix`）。刺激即時產生，不落地；只寫分數表。
- MuQ 只評分，不訓練、不引導。配方只在 dev 選，凍結後才評 test。
- CLAP 逐檔（batch 1）。主端點 A1 MI，m = 0.055；CE 平手帶 0.10 × S_CE（S_CE 1.1905）。

### 底（10 個）

| 底 | 內容 | 筆數 |
|---|---|---|
| `real` | MusicCaps 真實錄音 | 1,000 |
| `ma_cfg0_s{14159265,16180339,27182818}` | MeanAudio 三個訓練 seed，CFG0 | 各 1,000 |
| `ma_cfg3neg_s{…}` | 同三個 seed，CFG3＋fidelity8 負向 | 各 1,000 |
| `musicgen` | MusicGen（32 kHz 重採到 16 kHz） | 1,000 |
| `musicldm` | MusicLDM | 1,000 |
| `musdb` | MUSDB18-HQ 混音 150 首 × ≤4 窗（語料外複製） | 約 551 |

MusicCaps 九個底共用 Part B 的 1,000 個 prompt id。

### 疊加聲（7 種）

| 名稱 | 角色 | 內容 |
|---|---|---|
| `osynth` | 處理 | 188 凍結的 `P3_sawlp_const` |
| `osynth2`、`osynth3` | 處理（配方複製） | 不同波形族（鋸齒波族、正弦族）裡，在 dev 上選出的配方 |
| `sine1k` | 對照 | 1 kHz 正弦 |
| `music` | 對照 | 另一段真實音樂（dev 池） |
| `osynth_tr2` | 對照（最嚴） | `osynth` 整體移調 +2 半音：音色、包絡、響度相同，只差音高 |
| `specnoise` | 對照 | 與 `osynth` 長期頻譜相同（1/3 八度平滑）、隨機相位的噪音 |

### 音量

r ∈ {24, 18, 12, 6, 0} LU（疊加聲比底小 r LU）。24 LU 用來找下限。

### dev 選擇（凍結後才碰 test）

每個波形族取 188 Part C 單獨評分 A1 MI 最高的 3 個配方（排除 `P5_`、橋接與噪音刺激），以 r = 6 疊進 188 的 200 筆 dev 真實錄音，選 dev ΔA1 最大者。寫入 `selection.json` 並附 sha256；驗證步驟檢查所有 test 分數表的時間晚於凍結時間。

## 2. 評分器

| 家族 | 評分器 |
|---|---|
| MuQ 骨幹 | MuQ-Eval A1／A3a（MI、TA）、CMI-RM quality（caption／empty）、SongEval 五維 |
| AES | PQ／CE／CU／PC |
| 其他 | PAM、MEva、CLAP、FAD |

MuQ／CMI-RM／SongEval 共用 MuQ 家族骨幹，不是彼此獨立的證據；結果表按家族分欄。

## 3. 預登錄判讀

逐段配對差 d = 疊加後 − 同一段底。bootstrap 5,000 次（種子 20261111）；MUSDB 以曲為 cluster。

1. **單格判定（v3 判準）**：ΔA1 CI 下界 < m → `no_effect`；否則 ΔCE CI 上界 < 0 → `success`，不然 `not_success_CE_not_down`。另報 ΔCE 是否超出平手帶。
2. **扣對照**：同底、同音量下對 `sine1k` 與 `music` 兩個對照都要過。A1 多漲 ≥ m（CI 下界）且 CE 多降（CI 上界 < 0）→ `success_specific`；只過一半 → `A1_specific_CE_nonspecific`／`CE_specific_A1_nonspecific`；都不過 → `nonspecific`。
3. **主端點**：`osynth`、r = 18，九個 MusicCaps 底中 `success_specific` 的個數：
   - ≥ 7 且 MUSDB 的 ΔA1、ΔCE 同號 → **普遍成立**；
   - 3–6 → **依底而定**，報哪些底；
   - ≤ 2 → **只在 188 那個底成立**。
4. **音高專一**：(`osynth` − `osynth_tr2`) 的 ΔA1 CI 下界 ≥ m 才寫「音高專一」。`specnoise` 同法報「不只是頻譜」。
5. **配方複製**：`osynth2`、`osynth3` 在 r = 18 各自報幾個底過第 1 條；兩者都 ≤ 2 個底 → 寫「單一配方」。
6. **劑量**：每個底報 ΔA1 過 m 的最小音量（最大 r）。
7. **獨立守門**：PAM 與 CMI-RM caption 各自報 CI 是否 < 0、是否超出各自的平手帶；兩個都不動 → 寫「獨立評分器看不到」。
8. A3a 跟漲不算失敗（只描述）。

## 4. 不可寫

- 無人類評分：不寫疊加後「變好／變差」，只寫評分器的方向。
- 不寫「MuQ 被攻破」；只寫 A1 在哪些底、哪些音量被專一推高。
- 不把 MuQ 家族內的一致當成獨立複製。
- dev 選擇以外不看 test 分數調任何東西。

## 5. 實作偏差與說明（對上游設計）

- **D1 分層評分**：上游寫「全部評分器全量」，估 20–24 h。為了讓單一 job 在約 13 h 內完成（smoke 外推約 7.5 h：MuQ＋AES＋CLAP 層 1.2 h、加 PAM＋CMI-RM 層 3.6 h、全評分器層 2.3 h、MEva 0.3 h、FAD 0.2 h），慢的評分器分層：
  - 每個底的原版：全部評分器。
  - `osynth`／`sine1k`／`music`：r = 18 與 0 全部評分器；其他音量加 PAM＋CMI-RM caption。
  - 其餘疊加聲：r = 18 與 0 加 PAM＋CMI-RM caption；其他音量只有 MuQ＋AES＋CLAP。
  - MEva：每個底的原版與 `osynth` r = 18。
  - FAD：九個 MusicCaps 底的原版、`osynth` r = 18、`osynth` r = 0。
  主判定（第 1–6 條）只用 MuQ＋AES，全量不受影響；第 7 條在所有處理與對照格都有 PAM＋CMI-RM caption。
- **D2 平手帶**：PAM／CMI-RM／SongEval／MEva 的 S 用本次原版格的五個系統平均的全距（real、三 seed 平均的 MeanAudio CFG0、CFG3＋neg、MusicGen、MusicLDM），m = 0.10 S。A1 與 AES 沿用既有 S。
- **D3 MUSDB 無 caption**：CMI-RM 在 MUSDB 用 empty prompt，CLAP 不算。
- **D4 移調對照**：`osynth_tr2` 的音符整體 +2 半音，低通截止頻率不跟著移（對照只差音高）。
- **D5 `sine1k` 沿用 188 的產生器**（`P5_sine_const`）：與 188／188b 的對照聲相同，不是純 1 kHz 單頻（頻譜峰在 1／3／5／7 kHz）。跨實驗可直接對讀。
- **D6 波形族的定義**：以 188 Part C 刺激名稱的第二段為族名；`sawlp`（`osynth` 所在族）不算鋸齒波族，所以 `osynth2` 一定來自不同族。
- **D7 CMI-RM 與 MuQ-Eval 共用模組名 `muq`**：driver 在兩者之間切換時換掉 `sys.modules['muq']`，分數與各自單獨跑一致（smoke 驗過）。
- **D8 `summary.json` 的額外欄位**：`recipe_replication_r18`（第 5 條）、每個底的 `dose.min_dose_r`（第 6 條）、每格 `independent_unseen` 與主端點的 `n_independent_unseen_of_9`（第 7 條）。主端點「普遍成立」要求 MUSDB 的 ΔA1 CI 下界 > 0 且 ΔCE CI 上界 < 0。
- **D9 依賴未 commit 的工作樹檔案**：`scripts/eval/metric_win_tie_126d.py`（SongEval 評分函式；工作樹有 10-22 的 `_judge_126c` 修正尚未 commit）與 `muq_lowpass_probe_20261028.py`。contract 以排入時的 sha256 綁定。
- **D10 smoke**：`--smoke` 每個底 24 筆（MUSDB 8 筆）、音量只有 18／0、dev 16 筆，輸出在 `muq_overlay_matrix_189_smoke/`；只驗流程與欄位，數字不引用。

## 6. 產出

- 分數表：`~/eval_output_nvme/muq_overlay_matrix_189/`（逐段 TSV、`selection.json`、`summary.json`）。
- 結果文件：`docs/experiments/results/muq_overlay_matrix_189_results.md`。
