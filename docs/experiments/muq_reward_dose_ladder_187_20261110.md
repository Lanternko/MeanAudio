# 187：MuQ A1 獎勵微調的小劑量階梯（預登錄設計，2026-11-10）

狀態：已排入 p2 187（接在 185 arm R chain、186 評分之後）。本檔在任何 187 的訓練或評分開始前寫定並 commit。
上游設計：`docs/experiments/muq_train_exploit_183_20261108.md`（arm R；本檔不改它，它被 185／186 contract 綁 sha）。

## 0. 為什麼要做

183 的 arm R（1-NFE 可微 MuQ A1 獎勵微調）在 pilot 就把 Jamendo val 的 A1 MI 從 2.86 推到 10.93（λ 0.1、3000 步）。
A1 的訓練標籤是 1–5 分，10.93 是迴歸頭沒有上界造成的尺度外數字。這個結果有一個很容易的反駁：
「把輸出夾在 1–5 就好」。它也沒有回答「在量表內、其他指標都看不出變化的情況下，A1 能不能被推高」。

187 問的是後者：**訓練只多走 10–160 步，A1 的漲幅落在量表內（+0.1～+1 分），這時 PQ／CE／CLAP／FAD／A3a 有沒有跟著動？**
如果 A1 漲了一個有意義的量（≥ m）而其他讀數都在平手帶內，這就是不能用「夾住輸出」擋掉的專攻證據。

pilot λ 0.1 的訓練 log（1-NFE 訓練 prompt 上的 A1 MI）：step 50 3.73、100 4.41、150 4.72、200 5.52、250 6.56；
val step 0 是 2.86。所以量表內的區間大約是前 10–160 步，階梯就放在這裡。

## 1. 共同設定（全部沿用 183 arm R，除了步數與驗證間隔）

| 項目 | 值 |
|---|---|
| 訓練腳本 | `scripts/training/muq_reward_ft_183.py --mode train`（不改，185 contract 綁 sha） |
| 起點 | control S2 `ema_final`：`phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{seed}_stage2_50000` |
| seed | 14159265／16180339／27182818（與 control、185 相同） |
| λ | 0.1（185 pilot 依 Jamendo val 選定，`~/exps_nvme/muq_reward_ft_183/selection.json`；preflight 檢查） |
| LR／batch／B_R／EMA | 1e-5 常數／8／`gate.json` 的 br（4）／關 |
| 獎勵 | 1-NFE CFG0 樣本的 A1 MI，只用 Jamendo 訓練 prompt |
| 步數 | **160**，每 **10** 步驗證並存檔（`--save_each_val`）→ 每個 seed 16 個 checkpoint |
| 驗證 | track-disjoint Jamendo caption 前 **500** 筆（183 用 1,000 筆的前半）、1-NFE、val seed 183、PCM16、lvl30 |
| λ=0 對照 | 同一支腳本 `--lam 0`、160 步、只在 step 160 存檔（arm `Z160`）：同樣的 LR／batch／EMA 關，只少獎勵項 |

LR 是常數、沒有 warmup 之外的排程，所以「160 步 run 的 step N checkpoint」等於「只訓練 N 步」。
同 seed、同 λ 之下，187 的階梯約等於 185 final seed run 的前 160 步（GPU 非決定性，不是逐位相同）。

輸出：訓練 `~/exps_nvme/muq_reward_dose_187/{ladder,zero}/s{seed}/`；評分 `~/eval_output_nvme/muq_reward_dose_187/`。

## 2. 劑量與評分

**MusicCaps 評分的 arm**（全部 CFG0、MeanFlow 25 步、seed 42、Part B 1,000 prompt，與 183／186 同一支評分 harness）：

| arm | checkpoint | 說明 |
|---|---|---|
| `C` | control `ema_final` | 在 187 自己的 root 重生一份（同一個 job、同 batch 組成）；先過 harness 閘（對 181 的 A1，容差 0.02） |
| `D010`／`D020`／`D040`／`D080`／`D160` | 階梯 step 10／20／40／80／160 | 固定檔位，描述整條劑量曲線 |
| `D{主要劑量}` | 見下 | 不在固定檔位裡才另外評 |
| `Z160` | λ=0、160 步 | 區分「獎勵項」與「單純多訓練 160 步」 |

共 21–24 格 × 3 seed 已含在內（7–8 個 arm × 3 seed）。

**主要劑量只用 Jamendo val 決定，並在任何 MusicCaps 生成之前凍結**（`dose_selection.json`）：

- 對每個存檔步數算三個 seed 的 val ΔA1 MI（lvl30，對同一個 run 的 step 0）平均。
- 合格條件沿用 183：每個 seed 的 LUFS 中位數漂移 ≤ 3 LU、靜音比例 ≤ 5%。
- 主要劑量＝合格步數中 **val ΔA1 ≥ +0.5 的最小步數**（0.5 ≈ 0.9 × S_A1；val 起點 2.86，+0.5 仍在量表中段）。
  沒有任何一檔到 +0.5 → 取合格步數中 val ΔA1 最大的一檔。沒有合格步數 → 取 val ΔA1 最大的一檔並標記不合格（判定不能是 `in_range_exploit`）。
- 門檻訂在 val 上而不是 MusicCaps 上：val 是 1-NFE、Jamendo prompt，MusicCaps 是 25 步；轉移比事前不知道（186 的 R-G 還沒出來），
  +0.5 讓轉移比只要 ≥ 0.11 就能在 MusicCaps 上超過 m。

評分順序：C × 3 → harness 閘（不過就停，rc 1）→ 主要劑量 → Z160 → 其餘固定檔位。

## 3. 預登錄判準

尺度沿用 183：S_A1 0.551、S_A3a 0.623、S_PQ 0.914，S_CE／S_CLAP 由 TTM115 五系統全距算（約 1.19／0.071）；m ＝ 0.055（0.10 × S_A1）。
每個劑量都對同 seed 的 `C` 逐 prompt 配對。主要判定只看**主要劑量**；其他檔位是描述。

| 代號 | 條件 |
|---|---|
| **E**（A1 被推高） | ΔA1 MI（lvl30）三 seed 平均 ≥ 0.055，且三個 seed 都 > 0 |
| **E_clamp**（夾住也還在） | 每段 A1 先夾到 [1, 5] 再算差，平均 ≥ 0.055 且三個 seed 都 > 0；同時報 A1 > 5 的片段比例（arm 與 control） |
| **S**（held-out 沒跟著走） | T_A3a < 0.5、T_PQ < 0.5、ΔCLAP < +0.10 × S_CLAP、ΔFAD > −0.30（T_X ＝ (ΔX／S_X)／(ΔA1／S_A1)） |
| **stealth**（其他指標看不出來） | \|ΔPQ\| ≤ 0.10 × S_PQ、\|ΔCE\| ≤ 0.10 × S_CE、\|ΔCLAP\| ≤ 0.10 × S_CLAP、\|ΔFAD\| ≤ 0.30，且主要劑量在 val 上合格 |

FAD 的 0.30 取自 123 的 seed 底線（0.08–0.32）上緣；`Z160` 的 ΔFAD 一併報，當作「多訓練 160 步」本身造成多少 FAD 變化的實測參考。

判定（主要劑量）：

| 判定 | 條件 | 可以寫的話 |
|---|---|---|
| `in_range_exploit` | E、E_clamp、S、stealth 全過 | 在量表內把 A1 推高 ≥ m，其他讀數都在平手帶內；夾住輸出擋不掉 |
| `exploit_with_side_effects` | E、S 過，但 E_clamp 或 stealth 不過 | A1 被專攻推高，但要嘛靠尺度外片段、要嘛其他指標有可測的變化（寫出是哪一項） |
| `general_shift` | E 過、S 不過 | held-out 跟著走，分不出是專攻還是整體變化 |
| `no_effect` | E 不過 | 這個劑量在 MusicCaps 25 步上推不動 A1 |

**與 183 S 規則的差異（刻意）**：183 用 ΔCLAP ≤ 0 與「FAD 沒變好」這兩條符號條件。小劑量下這兩個差值預期在零附近，
符號條件會有一半機率因雜訊不過，所以 187 改成平手帶（CLAP +0.10 × S、FAD −0.30）。183 的原始 S 規則仍然算出來放在 summary（`s_rule_183`），不進判定。

描述性讀數（不進判定）：

- 劑量曲線：每個檔位的 ΔA1、夾住後的 ΔA1、ΔA3a、ΔPQ、ΔCE、ΔCLAP、ΔFAD、LUFS、靜音數、各自的 E／S／stealth。
- ΔA1 對步數的單調性（Spearman）。
- val（1-NFE、Jamendo）對 MusicCaps（25 步）的轉移比，每個檔位一個。
- `D160 − Z160`：獎勵項的淨效果。
- MusicCaps 上過 E 的最小檔位——這是看了 MusicCaps 才知道的，**只能當描述，不能當作選劑量的依據**。

## 4. 執行順序與成本

1. 階梯訓練 × 3 seed：每個 seed 160 步（約 7 分鐘）＋17 次 500 筆驗證（約 32 分鐘）→ 約 2.1 h。
2. λ=0 對照 × 3 seed：約 0.4 h。
3. 凍結主要劑量（`dose_selection.json`）。
4. MusicCaps 評分 21–24 格 × 約 10 分鐘 → 約 3.5–4 h。
5. `summary.json`。

合計約 6.5 h GPU。磁碟：48＋3 個 checkpoint × 0.48 GB ≈ 25 GB（線收掉前保留，未評分的也留著，之後要補檔位不必重訓）。
可暫停：訓練以 run 為單位（有 `DONE` 就跳過，殘缺目錄整個重來），評分以格為單位。

走 queue 的理由：> 30 分鐘。187 只依賴 `gate.json`、`selection.json` 與 control checkpoint，不依賴 185／186 的結果；
排在它們後面只是不搶 GPU。

## 5. 不可寫

- 沒有人類評分，不可寫「MuQ 獎勵較差的音訊」或「聽不出差別」；只能寫「A1 上升而 PQ／CE／CLAP／FAD／A3a 在平手帶內」。
- stealth 的平手帶是 0.10 × 系統間全距，不是聽覺 JND。
- 夾到 [1, 5] 是事後分析，不代表 MuQ 作者的用法。
- val 與 MusicCaps 的評分路徑不同（1-NFE vs 25 步、batch 16 vs 1、GPU resample vs 評分 harness、Jamendo vs MusicCaps prompt），兩邊的數字不能直接相減，只報比例。
- 主要劑量以外的檔位都是描述；不可事後改挑 MusicCaps 上最好看的檔位當主結果。
- 這是對 MeanAudio 自身的推高（within-MeanAudio ΔA1），不是贏過 MusicGen。
- 推論成本與 control 完全相同（同一個模型大小、25 步、沒有 best-of-N、推論期不碰 MuQ）。

## 6. 實作偏差與說明

- **D1　驗證集 500 筆**（183 是 1,000）：每 10 步驗證一次，1,000 筆會讓驗證時間變成訓練的 9 倍。500 筆是同一份 TSV 的前半，逐 prompt 配對同一組 noise，足以排劑量。
- **D2　`log_every 10`**（183 是 50）：只影響 log 密度與每 10 步多算一次梯度比。
- **D3　control 在 187 root 重生**：183／186 root 已有 `C_cfg0`，但 AES 對 batch 組成敏感、且要同一個 job 的 harness 閘，重生一份（約 30 分鐘）。
- **D4　新的 guest**（`scripts/eval/muq_reward_dose_ladder_187_guest.py`）：183 guest 寫完成證據時少了 `eval_only: true`，
  host 會用訓練 job 的 EMA 閘判它，183／184 因此完成後落在 `held/`。183 guest 被 183–186 的 contract 綁 sha 不能改，187 複製一份補上這個欄位（其餘相同）。
- **D5　沒有 GPU smoke**：寫腳本時 185 正在佔用 GPU（剩 8.7 GiB）。只在 CPU 上驗證：import、preflight、每一格的生成指令、checkpoint key 與 control 一致、summary 邏輯（合成資料）。
  訓練與評分的 GPU 路徑都是 183 已經跑過的程式（`muq_reward_ft_183.py`、`muq_train_exploit_183.py` 的 `run_cell`），187 只換步數與 checkpoint 路徑。
- **D6　評分 harness 的重用方式**：187 的 action import `muq_train_exploit_183` 並替換它的輸出 root 與 `cell_spec`（checkpoint 對應）；183 的檔案不改。
