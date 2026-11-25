# Round-2 benchmark screening：人類天花板與地板（2026-10-27）

預登錄：`docs/experiments/benchmark_screening_r2_20261025.md` §6.2 第 4 點（人類天花板只當參考、不進門檻）與 §7（門檻）。**本檔不改變 round-2 的任何判定、門檻或 summary**；所有數字都是描述性參考。
腳本：`scripts/eval/bench_r2/human_ceiling_r2.py`（新檔；重用 `analyze_r2.py` 的 `corr_stats`／`boot_idx`／`center_by`／`bradley_terry` 與 `bench_datasets.py` 的 manifest／PLAN，不改這兩支）。
抽樣：split-half 每資料集 **2,000 次、seed `20261025`**；報中位數與 percentile 95% 區間。地板＝ρ 0／一致率 0.50；「perm 95%」是把 item 對齊或票的極性打亂後同一統計量的 95th percentile（＝在這個樣本大小下，單靠雜訊可以達到的上界）。

**怎麼讀 split-half 與 SB**：`r_half` 是兩位「半panel」評分之間的相關；`SB = 2r/(1+r)`（Spearman-Brown）把它外推到完整 panel 的 reliability。指標是對**觀測到的人類平均**算相關，古典測量理論下它的上限是 **√SB**，不是 SB 本身；因此「佔天花板比例」用 `指標值 / √SB`。表中兩者都列，避免把 reliability 當成 correlation 直接比。
未做 SB 以外的不確定性修正；每個資料集的 rater 結構不同（ODAQ 26 位固定聽眾、MC522 10 個匿名 slot／clip、SongEval 每首 4 位、SingMOS-Pro 每 clip 5 位、AIME 每 snippet 12 票），細節見各節。

## 1. 對照表（每個 dataset 一列；主要統計量）

| dataset | split-half SB（中位數 [95%]） | √SB | 單一評分者 vs 其餘（中位數 [95%]） | 地板 | perm 95% | 現行門檻 | r2 最佳獨立指標（值） | 佔天花板 |
|---|---|---|---|---|---|---|---|---|
| ODAQ（G6a within-trial, Q1–Q5；對應 pam 0.603） | 0.986 [0.978, 0.993] | 0.993 | 0.868 [0.837, 0.897] | 0 | 0.152 | G6a 0.50 | pam 0.603 | 0.61 |
| MC522（PQ, 10 raters） | 0.896 [0.883, 0.908] | 0.947 | 0.659 [0.627, 0.689] | 0 | 0.073 | G1 0.50 | pam 0.610 | 0.64 |
| SongEval S800（Musicality, 4 annotators） | 0.847 [0.834, 0.860] | 0.921 | 0.686 [0.660, 0.712] | 0 | 0.055 | G1 0.50（無獨立指標） | — n/a | n/a |
| SingMOS-Pro test（MOS） | 0.844 [0.836, 0.852] | 0.919 | 0.707 [0.692, 0.722] | 0 | 0.028 | G1 0.50 | singmos 0.573 | 0.62 |
| AIME（G1：snippet 勝率, 12 votes） | 0.658 [0.632, 0.682] | 0.811 | 0.299 [0.256, 0.342] | 0 | 0.045 | G1 0.50 | pam 0.473 | 0.58 |
| ODAQ（G6a, 16 kHz 規則：AES／SingMOS，26 trials） | 0.984 [0.974, 0.992] | 0.992 | 0.857 [0.824, 0.887] | 0 | 0.152 | 參考 | — n/a | n/a |
| ODAQ（G6a, 24 kHz 規則：MuQ-Eval／CMI-RM／SongEval，30 trials） | 0.986 [0.978, 0.993] | 0.993 | 0.863 [0.832, 0.891] | 0 | 0.157 | 參考 | — n/a | n/a |
| ODAQ（item level, 參考） | 0.989 [0.983, 0.992] | 0.995 | 0.892 [0.872, 0.909] | 0 | 0.103 | — | — n/a | n/a |
| ODAQ（within-trial, all 8 conditions） | 0.987 [0.978, 0.991] | 0.993 | 0.894 [0.880, 0.908] | 0 | 0.116 | — | — n/a | n/a |

pairwise 的 G3 天花板（人類共識能預測單票的上限；BT 配適一半、預測另一半）：

| dataset | BT held-out 準確率（中位數 [95%]） | 地板 | perm 95% | 現行門檻 | r2 最佳獨立指標（值） | 佔天花板 |
|---|---|---|---|---|---|---|
| AIME | 0.685 [0.675, 0.696] | 0.500 | 0.513 | G3 0.55 | pam 0.621 | 0.91 |
| Music Arena | 0.742 [0.715, 0.765] | 0.500 | 0.533 | G3 0.55 | aes 0.584 | 0.79 |

- 「r2 最佳獨立指標」取自 `results/benchmark_screening_r2_20261025_summary.json` 的 `decision`，只考慮 `analyze_r2.INDEP` 標為獨立的指標與 dataset 配對；ODAQ 用 G6a、pairwise 用 G3、其餘用 G1。SongEval 依 §5 沒有獨立指標（B3 重疊），故為 n/a。
- Arena 的 G1 無法算 split-half：3,998 個音檔、2,000 個 battle；扣掉 TIE／BOTH_BAD 後 2,824 個音檔各只有 1 張決定票。
- AIME 每個 unique pair 只有 1 票，沒有 pair 內重複票；它的 vote-level 上限用系統層級 BT（同 Arena），另加 snippet 勝率的 split-half（見 §3）。
- SingMOS × Arena：已完成，見 §4。

## 2. 各資料集（item-level）

### 2.1 B1 ODAQ（MUSHRA；26 位聽眾 × 30 trials × 8 conditions）

G1/G2/G3 不適用，round-2 用 G6a（trial 內 Spearman）。`analyze_r2` 的 G6a **只在 trial 內對 Q1–Q5 排名**（LP3.5／LP7／Ref 不進排名），並用 `lp_visible` 把頻寬看不到的 Q 條件拿掉（LP trials 才有這個限制；16 kHz 模型只剩 Q1，24 kHz 模型剩 Q1–Q3，44 kHz 以上全可見）。人類沒有頻寬限制，所以主列用**全部 Q1–Q5、30/30 trials**（對應最好的 r2 獨立指標 pam，path 44）；另列 16 kHz（AES／SingMOS；4 個 LP trials 只剩 1 個 Q 條件、不足 3 條件，實際 26 trials）與 24 kHz（MuQ-Eval／CMI-RM／SongEval；30 trials）規則下的子集，及 item level／全 8 條件當對照。

| 層級 | split-half r（trial 平均） | SB [95%] | √SB | 單人 vs 其餘 | perm 95% |
|---|---|---|---|---|---|
| **within-trial, Q1–Q5（G6a 主列；30 trials）** | 0.973 | 0.986 | 0.993 | 0.868 [0.837, 0.897] | 0.152 |
| within-trial, 16 kHz 規則（26 trials） | 0.969 | 0.984 | 0.992 | 0.857 [0.824, 0.887] | 0.152 |
| within-trial, 24 kHz 規則（30 trials） | 0.973 | 0.986 | 0.993 | 0.863 [0.832, 0.891] | 0.157 |
| within-trial, all 8 conditions | 0.974 | 0.987 | 0.993 | 0.894 [0.880, 0.908] | 0.116 |
| item level（240 檔，參考；不是 G6a 的單位） | 0.979 | 0.989 | 0.995 | 0.892 [0.872, 0.909] | 0.103 |

trials 數：Q1–Q5＝30、16 kHz 規則＝26、24 kHz 規則＝30、all8＝30。單人 vs 其餘在 within-trial 版是「單一聽眾在該 trial 的可見 Q 條件上排名」對「其餘 25 人的平均排名」，對該子集所有 trial × 26 人取平均。within-trial 的 split-half r 用 trial **平均**（G6a 本身就是 30 個 trial Spearman 的平均；這裡中位數會飽和在 1.0，已改報平均）。

### 2.2 B5 MC522（522 首真實錄音；10 位評分者 × 4 軸）

人類逐評分者分數在 `research/eval/aes_human_ratings/AES_natural_music.jsonl`（`/musiccaps/` 那 549 筆；與 522 個 manifest item 的交集＝522）。manifest 的 `human_PQ` 與這份 10 人平均的最大絕對差＝0.00e+00（同一來源）。這份檔案只給每個 clip 的 10 個分數（沒有跨 clip 可追蹤的評分者 ID），因此無法確認 slot 是否同一人；每個 item 獨立隨機對半（5/5）。

| 軸 | split-half r | SB [95%] | √SB | 單人 vs 其餘 |
|---|---|---|---|---|
| **PQ（primary）** | 0.812 | 0.896 [0.883, 0.908] | 0.947 | 0.659 [0.627, 0.689] |
| PC | 0.722 | 0.839 [0.818, 0.857] | 0.916 | 0.548 |
| CE | 0.760 | 0.863 [0.846, 0.879] | 0.929 | 0.580 |
| CU | 0.782 | 0.878 [0.862, 0.893] | 0.937 | 0.624 |

### 2.3 B3 SongEval（S800；每首 4 位標註者，5 維取 4 位平均）

每首獨立隨機分 2/2；primary＝Musicality。標註者有 4 個不相交的 block（a01–04、a05–08、a09–12、a13–16），所以「單人 vs 其餘」主報每首隨機抽一位的 2,000 次分布，另列同一位標註者在其負責歌曲上的 ρ 中位數當穩健性檢查（per-ID 見 JSON）。

| 維度 | split-half r | SB [95%] | √SB | 單人 vs 其餘 |
|---|---|---|---|---|
| **Musicality（primary）** | 0.735 | 0.847 [0.834, 0.860] | 0.921 | 0.686 [0.660, 0.712] |
| Coherence | 0.705 | 0.827 [0.813, 0.840] | 0.909 | 0.657 |
| Memorability | 0.749 | 0.856 [0.844, 0.868] | 0.925 | 0.700 |
| Clarity | 0.719 | 0.836 [0.822, 0.850] | 0.915 | 0.666 |
| Naturalness | 0.699 | 0.823 [0.807, 0.837] | 0.907 | 0.645 |

### 2.4 B4 SingMOS-Pro test（3,101 clips；每 clip 5 位評審）

rater 用匿名 slot（`judge_id` 跨 clip 不保證同一人），每 clip 獨立隨機、平衡對半（5 位者用 2/2；若 n 為奇數會丟 1 位，本 test split 沒有這種 clip）。n_judge 分布：5 位 × 3101。

- split-half r＝0.730、SB＝0.844 [0.836, 0.852]、**√SB＝0.919**
- 單一評審 vs 其餘＝0.707 [0.692, 0.722]
- perm 95% r＝0.028
- 對照：SingMOS predictor 在 B4 的 G1＝0.573 [0.547, 0.599]，約為 √SB 的 0.62。

## 3. 成對資料集：vote-level 天花板

AIME：7,800 張 Music Quality 票、**7,800 個 unique pair（每 pair 恰 1 票）**；每 snippet 有 12 票（對 12 個不同對手）。所以 pair 內沒有重複票可算一致率；vote-level 上限用「系統層級 BT 配適一半票、預測另一半」（同 Arena 的作法），另用 12 票對半算 snippet 勝率的 split-half（這是 AIME G1 人類 target 的 reliability）。

Music Arena：2,000 battles、1,412 張決定票（A 670／B 742／TIE 155／BOTH_BAD 433；決定率 71%）。每 battle 恰 1 票且沒有重複 pair，因此同樣用系統層級 BT。**BT 上限只利用「系統共識」，不含 prompt 或音檔內容資訊；它是任何指標能達到的系統層級共識上限，不是任何單一 audio-based metric 的硬上限。** TIE／BOTH_BAD 佔 29%，G3 只評決定票，這一點本身限制了 G3 的意義。

| | AIME | Music Arena |
|---|---|---|
| 可用的票 | 7,800（13 systems） | 1,412（15 systems） |
| BT held-out 準確率（中位數 [95%]） | 0.685 [0.675, 0.696] | 0.742 [0.715, 0.765] |
| 地板／perm 95% | 0.500／0.513 | 0.500／0.533 |
| G1 人類 target split-half SB（√SB） | 0.658（0.811） | 不可算（≤1 票／音檔） |
| G1 單票 vs 其餘 | 0.299 [0.256, 0.342] | 不可算 |
| r2 最佳獨立指標 | G3 pam 0.621；G1 pam 0.473 | G3 aes 0.584；G1 aes 0.149 |

## 4. SingMOS × Music Arena（B6；描述性，不進判定）

| 讀數 | 值 |
|---|---|
| G1 Spearman（item = audio vs. 該 audio 的 battle 勝率） | **-0.057 [-0.095, -0.021]**（n_items=2824） |
| G2 系統內 Spearman | -0.061 [-0.098, -0.022] |
| G3 一致率 m=0.10S（S=3.672, m=0.367） | **0.446 [0.430, 0.463]**（n_pairs=1412） |
| G3 forced-choice（m=0） | 0.451 [0.425, 0.479] |
| G3 tie_frac（\|Δ\| < m） | 0.608 |
| 系統排名 Kendall τ（人類 BT vs. SingMOS 系統平均） | 0.048（15 systems） |

**描述性，不進判定。** 預登錄 §148 只讓 SingMOS predictor 在 B4（SingMOS-Pro）進閘門；Arena 這一格永遠不計入 round-2 判定，也不改 round-2 的「未定」。


## 5. 讀法與限制

1. **天花板不是門檻**：G1 0.50／G3 0.55 是預登錄門檻；本檔只回答「人類評分本身有多穩定」，不改變任何判定。
2. **√SB 才是可比的相關上限**：SB 是完整 panel 的 reliability。若把 SB 直接和指標的 ρ 比，會把上限高估（例：MC522 PQ SB=0.896，但觀測平均的可預測上限是 √SB=0.947）。表中「佔天花板」一律用 √SB（G3 用 BT held-out 準確率）。
3. **split-half 的 rater 結構不同**：ODAQ 用固定的 26 人 panel 對半（同一群人跨 trial）；MC522／SingMOS 只有匿名 slot、無法跨 item 追蹤同一人，所以每個 item 獨立隨機對半；SongEval 每首 4 位、每首獨立分半。這些都是標準 split-half 的合理延伸，但數字不能跨資料集當成同一種「rater 一致性」直接排序。
4. **AIME／Arena 沒有重複 pair**：兩者的 vote-level 上限是系統層級 BT 的 held-out 準確率。它忽略 prompt／音檔內容，所以對「看得到音檔的指標」是一個偏保守（只給系統共識）的上限；反過來，它也可能是任何只依賴系統身份的模型的樂觀值。
5. **Arena 的 TIE／BOTH_BAD**：29% 的 battle 不是 A/B 二選一，G3 的 0.55 只在決定票上有意義；人類自己在這 29% 上沒有可用的 forced-choice 答案。
6. **SingMOS 的域外警告**：預登錄 §148 只讓 SingMOS predictor 在 B4 進閘門。Arena 這格只描述，不能拿來推翻或支持「SingMOS 可用於純器樂」。
7. **沒有針對我們 arm 的人類評分**：本檔全部是公開 benchmark 的人類評分，與 round-1／round-2 的「未定」判定無關。

## 6. 重現

```bash
cd /home/kojiek/MeanAudio
~/venvs/dac/bin/python -I scripts/eval/bench_r2/human_ceiling_r2.py \
  --out-md docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.md \
  --out-json docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.json
# SingMOS x Arena（job 152）完成後同一條指令重跑即會補上 §4
```

輸出 JSON：`docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.json`（每個資料集的逐項數字、per-rater/per-annotator 值、permutation 上界）。
