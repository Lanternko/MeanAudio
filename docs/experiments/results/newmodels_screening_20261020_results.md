# 新增 eval model × 人類評分 benchmark 海選：結果（2026-10-20）

預登錄：`docs/experiments/newmodels_screening_20261020.md`（commit `ebc9a21`，早於本檔與所有 benchmark 評分）。
逐 clip 分數：`research/eval/newmodels_20261020/scores_<model>_<bench>.tsv`；統計：`analysis_cells.json`（`analyze.py`，cluster bootstrap 2000 次，seed 20261020）；完整表：`gate_tables.md`（`gates.py`）。
**以下只是在這三個 benchmark／這組協定下的相關與 CI；沒有針對我們 arm 的人類評分，不構成「準確／不準確」的結論。**

## 1. 結論（依預登錄規則）

**沒有任何指標通過海選**（含現役 AES、MEva）。因此**不寫受控效度設計、contract，也不排 queue job**。

| 指標 | 判定 | 主要原因（只列獨立基準上的格子） |
|---|---|---|
| AES PQ / CE / CU | 未定 | PAM400：PQ 的 G1 ρ 0.45 [0.37,0.52] 未達 0.50（CE 0.57、CU 0.52 過）；ME 的系統內 ρ 0.36–0.45（< 0.50）。MC522 AES 重疊未知，列描述 |
| AES PC | **排除** | PAM400 ρ 0.18、ME ρ 0.03（CI 上界 < 0.50）；成對 0.50–0.51 |
| MEva | **排除** | PAM400 ρ 0.01 [−0.08,0.11]、成對 0.50；MC522 0.35 [0.27,0.43]。ME 上高（0.86）但為訓練 benchmark，描述性 |
| MuQ-Eval A1 MI | 未定 | PAM400 ρ 0.51 過、系統內 0.41 未過；MC522 0.47 [0.39,0.53] 未過 |
| MuQ-Eval A1 TA | **排除** | PAM400 ρ 0.42 [0.33,0.50]，上界 < 0.50 |
| MuQ-Eval A3a MI / TA | 未定 | PAM400 ρ 0.58 過；系統內 0.42–0.43 未過；MC522 0.45／0.23 未過 |
| CMI-RM empty quality | 未定（最接近） | PAM400 ρ 0.81 [0.78,0.84]、系統內 0.66、成對 0.85 全過；ME-test ρ 0.83 過但系統內 0.47 [0.38,0.56] 未過；MC522 ρ 0.37 [0.30,0.44] 未過 |
| CMI-RM empty alignment | 未定 | PAM400 全過；ME-test 系統內 0.42；MC522 0.25 |
| CMI-RM caption quality / alignment | 未定 | PAM400 系統內 0.46／0.40 未過；caption alignment 在 ME-test ρ 0.46 [0.36,0.55] 未過 |
| PAM | 未定 | MC522 ρ 0.61 過、ME ρ 0.67 過；ME 系統內 0.39 未過（PAM400 為設計用資料，描述） |

「未定」的意思是 CI 跨門檻或有格子未過但沒有被證據排除，不得寫成通過。

## 2. 主要格子（Spearman 與 95% CI，primary 人類目標）

| 指標 | MC522（人類 PQ，n=522） | PAM400（PAM OVL，n=400） | ME（OVL，n=2748） | ME-test（n=413） |
|---|---|---|---|---|
| AES PQ | 0.82 [0.79,0.85]（重疊未知） | 0.45 [0.37,0.52] | 0.66 [0.63,0.68] | 0.63 [0.57,0.69] |
| AES CE | 0.63 [0.57,0.69] | 0.57 [0.49,0.63] | 0.68 [0.66,0.71] | 0.66 |
| AES CU | 0.80 [0.76,0.83] | 0.52 [0.46,0.58] | 0.69 [0.67,0.72] | 0.68 |
| AES PC | −0.09 [−0.17,0.00] | 0.18 [0.08,0.27] | 0.03 [−0.02,0.07] | 0.08 |
| MEva | 0.35 [0.27,0.43] | 0.01 [−0.08,0.11] | 0.86（訓練，描述） | 0.87（描述） |
| MuQ-Eval A1 MI | 0.47 [0.39,0.53] | 0.51 [0.43,0.58] | 0.89（重疊，描述） | 0.82（描述） |
| MuQ-Eval A3a MI | 0.45 [0.38,0.52] | 0.58 [0.51,0.65] | 0.85（描述） | 0.83（描述） |
| CMI-RM empty quality | 0.37 [0.30,0.44] | 0.81 [0.78,0.84] | 0.85（train 重疊，描述） | 0.83 [0.80,0.86] |
| PAM | 0.61 [0.55,0.67] | 0.63（設計資料，描述） | 0.67 [0.65,0.69] | 0.69（描述） |

Pearson、Kendall τ-b、系統內 ρ、成對一致率（整體與 <0.5／0.5–1／≥1 三段，附配對數）、與 AES PQ／CE 的成對 bootstrap 差，全在 `gate_tables.md`／`analysis_cells.json`。

## 3. 觀察（描述）

- 在獨立的生成音樂基準 PAM400 上，CMI-RM（empty）與 AES 的差距最大（ρ 0.81 vs AES PQ 0.45）；但 PAM400 只有 4 個系統、100 prompt，且 CMI-RM 以 PAM(music) 為其 RewardBench 測試子集（未訓練）。
- 系統內（扣掉系統平均）所有指標在 ME 上都明顯下降：AES 0.36–0.45、MEva 0.67、CMI-RM empty quality 0.53（全量，描述）；ρ ≥ 0.5 的門檻在系統內只被 MEva／MuQ-Eval A1 MI／CMI-RM 在 ME「描述」格達到。
- MC522（真實錄音、單一刻度窄）上 AES PQ/CU 的 ρ 最高（0.80–0.82，但訓練重疊未知）；新模型 0.23–0.61，PAM 0.61。MC522 人類 PQ 的天花板為 0.898（既有文件）。
- CMI-RM `alignment` 在 empty prompt 下與 quality 高度重疊，不是文字對齊量測；加 caption 後兩者在 PAM400 的 ρ 反而下降（quality 0.81→0.65，alignment 0.75→0.55）。協定自由度（empty vs caption）已記，未做多重比較校正。
- 序數／粗階：MuQ-Eval A1 TA 在 MC522 只有 0.26；TA 是純音訊分支，不是文字對齊。

## 4. 實作與稽核

- 權重 sha256：`/mnt/seagate/newmodels_20261020/weights_sha256.txt`（MuQ-Eval-A1 `4163ec9b…`、MuQ-Eval(A3a) `aaec8558…`、MuQ-large `273febab…`、CMI-RM `e816d99f…`、MS-CLAP 2023 `2cef4016…`）；版本與授權在各 `scripts/eval/newmetrics_*.py` docstring 與預登錄 §1。CMI-RM、MuQ 編碼器為 **CC-BY-NC-4.0，只限非商用**。
- 輸出範圍：CMI-RM -2.7～4.7、MuQ-Eval 0.4～5.8、PAM 0.02～0.998，皆有限；每組唯一值數 ≈ n（無量化假平手）。
- 預登錄後實作細節（非結果導向）：成對分析 G3 的最小配對數門檻訂為 500（PAM400 575、ME-test 653 皆適用）；`gates.py` 第一版寫成 1000 使 PAM400／ME-test 漏掉 G3，已在看過 gate 表後更正為 500 並重跑。**這個更正發生在看過第一版表之後**，但 G3 在預登錄 §4 本來就適用於 PAM400，不是新增門檻；更正後沒有任何判定翻轉（第一版與更正版判定欄相同）。
- MEva 用 `runtime/meva_20261003` 的 pinned venv（`check_binding` 通過）；MC522 的 MEva 為新算。

## 5. 對照行：LALM 評審（124，不重跑）

Music Flamingo／Qwen2.5-Omni 在 MusicEval clip 層 ρ 0.46–0.66（描述；與本表 AES 0.66、MEva 0.87 同量級或更低），但 124 已判兩個評審噪音階梯不單調、未過最低門檻 → 見 `results/lalm_quality_judge_124_results.md`。協定與母體不同，不與本表閘門比較。

## 6. 限制與未完成

- 人類評分天花板與 MC522 刻度窄（真實錄音）使 MC522 的 ρ 偏低是預期現象之一，不單歸因於指標。
- PAM400 只有 4 個系統；系統內 ρ 的 CI 基於 prompt cluster，系統數少，外推力有限。
- ME-test 只有 413 筆，CMI-RM 獨立格 CI 寬。
- 沒有針對我們 TTM arm（16 kHz）的人類評分；2026-10-15 的人類聽測 pilot 若完成，才能做 arm 層級檢驗。
- 沒做：SongBench、HEAR、SongEval 測試集、CMI-RewardBench 公開 quality 子集（理由見預登錄 §6）。
- 未寫受控效度實驗／contract／queue job（無指標通過）。若使用者想放寬規則（例如只用 PAM400＋ME-test 兩個生成基準、忽略系統內門檻），CMI-RM empty quality 是唯一候選；這需要使用者決定並另行預登錄，不能由本結果事後改規則。
