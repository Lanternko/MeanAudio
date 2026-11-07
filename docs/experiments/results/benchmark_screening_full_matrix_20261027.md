# 指標 × 人類評分：完整矩陣（2026-10-27）

**這份文件把 round 2 海選從 4 個資料集擴成 9 個，但沒有改任何 r2 預登錄的門檻、種子或判定。**r2 的結論（六個指標排除、SingMOS 未定、本輪沒有候選）原樣成立；這裡多出來的格子是**事後描述**，不進 G1／G2／G3／G6 的判定，也不改 `bench_r2_*_summary.json`。

* 產生時間：2026-10-27 19:09；由 `scripts/eval/bench_r2/build_full_matrix.py` 自動組表（重跑即更新）。
* 格子狀態：**105 格有數**、**0 格排隊中**、**3 格不適用**（共 108 格）。
* 種子：r2 `20261025`／r1 `20261020`；cluster bootstrap B=2000。

## 0. 三種數字不要混著讀

| 統計量 | 用在哪幾欄 | 意思是什麼 |
|---|---|---|
| G6a within-trial ρ | ODAQ（B1） | 同一 trial 內，指標排序是否跟專家排序一致（ODAQ 沒有逐檔絕對分數） |
| G1 ρ（逐檔／逐曲） | MC522、SongEval、SingMOS、AIME、Arena、PAM400、MusicEval、MusicPrefs | 指標分數 vs 人類分數（成對偏好資料集的人類分數＝該檔在成對比較裡的勝率） |
| 成對一致率 agree_m | AIME、Arena、MusicPrefs（另表） | 指標說「A 比 B 好」時跟人類同向的比例；差距 < m = 0.10×S 算平手給 0.5 |

人類天花板（`benchmark_screening_r2_human_ceiling_20261027.md`）：G1 欄位用 √SB，成對欄位用 BT held-out。**「佔天花板」＝ ρ ÷ 天花板**，這才是可比的分數；不同資料集的天花板差很多（Arena 的 ρ 上限本來就低，因為每檔只有 1–2 票）。

## 1. 主表 A — 生成音樂（人類聽生成系統）

| 指標 | SongEval | AIME | Music Arena | PAM400 | MusicEval | MusicPrefs |
|---|---|---|---|---|---|---|
| AES PQ | 0.65 [0.60, 0.69] | 0.40 [0.36, 0.45] | 0.15 [0.11, 0.19] | 0.45 [0.37, 0.52] | 0.66 [0.63, 0.68] | 0.13 |
| AES CE | 0.56 [0.51, 0.61] | 0.38 [0.34, 0.43] | 0.11 [0.07, 0.15] | 0.57 [0.49, 0.63] | 0.68 [0.66, 0.71] | 0.13 |
| AES CU | 0.71 [0.67, 0.75] | 0.45 [0.41, 0.50] | 0.17 [0.13, 0.20] | 0.52 [0.46, 0.58] | 0.69 [0.67, 0.72] | 0.16 |
| AES PC | 0.29 [0.22, 0.35] | 0.01 [-0.05, 0.07] | 0.11 [0.07, 0.14] | 0.18 [0.08, 0.27] | 0.03 [-0.02, 0.07] | -0.01 |
| PAM（零樣本 CLAP） | 0.61 [0.56, 0.65] | 0.47 [0.43, 0.52] | -0.06 [-0.10, -0.03] | 0.63 [0.56, 0.69] (d) | 0.67 [0.65, 0.69] | 0.21 |
| MEva | 0.53 [0.48, 0.58] (t) | 0.83 [0.81, 0.85] (t) | 0.25 [0.22, 0.29] (t) | 0.01 [-0.08, 0.11] | 0.86 [0.85, 0.88] (t) | 0.23 (t) |
| MuQ-Eval A1（MI） | 0.75 [0.72, 0.78] | 0.54 [0.50, 0.58] | 0.22 [0.19, 0.26] | 0.51 [0.43, 0.58] | 0.89 [0.88, 0.90] (t) | 0.15 |
| MuQ-Eval A3a（MI） | 0.79 [0.77, 0.81] | 0.64 [0.60, 0.67] | 0.22 [0.18, 0.26] | 0.58 [0.51, 0.65] | 0.85 [0.83, 0.86] (t) | 0.20 |
| CMI-RM（empty quality） | 0.69 [0.66, 0.73] | 0.59 [0.55, 0.62] | 0.22 [0.18, 0.25] | 0.81 [0.78, 0.84] | 0.85 [0.83, 0.86] (t) | 0.15 |
| SongEval（se_mus） | 0.50 [0.45, 0.55] (t) | -0.05 [-0.10, 0.01] | -0.01 [-0.05, 0.03] | 0.16 [0.07, 0.26] | -0.07 [-0.11, -0.03] | -0.02 |
| SingMOS predictor | 0.00 [-0.06, 0.07] | 0.02 [-0.03, 0.08] | -0.06 [-0.10, -0.02] | -0.24 [-0.34, -0.14] | 0.14 [0.10, 0.18] | 0.09 |
| CLAP（只描述） | — | 0.31 [0.26, 0.36] | 0.04 [0.00, 0.07] | 0.48 [0.40, 0.55] | 0.20 [0.15, 0.24] | 0.10 |
| **人類天花板** | 0.92 (√SB) | 0.81 (√SB) | 0.74 (G3 BT) | n/a | n/a | n/a |
| **n** | 800 | 1,300 | 3,998 | 400 | 2,748 | 5,030 |

旗標：`i` 標籤獨立；`t` 訓練／重疊過（不算證據）；`d` 該指標自己的設計資料（只描述）。格內是 ρ [95% CI]；「排隊中」表示分數還沒算（job 159／160），不是 0 也不是沒過。

## 2. 主表 B — 訊號、真實錄音、歌唱

| 指標 | ODAQ | MC522 | SingMOS-Pro |
|---|---|---|---|
| AES PQ | 0.554 [0.396, 0.700] | 0.825 [0.792, 0.853] | 0.208 [0.172, 0.240] |
| AES CE | 0.715 [0.573, 0.831] | 0.633 [0.571, 0.689] | 0.233 [0.199, 0.267] |
| AES CU | 0.677 [0.538, 0.796] | 0.803 [0.766, 0.836] | 0.116 [0.078, 0.151] |
| AES PC | -0.465 [-0.712, -0.192] | -0.086 [-0.178, 0.002] | -0.250 [-0.281, -0.215] |
| PAM（零樣本 CLAP） | 0.603 [0.460, 0.730] | 0.610 [0.549, 0.663] | 0.257 [0.223, 0.290] |
| MEva | -0.053 [-0.280, 0.167] | 0.355 [0.276, 0.433] | 0.138 [0.103, 0.173] |
| MuQ-Eval A1（MI） | 0.183 [-0.050, 0.407] | 0.466 [0.390, 0.529] | 0.358 [0.326, 0.388] |
| MuQ-Eval A3a（MI） | 0.257 [0.053, 0.457] | 0.451 [0.375, 0.515] | 0.411 [0.381, 0.441] |
| CMI-RM（empty quality） | 0.103 [-0.160, 0.373] | 0.372 [0.298, 0.442] | 0.278 [0.246, 0.311] |
| SongEval（se_mus） | -0.190 [-0.383, 0.013] | 0.098 [0.014, 0.185] | -0.047 [-0.082, -0.011] |
| SingMOS predictor | 0.269 [0.008, 0.500] | 0.338 [0.260, 0.409] | 0.573 [0.547, 0.599] (t) |
| CLAP（只描述） | — | 0.095 [0.007, 0.182] | — |
| **人類天花板** | 0.99 (√SB) | 0.95 (√SB) | 0.92 (√SB) |
| **n** | 960 | 522 | 3,101 |

旗標：`i` 標籤獨立；`t` 訓練／重疊過（不算證據）；`d` 該指標自己的設計資料（只描述）。格內是 ρ [95% CI]；「排隊中」表示分數還沒算（job 159／160），不是 0 也不是沒過。

## 3. 逐格明細（含 n、來源檔）

### ODAQ（訊號劣化）

* 人類評分：專家 MUSHRA（壓縮／分離瑕疵）
* 統計量：G6a within-trial ρ；人類天花板：0.993（√SB）；規劃單位數 960

| 指標 | 值 | 95% CI | n（trial 數） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.554 | [0.396, 0.700] | 26 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| AES CE | 0.715 | [0.573, 0.831] | 26 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| AES CU | 0.677 | [0.538, 0.796] | 26 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| AES PC | -0.465 | [-0.712, -0.192] | 26 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| PAM（零樣本 CLAP） | 0.603 | [0.460, 0.730] | 30 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| MEva | -0.053 | [-0.280, 0.167] | 30 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| MuQ-Eval A1（MI） | 0.183 | [-0.050, 0.407] | 30 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| MuQ-Eval A3a（MI） | 0.257 | [0.053, 0.457] | 30 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| CMI-RM（empty quality） | 0.103 | [-0.160, 0.373] | 30 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| SongEval（se_mus） | -0.190 | [-0.383, 0.013] | 30 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| SingMOS predictor | 0.269 | [0.008, 0.500] | 26 | i | analysis_extra/odaq_g6.tsv · native raw / 30 trials |
| CLAP（只描述） | 此資料集沒有人類文字，CLAP 無法定義 | — | — | — | — |

### MC522（真實錄音）

* 人類評分：MusicCaps 522 首品質評分（10 人）
* 統計量：G1 ρ（human_PQ）；人類天花板：0.947（√SB）；規劃單位數 522

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.825 | [0.792, 0.853] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| AES CE | 0.633 | [0.571, 0.689] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| AES CU | 0.803 | [0.766, 0.836] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| AES PC | -0.086 | [-0.178, 0.002] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| PAM（零樣本 CLAP） | 0.610 | [0.549, 0.663] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| MEva | 0.355 | [0.276, 0.433] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| MuQ-Eval A1（MI） | 0.466 | [0.390, 0.529] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| MuQ-Eval A3a（MI） | 0.451 | [0.375, 0.515] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| CMI-RM（empty quality） | 0.372 | [0.298, 0.442] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| SongEval（se_mus） | 0.098 | [0.014, 0.185] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| SingMOS predictor | 0.338 | [0.260, 0.409] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |
| CLAP（只描述） | 0.095 | [0.007, 0.182] | 522 | i | analysis_extra/mc522_corr.tsv · V10 raw / target human_PQ |

### SongEval（生成音樂（整首））

* 人類評分：SongEval Musicality（4 標註者）
* 統計量：G1 ρ（Musicality, VFULL）；人類天花板：0.921（√SB）；規劃單位數 800

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.646 | [0.604, 0.687] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| AES CE | 0.560 | [0.505, 0.612] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| AES CU | 0.711 | [0.673, 0.746] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| AES PC | 0.289 | [0.221, 0.354] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| PAM（零樣本 CLAP） | 0.609 | [0.563, 0.654] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| MEva | 0.533 | [0.478, 0.585] | 795 | t | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| MuQ-Eval A1（MI） | 0.753 | [0.724, 0.777] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| MuQ-Eval A3a（MI） | 0.794 | [0.768, 0.815] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| CMI-RM（empty quality） | 0.692 | [0.657, 0.726] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| SongEval（se_mus） | 0.501 | [0.450, 0.552] | 800 | t | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| SingMOS predictor | 0.003 | [-0.061, 0.069] | 800 | i | analysis_extra/songeval_corr.tsv · VFULL raw / target Musicality |
| CLAP（只描述） | 此資料集沒有人類文字，CLAP 無法定義 | — | — | — | — |

### SingMOS-Pro（歌唱）

* 人類評分：SingMOS MOS（官方 test）
* 統計量：G1 ρ（mos）；人類天花板：0.919（√SB）；規劃單位數 3,101

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.208 | [0.172, 0.240] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| AES CE | 0.233 | [0.199, 0.267] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| AES CU | 0.116 | [0.078, 0.151] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| AES PC | -0.250 | [-0.281, -0.215] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| PAM（零樣本 CLAP） | 0.257 | [0.223, 0.290] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| MEva | 0.138 | [0.103, 0.173] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| MuQ-Eval A1（MI） | 0.358 | [0.326, 0.388] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| MuQ-Eval A3a（MI） | 0.411 | [0.381, 0.441] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| CMI-RM（empty quality） | 0.278 | [0.246, 0.311] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| SongEval（se_mus） | -0.047 | [-0.082, -0.011] | 3,101 | i | analysis_extra/singmos_corr.tsv · native raw / target mos |
| SingMOS predictor | 0.573 | [0.547, 0.599] | 3,101 | t | analysis_extra/singmos_corr.tsv · native raw / target mos |
| CLAP（只描述） | 此資料集沒有人類文字，CLAP 無法定義 | — | — | — | — |

### AIME（生成音樂）

* 人類評分：13 系統成對偏好（12 票）
* 統計量：G1 ρ（逐檔勝率）；人類天花板：0.811（√SB）；規劃單位數 1,300

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.403 | [0.356, 0.450] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| AES CE | 0.382 | [0.335, 0.428] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| AES CU | 0.454 | [0.410, 0.500] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| AES PC | 0.008 | [-0.050, 0.068] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| PAM（零樣本 CLAP） | 0.473 | [0.433, 0.515] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| MEva | 0.829 | [0.806, 0.852] | 1,300 | t | analysis_extra/aime_pairwise.tsv · native raw |
| MuQ-Eval A1（MI） | 0.540 | [0.500, 0.580] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| MuQ-Eval A3a（MI） | 0.637 | [0.604, 0.671] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| CMI-RM（empty quality） | 0.586 | [0.549, 0.622] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| SongEval（se_mus） | -0.048 | [-0.101, 0.006] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| SingMOS predictor | 0.023 | [-0.031, 0.081] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |
| CLAP（只描述） | 0.307 | [0.256, 0.356] | 1,300 | i | analysis_extra/aime_pairwise.tsv · native raw |

### Music Arena（生成音樂）

* 人類評分：15 系統成對偏好
* 統計量：G1 ρ（逐檔勝率）；人類天花板：0.742（G3 BT）；規劃單位數 3,998

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.149 | [0.112, 0.186] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| AES CE | 0.110 | [0.073, 0.147] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| AES CU | 0.167 | [0.130, 0.203] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| AES PC | 0.107 | [0.071, 0.143] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| PAM（零樣本 CLAP） | -0.060 | [-0.098, -0.025] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| MEva | 0.252 | [0.216, 0.287] | 2,809 | t | analysis_extra/arena_pairwise.tsv · native raw |
| MuQ-Eval A1（MI） | 0.221 | [0.186, 0.258] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| MuQ-Eval A3a（MI） | 0.220 | [0.183, 0.256] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| CMI-RM（empty quality） | 0.216 | [0.180, 0.253] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| SongEval（se_mus） | -0.007 | [-0.046, 0.030] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| SingMOS predictor | -0.057 | [-0.095, -0.021] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |
| CLAP（只描述） | 0.037 | [0.001, 0.073] | 2,824 | i | analysis_extra/arena_pairwise.tsv · native raw |

### PAM400（生成音樂（5 s））

* 人類評分：PAM OVL（每檔 10 位 MTurk）
* 統計量：ρ（round-1 協定）；人類天花板：n/a（每檔是 10 人平均，沒有逐位評分可算 reliability）；規劃單位數 400

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.447 | [0.371, 0.519] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| AES CE | 0.566 | [0.491, 0.634] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| AES CU | 0.521 | [0.458, 0.580] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| AES PC | 0.178 | [0.083, 0.269] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| PAM（零樣本 CLAP） | 0.625 | [0.557, 0.687] | 400 | d | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| MEva | 0.015 | [-0.084, 0.111] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| MuQ-Eval A1（MI） | 0.508 | [0.431, 0.578] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| MuQ-Eval A3a（MI） | 0.580 | [0.510, 0.646] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| CMI-RM（empty quality） | 0.812 | [0.780, 0.840] | 400 | i | research/eval/newmodels_20261020/analysis_cells.json · target pam_OVL |
| SongEval（se_mus） | 0.161 | [0.070, 0.258] | 400 | i | analysis_extra/pam400_extra_cells.tsv · target pam_OVL |
| SingMOS predictor | -0.239 | [-0.336, -0.143] | 400 | i | analysis_extra/pam400_extra_cells.tsv · target pam_OVL |
| CLAP（只描述） | 0.477 | [0.400, 0.554] | 400 | i | analysis_extra/pam400_extra_cells.tsv · target pam_OVL |

### MusicEval（生成音樂）

* 人類評分：MusicEval OVL（31 系統）
* 統計量：ρ（round-1 協定）；人類天花板：n/a（同上，只有平均值）；規劃單位數 2,748

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.656 | [0.631, 0.679] | 2,748 | i | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| AES CE | 0.683 | [0.658, 0.707] | 2,748 | i | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| AES CU | 0.692 | [0.667, 0.715] | 2,748 | i | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| AES PC | 0.026 | [-0.022, 0.074] | 2,748 | i | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| PAM（零樣本 CLAP） | 0.673 | [0.649, 0.694] | 2,748 | i | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| MEva | 0.865 | [0.851, 0.877] | 2,748 | t | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| MuQ-Eval A1（MI） | 0.893 | [0.883, 0.902] | 2,748 | t | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| MuQ-Eval A3a（MI） | 0.847 | [0.834, 0.860] | 2,748 | t | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| CMI-RM（empty quality） | 0.846 | [0.833, 0.856] | 2,748 | t | research/eval/newmodels_20261020/analysis_cells.json · target human_OVL |
| SongEval（se_mus） | -0.071 | [-0.109, -0.033] | 2,748 | i | analysis_extra/me_extra_cells.tsv · target human_OVL |
| SingMOS predictor | 0.140 | [0.103, 0.178] | 2,748 | i | analysis_extra/me_extra_cells.tsv · target human_OVL |
| CLAP（只描述） | 0.197 | [0.155, 0.241] | 2,748 | i | analysis_extra/me_extra_cells.tsv · target human_OVL |

### MusicPrefs（生成音樂（成對偏好））

* 人類評分：2,515 對偏好（7 系統、2 軸）
* 統計量：ρ（逐檔勝率）；人類天花板：待 job 160 算 BT held-out；規劃單位數 5,030

| 指標 | 值 | 95% CI | n（評分單位／item） | 旗標 | 來源 |
|---|---|---|---|---|---|
| AES PQ | 0.132 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| AES CE | 0.134 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| AES CU | 0.163 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| AES PC | -0.010 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| PAM（零樣本 CLAP） | 0.206 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| MEva | 0.232 | — | 4,098 | t | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| MuQ-Eval A1（MI） | 0.148 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| MuQ-Eval A3a（MI） | 0.196 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| CMI-RM（empty quality） | 0.151 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| SongEval（se_mus） | -0.023 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| SingMOS predictor | 0.086 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |
| CLAP（只描述） | 0.100 | — | 4,098 | i | analysis_extra/musicprefs_pairwise.tsv · musicality 軸 / 成對 2049 |

## 4. 成對偏好：一致率與人類天花板

三個資料集的人類標籤是「A 比 B 好」，逐檔 ρ 只是它的投影；真正對得上標籤的是成對一致率。

| 資料集 | 人類天花板 | 指標 | 一致率（m=0.10S） | 95% CI | 強制比較 | 平手率 m | 對數 | 旗標 |
|---|---|---|---|---|---|---|---|---|
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | AES PQ | 0.605 | [0.596, 0.614] | 0.628 | 0.428 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | AES CE | 0.612 | [0.603, 0.622] | 0.640 | 0.461 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | AES CU | 0.612 | [0.604, 0.621] | 0.648 | 0.515 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | AES PC | 0.491 | [0.481, 0.501] | 0.491 | 0.402 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | PAM（零樣本 CLAP） | 0.621 | [0.612, 0.630] | 0.657 | 0.496 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | MEva | 0.714 | [0.706, 0.724] | 0.733 | 0.289 | 7,800 | t |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | MuQ-Eval A1（MI） | 0.641 | [0.632, 0.650] | 0.665 | 0.376 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | MuQ-Eval A3a（MI） | 0.664 | [0.655, 0.673] | 0.685 | 0.340 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | CMI-RM（empty quality） | 0.667 | [0.658, 0.677] | 0.684 | 0.302 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | SongEval（se_mus） | 0.485 | [0.475, 0.495] | 0.483 | 0.395 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | SingMOS predictor | 0.500 | [0.491, 0.509] | 0.508 | 0.528 | 7,800 | i |
| aime | 0.685（BT held-out，G3）[0.675, 0.696] | CLAP（只描述） | 0.586 | [0.576, 0.595] | 0.601 | 0.382 | 7,800 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | AES PQ | 0.584 | [0.570, 0.598] | 0.598 | 0.688 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | AES CE | 0.564 | [0.549, 0.579] | 0.585 | 0.654 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | AES CU | 0.565 | [0.553, 0.577] | 0.605 | 0.771 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | AES PC | 0.556 | [0.539, 0.572] | 0.575 | 0.573 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | PAM（零樣本 CLAP） | 0.476 | [0.456, 0.496] | 0.472 | 0.441 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | MEva | 0.636 | [0.618, 0.654] | 0.665 | 0.438 | 1,397 | t |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | MuQ-Eval A1（MI） | 0.605 | [0.587, 0.623] | 0.626 | 0.497 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | MuQ-Eval A3a（MI） | 0.598 | [0.583, 0.614] | 0.667 | 0.593 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | CMI-RM（empty quality） | 0.633 | [0.615, 0.653] | 0.654 | 0.385 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | SongEval（se_mus） | 0.497 | [0.479, 0.517] | 0.508 | 0.446 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | SingMOS predictor | 0.446 | [0.431, 0.462] | 0.451 | 0.608 | 1,412 | i |
| arena | 0.742（BT held-out，G3）[0.715, 0.765] | CLAP（只描述） | 0.521 | [0.503, 0.539] | 0.520 | 0.488 | 1,412 | i |
| musicprefs | 待 job 160 算（BT held-out） | AES PQ | 0.580 | [0.559, 0.600] | 0.596 | 0.451 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | AES CE | 0.584 | [0.564, 0.605] | 0.604 | 0.503 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | AES CU | 0.580 | [0.562, 0.598] | 0.624 | 0.580 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | AES PC | 0.499 | [0.475, 0.525] | 0.483 | 0.345 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | PAM（零樣本 CLAP） | 0.619 | [0.600, 0.638] | 0.637 | 0.480 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | MEva | 0.608 | [0.587, 0.629] | 0.631 | 0.300 | 2,049 | t |
| musicprefs | 待 job 160 算（BT held-out） | MuQ-Eval A1（MI） | 0.583 | [0.563, 0.603] | 0.589 | 0.393 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | MuQ-Eval A3a（MI） | 0.600 | [0.579, 0.621] | 0.608 | 0.438 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | CMI-RM（empty quality） | 0.598 | [0.575, 0.621] | 0.593 | 0.353 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | SongEval（se_mus） | 0.486 | [0.466, 0.507] | 0.490 | 0.392 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | SingMOS predictor | 0.534 | [0.520, 0.548] | 0.556 | 0.703 | 2,049 | i |
| musicprefs | 待 job 160 算（BT held-out） | CLAP（只描述） | 0.561 | [0.539, 0.584] | 0.572 | 0.367 | 2,049 | i |

**Arena 的 ρ 為什麼一定低**：每檔只出現在 1–2 場比較，人類「分數」其實是 0/0.5/1 的勝率；G1 的 ρ 上限由 BT held-out 0.742 決定，不是 1.0。AES PQ 在 Arena 的 ρ 0.149 看起來很差，但它的成對一致率 0.584 高於 0.55 門檻——兩者並不矛盾（ρ 對只有兩三種取值的 target 很敏感）。

## 5. 排隊中的格子（job 159／160）

| job | 腳本 | contract | 補哪些格 | 預估 |
|---|---|---|---|---|
| 159 | `gpu_queue/p2/pending/159_bench_r2_J6_extra.sh` | `docs/experiments/bench_r2_J6_extra_contract.json` | SongEval predictor × ODAQ／MC522／AIME／SingMOS／Arena；SingMOS predictor × SongEval；SongEval／SingMOS／CLAP × PAM400／MusicEval；**另外重跑 arena 成對分析，補 SingMOS × Arena 那一列** | 1.5–2.5 h |
| 160 | `gpu_queue/p2/pending/160_bench_r2_J7_musicprefs.sh` | `docs/experiments/bench_r2_J7_musicprefs_contract.json` | MusicPrefs 整欄 9 個指標（5,030 檔、2,515 對、2 軸） | 2–3.5 h |

編號從 **159** 起，因為 **153–158 已被 autoresearch 推論線預訂**（`docs/experiments/autoresearch_inference_clap_20261027.md` §10.3）。
SingMOS × Arena 的分數早在 job 152（J5c）就評完（`scores/arena/singmos__native_raw.tsv`，3,998 檔、0 缺），但 `analysis/arena_pairwise.tsv` 寫得比它早，所以表上暫時是「排隊中」；159 重跑 arena 分析就會補上。
兩支 job 都不改 r2 的格子；跑完後重跑 `build_full_matrix.py` 就會把「排隊中」換成數字。

## 6. 不適用的格子

* **CLAP × ODAQ／SingMOS／SongEval**：這三個資料集沒有人類文字（ODAQ 只有參考音、SingMOS 只有 MOS、SongEval 只有五維評分），CLAP 是文字–音訊對齊分數，沒有 caption 就沒有定義。
* **PAM400／MusicEval 的人類天花板**：釋出的是「每檔 10 人平均」，沒有逐位評分者資料，算不出 split-half reliability。

## 7. 旗標表（誰訓練過誰）

沿用 `benchmark_screening_r2_overlap_audit_20261027.md`；本表只多加 PAM400／MusicEval 兩欄，並把該檔的證據逐格列出。

| 指標 | ODAQ | MC522 | SongEval | SingMOS-Pro | AIME | Music Arena | PAM400 | MusicEval | MusicPrefs |
|---|---|---|---|---|---|---|---|---|---|
| AES PQ | i | i | i | i | i | i | i | i | i |
| AES CE | i | i | i | i | i | i | i | i | i |
| AES CU | i | i | i | i | i | i | i | i | i |
| AES PC | i | i | i | i | i | i | i | i | i |
| PAM（零樣本 CLAP） | i | i | i | i | i | i | d | i | i |
| MEva | i | i | t | i | t | t | i | t | t |
| MuQ-Eval A1（MI） | i | i | i | i | i | i | i | t | i |
| MuQ-Eval A3a（MI） | i | i | i | i | i | i | i | t | i |
| CMI-RM（empty quality） | i | i | i | i | i | i | i | t | i |
| SongEval（se_mus） | i | i | t | i | i | i | i | i | i |
| SingMOS predictor | i | i | i | t | i | i | i | i | i |
| CLAP（只描述） | · | i | · | · | i | i | i | i | i |

* MEva 的五個訓練 benchmark 是 MusicEval／SongEval／MusicPref／AIME／Music Arena（`runtime/meva_20261003/paper_layout.txt:55`，pooled checkpoint），所以那五格都是 `t`；B1／B4／B5 不在其中 → `i`。
* MuQ-Eval A1／A3a、CMI-RM 只訓練在 MusicEval → MusicEval 欄 `t`，其餘 `i`。
* SongEval predictor 訓練在 SongEval → 只有 SongEval 欄 `t`；SingMOS predictor 訓練在 SingMOS-Pro → 只有 SingMOS 欄 `t`。
* PAM 是零樣本 CLAP，不是學出來的評分器；但 PAM400 是它論文裡選 prompt／調設計用的資料 → `d`（只描述）。
* AES 對 B2／B3／B4／B6 的**生成軌**是標籤＋音訊獨立；B1／B5 與真實錄音子集的音訊重疊仍無法核實（維持 `i`，但別寫「沒見過」）。

## 8. 假設與限制

1. **MusicPrefs 沒有授權宣告**：只供內部研究，不重新散布；音訊快取在 `~/eval_output_nvme/bench_r2_derived/musicprefs`（8.7 GB），不放進任何對外產物。
2. **PAM400／MusicEval 的 audio cache 走 symlink**（`derived/<ds>/` → 原始檔），沒有 re-encode；MusicPrefs 是從 parquet 一次性解成 float32 wav（44.1／32／16 kHz 原樣保留）。
3. **MusicPrefs 的 ρ 是事後投影**：人類標籤是成對偏好，逐檔 ρ 用「該檔的勝率」當 target，與 AIME／Arena 在 r2 的做法一致；主讀數仍以成對一致率為準。
4. **MusicEval 的 ρ 用全部 2,748 檔**（round-1 協定），成對一致率只在 `shared=True` 的子集上算。
5. 這份矩陣**不是**新的預登錄：新增的 12 格以上全部是描述性；任何「某指標在某格看起來過了」都不能回頭改 r2 判定。

## 9. 重跑指令

```bash
cd ~/MeanAudio
# 1) 只重組表格（不快取任何模型；讀 analysis/ + analysis_extra/）
/home/kojiek/venvs/dac/bin/python -I scripts/eval/bench_r2/build_full_matrix.py

# 2) 格子跑完後（job 159／160）先驗證再重組
GPU_QUEUE_CONTRACT=docs/experiments/bench_r2_J6_extra_contract.json \
  /home/kojiek/venvs/dac/bin/python -I scripts/eval/bench_r2/run_extra.py --validate-only
```
