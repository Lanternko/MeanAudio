# 182：MuQ A1 能刷多高？疊加已知 PQ 導向手法＋best-of-N（2026-11-08）

設計登錄：2026-11-08，**生成與評分前登錄**。走 p2 queue（預估約 4–5 h，超過 30 min）。腳本與 contract 在本檔 commit 之後才寫，判準不會依結果修改。

## 0. 為什麼做

- 181 已確認：三種 PQ 導向手法（推論期負向、品質標籤前綴、負向蒸餾）都會讓 MuQ A1 MI 一起漲，幅度約為 PQ 的一半。但最好的格（081 HQ＋LQ，quarter）在 1,000 個 Part B 共同 prompt 上只有 3.45，仍低於 MusicGen medium 的 3.54 和真實錄音的 3.83。
- 本次要問的是：**不訓練新模型，只用現成的槓桿，MuQ A1 最多能刷到多高？** 分兩部分：
  - **A（疊加）**：把已知手法疊在一起，包括 HQ 正向前綴＋LQ／fid8 負向＋較高 cfg，以及 full recipe 的 NegMF N100。另外做事後 6 kHz 低通，179 在 A0 上量到 +0.15。
  - **B（best-of-N）**：同一個 prompt 用 8 個推論 seed 各生成一次，挑 A1 最高的那段。這是「拿評分器當篩子」最直接的刷分方式，也是 RL／rejection sampling 的上界代理。
- 用途：
  - A 若超過 MusicGen，就得到「只用 PQ 導向手法、不碰 MuQ，就能在 MuQ 上贏過外部系統」的例子。
  - B 量化「直接對 MuQ 優化」的空間，並看沒拿來挑選的指標（A3a、PQ、CE、CLAP、FAD）有沒有跟著漲。若 A1 漲、其他指標不漲，就是刷分的特徵。
- 只有 A、B 都不夠時，才考慮 C（用 MuQ 分級標籤重訓，另行登錄）。

## 1. 刺激

- **Prompt**：Part B 的 1,000 個共同 prompt（`muq_attack_fix_20261103/part_b_selection.json`，seed 0）。外部參照值（MusicGen、MusicLDM、真實錄音）都在這組 prompt 上。依 MusicCaps 的 id 切出子集 TSV（按 CSV record 切，不按行切）。hqpos 格用 `musiccaps_test_hqprefix.tsv` 同一組 id 的子集，該檔與原 TSV 的 id 順序逐列相同。
- **生成**：`eval.py` 的旗標與 112／181 相同：MeanFlow 25 步、fp32、NoMask、`--no_q`、duration 9.975。子集生成的音訊與全量 run 不會逐段相同，因為 eval.py 整個 run 只 seed 一次，所以不做身分閘，改用下面的 harness 閘。
- **Part A 的格**：每格都跑 3 個訓練 seed（14159265／16180339／27182818），推論 seed 42。

| 代號 | 模型 | 正向 TSV | 負向 | cfg |
|---|---|---|---|---|
| A_ref | 081 arm quarter | hqprefix | `Low quality recording.` | 3.0 |
| A_fid | 081 arm quarter | hqprefix | fid8 | 3.0 |
| A_comb | 081 arm quarter | hqprefix | `Low quality recording. ` ＋ fid8 | 3.0 |
| A_comb45 | 081 arm quarter | hqprefix | `Low quality recording. ` ＋ fid8 | 4.5 |
| A_n100 | N100 **full recipe**（120–122，S2 200k） | 原始 | fid8 | 3.0 |

  - fid8 ＝ `low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi`。
  - 每格另外評一次「事後 6 kHz 低通」：8 階 Butterworth `sosfiltfilt`，作用在 lvl30 音訊上，只多評分、不另外生成。
- **Part B**：
  - 選格規則（預登錄）：在 A 的 5 格中（不含低通），挑三 seed 平均 A1 MI lvl30 最高的一格。
  - 用該格的訓練 seed 14159265 跑推論 seed 42／101／202／303／404／505／606／707。seed 42 直接沿用 A 的音檔，其餘 7 個另外生成。
- 生成數：A 15 格＋B 7 格，共 22 次 × 1,000 段。

## 2. 評分

- **MuQ-Eval**：A1（主）與 A3a，MI 與 TA 兩個 head。用 `newmetrics_muqeval.score_muqeval`：mono、24 kHz、中央 10 s，batch 1（181 的 D1：batch 16 會漂 3.2e-3）。
- **音量條件**：lvl30（主，`level_match_rescore._gain_one`：−30 LUFS，peak 上限 0.999）與 raw（次）。
- **其他讀數**（都是 held-out，沒有拿來挑選）：
  - AES PQ／CE lvl30
  - CLAP，batch 1，對**原始** caption（不含 HQ 前綴）
  - FAD，`--ref_dir /mnt/HDD/kojiek/musiccaps_reference`，對全部 1,000 段
  - level：LUFS、靜音 < −45 dBFS
- **Part B 的 best-of-k**：
  - 對 k ∈ {1, 2, 4, 8}，每個 prompt 在前 k 個推論 seed 中挑 A1 MI lvl30 最高的一段，再對挑出的音檔算上述所有讀數。FAD 用 symlink 目錄計算。
  - 另外兩個描述讀數：
    - 以 A3a 挑選、讀 A1：看換一個 MuQ 變體挑，增益能不能轉移到 A1。
    - best-of-8 再加 6 kHz 低通。
- 評分完刪掉 A 的非選定格音檔；B 的 8 個 seed 留到 summary 寫完才刪，只保留分數。

## 3. 對比與預登錄判準

主讀數是 A1 MI lvl30。m_A1 = 0.055、m_A3a = 0.062（S 為 115 的 5 系統全距，與 181 相同）。

- **Harness 閘（先過才判讀）**：A_ref 三 seed 平均，與 181 同格（`arm081_hqpos_mc_mf25_cfg3_lqneg`）在同一組 1,000 prompt 子集上的三 seed 平均相差 |Δ| ≤ 0.05，而且每個 seed 的 |Δ| ≤ 0.10。不過就停下來查 bug，不判讀。
- **Q1（疊加能否贏過 MusicGen）**：取 A 中最好的格（不含低通），看它的三 seed 平均 − MusicGen medium 的 3.541。
  - ≥ m，而且三個 seed 都 > 3.541 → **贏過**
  - |差| < m → **追平**
  - ≤ −m → **仍輸**
  - 其他 → **未定**
  - 低通版另外用同一套規則判一次。
- **Q2（best-of-N 的空間）**：ΔA1(k) = bo-k − bo-1，對 prompt 做 bootstrap 95% CI（2,000 次）。
  - 報 bo-8 的 A1 相對 MusicGen 3.541 與真實 3.832 的位置，判法同 Q1，但只有一個訓練 seed，所以用 prompt bootstrap 的 CI 下界取代「三 seed 同號」：比 MusicGen 高 ≥ m 且 CI 下界 > 3.541 → 贏過。
- **Q3（刷分特徵，主要的判讀）**：用轉移比 T_X = (ΔX(8)/S_X) ÷ (ΔA1(8)/S_A1) 衡量。X 包括 A3a MI、PQ lvl30（S_PQ 0.914）、CE、CLAP；FAD 看方向。
  - T_A3a < 0.5、PQ 與 CLAP 的 T < 0.25，或 FAD 變差 → **刷分特徵成立**：被挑上的是 A1 特別偏好的段落，不是各指標都公認較好的段落。
  - T_A3a ≥ 0.8，而且 PQ 與 CLAP 都 ≥ 0.5 → **一般性較好**：挑選挑到的是「普遍較好的樣本」。
  - 其他 → **部分轉移**，分開描述。
- **描述讀數**（不改判定）：
  - A 各格之間的差（例如 A_comb − A_ref、A_comb45 − A_comb）
  - 各格的低通增益
  - 以 A3a 挑、讀 A1 的轉移
  - 各格的 FAD、CLAP、靜音數

### 預測（評分前寫下）

- A：最好的是 A_comb 或 A_comb45，約 3.50–3.60，對 MusicGen 判「追平」。低通再加 +0.1～+0.15，可能判「贏過」。A_n100 與 control 的 CFG3＋neg 同量級（約 3.35）。
- B：bo-8 − bo-1 約 +0.2～+0.3，bo-8 超過 MusicGen，但仍低於真實錄音。
- Q3：T_A3a 約 0.5、PQ／CLAP 的 T 接近 0，FAD 持平或變差，判「刷分特徵成立」或「部分轉移」。

## 4. 不可寫

- 沒有人類評分。bo-N 的 A1 增益不能寫成「音質變好」或「MuQ 被騙」，只能寫「被 A1 挑上的段落，其他指標漲了多少」。
- bo-N 是用同一個評分器挑選又評分，A1 的增益從構造上就偏樂觀。主要判讀只能放在 held-out 讀數（Q3）。
- 081 arm 是 quarter 模型，N100 是 full recipe，兩者不同，不能混成一張「我們的系統」表。Part B 的外部參照沿用 181 的作法，只當描述用的位置對照。
- 只限 MusicCaps 的 1,000 個 prompt；B 只有一個訓練 seed。

## 5. 執行

- queue：`~/gpu_queue/p2/pending/182_muq_stack_bestofn.sh`，搭配 contract `docs/experiments/muq_stack_bestofn_182_20261108_contract.json`。
- 腳本：`scripts/eval/muq_stack_bestofn_182.py`（重用 181／112 的 `eval_args`／`eval_cmd`）與 `_guest.py`。
- 流程：
  1. preflight：磁碟 ≥ 20 GiB，ckpt、TSV、181 per_cell 都在。
  2. 每格依序跑生成、lvl30、MuQ、AES、CLAP、level、FAD，再寫逐格 tsv。逐格標記完成，可以續跑。
  3. 依選格規則挑出 B 格，生成 B 的 7 個 seed。
  4. 算 best-of-k，寫 summary。
- 輸出：`~/eval_output_nvme/muq_stack_bestofn_182/{per_cell/*.tsv, per_cell/*.json, summary.json, progress.json}`。
- 預估：生成約 3 min／格，評分約 6–8 min／格，FAD 約 2 min，22 格加 bo-k 共約 **4–5 h**。暫存音檔每格約 0.4 GB。
- 結果寫成 `results/muq_stack_bestofn_182_results.md`，三段式回報。
