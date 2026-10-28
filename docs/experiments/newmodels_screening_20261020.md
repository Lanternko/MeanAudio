# 新增 eval model × 人類評分 benchmark 海選（2026-10-20）

**預登錄文件**：下面所有門檻、主指標、母體定義在**任何新模型對 benchmark 評分之前**寫定並 commit；之後不得改動。
結果寫在 `results/newmodels_screening_20261020_results.md`。候選來源：`docs/literature/music_eval_models_beyond_aes_meva_2026_10_20.md`。

背景：2026-10-20 會議（ASR 草稿，未聽校）定下「現有客觀音樂品質評分器在哪些情況可信、哪些失效？先用公認、有人類評分的 benchmark 海選，再做受控介入」。本實驗只做海選那一步；不碰 S 的重新定義（另一任務 `metric_std_20261020`）。

smoke test 階段（單檔，三個模型各在 1–3 個 PAM 檔上跑通）已做，**沒有**對任何 benchmark 批次評分，也沒有看過新模型與人類分數的相關。

## 1. 模型與固定輸入處理（寫死，不再改）

統一介面在 `scripts/eval/newmetrics_<model>.py`，共用 `newmetrics_common.py`：`score_<model>(wav_path_or_array, sr, ...) -> dict`。
共用：讀檔不改取樣率，多聲道取通道平均成單聲道，**不做任何響度／峰值正規化**。其餘處理沿用各上游實作。

| 模型 | 版本 | 輸入處理 | 輸出 | 授權 |
|---|---|---|---|---|
| MuQ-Eval A1（凍結 MuQ + attention pool + MLP，MSE） | code `dgtql/MuQ-Eval@60a88f8`；權重 `zhudi2825/MuQ-Eval-A1@b082e23` | 重採 24 kHz、**中央裁 10 s**、不足補零（上游預設是 random crop，為可重現改 center；README 的 `load_and_process` 不存在） | MI、TA（1–5 純量） | MIT；MuQ 編碼器 `OpenMuQ/MuQ-large-msd-iter` CC-BY-NC-4.0 |
| MuQ-Eval A3a（LoRA + ordinal CE） | 權重 `zhudi2825/MuQ-Eval@cc0cb22` | 同上 | MI、TA（ordinal 期望值，bin 中心 1..5） | 同上 |
| CMI-RM | code `Haiwen-Xia/CMI-RewardBench@9235ef5`（Apache-2.0）；權重 `HaiwenXia/CMI-RM@704f2dc`；底層 `OpenMuQ/MuQ-MuLan-large`（CC-BY-NC-4.0） | 官方 `RewardModelInference(mode='final')`：24 kHz、截前 30 s、30 s 分塊編碼；**fp32**（上游預設 bf16 autocast 會把輸出量化成 1.75、0.3965 這種粗階，製造假平手；`bf16=True` 可重現上游） | `quality`、`alignment` | **權重 CC-BY-NC-4.0：只限非商用** |
| PAM | code `soham97/PAM@ead6e7f`；MS-CLAP `CLAP_weights_2023.pth`；gpt2 文字骨幹 | 44.1 kHz；<7 s 重複補到 7 s；≥7 s 切 7 s 塊＋尾端一塊（上游 `repro=False` 預設），塊平均 | `pam` = P(「clear and clean」\|兩句 softmax) | MS-PL |

**CMI-RM 的 prompt 固定處理（兩種都跑，之後不再改）**：
- `empty`：text=''、lyrics=''、無參考音訊。
- `caption`：text=該 benchmark 的原始 caption/prompt（MusicCaps 用 `musiccaps_test.tsv` 的 caption；PAM 用 `scores.csv` 的 Text；MusicEval 用 `prompt_info.txt`／`demo_prompt_info.txt`），lyrics=''。
- `alignment` 在 `empty` 下**不是**文字對齊量測。
權重 sha256、版本寫在結果檔。SongBench、HEAR：本輪**不接**（見 §6），不在預登錄範圍。

## 2. Benchmark（只用有獨立人類評分的資料）

| 代號 | 內容 | n | 主要人類目標（primary） | 次要 |
|---|---|---|---|---|
| **MC522** | MusicCaps 真實錄音，Meta AES-natural 10 位標註者（我們自己重抓的 16 kHz 10 s 音檔） | 522 | 人類 PQ | CE、CU、PC |
| **PAM400** | PAM human_eval 生成音樂（audioldm2、musicldm、musicgen_large、musicgen_melody × 100 prompt；5 s，16/32 kHz） | 400 | **PAM 自家 OVL 面板**（與 AES 標註完全獨立） | AES 面板 PQ/CE/CU/PC；PAM REL |
| **PAM500** | 同上＋real 100（只用於描述性對照） | 500 | 同上 | |
| **ME** | MusicEval（25 系統共用 100 prompt＋6 個額外系統；5–349 s） | 2748（共用集 2500） | 專家 OVL | REL |

母體：成對分析只用「同 prompt、不同系統」；PAM400 每 prompt 6 對，MusicEval 共用集每 prompt 300 對。人類分數相等的配對排除；指標平手算 0.5。
**不納入**：SongEval 測試集、CMI-RewardBench 的 quality 子集（見 §6）。

### 訓練重疊旗標（逐模型，**預先寫定**；重疊集合只能當描述性，不得列為獨立泛化證據）

| 指標 | MC522 | PAM400 | ME |
|---|---|---|---|
| AES (PQ/CE/CU/PC) | 無法排除（論文未說 MusicCaps 是否在 97k 訓練語料內）→ 描述性 | 獨立（PAM 面板；AES_PAM 是 AES 自己的評測集，不是訓練集） | 獨立（零樣本） |
| MEva | 獨立（其五個訓練 benchmark 為 MusicEval、SongEval、MusicPref、AIME 等，不含 MusicCaps 人類 AES 分數；仍無法排除音訊本身） | 獨立（審計文件已記錄 PAM 不在其訓練／評測 benchmark） | **重疊**（MusicEval 是其訓練 benchmark）→ 描述性 |
| MuQ-Eval A1 / A3a | 獨立（標籤） | 獨立（標籤；MusicEval prompts 與 PAM 不同） | **重疊**（MusicEval 5-fold CV，釋出 checkpoint 見過 ME；A3a 另含 SongEval 維度設定）→ 描述性 |
| CMI-RM | 獨立（標籤；訓練含 CMI-Pref 人類標註、SongEval、MusicEval train） | 獨立（PAM(music) 是 CMI-RewardBench 的**測試**子集，不在訓練集） | train 1923 筆**重疊**；官方 test split 413 筆未見 → 只在 test split 上算「獨立」，且 n=413 CI 寬 |
| PAM | 獨立（零樣本 CLAP） | **設計用資料**（PAM 論文在這份資料上調 prompt／報告）→ 描述性 | 獨立（零樣本） |

「獨立」只指標籤未用於該模型訓練，不保證音訊或系統分布從未見過。

## 3. 統計量

每個 指標 × benchmark × 人類目標 報：
1. Pearson、Spearman、Kendall τ-b。
2. **系統內**相關：分數與人類目標都先扣掉各自「系統平均」再算 Spearman（PAM400：4 系統；ME：31 系統）。MC522 無此項。
3. 同 prompt 成對勝負一致率（整體、人類差 <0.5、0.5–1、≥1，各附配對數）。人類差用該目標的刻度（PAM/ME OVL 約 1–5；AES 面板 1–10）。
4. 對 AES PQ 與 AES CE 的差：每個相關／一致率與 AES 的**成對** bootstrap 差。

CI：cluster bootstrap 2000 次、seed 20261020、percentile 95%；cluster = prompt（PAM、ME）或單一 clip（MC522）。成對一致率的 cluster 也是 prompt。同一組 bootstrap 索引用於所有指標，所以差值 CI 是成對的。
人類天花板沿用既有文件（MC522 PQ 0.898；PAM OVL 見 aes_human_corr_pam；ME OVL 0.909），只當參考，不進門檻。
方向：預期為正相關；負相關一律視為不通過。

## 4. 通過／排除門檻（預登錄）

只用該指標「獨立」的 benchmark（§2 表）作閘門；描述性格子照樣報數字，但不計入通過、也不計入排除。

**G1（相關）**：對該 benchmark 的 primary 人類目標，Spearman 點估計 ≥ 0.50 且 95% CI 下界 > 0.30。
**G2（系統內，只有 ME 與 PAM400）**：系統內 Spearman 點估計 ≥ 0.50 且 CI 下界 > 0.30（使用者建議值；此門檻對 AES 也很嚴，我們接受這個結果）。
**G3（成對，只有 PAM400 與 ME 共用集）**：成對一致率（整體）CI 下界 > 0.55。

判定：
- **通過海選**：在所有「獨立」且可算的 benchmark 上，適用的 G1／G2／G3 全部成立，且獨立的 benchmark 至少 2 個、其中至少 1 個是生成音樂（PAM400 或 ME-test-split）。
- **排除**：在至少 1 個獨立的生成音樂 benchmark 上 G1 或 G3 的 CI **上界** < 對應門檻（0.50 或 0.55）——即 benchmark 證據顯示未達門檻，而非只是不確定；附該格數字。
- **未定（需外部資料）**：其餘情況（含獨立 benchmark 不足 2 個、CI 跨門檻）。**不得寫成準確或不準確**。

注意：
- 沒有多重比較校正（約 10 個候選分數 × 3 個 benchmark）；CMI-RM 的 `empty`／`caption` 兩種協定都算候選，因此通過者要註記這個自由度。
- 「人類差 <0.5」那一格人類自己也常分不出（MusicEval |Δ|≤0.4 兩半一致率 0.285），該格只描述，不進閘門。
- AES、MEva 是現役對照，用同一規則判定，但它們是否替換不由本文決定。
- Music Flamingo／Qwen2.5-Omni（124）：不重跑，引用 `results/lalm_quality_judge_124_results.md` 的既有數字列為對照行（該評審的協定與母體不同，只能並列不能跟閘門比）。

## 5. 通過者的下一步

通過者寫受控效度實驗設計（沿用 126 的兩條要求框架，S 先沿用 126）、contract 與 queue job；沒有任何指標通過就明寫並停在此處。

## 6. 範圍限制與沒做的事

- SongBench 評分器、HEAR（Track 1/2）本輪不接：時間優先放在三個第一優先候選與 benchmark 統計；兩者是歌曲導向、短純器樂有效性未知。
- SongEval 測試集與 CMI-RewardBench quality 子集未納入：SongEval 是整首歌曲（與 10 s 片段不同域，126d 已記錄域外效應），且 CMI-RM／MuQ-Eval(A3a) 訓練已含 SongEval；CMI-RewardBench 的 PAM/MusicEval 子集與本表重複，其餘子集是成對偏好、無逐 clip 分數。
- 人類評分資料沒有的結論不下「準確」；只寫相關與 CI。
- 不改 `eval_metrics.py`、`mc_mf25_eval*.sh`、`metric_win_tie_126*.py` 等被 contract 綁 sha 的檔案。
- 為何直接跑、不走 queue：三模型 × 2 協定共約 3,770 檔 × 數個變體，預估 GPU 合計 < 30 分鐘；腳本掛 `notify_on_exit`。超過 30 分鐘的受控實驗走 queue。
- 權重與外部 repo 放 `/mnt/seagate/newmodels_20261020/`（NVMe 剩 42 G）。
