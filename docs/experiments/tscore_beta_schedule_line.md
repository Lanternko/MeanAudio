# Score-aware Beta timestep schedule — 復現 arXiv 2606.07387

> 2026-09-13 設計。階段 A queue job `053_tscore_beta_probe.sh`，
> contract `docs/experiments/tscore_beta_probe_contract.json`。

## 論文在說什麼

Cheng, Huang, Tan（NTU），*Making the Most of Limited Data: Score-Aware Training for Text-to-Music Generation*，ICME 2026 ATTM Efficiency Track（客觀第 2、MOS 第 3）。骨幹就是 FluxAudio-S。四個元件：

| 元件 | 論文證據 | 我們怎麼處理 |
|---|---|---|
| (i) 每首切 15 段 10s，按 CLAP 分三檔，留 6 段 | 無消融 | **不做**。要重 encode latent（HDD 100%）；10-exp benchmark 已顯示 hard filter 無幫助；用 CLAP 篩又用 CLAP eval = leakage |
| (ii) CLAP 條件化 Beta timestep schedule | λ=0.2 CLAP +0.003、λ=1.0 −0.001（n=100 val、單 seed）；正面證據只有 val loss（base 在 ~7.5k 後 overfit） | **本線主題** |
| (iii) LLM 改寫成推論風格 caption 再 fine-tune | +0.013 CLAP | 與 050 slot4 剝數字線重疊，不另開 |
| (iv) REPA（CLAP / MuQ 對齊） | CLAP-REPA +0.018；MuQ-REPA 大幅變差 | 階段 B 選配；只能加 S1（不動 `MeanAudio` 類別），對齊目標不得用 CLAP |

排程公式：P(t|S) = Beta(α(S), 1)，α(S) = 1 + λ(1 − S)，t=1 是噪聲，S 截到 p75 以上為 1。

## 要回答什麼

「按對齊分數把樣本分配到不同 t」在我們的 pipeline 能不能改善泛化？如果能，功勞是**分數資訊**，還是只是**整體 t 分布偏向高噪聲**？

先驗偏懷疑：論文的 Beta 結果本身落在雜訊內，最終提交用的 λ=1.0 在消融裡是負的；而且它的正則化只在「2k rows、會 overfit」的情境下被觀察到。

## 實作（2026-09-13）

- **S1 其實走 `RunnerMeanFlow` + `MeanFlow.loss`**（`train_config.yaml: use_meanflow: True`），不是 `runner_flowmatching.py`。t 在 `MeanFlow.sample_t_r` 取樣（兩個 logit-normal(−0.4, 1) 取 max/min，75% 設 r=t）。
- 傾斜寫成 Beta(α,1) 的反 CDF `u ** (1/α)`，**套在兩個 logit-normal 樣本上、取 max/min 之前**：單調所以 t ≥ r 保持；S=1 時與 base 逐位元相同（已驗證），不多吃 RNG → 同 seed 的 arm 共用同一條 t 亂數流，是 paired 設計。
- `mean_flow.py` 的改動避開 `set_training_stage.py` 的 5 個 patch 片段（`--check` 仍正確判定 stage）。同一段 code 也適用 S2。
- Dataset：`data.<split>.t_score_column` 指名 TSV 欄位才輸出 `t_score`（值必須在 [0,1]，否則 fail closed）；runner：`+t_score_beta_lambda`（預設 0 → 完全不動）。有 `t_score` 時 train log 多一欄 `t_mean`。
- **Val 指標**：MeanFlow 的 val loss 是 adaptive L2 `sg(1/(mse+c))·mse`，恆 ≈ 0.994，而且 t 取自未重設的 numpy RNG — 看不出 overfit。新增 `+val_fm_mse=true`：固定 t 格點 {0.1,0.3,0.5,0.7,0.9} 的條件 velocity MSE，noise 來自每次 val 前都會重設的 `self.rng`，所以每次 val 輸入完全相同。預設關閉。
- **分數 S 用 PE-AV，不用 LAION-CLAP**：`scripts/preprocess/score_peav_alignment.py`，cos(caption, 前 10 秒 @48 kHz)，與 c2p0 captioner 看到的視窗一致。正規化 S = clip((s − p05)/(p75 − p05), 0, 1)（train split 上算）。
- Shuffled 對照：同一組 S 值在 train 列之間用 seed 424242 打亂（multiset 不變 → t 分布整體偏移完全相同，只拿掉逐列資訊）。

煙霧測試（2026-09-13，fake S，40 iter）：dataset `t_score` 載入、`t_mean` ≈ 0.60（base ≈ 0.53）、`fm_mse` 按格點記錄、EMA 合成、rc=0。

## 階段 A：機制探針（P0 級，本 job）

| 項目 | 設定 |
|---|---|
| 語料 | c2p0 slot0（`phase8_qwen_caption10s_multisent_train.tsv` + `true_random` overlay、`cap_index_fixed=0`、`require_text_overlay=true`） |
| 子集 | 每個 track 抽一段；100 val + 2,000 train，track 不重疊；select seed 20260913（`scripts/preprocess/build_tscore_beta_probe_inputs.py`） |
| 訓練 | S1 `fluxaudio_s` 20k iter，batch 8，lr 1e-4，warmup 1000，NoQ，NoMask；val 每 500 iter |
| Arms | `base`（λ=0）、`lam0p2`、`lam1p0`、`lam1p0shuf` × 訓練 seed {14159265, 27182818} = 8 run |
| 次要 | 每個 EMA：MusicCaps 前 500 列 + val 100 列，fluxaudio 25 步、CFG 0、seed 42 → CLAP/AES（只報，不當 gate） |
| 成本 | ~0.15 s/it → 每 run ~50 min + eval ~10 min；總計約 8–9 GPU-h |
| 磁碟 | 每 run 留 ema_final（0.48G）；ckpt_last / ema_ckpts 在 eval 後刪 |

### 判讀（preregistered，`scripts/analysis/summarize_tscore_beta_probe.py`）

主指標：`fm_mse` 最後 4 次 val 的平均（final_val）。雜訊底線 floor = |final_val(base, seed A) − final_val(base, seed B)|。

| 規則 | 條件 |
|---|---|
| R1 base overfit | 兩個 seed 的 base argmin ≤ 15k，且平均 overfit_gap > 2 × seed 間 gap 差 |
| R2 λ 有正則化效果 | final_val(arm) < final_val(base) 兩個 seed 皆成立，且平均 paired 差 > 2 × floor |
| R3 分數資訊有用 | final_val(lam1p0) < final_val(lam1p0shuf) 兩個 seed 皆成立，且平均差 > 2 × floor |

| 結果 | 下一步 |
|---|---|
| R2 兩個 λ 都不過 | **收線**，不開階段 B |
| R2 過、R3 過 | 開階段 B（B0–B3） |
| R2 過、R3 不過 | 效果是 t 分布偏移不是 score-aware；階段 B 的 B1 vs B3 預期 null，是否跑交由使用者決定 |

注意：傾斜的 arm 在高 t 訓練得多，格點 MSE 的高 t 項天然占優；summary 會附 per-t 數字，判讀時要看是不是只贏在 t=0.7/0.9。R1 不過時 R2/R3 仍報，但「正則化」的解釋站不住。

## 階段 B：全量因果對照（P2，待階段 A）

quarter 協定（S1 100k / S2 50k），c2p0 slot0 NoQ，PE-AV S 需補算 251,599 列（估 ~8 GPU-h，不佔磁碟）。

| Arm | 內容 |
|---|---|
| B0 | base（先查能否沿用現成 slot0 quarter，注意 recipe drift） |
| B1 | Beta 只在 S1 |
| B2 | Beta S1 + S2 |
| B3 | B1 + S 打亂 |
| B4（選配） | S1 加 REPA（目標 PE-AV audio embedding；`extracted_audio.py` 已有 `repa_npz_dir` 殘留介面） |

判讀：CFG0 與 CFG3+neg 兩個協定都報；效果 ≥ 2× 同協定訓練 seed floor；主張「分數資訊有用」須 B1 > B0 **且** B1 > B3。
