# 089 PE-AV 定義的品質標籤前綴重訓（081 交叉定義；2026-09-29）

## 問題

081 用語料自身的 **AES PQ** 分級加 `Low/High quality recording.` 前綴，結果 LQ 負向差中差 +1.061、HQ 正向 CFG0 +0.824 PQ（3 seed 同號）。
但**標籤與端點同為 AES PQ**：模型可能只是學到「PQ 評分器喜歡的東西」（自我洩漏），無法區分「真品質標籤有效」和「PQ 洩漏」。

089 換一個**非 PQ** 的評分器來定義分級：**PE-AV（`facebook/pe-av-large`）caption↔音訊 cosine**。
其餘（語料、前綴字、20% 分位、recipe、seed、eval 格）與 081 逐項相同。

**要回答的**：非 PQ 定義的「Low/High quality」標籤，是否也讓 LQ 負向／HQ 正向拉高 PQ？
- 會，且超過「分級裡剩下的 PQ 差」能解釋的量 → 081 不是純 PQ 自我洩漏。
- 只到 PQ 差能解釋的量 → 與「分級帶的 PQ 內容驅動效果」一致。
- CLAP 讀數：PE-AV 與 CLAP 同屬 caption↔audio 對齊，但模型不同，CLAP 可當部分相關的讀數；PQ 在 089 是**獨立**於標籤的讀數（081 則不是）。

## 設定（與 081 的差異只有分級分數）

- **語料／列序／前綴／分位**：同 081（`slot0clean_nmv2matched` 251,596 列；bottom/top 20%）。
- **分級分數**：PE-AV cos(normalize(audio_embeds), normalize(text_audio_embeds))。
  - 音訊 = 模型實際訓練的 10 s 窗口（`load_window` 逐行複製自 075 builder，builder 會對 32 個 stem 驗證陣列完全相等），以 librosa 16k→48k 重採樣（同 `research/eval/peav_eval.py`）。
  - 文字 = 該列**原始**訓練 caption（無前綴）。
  - 排序集合 = 081 有排序的列（`tier != unranked`，靜音排除規則完全相同）。
  - batch 閘：同 16 列 batch 8 vs 逐筆 |d| ≤ 2e-3（smoke：1.1e-6）。
- **Overlay**：前綴列若 089 分級 = 081 分級，caption 逐位元相同 → 直接用 081 的 HDD overlay（唯讀依賴 `/mnt/HDD/kojiek/quality_label_081/overlay_new/`，**不可刪**）；其餘前綴列新編到 NVMe `~/exps_nvme/quality_label_089/overlay_new/`。
- **Control**：同 081 的 nmv2pair control；它的 5 格（stock、lqneg、hqpos、hqpos lqneg，raw 與 lvl30）081 時已全部跑完，**089 不再跑 control**，只跑 arm 5 格。
- **Seed**：14159265 / 16180339 / 27182818。

## Eval 格（每 seed；同 081：MusicCaps 5521、MeanFlow 25、seed 42、fp32、NoMask、`--no_q`；CLAP batch 1 對原始 caption）

cfg0、cfg3_neg（fidelity8）、cfg3_lqneg、hqpos cfg0、hqpos cfg3_lqneg；每格 −30 LUFS 重評後刪音檔。

## 端點（預先登記；同 081 的定義與門檻）

- **E1（主）**：[arm lqneg − arm cfg0] − [ctrl lqneg − ctrl cfg0] ≥ 0.19、CI 下界 > 0、3 seed 各 > 0。
- **E3**：hqpos CFG0 差中差 ≥ 0.155。
- **E2／E4／E5**：同 081（E5 CLAP 非劣性 CFG0 −0.004、CFG3+neg −0.0158）。
- **交叉比較（089 − 081，逐 clip 配對）**：`X_E1`、`X_E3`、`X_hqlq`、`X_cfg0`。
- **PQ 比例預測（判讀用，預先登記）**：若 081 效果純由 PQ 驅動，089 的 E1／E3 預期 ≈ `E_081 × ΔPQ_089 / ΔPQ_081`，ΔPQ = 該分級下 high 與 low 層的語料 PQ 均值差（manifest `stats.pq_contrast_high_minus_low`）。
  - 觀測值明顯高於預測（超出其 CI）→ 非 PQ 標籤本身也有效。
  - 觀測值 ≈ 預測 → 與 PQ 內容驅動一致。
  - smoke（200 列）：Spearman(PE-AV, PQ) ≈ 0.19、ΔPQ 089/081 ≈ 0.43/1.74；全量數字建完才知道。
- 分析：`scripts/analysis/quality_label_peav_089_analysis.py` → `docs/experiments/results/quality_label_peav_089_summary.json`（091 結尾自動跑）。

## 預先登記的限制

1. **PE-AV 低分 ≠ 音質差**：PE-AV 低分表示 caption 與音訊不對齊（caption 錯、或音樂不典型），標籤語意是「Low quality recording.」但實際綁到的是「描述對不上」。這正是本實驗要的交叉定義，但 CLAP 讀數會被這層語意牽動（例如 LQ 負向可能推向「更像 caption」而拉高 CLAP）。
2. **CLAP 不是完全獨立**：PE-AV 與 CLAP 同為 text↔audio 對齊模型；教授原則禁的是「CLAP 定義訓練資料＋CLAP eval」，089 用 PE-AV 定義，CLAP eval 允許但解讀要標「部分相關」。
3. **Control 是 081 時跑的**：同 checkpoint、同 wrapper、同 sha；eval 生成具決定性，不重跑。
4. **081 的 HDD overlay 成為依賴**：刪掉 `/mnt/HDD/kojiek/quality_label_081/overlay_new` 會讓 089 farm 斷鏈。
5. **磁碟**：開排時（2026-09-29）NVMe 51 GB、HDD 3.7 GB free。新 overlay 約 20–25 GB（smoke：前綴列 68% 要新編）＋ builder 要求再留 25 GB 給 checkpoint；HDD 滿 → 每 seed 的 S1/S2 run（瘦身後各約 5 GB）留在 NVMe。**builder 空間不足會 exit 3（不寫半套）**；上座前要先清出空間。
6. 單一語料、quarter 預算、單一生成 seed；FAD 不在端點內（可事後補）。

## 流程與資源

- Queue：p2 `089_quality_label_peav_quarter_s14159265.sh` → `090_…_s16180339.sh` → `091_…_s27182818.sh`，排在 087/088（NegMF Stage B）之後。
- 第一個 seed 多做：PE-AV 全語料打分（~251k 列，估 ~8 h，可續跑）＋ 新編 overlay ＋ verify。
- 每 seed：S1 100k + S2 50k（約 9 h）＋ arm 5 格 eval（約 1 h）＋ lvl30 重評。
- queue evidence = 該 seed arm 的 S2 ema_final ＋ arm stock cfg0 REPORT。

## Smoke 驗證（2026-09-29，200 列，scratchpad root）

- `score_peav_corpus_089.py`：199 列（1 列 081 unranked）全打分；batch 閘 1.07e-6。
- builder `all`：window 閘 32/32 陣列相等；分級 40/119/40；overlay 26 列重用 081、54 列新編；081 `verify()` 全過（caption sha、clip_id、重編 max diff 1.5e-6）。
- action 的 control 格檢查三個 seed 全數找到（raw REPORT ＋ lvl30 per_clip）。
- 分析腳本在 arm 尚無資料時正常輸出 control 對照列。
