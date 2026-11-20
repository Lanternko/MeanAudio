# 135–138 表 B 推論 seed 穩健性＋139 切段長度（預登錄，2026-10-21）

狀態：**預登錄，尚未產生任何新結果。** 本檔在任何新音檔生成之前 commit；commit 之後只能加「偏離紀錄」段，不能改門檻。

對應 2026-10-20 會議待辦第 3 項（小差異的可靠性與實際影響）與「後續探索：輸入長度／切段」。

## 0. 為什麼要做（上週的漏洞）

表 B（`ttm_quality_comparison_plan_20261007.md`、128b）的主結論是 ours CFG3＋neg 勝 MusicGen：lvl30 PQ +0.155／+0.164／+0.115（三個訓練 seed），平手邊界 m₀ = 0.0919；CLAP +0.0161／+0.0195／+0.0215，m₀ = 0.0104。

問題：

1. **所有系統都只跑過推論 seed 42。** 三個訓練 seed 的比較共用同一份 MusicGen 音檔，MusicGen 那一次抽樣的偏差同時進入三個比較，三 seed 一致不能當作獨立重複。
2. 已知 MusicCaps 全量的推論 seed PQ 差可到 0.142（`reference_inference_seed_noise_floor`，raw、CFG3＋neg、另一個 checkpoint），與 +0.115～+0.164 同一量級。
3. 逐段 CI（128b）只反映 prompt 抽樣，不含 seed 變異，會低估不確定性。

所以「表 B 的 PQ 優勢」目前只能算「在 seed 42 下成立」。

## 1. 假說與主端點

全部在 115 協定的 lvl30（−30 LUFS 純量增益，peak cap 0.999）上算；指標程式碼直接 import 115／118 用過的函式，不改。

| 編號 | 比較 | 指標 | 預登錄邊界 | 來源 |
|---|---|---|---|---|
| H1（主） | ours CFG3＋neg − MusicGen-medium | AES PQ | m₀ = 0.0919 | 128b K1 |
| H2（主） | ours CFG3＋neg − MusicGen-medium | CLAP（batch 1） | m₀ = 0.0104 | 128b K2 |
| H3（次） | ours CFG0 − MusicLDM | AES PQ | 平手：\|Δ\| < m₀ = 0.0919 | 128b K3 |
| H4（次） | ours CFG3＋neg − MusicLDM | CLAP | 預期 < 0（表 B：CLAP 輸 MusicLDM） | 表 B |

S 定義（D2 vs D3）使用者尚未決定，本實驗一律沿用 m₀，不在這裡重新定義。

### 判定規則（H1、H2；H3、H4 同法換方向）

設 ours 有 3 個訓練 seed × 5 個推論 seed = 15 份輸出，MusicGen 有 5 份。所有 15 × 5 = **75 個系統均值差**都算出來。

- **`robust`**：(a) 75 個差裡 ≥ 95% 大於 m₀，且 (b) 階層式 CI 下界 > m₀。
- **`holds_mean_only`**：平均差 > m₀，但 (a) 或 (b) 不成立。
- **`seed_dependent`**：平均差 > 0 但 ≤ m₀，或 75 個差裡有 > 5% 小於 0。
- **`reversed`**：平均差 < 0 且 CI 上界 < 0。

階層式 CI：每次 bootstrap 重抽 prompt（5,521 個，同一組 prompt 套到所有系統，保留配對）、重抽 ours 的訓練 seed（3 取 3）、重抽各系統的推論 seed（5 取 5，MusicLDM 3 取 3），2,000 次，95% 百分位。seed 層級太少，CI 會偏寬；這是刻意的保守選擇，寫進結果的限制段。

多重比較：H1–H4 以 Holm 校正 bootstrap 的單尾 p（以 0 或 m₀ 為界，依假說方向）。

### 次要（描述性，不設門檻）

- 變異分解：每個指標 × 系統，推論 seed 間的系統均值 SD、訓練 seed 間 SD、prompt 層級 SD。
- **最小可宣稱差（MDD）表**：每個指標，2 × √(推論 seed 方差＋訓練 seed 方差)，對照 m₀。這張表回答「多小的差在論文裡不能講」。
- 逐段平手底線：同 prompt、同系統、不同推論 seed 的逐段差分布（中位數、90 百分位）。這是「同一模型換 seed」的平手參考，給 S 定義討論用，不在本實驗裡定 S。
- 逐段 better 比例（128b 報 51–53%）在不同 seed 組合下的範圍。
- MEva、AES CE／CU／PC、raw FAD（`--ref_dir /mnt/HDD/kojiek/musiccaps_reference`）同樣報 75 差的分布，不判定。

## 2. 生成設定

| 系統 | 推論 seed | 新生成 | 設定 |
|---|---|---|---|
| ours s14159265／s27182818／s16180339 × CFG0／CFG3＋neg | 42（既有）＋ 101、202、303、404 | 3 × 2 × 4 = 24 格 | 與 `mc_mf25_eval.sh` 完全相同，只換 `--seed` |
| MusicGen-medium | 42（既有）＋ 101、202、303、404 | 4 份 | 與 113 完全相同，只換 base seed（batch 大小不變） |
| MusicLDM（無 fidelity8） | 42（既有）＋ 101、202 | 2 份 | 與 113 完全相同，只換 base seed |

seed 清單在本檔 commit 時固定，不因結果增減。

- 既有的 seed 42 輸出不重生，直接沿用 115／118 的 lvl30 分數與音檔。
- `mc_mf25_eval.sh`、`eval.py`、`ttm_external_generate.py`、`eval_metrics.py`、`ttm_metrics_115.py` 都被 contract 綁 sha，**不改**。新 driver 是複本＋參數化 seed。
- 輸出放 `/mnt/seagate/meanaudio/seed_robustness_135/`（NVMe 只剩 87 GB）。

## 3. 嚴謹性關卡（上週缺的）

1. **重現閘（gate，不過就停）**：新 driver 用 seed 42 重生前 256 個 prompt：
   - ours（s14159265，CFG0 與 CFG3＋neg）、MusicGen、MusicLDM 各一份。
   - 與既有音檔比：逐樣本 max abs 差 ≤ 1e-4，或（GPU 非決定性時）逐段 PQ 差絕對值的中位數 ≤ 0.01 且系統均值差 ≤ 0.005。
   - 不過 → job 失敗、不生成新 seed。證明新 driver 就是表 B 的協定，換 seed 是唯一變因。
2. **評分重現閘**：對既有 seed 42 的一個 ours 格重算 lvl30 PQ／CLAP，與 118 summary 差 ≤ 0.002（AES batch 組成會漂 ~1e-3）。
3. **覆蓋檢查**：每份輸出 5,521 首、缺 0；靜音數、NaN 數寫進 manifest；有缺就整份重跑（eval.py 續跑不具決定性）。
4. **分析腳本先寫好、先凍結**：分析程式在生成前完成，用重現閘的 256 首資料跑通（產生假表），sha 綁進 contract。結果出來後不能改分析程式；要改就是「偏離紀錄」並同時報原版結果。
5. **吞吐 pilot 只量時間**：256 首的生成時間只用來估預算，不看分數、不改設定。

## 4. 139：切段長度敏感度（探索，不做主張）

會議只提到「10 秒與 5 秒是否影響評分」，預期方向不明（切段會改變內容），所以本項**只做描述、不設通過門檻**，結果檔標明 exploratory。

- 素材：表 B 的 5 個系統（ours 三 seed CFG3＋neg、MusicGen、MusicLDM、真實錄音），seed 42 既有 lvl30 音檔，5,521（真實 5,131）首。
- 版本：完整 10 s；前 5 s；後 5 s；兩個 5 s 分數平均。切段後各自重新做 −30 LUFS。
- 指標：AES 四軸、CLAP、MEva、SongEval（若 126d 的 runner 可直接用）、MIR 兩個（pulse_clarity、key_cnn_conf）。
- 報：(a) 10 s 與 5 s 平均的逐段 Spearman；(b) 系統排名是否改變（Kendall τ 與每對系統差的符號）；(c) 表 B H1／H2 的差在 5 s 版本下的值；(d) 前後半差（曲頭效應）。
- 用 seed 42 單一推論 seed，結果只能寫「在 seed 42 下」。

## 5. 預算（重現閘 pilot 後凍結）

| job | 內容 | 預估 GPU（2026-10-21 重現閘 pilot 實測後） |
|---|---|---|
| 135 | 重現閘＋ours 24 格生成＋lvl30 評分 | 約 18.5 h（cfg0 0.327 s／首＝30 min／格、cfg3_neg 0.435 s／首＝40 min／格，各加 lvl30 評分約 11 min／格；12×41 min＋12×51 min＋閘約 10 min） |
| 136 | MusicGen 4 seed 生成＋評分 | 約 4.6 h（0.634 s／首＝58 min／格＋11 min；原估 2 s／首偏高） |
| 137 | MusicLDM 2 seed 生成＋評分 | 約 3.9 h（1.147 s／首＝106 min／格＋11 min；原估 2.5 s／首偏高） |
| 138 | 分析（CPU 為主，含 30 格 raw FAD） | 約 1.5 h（FAD 256 首 47 s，全量約 2–4 min／格；階層 bootstrap 2000 次 < 1 min） |
| 139 | 切段長度（只評分） | 約 4 h（pilot 64 首／實體：每個單位 35–44 s 幾乎全是模型載入；全量以 lvl30 評分 11 min／格＋MIR 約 10 min／格＋SongEval 約 2 min 推估） |
| 合計 | | **約 32–33 h**（原估 33–38 h；生成 s／首含程序啟動，是上界） |

重現閘 pilot 實測（前 256 首、seed 42、與既有音檔比；ours 兩格各 256/256 逐位相同，max|Δ| = 0）：ours cfg0／cfg3_neg、MusicGen、MusicLDM 全部 bit-identical（max abs diff 0.0、PQ 差 0.0000），評分關（重新評 s27182818 cfg3_neg 全量 lvl30）CLAP／PQ／MEva 與 118 差 0.0。

job 依 135 → 136 → 137 → 138 → 139 排在 p2，各自可在「份」的粒度續跑（已有完整 manifest 的份跳過，殘缺份整份重跑）。

## 6. 結果怎麼寫

- H1／H2 是 `robust` 才能在論文寫「ours 的 PQ／CLAP 優勢對推論與訓練 seed 穩健」。
- `holds_mean_only` → 只能寫平均值並附 75 差的範圍。
- `seed_dependent` → 表 B 的 PQ 優勢改寫成「在 seed 42 下觀察到，換 seed 不穩定」，並更新 `ttm_quality_comparison_plan_20261007.md` 與 128 結果檔的引用。
- 無人類評分；所有結論只關於指標分數，不關於感知品質。

## 偏離紀錄

（commit 後的任何改動記在這裡，附日期與理由。）

- 2026-10-21 實作決定（不改門檻、seed 清單、判定規則；實作 138 時為規則沒覆蓋的細節定案，分析腳本 sha 在 contract 綁定後不再改）：
  1. 判定類別順序 robust → holds_mean_only → reversed → seed_dependent（其餘兜底）；~~平均 > m0 但 > 5% 的差 < 0 者依規則順序是 holds_mean_only~~ → 2026-10-21 審閱改正（尚無任何新 seed 結果）：這種情況判 `seed_dependent`，「> 5% 的差 < 0」子句優先於 holds_mean_only，符合預登錄原意（較嚴）；結果表仍另標 `more_than_5pct_below_zero`。
  2. H3／H4 的差數是 45（ours 15 組合 × MusicLDM 3 seed），不是 75。
  3. 階層 bootstrap：每次重抽先抽 prompt（所有系統同一組索引，保持配對）、再抽 ours 訓練 seed（3 取 3）並在被抽中的訓練 seed 內巢狀重抽推論 seed、對照組推論 seed 獨立重抽；numpy default_rng(20261021)。
  4. MusicGen 沿用 113 的 PCM_16 FLAC 存檔，使 seed 42 可 bit-identical 重現（重現閘因此能過）；`feedback_external_generator_write_float` 提過的 PCM_16 硬削波因此仍在所有 MusicGen seed 上一致存在，不影響跨 seed 比較，但表 B 的絕對 MusicGen 分數仍帶這個存檔格式。
  5. 重現閘 pilot 結果見 §5。

