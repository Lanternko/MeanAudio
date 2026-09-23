# 073 預註冊：guidance 幾何（ADG / APG）取代樸素 CFG 外插

2026-09-22 設計。推論期介入、不重訓、單一 checkpoint。
結果將寫在 `docs/experiments/results/guidance_geometry_adg_apg_20260922_results.md`。
腳本 `scripts/eval/guidance_geometry_adg_apg_20260922.py`；產物
`/home/kojiek/nvme_experiment_artifacts/meanaudio/guidance_geometry_adg_apg_20260922/`。

## 為什麼是這一條

caption 內容編輯線 2026-09-22 收線，理由是 ROI：4× 訓練預算換 +0.012 CLAP，同一個
checkpoint 加 negative prompt 換 +0.046 CLAP / +0.95 PQ
（`project_step_budget_vs_negprompt_roi`）。推論期那側高一個量級，而那一側只動過
**negative prompt 的文字內容**與 **cfg 的大小**，從來沒有動過 **guidance 的幾何**。

### 樸素 CFG 在做什麼

現行 `MeanAudio.ode_wrapper`（`meanaudio/model/networks.py:593-602`）：

```
v = c·A + (1−c)·B        等價於    v = A + (c−1)·(A − B)
```

`A` = 條件分支預測、`B` = negative（或模型 stored null）分支預測、`c` = `cfg_strength`。
把 `(A−B)` 誇大 `c` 倍同時做了兩件事：**轉方向**（貼 prompt，這是想要的）與
**放大範數**（向量變長）。在音訊裡，latent 範數被放大出去就是波形幅度飽和。

### 我們自己的資料已經顯示這件事在發生

`negprompt_cfg_content_ablation_2026_08_31`（MusicCaps 1,024 seeded subset, seed 20260830,
CLAP b32）：

| 觀察 | 數字 |
|---|---|
| 純 CFG（對 stored null，無 negative 文字）PQ | cfg 0→3.0 **完全不動**（6.579~6.598），cfg 4.5 反而低於基準 0.05 |
| 純 CFG CLAP | 隨 cfg **單調 +0.020** |
| 純 CFG cfg 4.5 的 `crest_min` | **1.85**（失真） |
| ＋fidelity8 cfg 4.5 的 `crest_min` | **2.90**（健康） |
| 純 CFG 隨 cfg 上升 | 重心 1551→1860 Hz（變亮）、crest 2.93→1.85（變壓縮） |

也就是：**純 CFG 的 headroom 全部換成了 CLAP，PQ 一分都沒拿到，而且在 4.5 開始賠錢，
同時波形被壓扁。** 這正是「範數放大」的指紋。當時的解讀是「negative 文字是波形的穩定劑」
（該文件寫明是推測、未證）。ADG/APG 給的是另一個、可直接檢定的解釋：**問題不在於
negative 槽有沒有文字，而在於外插把向量拉長了**；幾何修正應該能在**不靠文字**的情況下
拿回同樣的穩定性。

### 文獻位置

`docs/literature/negative_prompting_and_prompt_engineering_2026_09_04.md` §2.3 與 §7b④
已把這條線列為優先，建議順序 **ADG → APG → LF-CFG**（三者都 training-free）：

- **ADG**（Angle Domain Guidance, ICML 2025, arXiv:2506.11039）：高 guidance 的失真來自
  **latent 樣本範數被放大**；約束幅度變化、只優化角度對齊。
- **APG**（ICLR 2025, arXiv:2410.02416）：把更新項拆成**平行於條件分支**（造成過飽和）與
  **正交**（提升品質）兩塊，只留正交。影像上 FID −10~50%、saturation −20~60%。
- LF-CFG（arXiv:2506.21452）走頻域，本實驗**不做**。

本實驗做 ADG 與 APG 兩者，ADG 排前面（診斷更貼 crest 崩塌）。

> **延伸（2026-09-23）**：方向備忘裡的第三種順序「**先 normalize 再減**」本預註冊未涵蓋
> （ADG 縮的是合成後的向量、APG 沿 `A` 拆，兩者都移不掉兩個分支之間的純幅度差）。
> 已另跑並收線，沿用本檔的 checkpoint／協定／driver、vanilla 對照讀本實驗的 cell cache：
> `results/guidance_geometry_prenorm_20260923_results.md`。結論：純 CFG 上 no-op；
> fidelity8 上拆掉了 negative 分支的範數煞車（crest −1.02、LUFS +1.69），early-kill 未過。

## 基準 arm 與既有錨點

固定 checkpoint：`phase8_qwen_caption10s_multisent_noq_full_stage2_200000`
（`c2p0_slot0` full, NoQ），也就是整條 negprompt 線的 arm。

全量 MusicCaps 5521 / MF25 / seed 42 / fp32 / NoMask、CLAP **batch 1**：

| 格 | CLAP | CE | CU | PC | PQ |
|---|---:|---:|---:|---:|---:|
| CFG 0 | 0.21494 | — | — | — | ≈6.579 |
| CFG 3.0 ＋ fidelity8 | **0.24624** | 7.2114 | 7.6251 | 5.1059 | **7.5992** |

逐檔配對 vs CFG 0：ΔPQ **+1.0199**（93.2% clip 改善）、ΔCE +0.924、ΔCU +0.903、
ΔPC −0.033。64 clip 抽樣 `crest_mean` 6.689 / `crest_min` 3.435 / clipped 0。

**唯一的負帳**：negprompt 讓 FAD **+0.0458**（越低越好，ATTM 協定），機制是把音檔推離
YouTube 錄音的參考分布（`project_negprompt_hurts_fad_2026_09_04`）。這是本實驗的第三端點。

## 介入定義（數學寫死）

令 `Â = A/‖A‖`，`delta = A − B`，vanilla `v = A + (c−1)·delta`。

**ADG（範數保持）**
```
v_adg = v · ( (1−γ) + γ·‖A‖/‖v‖ )       γ ∈ [0, 1]
```
γ=0 → 數值上等於 vanilla；γ=1 → 完全把合成向量縮回條件分支的範數。
參考範數用 `‖A‖`（不是 `‖B‖`）；pilot 另測 `‖B‖` 一格。

> 誠實註記：這是 ADG 論文**實務核心**（幅度約束、只保留角度變化）的最小實作，
> 不是論文完整演算法。結果文件一律寫成 “norm-preserving guidance (ADG-style)”，
> 不得寫成「復現 ADG」。

**APG（正交投影）**
```
delta_∥ = (delta · Â)·Â
delta_⊥ = delta − delta_∥
v_apg   = A + (c−1)·( delta_⊥ + η·delta_∥ )     η ∈ [0, 1]
```
η=1 → 數值上等於 vanilla；η=0 → 完全丟掉平行分量。APG 論文的 rescale 與 momentum
兩個附加項**本實驗關閉**，只測投影本身。

**範數／投影的軸**：預設**逐樣本、對所有元素**取（global）。pilot 另測**逐 latent frame**
一格；全量只用 pilot 選定的那一種，並在結果文件寫明。

## 實作約束（NEVER 相容）

- **不改 `meanaudio/model/networks.py`**（CLAUDE.md NEVER：不動 MeanAudio 類別）。
- **不改 `eval.py`**（被歷史 contract 綁 sha）。
- 驅動腳本在**自己的 process 內**用 `runpy.run_path('eval.py', run_name='__main__')`
  跑原封不動的 `eval.py`，並在跑之前 monkeypatch `MeanAudio.ode_wrapper`。
  `geometry='vanilla'` 時 patch 回傳**逐字相同的原式**，不是 γ=0／η=1 的等價式。
- 兩層檢查分開：**vanilla 模式**要音檔逐位元相同；**γ=0／η=1** 只要求 latent 相對誤差
  ≤1e-5（浮點結合律不同，不可能逐位元相同，事前寫明以免被誤判成 bug）。

## 階段與格子

### Stage A — pilot（MusicCaps 1,024 seeded subset, seed 20260830）

沿用 36 格矩陣那份 `musiccaps_subset1024.tsv`（已存在於
`nvme_experiment_artifacts/meanaudio/negprompt_ablation/`），目的是**選超參**，不是下結論。

兩個 negative 家族：
- **N0** = 純 CFG（negative 走模型 stored null，`--negative_prompt` 不傳）
- **N8** = canonical fidelity8 字串

每個家族在 **cfg 4.5**（headroom 最大、也是純 CFG 出事的那格）跑：

| 格 | 設定 |
|---|---|
| vanilla | 現行式，作為 run 內配對基準 |
| ADG γ=0.5 / γ=1.0 | 範數保持 |
| ADG γ=1.0（`‖B‖` 參考） | 參考範數的敏感度 |
| ADG γ=1.0（逐 frame） | 軸的敏感度 |
| APG η=0.5 / η=0.25 / η=0 | 正交投影 |

再加兩家族的 **cfg 3.0 vanilla** 各一格作為錨點。合計 **18 格 × 1,024 列**，
每格約 6 min 生成 ＋ 3 min 評分 → **約 2.7 h**。

### Stage B — 全量（MusicCaps 5521），只在 early-kill 未觸發時跑

每家族各取 pilot 最佳的 1 個 ADG 設定與 1 個 APG 設定，在 cfg 3.0 與 4.5 各跑一格
（8 格），另補 vanilla 參考 3 格（N8 cfg4.5、N0 cfg3.0、N0 cfg4.5；N8 cfg3.0 已存在且
沿用既有音檔）。**11 格 × 約 45 min ≈ 8.3 h**，FAD 另計。

### Early-kill gate（寫進 Stage A 的腳本，不靠人盯）

Stage A 結束後，若**沒有任何一格**同時滿足下列兩條，**不啟動 Stage B**，直接收線寫 null：

1. `ΔPQ ≥ +0.28`（= 2× 推論 seed 底線 0.142，`reference_inference_seed_noise_floor`）
   **或** `Δcrest_mean ≥ +0.5` 且 `crest_min` 不下降；
2. `ΔCLAP ≥ −0.005`（不拿 CLAP 換上面那些）。

兩條都對**同家族同 cfg 的 run 內 vanilla 格**配對比較。

## 端點

**Primary — 幾何能不能把「純 CFG 的浪費」換成 PQ。**
N0 家族、cfg 4.5、`ADG − vanilla` 與 `APG − vanilla` 的逐檔配對 ΔPQ、Δcrest_mean、
Δcrest_min、ΔCLAP。事前預測（若範數放大是主因）：PQ 上升、crest 回升、CLAP 大致持平。

**Secondary — 幾何能不能把 N8 家族的可用 cfg 往上推。**
N8 家族 cfg 3.0 → 4.5 的 PQ 變化，vanilla 是 7.6495 → 7.6071（**已見頂並回落**，b32 子集），
而 CLAP 仍在升（0.2598 → 0.2627）。問：幾何修正後 cfg 4.5 的 PQ 還會不會回落。

**Tertiary — FAD。**
Stage B 的入選格補算 FAD（n=2048，MusicCaps 官方參考音訊，與 `best_results.md` 同抽樣）。
negprompt 線唯一的負帳是 FAD +0.0458；若幾何修正能在保住 CLAP 的同時收回一部分 FAD，
那是這條線目前拿不到的東西，也直接改寫「對打 ATTM」的帳目。

**Covariates（每格必報）**：`level_lufs_mean`、`level_rms_mean`、`level_crest_mean`、
`level_crest_min`、`level_clipped_n`、`level_silent_n`、頻譜重心。

## 閘門

| 閘門 | 判準 |
|---|---|
| vanilla 複製（音訊層） | `geometry='vanilla'` 的 N8 cfg 3.0 全量格，與既有 `c2p0_slot0_full_noq` 音檔逐檔 `sha256` **完全相同**，容忍 0 筆 |
| 等價式數值檢查 | γ=0 與 η=1 對 vanilla 的 latent 相對誤差 ≤ 1e-5（32 個 clip × 25 步抽查） |
| 響度閘門 | 任一 geometry 格與其配對 vanilla 的 `level_lufs_mean` 差 > 0.5 LU 時，**必須同時報響度對齊後的分數**（兩邊 normalize 到同一 LUFS 重評） |
| 靜音閘門 | `level_silent_n` 相對 vanilla 增加 > 2× 即標記（`project_slot0nm_silence_mode` 的教訓：平均持平底下可能藏劣化模式） |
| AES 分數複製容忍 | 5e-3（batch 組成敏感，`reference_aes_batch_composition_sensitivity`） |

第一與第二道任一未過 → 視為實作 bug，所有 geometry 數字不可讀
（`feedback_suspect_bug_before_explaining`）。

**響度閘門為什麼是必要的而不是裝飾**：063/065 已定案 **PQ/CU 越小聲越高、CLAP 越大聲越高**
（`project_gain_ladder_063_result`、`project_floor_ladder_065`）。ADG 的整個作用就是
**改變輸出的幅度**，所以它必然同時移動 LUFS。**不鎖響度就無法分辨「幾何讓音訊變好」與
「幾何讓音訊變小聲、而 PQ 本來就偏好小聲」。** 這是本實驗最容易自欺的一格。

## 判定規則（事前寫定）

| 觀察 | 寫法 |
|---|---|
| N0 cfg4.5 的 ΔPQ 過門檻、crest 回升、CLAP 持平，且響度對齊後仍成立 | 支持「範數放大是高 guidance 失真的主因」，且 negative 文字的穩定作用**可被幾何取代** |
| PQ 上升但響度對齊後消失 | 寫成**響度效應**，不是品質改善；照 063 的形狀報 |
| crest 回升但 PQ / CLAP 不動 | 只能寫「幾何修正了波形形狀，兩個指標量不到」；crest 本身不是目標（061 已定 crest 不可當訓練目標） |
| ADG 與 APG 方向不一致 | 分開報，不合併宣稱；並報兩者的 LUFS 差 |
| 全部 CI 跨零 | **證據不足，不是等效**；逐點區間、不做多重比較校正、不自動晉升 |
| FAD 改善但 CLAP 下降 | 寫成 trade-off 的移動，不是免費午餐（與 `project_negprompt_hurts_fad` 同一寫法） |

## 限制

1. 單一 checkpoint（`c2p0_slot0` full NoQ）、單一資料集（MusicCaps）、單一推論 seed（42）。
   `reference_inference_seed_noise_floor`：全量推論 seed 的底線是 PQ 0.142 / CE 0.296 /
   CLAP 0.0003，所以 CLAP 的小效果可讀、PQ 的小效果不可讀。
2. ADG 是論文實務核心的最小實作，不是完整復現（見上）。負面結果**不能**寫成
   「ADG 在音訊上無效」，只能寫成「範數保持這一項在本設定下無效」。
3. CLAP 與 AES 都不是聽感 ground truth（`reference_quality_metric_validity_literature`：
   SongEval 的 PC 相關性只有 0.408、乾淨音訊上指標會塌到 chance）。若 Stage B 出現
   正結果，必須補主觀試聽（`docs/eval/subjective_prompts.md`）才能對外宣稱。
4. MeanFlow 25 步是少步 regime。NAG/VSF 兩篇專門指出 CFG 的 negative 分支在少步下行為
   不同（`negative_prompting_and_prompt_engineering` §2.3），所以本實驗的結果**不可**
   外推到多步 flow matching。
