# Autoresearch：推論期 CLAP 搜尋線（設計登錄，2026-10-27）

> **狀態**：`preregistered_not_started`。本檔只定規格，**尚未啟動正式搜尋**、**尚未排任何正式 queue job**。
> **機器可讀的預登錄**：`docs/experiments/autoresearch_clap_20261027_prereg.json`（以下所有數字以它為準；本檔若與它不一致，以 JSON 為準並回頭修本檔）。
> **範圍**：只動推論期（`runpy` + monkeypatch），**不重訓、不改架構、不改 `networks.py`／`eval.py`／`eval_metrics.py`／`mc_mf25_eval*.sh`**。

---

## 0. 一頁摘要

| 項目 | 值 |
|---|---|
| 固定模型 | TTM 表 B 的 ours 列：`slot0clean nmv2matched NoQ full`，Stage 2 200k `ema_final`，seed **s14159265**（primary）；s16180339／s27182818 只做最終確認 |
| 目標函數 | **MusicCaps CLAP（laion_clap HTSAT-base，batch 1）**，在 **lvl30（逐檔純量增益到 −30.0 LUFS）** 音訊上 |
| 搜尋空間 | CFG scale、guidance interval、CFG schedule、negative prompt 文字、ADG／APG、MeanFlow steps／time grid（7 族，rung 1 共 38 組） |
| 選模資料 | `dev256` → `dev1024`（皆取自 records [0,1024)），**heldout1024 = records [1024,2048) 全程不參與選模** |
| 主要門檻 | lvl30 CLAP Δ ≥ **0.005**（dev）、heldout ≥ **0.004**、組合增益 ≥ **0.002**；FAD +0.30、AES PQ −0.10 為強制不採納閘門 |
| 迴圈 | Phase 0 直接跑（<30 min），之後**每個 rung 一個 p2 queue job**，編號 **153–157**；rung 之間由代理分析、寫下一輪 configs 與 contract |
| 單 trial 時間 | dev256 **163 s**／dev1024 **475 s**／heldout1024 **475 s**／full 5521 **2215 s** |
| 計畫 trial 數 | **66**（38 + ≤10 + ≤6 + 9 + 3），另加 Phase 0 的 4–6 個基線／校正 trial |
| 總 GPU 時數 | **7.75 h**（含 0.4 h 分析；GPU 本體 ≈ 7.35 h），soft 8 h、**hard cap 15 h** |
| 磁碟 | 每 trial 峰值 0.13／0.47／2.5 GB，FAD 暫存另 1.23 GB，最壞 **3.8 GB**；NVMe 目前 64 GB 可用 |

**一句話**：在一個固定 checkpoint 上，只改推論期的 guidance／prompt／步數／時間格，用 256 檔快篩、1024 檔複驗、1024 檔獨立 heldout 三關擋 winner's curse，再用 full 5521 × 3 checkpoint 做最終確認；全程 ≤ 8 h GPU，且 FAD／PQ／靜音／削波／響度五類閘門都可以單獨否決一個 CLAP 較高的設定。

---

## 1. 固定模型與目標函數

### 1.1 模型

| 角色 | 路徑 | sha256（前 12） |
|---|---|---|
| primary | `~/exps_nvme/phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s14159265_stage2_200000/…_ema_final.pth` | `7cc14d093480` |
| confirm 1 | `…_s16180339_stage2_200000/…_ema_final.pth` | `997077fbc386` |
| confirm 2 | `…_s27182818_stage2_200000/…_ema_final.pth` | `4f98eefc7de4` |

固定 eval 旗標（與 canonical `mc_mf25_eval.sh` 相同）：

```
--variant meanaudio_s --use_meanflow --num_steps 25 --seed 42 --full_precision \
--no_text_attention_mask --no_q          # 模型是 NoQ 訓練 → 一律 --no_q
```

這三個 checkpoint 就是 TTM 表 B 「ours」列的三個 seed（118 的 lvl30 數字來源）；確認階段若任何一個檔案的 sha 與上表不符，整條確認作廢。

### 1.2 為什麼 objective 是 lvl30 CLAP

- 063 已證明 **CLAP 隨播放響度上升**：不鎖響度時，「把音檔變大聲」是免費的 CLAP 槓桿，會把搜尋引導到響度而不是內容對齊。
- lvl30 是既有協定：`scripts/eval/level_match_rescore.py` 對每檔做一次純量增益到 −30.0 LUFS（peak_cap 0.999），所有 trial 與所有 baseline 走**同一條渲染路徑**，比較才成立。
- lvl30 的 CLAP 是**新尺度**（同一 checkpoint full 5521：cfg0 0.1927、cfg3+neg 0.2325），不可與歷史 raw CLAP（0.2148／0.2432）混用。這條寫進 prereg 的 risks。

### 1.3 Baseline（full 5521，三個 checkpoint，六格）

| checkpoint | cfg | lvl30 CLAP | lvl30 PQ | raw CLAP | raw PQ | raw silent_n | raw FAD（1924 pairs） |
|---|---|---|---|---|---|---|---|
| s14159265 | cfg0 | 0.1927 | 6.678 | 0.2148 | 6.445 | 14 | 3.2640 |
| s14159265 | cfg3+neg | **0.2325** | 7.593 | 0.2432 | 7.424 | 9 | 5.0836 |
| s16180339 | cfg0 | 0.1949 | 6.676 | 0.2141 | 6.459 | 22 | 3.1927 |
| s16180339 | cfg3+neg | **0.2365** | 7.603 | 0.2497 | 7.439 | 15 | 5.4385 |
| s27182818 | cfg0 | 0.1922 | 6.637 | 0.2130 | 6.414 | 38 | 3.3426 |
| s27182818 | cfg3+neg | **0.2375** | 7.548 | 0.2535 | 7.392 | 35 | 4.9986 |

- **cfg3−cfg0 的 lvl30 CLAP 差距 = +0.042**（三 checkpoint 平均；逐一為 0.0398／0.0415／0.0453）。這是「現有 guidance 值多少」的尺；最小可偵測效果 0.005 就是它的 ~12%。
- 三 checkpoint 的 lvl30 CLAP 散布：cfg3 **0.0050**、cfg0 **0.0027**（這是模型間差異，不是推論雜訊）。
- FAD 的三 checkpoint 散布：cfg3 **0.4399**、cfg0 **0.1499** → 256 檔子集上的 FAD 閘門必須用**同子集**的三 checkpoint 實測雜訊來定，不能沿用 full 的 0.30。

---

## 2. 需求 1：防止過擬合

### 2.1 資料切分（record-aware，無 shuffle）

| split | csv records | n | 用途 | sha256（前 8） |
|---|---|---|---|---|
| `dev1024` | [0, 1024) | 1024 | rung 2/3 選模 | `3bc818c8` |
| `dev256` | dev1024 的前 256 筆 | 256 | rung 1 快篩 | （dev1024 的前綴） |
| `heldout1024` | [1024, 2048) | 1024 | **只做 rung 4 確認，全程不參與選模** | `a87b20ad` |
| `rest3473` | [2048, 5521) | 3473 | 只出現在 full 5521 的報告裡 | `b230d552` |

- 切分由 `scripts/eval/autoresearch_clap_20261027/split_subsets.py` 產生，寫出 `manifest.json`（source sha256 `de567b13…`、每段 sha256、ids_sha256）。切分 commit 後不可漂移。
- **按 csv record 切，不按實體行**：MusicCaps TSV 有 5 條引號內跨行 caption（第一條在實體行 463），`head -n` 會少筆（memory `reference_musiccaps_tsv_multiline_captions`）。
- 不用任何舊的 `musiccaps_subset1024.tsv` 之類的檔案：舊子集抽樣規則不明，混用等於換了題目。

### 2.2 配對性（paired vs not）

`eval.py` **一次只生成一列**（`generate_mf([prompt], …)`），整個 run 只 seed 一次 RNG。因此：

| split | 與 canonical full cell 的關係 | 統計處理 |
|---|---|---|
| dev256／dev1024 | 是 full run 的**連續前綴** → x0 逐位元相同 | **逐 clip 配對**，Δ 的變異數小一個檔次（G0 用 256/256 sha 證明） |
| heldout1024 | 不是前綴 → x0 不同 | 兩樣本比較，誤差估 `σ√2` |
| full5521 | 與 canonical 相同 rows | 逐 clip 配對（但含 2048 筆選模過的 records → 只當論文數字，不當獨立檢定） |

推論 seed 7／161803 的 run 與 seed 42 不配對，只用來量 seed 雜訊與 rung 4 的 seed 方向一致性。

### 2.3 雜訊底線

| 量 | 值 | 出處 |
|---|---|---|
| 推論 seed 的 full CLAP 差 | **0.0003** | memory `reference_inference_seed_noise_floor`（一次 seed pair 的 \|Δ\|，保守當 ~1σ） |
| 推論 seed 的 full PQ 差 | **0.142** | 同上（PQ 的閘門用這個量級） |
| σ_seed(n) | 0.0014（n=256）、0.0007（n=1024）、0.0003（n=5521） | `0.0003·√(5521/n)` |
| C0 重複（seed 42） | 必須逐 clip sha256 相同 | 決定性閘門 |
| C1／C2（seed 7、161803） | 實測 Δ 進 `measured_floor_rule` | rung 1 的兩個 null 控制 |
| FAD 子集雜訊 | 用同子集三 checkpoint 的 baseline 實測，Phase 0 完成後才定 | 不沿用 full 的 0.30 |

**底線規則**：每個 rung 的 σ 用 `max(σ_seed(n), |ΔC1|, |ΔC2|)`（C1/C2 只在 dev256 量，高 rung 沿用並按 `√(n)` 縮放）。這樣「底線」不是文獻數字，而是這條線自己量出來的。

### 2.4 門檻

| 用途 | 門檻 | 理由 |
|---|---|---|
| dev（rung 1–3） | Δ lvl30 CLAP ≥ **0.005** | = cfg3−cfg0 差距（0.042）的 ~12%；在 n=256 是 3.6σ、n=1024 是 7.1σ |
| heldout（rung 4） | ≥ **0.004** | 獨立檢定，樣本 9 格（3 ckpt × 3 seed），留一點餘裕給兩樣本誤差 |
| 組合（rung 3） | 比最佳單項多 ≥ **0.002** | 小於此值就當「沒有額外貢獻」，直接用單項 |

### 2.5 多重比較與 winner's curse

- rung 1 有 **35 個真設定**（38 減 3 個控制）。
- n=256、σ=0.0014 時，35 個 null 的最大值期望值 ≈ **0.0031**，小於門檻 0.005。
- 門檻 0.005 對應的 familywise α ≈ **0.6%**（35 × P(Z>3.57)）。
- rung 2 最多 10 個候選，暫用 σ_used≈0.0013（Phase 0 量到 C1/C2 後若更大，`E[max]` 按比例上修）→ null 最大值期望 ≈ **0.0020**，仍低於 0.005。
- **偏差揭露**：rung 1–3 的 Δ 都是「選出來的」，必然上偏；唯一不偏的估計是 **heldout1024**。full 5521 內含 2048 筆選模 records，它的數字是**論文用數字，不是獨立檢定**。這條寫進 prereg 的 `multiplicity.bias_note`，報告時必須一起寫。

### 2.6 negative prompt 洩漏稽核

規則：**任何 negative prompt 字串不得是任何 dev caption 的大小寫無關子字串**；prompt 只能手寫，不能從 caption 挖。

| prompt id | 文字 | dev1024 命中數 |
|---|---|---|
| `stock` | `low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi` | 0 |
| `short` | `low quality recording, noisy` | 0 |
| `poorq_noisy` | `poor quality, noisy` | 0 |
| `ndm` | `noisy, distorted, muffled` | 0 |
| `ahc` | `amateur recording, hiss, clipping` | 0 |
| `pfld` | `poor fidelity, lo-fi, distant` | 0 |
| `extended` | stock + `, clipping, room noise` | 0 |
| `irrelevant` | `banana, submarine` | 0 |

**已否決的候選**：`low quality`（在 240 條 dev caption 裡出現）→ 註冊前就丟掉。這條留在 prereg 的 `rejected_candidate`，避免以後有人「只改短一點」又把它加回來。

### 2.7 閘門（mandated＋自訂）

| 閘門 | 規則 | 為什麼 |
|---|---|---|
| **G0 copy** | Phase 0：stock cfg3 在 dev256 的 256 個 flac 要與 canonical full cell 的 sha256 完全相同 | 先證明 harness 是 canonical 路徑（073 的做法） |
| **G1 determinism** | `C0_dup_seed42` 也要 256/256 相同 | patch 沒漏進 vanilla 路徑 |
| **G2 FAD**（強制） | `FAD(trial) − FAD(同子集 baseline) > +0.30` → 不採納。**在 raw 音訊上算**（canonical `*_fad` 格就是 raw），full trial 用 2048 抽樣→1924 對 | 使用者指定；CLAP 與保真度在 064／084 多次反向 |
| **G3 PQ**（強制） | `AES PQ(lvl30 trial) − PQ(lvl30 baseline) < −0.10` → 不採納 | 0.10 是推論 seed PQ 底線 0.142 的 0.7 倍，只會擋到真劣化 |
| **G4 silence** | `level_silent_n(trial) − level_silent_n(baseline) > max(3, 2×baseline)` → 拒絕 | 065／126：CLAP 給純靜音的分數高過 39% 的音樂，靜音逃逸是最便宜的作弊 |
| **G5 clipping/crest** | raw `clipped_n` 比 baseline 多 → 拒絕；raw `crest_mean` 比 baseline 低 0.5 → 拒絕 | 2026-04-21 的高 cfg 飽和事故；crest 是波形崩潰讀數 |
| **G6 loudness** | lvl30 `peak_capped_n > max(baseline+3, 0.005n)` → 拒絕；`|LUFS+30| > 0.05` 的 clip 超過 1% → 拒絕 | 響度對齊失敗會直接污染 objective（baseline 的 peak_capped_n 在 full 5521 是 3） |
| **G7 integrity** | metrics 全有限；音檔數 == n；id 集合與 split 完全一致、無重複 | 缺檔／多檔會讓 CLAP 均值靜默偏移 |
| **G8 prompt leak** | 2.6 的稽核（目前全 0） | 防止把 eval caption 的內容放進負向槽 |

**FAD 待查機制**：rung 1 的 256 檔 FAD 太吵，只差 G2 的設定可以標成 `FAD-pending` 帶進 rung 2（最多 2 個名額），但最終採納**仍必須**在 rung 5 過 G2。這是為了不讓雜訊殺掉真贏家，同時不讓它免驗。

### 2.8 最終判定（win criterion）

- **rung 4（唯一不偏的檢定）**：heldout1024、3 checkpoint × 3 seed（42／7／161803）共 9 格
  1. 9 格平均 Δ lvl30 CLAP ≥ **+0.004**
  2. ≥ 2/3 個 checkpoint 的平均為正
  3. 3 個 seed 的平均全為正
  4. G2–G7 在 primary checkpoint 的 heldout 子集全過
- **rung 5（論文數字）**：full 5521、seed 42、3 checkpoint
  1. 平均 Δ ≥ **+0.005**，且 ≥ 2/3 checkpoint 為正
  2. G2（FAD ≤ +0.30，對同 checkpoint 的 cfg3 baseline）、G3–G7 全過
- **判定語彙**：rung 4+5 都過 = win；rung 5 過但 checkpoint < 2/3 為正 = checkpoint-specific（不採納）；rung 4 沒過 = not replicated（不管 dev 多漂亮）；任何強制閘門沒過 = not adopted（即使 CLAP 上升）。

---

## 3. 搜尋空間（rung 1：38 組）

### 3.1 七族與理由

| 族 | 動什麼 | 為什麼值得搜 | 既有證據 |
|---|---|---|---|
| **F1 cfg scale** | `--cfg_strength` ∈ {0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0} | 現在只量過 0 與 3；曲線形狀未知 | 073：純 CFG cfg4.5 的 PQ 浪費；084：CFG0 也能拿 PQ |
| **F2 negative prompt** | 8 條已註冊字串（cfg 3.0） | 文字負向是這條專案最大的單一槓桿 | 086／107／108：短負向 ≈ 原 DiD 的 36–44%；無關文字也有 +0.33 |
| **F3 guidance interval** | 只在 `lo < t ≤ hi` 開 guidance，其餘走 conditional-only | 增益集中在早段，後段 guidance 可能只是花錢 | 075：fidelity8 增益 93–100% 來自前 9/25 步 |
| **F4 cfg schedule** | `c(t) = 1 + 2·r(t)`，r ∈ {1−t, t, 4t(1−t), √(1−t)} | 同樣的「早段多、晚段少」，但連續 | 同上；073 的 ADG 顯示晚段推力效率低 |
| **F5 geometry** | ADG γ ∈ {0.25, 0.5, 1.0}、APG η ∈ {0, 0.5, 1.0} | 改外插幾何，不重訓 | 073：ADG 只收回純 CFG 浪費的 2%；prenorm 在 fidelity8 上拆掉煞車 |
| **F6 steps** | `--num_steps` ∈ {12, 16, 50} | 步數是最貴的旋鈕，先問 12/16 掉多少 | 步數預算 vs negprompt ROI：加步數是最差的槓桿 |
| **F7 time grid** | cos／smoothstep（25 步） | 把解析度移到早段或中段 | 同 F3/F4 的機制假設 |

**patch 面（重要）**：38 組裡有 **20 組完全不 patch**（F1 7 + F2 7 + F6 3 + C 3），直接走 `eval.py` 原路徑；只有 F3／F4／F5 需要 patch `MeanAudio.ode_wrapper`（它收得到當步的 `t`，可依 `t` 決定 cfg），F7 需要 patch `MeanFlow.run_t0_to_t1`（換時間格）。這讓 copy gate 與大部分的 trial 都站在 canonical 路徑上。

| 族 | 需要 patch | 介面 |
|---|---|---|
| F1 cfg | 否 | `--cfg_strength` |
| F2 prompt | 否 | `--negative_prompt` |
| F3 interval | 是 | `MeanAudio.ode_wrapper`：`c(t)=3.0 if lo<t≤hi else 0.0`（程式上 `cfg<1` 只算 conditional，少一次 forward） |
| F4 schedule | 是 | `MeanAudio.ode_wrapper`：`c(t)=1+2r(t)`，`c≤1` 時走 conditional-only |
| F5 ADG/APG | 是 | `MeanAudio.ode_wrapper`（073 的公式，原封不動搬） |
| F6 steps | 否 | `--num_steps` |
| F7 grid | 是 | `MeanFlow.run_t0_to_t1`（`self.steps` 不變，只換 grid） |

### 3.2 rung 1 完整 38 組

| id | 族 | cfg | prompt | interval | schedule | geometry | steps | grid | 備註 |
|---|---|---|---|---|---|---|---|---|---|
| F1_cfg0.0 | F1 | 0.0 | stock | — | — | vanilla | 25 | uniform | 等於 eval.py cfg0 路徑（與 canonical cfg0 交叉驗證） |
| F1_cfg1.5 | F1 | 1.5 | stock | — | — | vanilla | 25 | uniform | |
| F1_cfg2.0 | F1 | 2.0 | stock | — | — | vanilla | 25 | uniform | |
| F1_cfg2.5 | F1 | 2.5 | stock | — | — | vanilla | 25 | uniform | |
| F1_cfg3.5 | F1 | 3.5 | stock | — | — | vanilla | 25 | uniform | |
| F1_cfg4.0 | F1 | 4.0 | stock | — | — | vanilla | 25 | uniform | |
| F1_cfg5.0 | F1 | 5.0 | stock | — | — | vanilla | 25 | uniform | 飽和風險，靠 G5 擋 |
| F2_short | F2 | 3.0 | short | — | — | vanilla | 25 | uniform | |
| F2_poorq_noisy | F2 | 3.0 | poorq_noisy | — | — | vanilla | 25 | uniform | |
| F2_ndm | F2 | 3.0 | ndm | — | — | vanilla | 25 | uniform | |
| F2_ahc | F2 | 3.0 | ahc | — | — | vanilla | 25 | uniform | |
| F2_pfld | F2 | 3.0 | pfld | — | — | vanilla | 25 | uniform | |
| F2_extended | F2 | 3.0 | extended | — | — | vanilla | 25 | uniform | |
| F2_irrelevant | F2 | 3.0 | irrelevant | — | — | vanilla | 25 | uniform | 對照：非空文字層本身值多少（107） |
| F3_early_third | F3 | 3.0 | stock | (0.667, 1.0] | — | vanilla | 25 | uniform | |
| F3_mid_third | F3 | 3.0 | stock | (0.333, 0.667] | — | vanilla | 25 | uniform | |
| F3_late_third | F3 | 3.0 | stock | (0.0, 0.333] | — | vanilla | 25 | uniform | |
| F3_early_two_thirds | F3 | 3.0 | stock | (0.333, 1.0] | — | vanilla | 25 | uniform | |
| F3_early_half | F3 | 3.0 | stock | (0.5, 1.0] | — | vanilla | 25 | uniform | |
| F3_late_half | F3 | 3.0 | stock | (0.0, 0.5] | — | vanilla | 25 | uniform | |
| F4_ramp_up | F4 | — | stock | — | 1−t | vanilla | 25 | uniform | t=1 起點 c=1，越晚越強 |
| F4_ramp_dn | F4 | — | stock | — | t | vanilla | 25 | uniform | 起點最強，越晚越弱 |
| F4_bump_mid | F4 | — | stock | — | 4t(1−t) | vanilla | 25 | uniform | 中段最強 |
| F4_sqrt_up | F4 | — | stock | — | √(1−t) | vanilla | 25 | uniform | |
| F5_adg_g0.25 | F5 | 3.0 | stock | — | — | ADG γ0.25 | 25 | uniform | |
| F5_adg_g0.5 | F5 | 3.0 | stock | — | — | ADG γ0.5 | 25 | uniform | 073 的 γ0.5 曾 +0.096 PQ |
| F5_adg_g1.0 | F5 | 3.0 | stock | — | — | ADG γ1.0 | 25 | uniform | |
| F5_apg_e0.0 | F5 | 3.0 | stock | — | — | APG η0.0 | 25 | uniform | |
| F5_apg_e0.5 | F5 | 3.0 | stock | — | — | APG η0.5 | 25 | uniform | |
| F5_apg_e1.0 | F5 | 3.0 | stock | — | — | APG η1.0 | 25 | uniform | **數值控制**：數學上等於 vanilla，但 float 順序不同（073 已註明） |
| F6_steps12 | F6 | 3.0 | stock | — | — | vanilla | 12 | uniform | |
| F6_steps16 | F6 | 3.0 | stock | — | — | vanilla | 16 | uniform | |
| F6_steps50 | F6 | 3.0 | stock | — | — | vanilla | 50 | uniform | 成本 2×，只有 CLAP 明顯上升才值得 |
| F7_grid_cos | F7 | 3.0 | stock | — | — | vanilla | 25 | cos | |
| F7_grid_smoothstep | F7 | 3.0 | stock | — | — | vanilla | 25 | smoothstep | |
| C0_dup_seed42 | C | 3.0 | stock | — | — | vanilla | 25 | uniform | 決定性：必須 256/256 sha 相同 |
| C1_null_seed7 | C | 3.0 | stock | — | — | vanilla | 25 | uniform | seed 底線 |
| C2_null_seed161803 | C | 3.0 | stock | — | — | vanilla | 25 | uniform | seed 底線 |

> F3 的區間寫法：`(lo, hi]` 對應程式 `lo < t ≤ hi`；`t` 是該步的起點時間（1.0 起、0.0 止）。

### 3.3 rung 2–3 的生成規則（adaptive，但規則先定）

1. **rung 2**：rung 1 過門檻的設定中取 **top 8、每族最多 2 個**（家族多樣性），加上最多 2 個 `FAD-pending`，最多 10 個；在 dev1024 重跑。
   - 若過門檻的少於 3 個：直接取 top 3（即使低於門檻），該輪標記 `underpowered`，結論只能寫「無設定過門檻」。
2. **rung 3**：把 rung 2 的優勝者做**兩兩組合**（最多 6 個）：互補族（F2×F3、F2×F4、F2×F5、F1×F2、F1×F3、F3×F5）；只在 dev1024 上跑，Δ 要贏最佳單項 ≥ 0.002 才採用組合。
3. **rung 4**：只有一個贏家（組合或單項），跑 heldout1024 × 3 checkpoint × 3 seed。
4. **rung 5**：同一個贏家，full 5521 × 3 checkpoint。

rung 2 之後不再擴大搜尋空間（避免「邊看邊發明」的無限膨脹）；若 rung 1 全滅，就報 null 並收線（E1）。

---

## 4. 需求 2：迴圈機制

### 4.1 形式：混合式（hybrid）

| 階段 | 形式 | 為什麼 |
|---|---|---|
| Phase 0（copy gate、FAD parity、子集 baseline、seed 校正） | **直接跑**（< 30 min） | contract 的固定成本比工作本身高；但仍要 `hold_lock.py`＋`notify_on_exit` |
| rung 1–5 | **每個 rung 一個 p2 queue job** | 每個 rung 0.8–2.1 h，值得有 contract、done/held 分類與 Discord 通知 |
| rung 之間的判斷 | 由派工代理（本 session 的下一輪）跑 `analyze.py`／`make_round_configs.py` | 需要判斷（例如 FAD-pending 要不要帶），不適合寫死進 queue |

### 4.2 Queue job 編號與 contract

- 編號 **153–157**（152 已被 `bench_r2_J5c_singmos` 佔用）：153=R1、154=R2、155=R3、156=R4、157=R5。
- 每個 job 的 contract 先寫在 `docs/experiments/`（`autoresearch_153_r1_contract.json` … `157_r5_contract.json`），再放 launcher 進 `gpu_queue/p2/pending/`（順序不可顛倒；`pending/` 裡不能有 dotfile）。
- contract 沿用 bench_r2 的 schema：`schema_version: 1`、`document_kind: eval_only_metrics_contract`、`launch_authorization`、`bindings`（每個用到的腳本算 sha256）、`commands.{preflight,run,postflight}`（`python -I`）、`completion_evidence.eval_only: true` + `reports`、`resume.from_scratch_with_autoresume`、`storage.min_free_bytes`、`watcher.stall_seconds`、`queue_role: p2`。
- **guest 要自己寫**（`scripts/eval/autoresearch_clap_20261027/guest.py`），不能直接用 `gpu_queue/harn_guest.py`：通用 guest 硬編碼 `validate_cfg0_report` 的閘門，對 eval-only 不適用（bench_r2 也是為此自己寫 `bench_r2_guest.py`）。

### 4.3 喚醒與交接

派工方在同一個 shell 起背景 watch（不可用 sleep 輪詢）：

```
python3 ~/.claude-dispatch/claude_dispatch.py after \
  --wait '/home/kojiek/gpu_queue/p2/*/<JOB>.terminal.json' \
  --task <task_id> --log <driver.log> \
  --message '<下一輪指令：analyze.py → make_round_configs.py → 寫 contract → 放 launcher>' \
  --poll-seconds 300 --check-minutes 60 --stall-minutes 45 --timeout-hours 6
```

job 結束（`*.terminal.json` 出現）→ 自動喚醒 → `dispatch_continue` 做 rung 之間的判斷。

### 4.4 Resume

- **trial 級**：`summary.json` 存在且其 `config_sha` 與本輪 configs 相同 → 跳過。
- **cell 級**：`eval.py` 的部分音檔不可續（單一 RNG、skip 不抽 noise）→ 一律整個 cell 重跑（wrapper 已這樣做）。
- 每個 queue job 的 guest 自己吃「同一個 configs.json 的剩餘 trial」，所以重跑 job 不會重複燒 GPU。

---

## 5. 需求 3：預算與停止規則

### 5.1 成本模型（每個數字的出處）

| 項 | 值 | 出處／指令 |
|---|---|---|
| `eval.py` 啟動（載模型、T5、vocoder） | 23.0 s（實測 cold 24.1／warm 14.7，取中間值） | `stat -c %W <cell>/audio` 與首個 flac mtime 差 |
| 生成 CFG0 | **0.1962 s/clip** | `…_cfg0/audio` first→last flac 1083.0 s / 5520；`gen.log` tqdm `18:03` |
| 生成 CFG≥1（每步 2 次 forward） | **0.3142 s/clip** | 同上，cfg3_neg 1734.4 s；tqdm `28:55` |
| `eval_metrics.py` 模型載入 | 27.0 s／次 | 一次子程序載 CLAP+AES+WavLM；5521 檔全量時被攤平，小 cell 不能忽略 |
| metrics 每 clip（載入除外） | 0.022 s | full cell last flac→metrics.json 147.6 s（cfg0）／154.9 s（cfg3） |
| raw level-only（`--skip_clap --skip_aes`） | **0.74 s／256 檔（含載入）** | 本輪實測（§10.1）；1024 檔按比例 ≈ 3 s |
| lvl30 純量增益 | **0.00644 s/clip** | 64 檔 8 workers 實測 0.412 s |
| FAD（估計，**未實測**） | 25 s（256 檔）／71 s（1024）／125 s（1924 對） | 由 `score_fad` 的 16 kHz 轉檔＋VGGish 推估；Phase 0 會換成實測 |
| MEva | 0.163 s/clip | 097/112 的實測 |

> **注意**：`level_match_rescore.py` 的 lvl30 cell 在 2026-10-12 的批次裡花了 ~577 s，是因為當時 36 個 cell 並行（GPU 爭用）。queue 內 trial 是**序列**跑的，單跑成本用上表的乾淨值；Phase 0 會再量一次序列值。

### 5.2 單 trial 時間（公式與結果）

rung 1–4（子集 n）：

```
23 + 0.3142·n + (27 + 0.022·n) + level_only(n) + 0.00644·n + FAD(n)
```

| split | n | 生成 | metrics | raw level | 增益 | FAD | **合計** |
|---|---|---|---|---|---|---|---|
| dev256 | 256 | 80.4 s | 32.6 s | 0.7 s | 1.6 s | 25 s | **163 s** |
| dev1024 | 1024 | 321.7 s | 49.5 s | 3.0 s | 6.6 s | 71 s | **475 s** |
| heldout1024 | 1024 | 321.7 s | 49.5 s | 3.0 s | 6.6 s | 71 s | **475 s** |
| full5521（rung 5） | 5521 | 1734.7 s | 148.5 s（raw，含 level）＋148.5 s（lvl30） | 0（已含在 raw） | 35.5 s | 125 s | **2215 s** |

（rung 1–4 只算 raw level 與 lvl30 全量；rung 5 才算 raw CLAP/AES＋MEva。rung 5 的 2215 s 已含 setup 23 s。）

### 5.3 階段、trial 數與總時數

| 階段 | 形式 | 工作 | trial 數 | 單 trial | 小時 |
|---|---|---|---|---|---|
| **P0** | 直接 | copy gate ×1、FAD parity ×1、子集 baseline（dev256／dev1024／heldout 的 CLAP/PQ/FAD）、seed 校正 | 4–6 | 163–475 s | **0.20** |
| **R1** | queue 153 | 38 組在 dev256（含 3 控制） | **38** | 163 s | **1.72** |
| **R2** | queue 154 | top 8（＋≤2 FAD-pending）在 dev1024 | **≤10** | 475 s | **1.32** |
| **R3** | queue 155 | ≤6 組合在 dev1024 | **≤6** | 475 s | **0.79** |
| **R4** | queue 156 | 贏家 × 3 ckpt × 3 seed 在 heldout1024 | **9** | 475 s | **1.19** |
| **R5** | queue 157 | 贏家 full 5521 × 3 ckpt＋FAD＋MEva | **3**（＋MEva） | 2215 s | **2.13** |
| 分析 | session | 匯總、圖表、文件 | — | — | **0.40** |
| **合計** | | | **66** | | **7.75 h** |

- **GPU 本體 ≈ 7.35 h**（分析 0.4 h 是 CPU）；soft target 8 h、**hard cap 15 h**。
- 66 個 trial 是「上限」（R2/R3 可能不到）；Phase 0 的 4–6 個不計入 66。

### 5.4 停止規則

| id | 條件 | 動作 |
|---|---|---|
| E1 | rung 1 沒有任何設定同時過 G2/G4–G7 且 Δ ≥ 0.005 | 停下，報 null result |
| E2 | rung 2 沒有設定達到 Δ ≥ 0.005 且配對 bootstrap 95% CI 下界 > 0 | 停下 |
| E3 | rung 3 沒有組合贏最佳單項 ≥ 0.002 | 帶最佳單項進 rung 4 |
| E4 | rung 4 未達 2.8 的四項要求 | 停下，報 not replicated |
| E5 | rung 邊界時 wall clock > 8 h | 跳過 rung 3；> 15 h | 當前 trial 結束後中止 |
| E6 | rung 5 任何強制閘門沒過 | 報閘門失敗，不得宣稱 win |

---

## 6. 需求 4：磁碟計畫

### 6.1 每 trial 峰值

| 內容 | 每 clip | dev256 | dev1024 | full5521 |
|---|---|---|---|---|
| raw flac | 225,882 B | 57.8 MB | 231 MB | 1.25 GB |
| lvl30 flac | 同上 | 57.8 MB | 231 MB | 1.25 GB |
| FAD 暫存（2×16 kHz wav） | — | ~24 MB | ~96 MB | 1.23 GB（1924 對） |
| 評分後立即刪 audio | | | | |
| **峰值** | | **0.13 GB** | **0.47 GB** | **2.5 GB**（FAD 同時在跑時 3.8 GB） |

### 6.2 保留與清理

- 每 trial 保留：`summary.json`、`config.json`、`per_clip.tsv`（raw＋lvl30）、`metrics.json`（raw＋lvl30）、`gain_status.tsv`、`fad.json`、`render.log` → 約 **20 MB／66 trial**。
- **音檔評分完就刪**（trial.py 的 `finally`）；FAD 暫存在 tempdir，函式結束就 `rmtree`。
- scratch 根：`~/nvme_experiment_artifacts/meanaudio/autoresearch_clap_20261027/`（與 `~/` 同一個 NVMe）。

### 6.3 磁碟閘門

| 閘門 | 值 | 動作 |
|---|---|---|
| 進新 trial 前 | free < **20 GB** | 拒絕並通知 |
| 整個 run | free < **10 GB** | 中止（guest 寫 held 證據） |
| 註冊時 free | 64 GB | 目前狀態 |

### 6.4 之後可搬清單（**這輪不動**）

只列候選、不動手（限制：不刪不搬任何現有資料）：

1. `~/eval_output_nvme`（110 GB）→ `/mnt/seagate/nvme_archive/eval_output_nvme`，保留六個 canonical full cells 在原位。
2. `~/nvme_experiment_artifacts`（52 GB）→ `/mnt/seagate/nvme_archive/nvme_experiment_artifacts`。
3. `~/exps_nvme` 的 `ckpt_last`（~163 個 × 2.4 GB）與 `ckpt_shadow`（6 個）→ `/mnt/seagate/nvme_archive/exps_nvme`（只留 `ema_final` 與本線三個 checkpoint）。

---

## 7. 需求 5：最終確認

### 7.1 兩段式

| 段 | 資料 | 格子 | 目的 |
|---|---|---|---|
| rung 4 | heldout1024 | 3 checkpoint × 3 seed = 9 | **唯一不偏**的檢定（見 2.5） |
| rung 5 | full 5521 | 3 checkpoint × seed 42 | 論文數字＋FAD/PQ/level 全量閘門 |

### 7.2 對照組

- 每個 checkpoint 的 **stock CFG0** 與 **stock CFG3+fidelity8** 都已在 `~/eval_output_nvme/<EXP>_mc_mf25_{cfg0,cfg3_neg}{,_lvl30,_fad}` 存在（六格），不重跑。
- heldout 的 baseline 由這些 full cell 的 per-clip 抽出（lvl30 CLAP/PQ 直接取 per_clip；FAD 用同一套 cache 在 heldout 子集上算一次；MEva 用 symlink 目錄指到 canonical lvl30 音檔，零複製）。
- rung 5 的 trial 只生成贏家設定的三個 checkpoint 的 full cell。

### 7.3 必須回報的指標

| 類別 | 指標 | 來源 |
|---|---|---|
| 主指標 | lvl30 CLAP（batch 1） | `eval_metrics.score_clap` |
| 次要 | raw CLAP、AES CE/CU/PC/PQ（raw 與 lvl30） | `eval_metrics.py` |
| 響度／波形 | LUFS、RMS、crest、silent（<−45 dBFS）、clipped | 同上 |
| FAD | full（`--fad --ref_dir /mnt/HDD/kojiek/musiccaps_reference`，1924 對） | `eval_metrics.py --fad`（**ref_dir 必須明確傳**，預設目錄不存在會靜默給 −1） |
| MEva | heldout1024（pinned f03 binding） | `runtime/meva_20261003/venv/bin/python scripts/eval/meva_score_dir.py` |

### 7.4 預登錄的勝負語句（先寫死）

- **win**：rung 4 與 rung 5 全過 → 可寫「在固定 checkpoint 上，只改推論期的 X，lvl30 CLAP 提升 +Y（heldout 獨立驗證），FAD/PQ 不劣化」。
- **checkpoint-specific**：rung 5 平均過但 < 2/3 checkpoint 為正 → 只能寫其中一個 checkpoint 的觀察。
- **not replicated**：rung 4 沒過 → 不论 dev Δ 多大，只能寫「dev 上看到 +Y，但 heldout 未重現」。
- **gate fail**：任何強制閘門沒過 → 不得寫「品質提升」，即使 CLAP 上升。

---

## 8. 需求 6：預測（執行前寫死）

| id | 預測 | 若錯代表什麼 |
|---|---|---|
| **P1** | F2（prompt）是最強族：`short` 在 lvl30 CLAP 贏 stock **+0.003～+0.010** | 文字負向在 lvl30 尺度上不成立，086/108 的短負向結論要重審 |
| **P2** | CLAP–cfg 曲線在 2.5–4.0 之間平坦（±0.003），5.0 下降 | cfg 是比預期更強的槓桿，或高 cfg 的飽和比想像晚 |
| **P3** | guidance 只開前 1/3（t>0.667）保住 ≥ 80% 的 stock CLAP，PQ 升 ≥ 0.05 | 「增益在早段」的機制不成立（075 的外推錯） |
| **P4** | ADG/APG 在 cfg 3.0 上與 vanilla 差 ≤ 0.003 | 幾何干預在 lvl30 CLAP 上有獨立效果（073 的「範數浪費只 2%」被推翻） |
| **P5** | 三個控制都在 baseline ±0.002 內，C0 逐檔 sha 相同 | harness 有問題，先停下查 |
| **P6** | 最後贏家是 prompt 或 interval 改動，full 5521 +0.004～+0.012；過 heldout 就是真效果 | 贏家落在其他族，或效果 <0.004（等於沒有可報告的推論期增益） |
| **P7** | 有設定通過全部閘門（含 heldout）的機率 ≈ **40%**；最可能的失敗是 dev-only 增益 | 若遠低於 40%，代表 0.005 門檻相對 seed 雜訊太鬆，需檢討 |
| **P8** | FAD 與 PQ 閘門至少否決一個高 CLAP 設定 | CLAP 與保真度在這條線上同向，與 064/084 相反 |

---

## 9. 實作介面、目錄與檔案

### 9.1 檔案清單

**已存在（本輪）**

| 路徑 | 狀態 |
|---|---|
| `docs/experiments/autoresearch_inference_clap_20261027.md` | 本檔 |
| `docs/experiments/autoresearch_clap_20261027_prereg.json` | 預登錄（38 組、閘門、預算、預測） |
| `scripts/eval/autoresearch_clap_20261027/split_subsets.py` | 已寫、已跑（manifest sha 已進 prereg） |
| `scripts/eval/autoresearch_smoke_20261027.sh` | 計時煙測腳本（**本輪未執行**，見 §10.3） |

**待寫（下一輪，實作階段）**

| 路徑 | 介面 |
|---|---|
| `scripts/eval/autoresearch_clap_20261027/patch_meanflow.py` | `make_ode_wrapper(spec)`、`make_run_t0_to_t1(spec)`、`patch(spec)` contextmanager |
| `…/render.py` | `--out DIR --tsv TSV --num-steps N --cfg C --prompt TEXT --seed S [--interval lo:hi] [--schedule NAME] [--grid NAME] [--geometry NAME --adg-gamma G / --apg-eta E]` → 用 `runpy` 跑 `eval.py` |
| `…/trial.py` | 單 trial：render → raw level → lvl30 → CLAP/AES → FAD → 閘門 → `summary.json` |
| `…/fad_cache.py` | `score(background_dir, eval_dir, background_embds_path=<cache.npy>)`；cache key = 排序後 ref id 清單的 sha256 |
| `…/driver.py` | 吃 `configs.json`，逐 trial 呼叫 `trial.py`，跳過已有 `summary.json` 且 sha 相符者 |
| `…/analyze.py` | 配對 bootstrap（10000 次、seed 20261027）、閘門、晉級清單 |
| `…/make_round_configs.py` | 由 analyze 輸出下一輪 `configs.json`（rung 2/3 規則見 §3.3） |
| `…/guest.py` | p2 guest：progress 行、stall watchdog、`evidence={'eval_only': True, 'reports': [...]}` |
| `scripts/queue_candidates/153_autoresearch_r1.sh` … `157_…_r5.sh` | launcher（`GPU_QUEUE_CONTRACT`、`PYTHONUNBUFFERED`、`HF_HUB_OFFLINE`、`OMP_NUM_THREADS=2`、`exec python -I … guest.py`） |
| `docs/experiments/autoresearch_153_r1_contract.json` … `157_…_r5_contract.json` | contract（先於 launcher 進 pending） |

### 9.2 一個 trial 的實際指令

```bash
PY=/home/kojiek/venvs/dac/bin/python
T=<scratch>/trials/r1/F2_short
RAW=$T/cells/F2_short_raw                    # 命名規則：<config_id>_raw

# 1) 生成（patched 或 unpatched；unpatched 時 argv 與 mc_mf25_eval.sh 完全相同）
$PY scripts/eval/autoresearch_clap_20261027/render.py \
  --out $RAW/audio --tsv <scratch>/tsv/dev256.tsv \
  --checkpoint <primary.pth> --num-steps 25 --cfg 3.0 --prompt "$PROMPT" --seed 42
# 內部等價 argv（unpatched）：
#   python eval.py --variant meanaudio_s --model_path <CKPT> --output $RAW/audio \
#     --tsv <split.tsv> --use_meanflow --num_steps 25 --cfg_strength 3.0 \
#     --negative_prompt "<PROMPT>" --no_text_attention_mask --encoder_name t5_clap \
#     --text_c_dim 512 --seed 42 --full_precision --no_q

# 2) raw level 只算響度（G4/G5）
$PY scripts/eval/eval_metrics.py --gen_dir $RAW/audio --tsv <split.tsv> \
  --exp_name F2_short_raw --out_dir $RAW --skip_clap --skip_aes

# 3) lvl30 增益＋全量 metrics（CLAP batch1 + AES + level）
#    輸出固定是 <out_root>/<cell 名>_lvl30/，所以會是 $T/lvl30/F2_short_raw_lvl30/
$PY scripts/eval/level_match_rescore.py --cell_dir $RAW --tsv <split.tsv> \
  --target -30 --peak_cap 0.999 --workers 8 --out_root $T/lvl30

# 4) FAD（raw 音訊；ref embedding 用 cache）
$PY scripts/eval/autoresearch_clap_20261027/fad_cache.py \
  --gen-dir $RAW/audio --tsv <split.tsv> --split dev256 --out $T/fad.json

# 5) 閘門＋summary
$PY scripts/eval/autoresearch_clap_20261027/trial.py --score-only \
  --trial-dir $T --configs <scratch>/rounds/r1.json --config-id F2_short
```

driver 一次跑整輪：

```bash
$PY scripts/eval/autoresearch_clap_20261027/driver.py \
  --configs <scratch>/rounds/r1.json --split dev256 --scratch <scratch> --resume
```

rung 4／5 用 `--cross`：

```bash
$PY …/driver.py --configs <scratch>/rounds/r4.json --split heldout1024 \
  --checkpoints s14159265,s16180339,s27182818 --seeds 42,7,161803 --scratch <scratch> --resume
$PY …/driver.py --configs <scratch>/rounds/r5.json --split full5521 \
  --checkpoints s14159265,s16180339,s27182818 --seeds 42 --fad --meva --scratch <scratch>
```

### 9.3 目錄結構

```
~/nvme_experiment_artifacts/meanaudio/autoresearch_clap_20261027/
  tsv/
    manifest.json
    dev256.tsv  dev1024.tsv  heldout1024.tsv  rest3473.tsv
  baselines/
    dev256/<ckpt>/<cfg>/{lvl30.json,fad.json}
    dev1024/…  heldout1024/…  full5521/…        # 由 canonical full cell 抽出，不重新生成
  fad_cache/
    <ref-ids-sha>.npy                           # ref embeddings
  rounds/
    r0.json  r1.json  r2.json  r3.json  r4.json  r5.json
  trials/
    r1/F2_short/
      config.json            # id、argv、checkpoint sha、split sha、started/finished
      cells/F2_short_raw/
        audio/*.flac
        F2_short_raw/{metrics.json,per_clip.tsv}     # raw level（G4/G5）
        gen.log
      lvl30/F2_short_raw_lvl30/
        audio/*.flac
        gain_status.tsv  gain_summary.json
        F2_short_raw_lvl30/{metrics.json,per_clip.tsv}   # CLAP batch1 + AES + level
      fad.json
      summary.json           # 閘門結果、Δ vs baseline、config_sha
  rounds_out/
    r1_summary.json          # analyze 的輸出（晉級清單）
```

- 上面是 Phase 0 之後的目標佈局。本輪只產生驗證用的 split：`runtime/autoresearch_smoke_20261027/tsv_probe/`（`dev1024/heldout1024/rest3473.tsv`＋`manifest.json`，`runtime/` 是 gitignore）與 `~/nvme_experiment_artifacts/meanaudio/autoresearch_clap_20261027/tsv/dev256.tsv`；它們的 sha 與 prereg 逐一相符（§10.1），Phase 0 用同一支 `split_subsets.py` 在 scratch 重建，sha 必須相同。

### 9.4 patch 介面（骨架）

```python
# patch_meanflow.py
def make_ode_wrapper(spec):
    import torch
    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg_strength, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        c = cfg_at(float(t[0]), spec, cfg_strength)      # interval / schedule / passthrough
        if c < 1.0:
            return self.predict_flow(latent, t, r, conditions, q)
        a = self.predict_flow(latent, t, r, conditions, q)
        b = self.predict_flow(latent, t, r, empty_conditions, q)
        delta = a - b
        if spec['kind'] == 'apg':
            a_hat = a / _norm(a, 'global')
            par = (delta * a_hat).sum(dim=tuple(range(1, a.ndim)), keepdim=True) * a_hat
            return a + (c - 1.0) * ((delta - par) + spec['eta'] * par)
        v = a + (c - 1.0) * delta
        if spec['kind'] == 'adg':
            v = v * ((1.0 - spec['gamma']) + spec['gamma'] * _norm(a, 'global') / _norm(v, 'global'))
        return v
    return ode_wrapper

def make_run_t0_to_t1(spec):
    def run_t0_to_t1(self, fn, x0):
        ts = grid(1.0, 0.0, self.steps, spec['grid'], x0)   # uniform: torch.linspace 逐位元相同
        for i in range(self.steps):
            t, nt = ts[i].expand(x0.shape[0]), ts[i + 1].expand(x0.shape[0])
            u = fn(t=t, r=nt, x=x0)
            x0 = x0 - (t - nt).mean() * u
        return x0
    return run_t0_to_t1
```

- `cfg_at` 只讀當步的 `t`；`vanilla` 的 spec 完全**不 patch**（走 073 的 `geometry('vanilla')`：yield 前後不動 `MeanAudio.ode_wrapper`）。
- F7 的 `grid`：`uniform` 直接 `torch.linspace(1,0,steps+1)`；`cos` 用 `0.5*(1+cos(πu))` 的重參數化；`smoothstep` 用 `u²(3−2u)`。端點固定 1.0／0.0。

### 9.5 `trial.py` 的 summary 格式（gate 證據）

```json
{
 "config_id": "F2_short", "config_sha": "…", "split": "dev256", "n": 256,
 "checkpoint": {"name": "s14159265", "sha256": "7cc14d09…"},
 "lvl30": {"clap": 0.1973, "pq": 7.58, "lufs": -30.001},
 "raw": {"clap": 0.2160, "silent_n": 11, "clipped_n": 0, "crest_mean": 6.19},
 "fad": {"value": 5.12, "pairs": 240},
 "delta_vs_baseline": {"lvl30_clap": 0.0046, "lvl30_pq": -0.01},
 "gates": {"G1": true, "G2": false, "G3": true, "G4": true, "G5": true, "G6": true, "G7": true},
 "status": "fad_pending", "runtime_s": 161.3
}
```

`analyze.py` 只讀 `summary.json`，輸出：每族最佳、晉級清單、配對 bootstrap CI、underpowered 標記。

### 9.6 contract 內容（每個 rung 一份）

```json
{
 "schema_version": 1,
 "document_kind": "eval_only_metrics_contract",
 "job_id": "153_autoresearch_r1",
 "queue_role": "p2",
 "launch_authorization": {"authorized_by": "Lanternko", "date": "2026-10-27"},
 "bindings": {"<script path>": {"sha256": "…"}, "prereg": {"sha256": "…"},
              "checkpoint_primary": {"sha256": "7cc14d09…"},
              "split_dev256": {"sha256": "…"}},
 "commands": {
   "preflight": "python -I …/guest.py --preflight --configs …/r1.json",
   "run":       "python -I …/guest.py --run --configs …/r1.json",
   "postflight":"python -I …/guest.py --postflight --configs …/r1.json"
 },
 "completion_evidence": {"eval_only": true, "reports": ["<scratch>/rounds_out/r1_summary.json"]},
 "resume": {"from_scratch_with_autoresume": true},
 "storage": {"min_free_bytes": 20000000000},
 "watcher": {"stall_seconds": 900}
}
```

> guest 必須自己驗：checkpoint sha、split sha、prereg sha、磁碟門檻、trial 完整性；任何一項不符 → held 並寫證據，不進 run。

---

## 10. 計時量測與煙測狀態

### 10.1 已實測（本輪，指令可重跑）

| 量 | 值 | 指令 |
|---|---|---|
| CFG0 生成 | 0.1962 s/clip（1083.0 s / 5520 間隔） | `stat -c %W`／mtime on `…_cfg0/audio`；`gen.log` tqdm `18:03` |
| CFG3+neg 生成 | 0.3142 s/clip（1734.4 s） | 同上；tqdm `28:55` |
| `eval.py` 啟動 | 24.1 s cold／14.7 s warm | audio 目錄 btime → 首個 flac mtime |
| 全套 metrics（raw, 5521） | 147.6 s（cfg0）／154.9 s（cfg3） | last flac mtime → `metrics.json` mtime |
| lvl30 增益 | 0.00644 s/clip（64 檔 8 workers 0.412 s） | 直接在 CPU 量 |
| raw level-only（256 檔，含載入） | **0.74 s** | `eval_metrics.py --skip_clap --skip_aes` |
| dev256 baseline（s14159265，由 canonical full cell 抽出） | cfg0：lvl30 CLAP **0.1847**／PQ **6.6368**；cfg3+neg：**0.2296**／**7.5786**；raw crest 6.3444／6.3043、silent 0、clipped 0 | 讀 `*_lvl30/<cell>_lvl30/per_clip.tsv` 的 256 個 id + `eval_metrics` level |
| lvl30 全流程（批次） | ~577 s/cell | **並行時**（36 cell 同時），非單跑值 |
| baseline 六格 | 見 §1.3 | canonical `*_lvl30/metrics.json` |
| split 重現 | 三段 sha 與 prereg 相同：dev1024 `3bc818c8`／heldout `a87b20ad`／rest `b230d552`（dev256 = dev1024 前 256 筆，逐列相同） | 比對 `runtime/autoresearch_smoke_20261027/tsv_probe/manifest.json` |
| 磁碟 | 64 GB free | `df -h ~` |

### 10.2 未實測（估計，Phase 0 會換成實測）

| 量 | 估計 | 風險 |
|---|---|---|
| FAD 秒數 | 25／71／125 s | 可能低估（16 kHz 轉檔＋VGGish）；Phase 0 用 256 檔實測 |
| 子集 FAD 雜訊 | 未定 | 必須先量三 checkpoint 的 baseline 散布，才定 G2 的實務門檻 |
| 序列 lvl30 全流程 | 183 s（5521） | 用乾淨單跑值；Phase 0 驗一次 |
| `--skip_clap --skip_aes` 的 raw level 成本 | **已實測 0.74 s（256 檔）**，1024 檔 ≈ 3 s | 已明列在 §5.2 的 `level_only(n)` 欄（163／475 都含它） |

### 10.3 煙測為什麼沒跑（誠實記錄）

- 本輪指定「若尚未跑，跑一次 ≤10 分鐘的計時煙測」。檢查時 **GPU 上已有 27,778 MiB 被佔用**：
  - `nvidia-smi` 顯示 PID **2292158** = `venvs/dac/bin/python -I scripts/eval/bench_r2/score_r2.py --dataset arena --job J5 --models singmos`
  - 它的環境是 `P2_JOB_ID=152_bench_r2_J5c_singmos`，父程序是 `gpu_queue/p2_host.sh`；`gpu0.owner.json` 記 `job_id=152_bench_r2_J5c_singmos`、`acquired_at=2026-10-26T20:54:46Z`。
  - 這是**另一個任務（benchmark r2）的 p2 job**，不是我的殘留程序 → **不動、不殺、不搶**。
- `gpu0.lock` 由該 job 持有，我的煙測腳本（`scripts/eval/autoresearch_smoke_20261027.sh`）用 `hold_lock.py --block`，會一直等；在別人的 job 結束前跑就是搶佔，違反規則。
- 因此本輪**沒有**煙測數字；§5.1 的成本模型全部來自**既有 canonical cell 的實測 timestamp**（§10.1），並在 Phase 0 補量 FAD、子集雜訊與序列 lvl30 成本。Phase 0 預算 0.2 h 已含這些量測。
- 煙測腳本保留在 repo（未執行、未 commit 任何音檔），等 GPU 空出來時可直接跑；它輸出的 `timing.json` 會是 Phase 0 的第一份證據。

### 10.4 Phase 0 要補的量測（開跑前）

1. copy gate：stock cfg3 在 dev256 的 256/256 sha（同時得到一次 dev256 trial 的實測秒數）。
2. FAD parity：canonical cfg3 full cell 用 cache 重算，必須得到 **5.0836**（誤差 < 1e-3）。
3. 子集 baseline：dev256／dev1024／heldout 的六個 baseline 的 CLAP/PQ/FAD（不重新生成）。
4. seed 校正：C1/C2 的實測 Δ（進 `measured_floor_rule`）。
5. 序列 lvl30 成本（一次 5521 的 lvl30 cell，用既有的 cfg0 raw cell 重跑 rescore；不寫新音檔到 canonical 目錄）。

---

## 11. 風險與限制

| 風險 | 影響 | 對策 |
|---|---|---|
| lvl30 CLAP 是新尺度 | 不能與歷史 raw CLAP 比 | 所有比較都在這條線內部；報告兩種尺度但只判 lvl30 |
| 子集 FAD 很吵 | G2 在 rung 1 可能誤殺 | `FAD-pending`（最多 2 個）＋最終 full FAD 必過 |
| 0.005 門檻相對 seed 雜訊 | n=256 是 3.6σ，不算離譜；但 35 組的最大 null 是 0.0031 | 用 heldout 做最終檢定；dev 的 Δ 不寫進結論 |
| 3 checkpoint 的 sha 若不符 | 確認作廢 | guest 的 preflight 硬驗 sha |
| 改動 `eval.py`／`eval_metrics.py`／`mc_mf25_eval.sh`／`level_match_rescore.py` | prereg 失效 | 這些檔案不動；contract 綁它們的 sha |
| GPU 被別的 job 佔用 | 排程延遲 | queue 本來就是排隊；不用 sleep 輪詢 |
| FAD 的 `ref_dir` 忘了傳 | 靜默得到 −1 | `fad_cache.py` 強制 `--ref-dir`，缺檔直接失敗 |
| CLAP 的 batch 不是 1 | 分數偏移且會翻排名 | 一律 `eval_metrics.score_clap`（batch 1）；不自己寫 batch 迴圈 |
| 響度對齊失敗 | objective 被污染 | G6 的 `peak_capped_n` 與 `|LUFS+30|` 檢查 |
| 靜音逃逸 | CLAP 假高 | G4（靜音數）與 G7（完整性） |

**限制（先講清楚）**：

- 這條線只搜**推論期**；它不可能改變模型的內容分布，只能在既有分布上換取 CLAP。
- MEva／FAD／PQ 都只是「不劣化」閘門，不是這條線的優化目標。
- full 5521 的 rung 5 數字含 2048 筆選模 records，不可當獨立檢定。
- 本輪沒有煙測（§10.3），所有秒數是**既有 artifact 的實測值＋Phase 0 的校準**，不是本輪新跑的。

---

## 12. 與 prereg 的一致性檢查表

| 項目 | prereg | 本檔 |
|---|---|---|
| 模型 sha | 7cc14d09／997077fb／4f98eefc | §1.1 同 |
| 六格 baseline | 見 `baseline_numbers_lvl30_full5521` | §1.3 同 |
| splits sha | dev1024 3bc818c8／heldout a87b20ad／rest b230d552 | §2.1 同 |
| 門檻 | 0.005／0.004／0.002 | §2.4 同 |
| 雜訊 | 0.0003／0.142／σ(n) | §2.3 同 |
| rung 1 configs | 38（F1 7、F2 7、F3 6、F4 4、F5 6、F6 3、F7 2、C 3） | §3.2 同 |
| 單 trial | 163／475／475／2215 | §5.2 同 |
| 總時數 | 7.75 h、soft 8、cap 15 | §5.3 同 |
| 盤 | 峰值 3.8 GB、guards 20/10 GB | §6 同 |
| 預測 | P1–P8 | §8 同 |
