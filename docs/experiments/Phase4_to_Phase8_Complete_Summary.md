# MeanAudio Phase 4–8 完整實驗記錄

**專案**：MeanAudio — 文字驅動音訊生成（Jamendo 訓練集，251K clips）
**評估集**：Jamendo test set，n=2048（random subset from 90,063 clips）
**主要指標**：CLAP ↑、CE ↑、PQ ↑（FAD 已降為歷史參考，不再作為主要指標）
**更新日期**：2026-04-02

---

## 一、Golden Baseline — Phase 4 V2

### 目標
建立可重現的基準線，供所有後續實驗比較。

### 設定
| 項目 | 值 |
|------|-----|
| 訓練集 | 251K clips |
| Caption 策略 | best-consensus（5 個候選中 inter-caption 相似度最高者） |
| Quality Conditioning | 無 |
| Stage 1 | 400k steps |
| Stage 2 | 200k steps |

### 數據

| 指標 | 數值 |
|------|------|
| CLAP | 0.1929 |
| FAD | 1.5853 |
| CE | 5.905 |
| PQ | 6.620 |

### 結論
所有後續實驗均以此為基準。CE 5.905、PQ 6.620 為感知品質底線。

---

## 二、Phase 5 — Hard Filtering 實驗

### 目標
驗證「過濾低品質 caption（mean_similarity < 0.80）是否提升模型效果」。

### 設定
| 版本 | 訓練集 | 過濾方式 |
|------|--------|---------|
| V1 | 117K clips | mean_similarity ≥ 0.80（硬過濾） |
| V2（對照） | 117K clips | 隨機抽樣（無過濾） |

### 數據

| 版本 | CLAP | FAD | 備註 |
|------|------|-----|------|
| Phase 4 V2（baseline） | 0.1929 | 1.5853 | 251K |
| Phase 5 V1（hard filter） | 0.1834 | 2.6603 | 117K |
| Phase 5 V2（random sample） | 0.1846 | 2.6446 | 117K |

### 結論
**過濾無效，退步原因是資料量 -53%，與過濾本身無關。** V1 與 V2 結果幾乎相同，說明問題出在 data quantity，非 data quality。Hard filtering 有害。

---

## 三、Phase 6 V1 — Quality Conditioning（Stage 2 Only）

### 目標
引入 q_embed，讓模型在 Stage 2 學習品質信號。

### 設定
- q_embed：nn.Embedding(11, hidden_dim)，index 0~9 = 品質等級，10 = null token
- **Stage 1 無 q conditioning，Stage 2 才加入**
- Caption 策略：best-consensus

### 數據

| q 設定 | CLAP | FAD |
|--------|------|-----|
| q=9 | 0.1898 | 1.7628 |
| native_q | 0.1895 | 1.8144 |
| q=8 | 0.1889 | 1.8567 |
| q=0 | 0.1379 | 2.2839 |

### 結論
q_embed 確實有效（q=0 崩潰驗證 q 信號被學到）。但 Stage 1 未接受 q 參數，限制了效果上限。指向 Stage 1+2 同步訓練的方向。

---

## 四、Phase 6 V2 — Quality Conditioning（Stage 1+2）

### 目標
Stage 1 與 Stage 2 同時引入 q conditioning，驗證全程 q 訓練的效果。

### 設定
- q_embed 從 Stage 1 就開始訓練
- Caption 策略：best-consensus
- 對 q level 0~9 全面 sweep

### 數據

| q 設定 | CLAP | FAD | CE | PQ | 備註 |
|--------|------|-----|----|----|------|
| q=0 | 0.0340 | 9.7771 | 3.897 | 5.882 | 全面崩潰 |
| q=5 | 0.1846 | 1.5495 | 6.061 | 6.869 | 接近 baseline |
| **q=6** | 0.1979 | **1.0549** | 6.175 | 6.859 | ⭐ FAD 最佳，最均衡 |
| q=7 | 0.1956 | 1.1327 | 6.116 | 6.824 | — |
| q=8 | 0.1939 | 1.1518 | 6.081 | 6.804 | — |
| native_q | 0.1950 | 1.1153 | 6.111 | 6.824 | 穩健，無需調參 |
| **q=9** | **0.2139** | 2.5849 | 6.109 | 6.821 | CLAP 最高，FAD 最差 |

**FAD U 型曲線說明**：test set 品質分佈均值 q≈7.3，q=6 最接近中心 → FAD 最低。q=9 強制高品質信號把生成分佈推離 reference 中心 → FAD 暴增但 CLAP 最高。

### 結論
Stage 1+2 同步 q conditioning 後，CE/PQ **首次全面超越 Baseline**（CE: 5.905 → 6.175，PQ: 6.620 → 6.859）。CLAP ≠ 感知品質首次量化驗證（q=9 CLAP 最高但感知指標非最佳）。

---

## 五、Phase 7 V1 — Random Caption + q

### 目標
測試 caption 選擇策略對模型效果的影響：從 best-consensus 改為 random selection。

### 設定
- Caption：5 個候選中隨機選 1 個（seed=42）
- TSV 生成腳本：`research/meanaudio_training/gen_phase7_v1_tsv.py`
- 其餘設定與 Phase 6 V2 相同

### 數據

| q 設定 | CLAP | CE | CU | PC | PQ |
|--------|------|----|----|----|-----|
| q=6 | 0.1980 | **6.276** | 7.028 | 5.053 | 6.936 |
| q=9 | 0.1984 | 6.254 | **7.035** | 4.972 | **6.939** |
| native_q | 0.1977 | 6.244 | 7.009 | 5.017 | 6.913 |

### 結論
**Phase 7 V1 是目前最佳模型。** CE 從 6.175（P6V2 q=6）升至 6.276，PQ 從 6.859 升至 6.939。

**為什麼 random > best-consensus？（多樣性假說）**
1. best-consensus 選出最「平均」的 caption，缺乏特異性
2. random 保留訓練集的文字多樣性，起到正則化效果
3. 部分非 representative caption 實際上更準確描述音訊細節

---

## 六、Phase 7 V2 — CLAP-Best Caption + q

### 目標
測試「選 caption-audio CLAP 相似度最高的 caption」是否優於 random。

### 設定
- Caption：對每個 clip 計算 5 個候選的 audio-CLAP 相似度，取最高分
- TSV 生成腳本：`research/meanaudio_training/gen_phase7_v2_tsv.py`
- ⚠️ 因訓練資料用 CLAP 選 caption，evaluation 不報 CLAP（data leakage 原則）

### 數據

| q 設定 | CLAP | CE | CU | PC | PQ |
|--------|------|----|----|----|-----|
| q=6 | 0.1943 | 6.230 | 7.002 | **5.110** | 6.938 |
| q=9 | 0.1916 | 6.093 | 6.876 | 5.093 | 6.786 |
| native_q | 0.1904 | 6.134 | 6.919 | 5.085 | 6.851 |

### 結論
**V2 全面劣於 V1（random）。** CLAP-best 雖語義最準確，但過度同質化 → 損失訓練多樣性。**多樣性假說成立：random > CLAP-best。**

---

## 七、Phase 7 V3 — Worst-Consensus Caption + q

### 目標
測試「選 inter-caption text similarity 最低的 caption（最偏離群組的描述）」的效果。

### 設定
- Caption：對 5 個候選計算 pairwise text-text CLAP 相似度，選 avg_similarity 最低者
- TSV 生成腳本：`research/meanaudio_training/gen_phase7_v3_tsv.py`

### 數據

| q 設定 | CLAP | CE | PQ |
|--------|------|----|----|
| q=6 | 0.1969 | 6.170 | 6.875 |
| q=9 | 0.1970 | 6.253 | **6.981** |
| native_q | 0.1959 | 6.156 | 6.876 |

### 結論
V3 幾乎追平 V1（CE q=9: V1 6.254 vs V3 6.253，PQ q=9: V3 6.981 > V1 6.939）。**「只要遠離 best-consensus，模型都能受益」——多樣性的方向（random 或 worst）比 CLAP-best 更重要。**

---

## 八、Phase 8 — Random Caption，無 q Conditioning

### 目標
分離 random caption 與 q embedding 的獨立貢獻，驗證 q conditioning 是否必要。

### 設定
- Caption：random（與 Phase 7 V1 相同，使用 phase7_v1_train.tsv）
- q conditioning：**完全關閉**（runner 加 `use_q_conditioning=False`，訓練時永遠 q=None → null token）
- eval 時加 `--no_q` 確保一致性（避免 untrained q_embed[0~9] 污染結果）

### 數據

| 指標 | Phase 8（random, no q） | Phase 7 V1 q=9（random + q） | 差距 |
|------|------------------------|------------------------------|------|
| CLAP | **0.1981** | 0.1984 | -0.0003（≈相同） |
| CE | 6.127 | **6.254** | **-0.127** |
| PQ | 6.772 | **6.939** | **-0.167** |

### 結論
**q embedding 有獨立且必要的貢獻。**
- Random caption 貢獻 CLAP 提升（0.1929 → 0.1981）
- q embedding 額外貢獻 CE +0.127、PQ +0.167
- 兩者缺一不可：只有 random caption 不夠，q conditioning 是獨立正向因素

---

## 九、貢獻分解總表

| 系統 | vs Baseline CLAP | vs Baseline CE | vs Baseline PQ |
|------|-----------------|----------------|----------------|
| Phase 4 V2（baseline） | — | — | — |
| + q conditioning（P6V2 q=6） | +0.0050 | +0.270 | +0.239 |
| + random caption（P8） | +0.0052 | +0.222 | +0.152 |
| **+ 兩者同時（P7V1 q=9）** | **+0.0055** | **+0.349** | **+0.319** |

---

## 十、累積學習原則

1. **CLAP score ≠ 感知品質**：q=9 CLAP 最高但 FAD/CE 不佳；CLAP 是輔助工具，CE/PQ 更可靠
2. **Hard filtering 有害，data quantity matters**：Phase 5 驗證，資料量 -53% 導致全面退步，與過濾品質無關
3. **Caption 多樣性提供正則化**：random > CLAP-best > best-consensus；多樣性比準確性更重要
4. **Quality conditioning 有效且必要**：Phase 8 量化驗證，q embedding 獨立貢獻 CE +0.127、PQ +0.167
5. **Stage 1 必須同步接受 q 信號**：Phase 6 V1 vs V2 的教訓，只有 Stage 2 加 q 效果受限
6. **Experiment design discipline**：每次只改一個變量（P7 V1/V2/V3 vs P8 各自對照清楚）

---

## 十一、未來待做清單

- [ ] **Phase 9**：以 Meta Audiobox PQ 分數取代 mean_similarity 作為 q conditioning 信號（需先對 251K 訓練 clip 跑 Audiobox inference 產生 PQ label）
- [ ] **Inference-only 實驗**：對同一音訊的 5 個 caption 取 text embedding 平均再輸入模型（zero-cost，不需重訓）
- [ ] **PE-AV 替換 CLAP**：評估以 Meta PE-AV 取代 LAION-CLAP 作為更可靠的 evaluation encoder（long-term）
- [ ] **為 Jamendo 訓練集跑 Audiobox Aesthetics inference**：產生全集 PQ label，為 Phase 9 準備資料
- [ ] **討論 Phase 10 方向**：Resonate 的 Flow-GRPO + LALM reward 作為後續 RL fine-tuning 方向
