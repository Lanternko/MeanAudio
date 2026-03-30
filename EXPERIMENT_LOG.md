# MeanAudio 實驗記錄

所有實驗在 Jamendo test set 評估（n=2048，random subset）。
主要指標：**CLAP ↑、CE ↑、PQ ↑**（FAD 僅歷史參考，不再作為主要指標）。

---

## 完整數字表

| Phase | Caption 策略 | q 設定 | CLAP ↑ | FAD ↓ | CE ↑ | PQ ↑ | 備註 |
|-------|------------|--------|--------|-------|------|------|------|
| phase4_v2 | best-consensus | — | 0.1929 | 1.5853 | 5.905 | 6.620 | **Baseline** |
| phase6_v1 | best-consensus | q=9 | 0.1898 | 1.7628 | 6.150 | 6.803 | CE/PQ 超越 baseline |
| phase6_v2 | best-consensus | q=0 | 0.0340 | 9.7771 | 3.897 | 5.882 | 全面崩潰 |
| phase6_v2 | best-consensus | q=5 | 0.1846 | 1.5495 | 6.061 | 6.869 | FAD 接近 baseline |
| phase6_v2 | best-consensus | q=6 | 0.1979 | 1.0549 | 6.175 | 6.859 | ⭐ FAD 最佳 |
| phase6_v2 | best-consensus | q=7 | 0.1956 | 1.1327 | 6.116 | 6.824 | — |
| phase6_v2 | best-consensus | q=8 | 0.1939 | 1.1518 | 6.081 | 6.804 | — |
| phase6_v2 | best-consensus | native_q | 0.1950 | 1.1153 | 6.111 | 6.824 | 穩健，無需調參 |
| phase6_v2 | best-consensus | q=9 | **0.2139** | 2.5849 | 6.109 | 6.821 | CLAP 最高，FAD 最差 |
| **phase7_v1** | **random** | **q=6** | 0.1980 | 1.0553 | **6.276** | 6.936 | **🏆 CE 最高** |
| **phase7_v1** | **random** | **q=9** | 0.1984 | 1.1997 | 6.254 | **6.939** | **🏆 PQ 最高** |
| **phase7_v1** | **random** | **native_q** | 0.1977 | 1.1110 | 6.244 | 6.913 | 穩健最佳 |
| phase7_v2 | CLAP-best | q=6 | 0.1943 | — | 6.230 | 6.938 | 劣於 V1 |
| phase7_v2 | CLAP-best | q=9 | 0.1916 | — | 6.093 | 6.786 | 劣於 V1（差距最大） |
| phase7_v2 | CLAP-best | native_q | 0.1904 | — | 6.134 | 6.851 | 劣於 V1 |

---

## Phase 6 V2 — Quality Conditioning 實驗

**核心改動**：引入 `q_embed`（nn.Embedding(11, hidden_dim)），index 0~9 為品質等級，index 10 為 null token。

**Caption 策略**：best-consensus（5 個候選中 inter-caption 文字相似度最高者）。

**q_level U 型曲線（FAD）**：
- Test set 品質分佈均值 q ≈ 7.3
- FAD 最低點在 q=6（1.05），接近分佈中心
- q=9 CLAP 最高（0.2139）但 FAD 最差（2.58）：強制高品質信號把生成分佈推離 reference 中心
- q=0 全面崩潰：CLAP 0.034，CE 3.9

**結論**：quality conditioning 使 CE/PQ 全面超越 baseline（CE: 5.905 → 6.175，PQ: 6.620 → 6.859）。但 caption 策略（best-consensus）仍有改進空間。

---

## Phase 7 V1 — Random Caption 實驗

**核心改動**：Caption 選擇策略從 best-consensus 改為 random（每個 clip 從 5 個候選中隨機選 1，seed=42）。

**資料生成**：`research/meanaudio_training/gen_phase7_v1_tsv.py` → `phase7_v1_train.tsv`（251,599 rows）。

**V1 vs 前代比較（q=6）**：

| 指標 | phase4_v2 (baseline) | phase6_v2 | phase7_v1 | 改善幅度 |
|------|---------------------|-----------|-----------|---------|
| CLAP | 0.1929 | 0.1979 | **0.1980** | +2.6% vs baseline |
| CE | 5.905 | 6.175 | **6.276** | +6.3% vs baseline |
| PQ | 6.620 | 6.859 | **6.936** | +4.8% vs baseline |

**結論**：Phase 7 V1 在所有主要指標全面超越 phase6_v2，是目前最佳模型。

---

## Phase 7 V2 — CLAP-Best Caption 實驗

**核心改動**：Caption 選擇策略改為 caption-audio CLAP 相似度最高者（每個 clip 計算 5 個候選 vs audio 的 CLAP 分數，取最高）。

**資料生成**：`research/meanaudio_training/gen_phase7_v2_tsv.py`（需先計算 ~251k × 5 CLAP 分數，約 3~4 小時）。

**⚠️ Data Leakage**：V2 用 CLAP 選 caption → evaluation 不報 CLAP（改用 AES）。

**V1 vs V2 詳細比較**：

| q 設定 | 指標 | V1 (random) | V2 (CLAP-best) | 勝者 |
|--------|------|-------------|----------------|------|
| q=6 | CLAP | **0.1980** | 0.1943 | ✅ V1 |
| | CE | **6.276** | 6.230 | ✅ V1 |
| | PQ | 6.936 | **6.938** | ≈ 平手 |
| q=9 | CLAP | **0.1984** | 0.1916 | ✅ V1 |
| | CE | **6.254** | 6.093 | ✅ V1 |
| | PQ | **6.939** | 6.786 | ✅ V1 |
| native_q | CLAP | **0.1977** | 0.1904 | ✅ V1 |
| | CE | **6.244** | 6.134 | ✅ V1 |
| | PQ | **6.913** | 6.851 | ✅ V1 |

**結論：random caption (V1) 全面優於 CLAP-best caption (V2)。**

**為什麼 random > CLAP-best？（多樣性假說）**

1. **best-consensus / CLAP-best 都是「中間值」**：highest inter-caption similarity（V0）和 highest audio similarity（V2）都會選出最「安全」、最通用的描述，缺乏特異性。
2. **random 保留文字多樣性**：5 個候選 caption 風格各異（詳細描述、簡短描述、情緒描述...），random 選擇讓模型見到更廣的文字分佈，起到正則化效果。
3. **部分非 representative caption 更準確**：某些被排除的 caption 其實捕捉了音訊細節，CLAP-best 選出的未必最準確，只是「最像典型描述」。

→ 多樣性 > 準確性，在 caption 選擇上「更正確」反而傷害訓練效果。

---

## 評估指標說明

| 指標 | 全名 | 意義 | 優先級 |
|------|------|------|--------|
| CLAP | Contrastive Language-Audio Pretraining score | 音訊與 caption 語義一致性 | 主要 |
| CE | Content Enjoyment (Audiobox AES) | 主觀聽感、藝術性、整體喜好，與人類 MOS 相關係數 0.528 | 主要 |
| PQ | Production Quality (Audiobox AES) | 技術品質：清晰度、保真度、無雜訊失真 | 主要 |
| FAD | Fréchet Audio Distance | 生成分佈 vs 參考分佈的距離 | 歷史參考 |
| CU | Content Usefulness (Audiobox AES) | 內容是否符合使用情境 | 次要 |
| PC | Production Complexity (Audiobox AES) | 製作複雜度 | 次要 |

**Data Leakage 原則**：若訓練資料用 CLAP 過濾 → evaluation 不能報 CLAP（改用 FAD + AES）。

---

## 待驗證方向（Phase 8 候選）

- **Caption 品質過濾**：丟棄 CLAP-audio 相似度低於門檻的 clip（不選最高，而是過濾最差）
  - 若採用 → evaluation 改用 AES（data leakage）
- **更長訓練**：S1 600k steps + S2 300k steps
- **CFG strength 調整**：目前 eval 用 cfg=0.5，主觀用 cfg=3.5，是否有更好設定
