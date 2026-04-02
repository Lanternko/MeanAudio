# 文獻參考筆記 — 對 MeanAudio 研究的直接啟示

**更新日期**：2026-04-02

---

## 1. Meta Audiobox Aesthetics（Tjandra et al., 2025）

**論文**：Audiobox Aesthetics: Automatic Quality Scoring for Audio Generation
**機構**：Meta AI
**程式碼**：`github.com/facebookresearch/audiobox-aesthetics`
**安裝**：`pip install audiobox_aesthetics`（CC-BY 4.0，無需申請，自動下載權重）

### 核心內容

四個評估維度：

| 指標 | 全名 | 物理意義 |
|------|------|---------|
| **PQ** | Production Quality | 技術品質：清晰度、保真度、無雜訊失真 |
| PC | Production Complexity | 製作複雜度 |
| **CE** | Content Enjoyment | 主觀享受、情感影響、藝術性、整體喜好 |
| CU | Content Usefulness | 創作可用性 |

與人類 MOS 相關係數（PAM-music，utterance-level）：
- CE ↔ 人類 OVL：PCC = **0.528**
- PQ ↔ 人類 OVL：PCC = 0.464

成對偏好預測準確率：CE、CU > 60%（顯著高於盲猜 50%）

### 關鍵發現

**Prompting strategy > Filtering strategy**：在資料量相同的前提下，prompting（用高品質描述引導生成）優於 filtering（直接丟棄低品質資料）。這與本研究 Phase 5 Hard Filtering 有害的結論一致。

### 對本研究的直接啟示

1. **取代 FAD**：CE/PQ 比 FAD 更能反映人類感知，已在 Phase 7+ 採用為主要指標
2. **Phase 9 方向**：用 Audiobox PQ 分數（1~10）取代 mean_similarity 作為 q conditioning 信號，讓 q_embed 直接對應感知品質而非 caption 一致性
3. **⚠️ Data Leakage 原則**：若訓練資料過濾用了 Audiobox，evaluation 就不能再用 Audiobox 指標

### 已採用此指標的論文（學術引用依據）
- **LeVo (2025)**：多偏好對齊歌曲生成
- **ACE-Step (2025)**：音樂生成基礎模型，Table 1 全面採用四指標
- **SongBloom (2025)**、**MIDI-SAG (2025)**
- **AudioMOS Challenge 2025（Track 2）**：官方評測框架

---

## 2. Resonate（Li et al., 2026，SJTU）

**論文**：Resonate: Flow Matching with Group Relative Policy Optimization for Audio Generation
**機構**：上海交通大學（SJTU）
**第一作者**：Xiquan Li — **即 MeanAudio codebase 的維護者**

> Resonate 直接引用 MeanAudio 為 reference [10]，並以它為 pre-training backbone。這是 MeanAudio 原班人馬的後續工作，與本研究方向平行且互補。

### 核心內容

**訓練流程**：
1. Pre-training：MeanAudio backbone（Flow Matching）
2. SFT（Supervised Fine-tuning）：在 AudioCaps 上微調
3. **Flow-GRPO**（Group Relative Policy Optimization）：用 LALM reward 取代 DPO + CLAP reward

**關鍵結果**：AQAScore（LALM-based）作為 reward 優於 CLAP reward：
- CU：+0.09
- PQ：+0.14

### 與本研究的直接關聯

| Resonate 發現 | 本研究對應 | 意義 |
|--------------|-----------|------|
| SFT on in-the-wild AudioCaps 導致 PQ 退步（5.923→5.764） | Phase 5 Hard Filtering 退步 | **External validation**：資料集錄音品質不均是共同問題 |
| LALM reward > CLAP reward | Phase 7 V2（CLAP-best）< V1（random） | CLAP 作為信號的局限性跨兩篇論文印證 |
| MeanAudio 為 backbone | 本研究即 MeanAudio | 研究脈絡直接延續 |

### 對本研究的直接啟示

1. **Phase 5 退步現象的 External Validation**：可在論文中引用 Resonate，說明「SFT on in-the-wild data 導致 PQ 退步」是已知現象，非本研究獨有問題
2. **Phase 10+ 方向**：Flow-GRPO + LALM reward 是 Phase 9 之後的可能路徑（RL fine-tuning）
3. **AQAScore 參考**：Resonate 用 LALM 評分，對應本研究考慮的 Audiobox PQ conditioning 方向

---

## 3. PE-AV / Perception Encoder Audiovisual（Vyas et al., 2025，Meta）

**論文**：Perception Encoder: The best visual embeddings are not at the output
**機構**：Meta AI
**版本**：PE-AV（Audiovisual variant）

### 核心數據

| Benchmark | LAION-CLAP | PE-AV | 提升 |
|-----------|-----------|-------|------|
| AudioCaps T→A R@1 | 35.4 | **45.8** | +10.4 |

支援 domain：speech、music、sound（三個 domain 統一模型）

### 對本研究的直接啟示

**長期替換目標**：LAION-CLAP 作為 evaluation encoder 有已知局限（CLAP score ≠ 感知品質，Phase 6/7/8 均驗證）。PE-AV 在音訊-文字對齊任務上顯著優於 LAION-CLAP。

**實施建議**：
- 短期（Phase 8 以前）：維持 LAION-CLAP，繼續用 CE/PQ 作為主要感知指標
- 長期（Phase 10+）：考慮以 PE-AV 取代 LAION-CLAP 作為 CLAP score 計算基礎

**注意**：替換 evaluation encoder 會使歷史數字不可直接比較，需完整重新跑所有 baseline eval。成本高，應作為長期規劃。

---

## 優先順序總結

| 優先級 | 行動 | 成本 | 預期效益 |
|--------|------|------|---------|
| ★★★ | 對 251K 訓練 clip 跑 Audiobox，產生 PQ label | 3~4 小時（CPU） | 為 Phase 9 準備資料 |
| ★★★ | Phase 9：Audiobox PQ → q conditioning | 全重訓（~15h） | 解決 CLAP ≠ 感知的根本問題 |
| ★★ | Inference-only：5 caption embedding 平均 | 零成本 | 快速驗證 diversity 假說邊界 |
| ★ | PE-AV 替換 CLAP evaluation | 高成本（需重跑全部 baseline） | 長期 evaluation 可靠性 |
| ★ | Flow-GRPO（Resonate 路線） | 非常高 | Phase 10+ 方向 |
