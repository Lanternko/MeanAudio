# PromptCC 文獻品質審核 × 論文缺口補調查（2026-07-20）

**對象論文**：ISMIR 2026 Paper 487 — *Improving Text-to-music Generation Model Training Through Prompt-consistency Conditioning*  
**來源**：`docs/reviews/ismir2026-487-promptcc/`（R1 WR / R2 WR / R3 WA / Meta WA）  
**配套內部結論**：`docs/reviews/ismir2026-487-promptcc/CORRECTNESS_VALIDATION_PLAN.md`  
**舊版速記**：`docs/literature/Literature_Insights.md`（2026-04-02；僅 AES / Resonate / PE-AV，**不足以**回應 reviewer）  
**本檔定位**：文章品質評級 → 對論文幫助度 → 不足處大範圍補調查 → 可執行寫作/實驗清單

---

## 0. 執行摘要

| 維度 | 結論 |
|------|------|
| **論文現況** | Meta Weak Accept；R1/R2 Weak Reject 的共識弱點是 **claim 過度 = 把 text-space agreement 講成 audio-grounded correctness** + hard-filter 不公平 + 單一 pipeline |
| **既有 related work** | 已有 QA-MDT、MR-FlowDPO、Noise2Music、LP-MusicCaps、AES；**缺** CosyAudio、Self-Consistency/Ding、BRACE、Manco diversity、MU-LLaMA/MuLan teacher 定位 |
| **可 defend 的 claim** | q = **單一 captioner 的 stochastic self-agreement**；在 LP-MusicCaps→MeanAudio 上 **empirically useful as auxiliary condition**（非 correctness、非 trust score） |
| **文獻策略** | Ding+BRACE 護城河（agreement/CLAP proxy ≠ correctness）+ CosyAudio audio-grounded 對照 + **Audiobox 一手** prompting≫filtering + QA-MDT（仍須 size-matched 實驗） |
| **P0 實驗（非文獻）** | random-bin control、size-matched filter、clean multicap（歷史 0.0650 **不可引用**） |
| **本檔狀態** | R1–5 完成；**lit×內部 phase 對齊**（§15）；無新 A 級外部 prior；**survey 收斂+凍結建議**；瓶頸＝寫作+P0 實驗 |

---

## 1. Reviewer 批評 → 文獻需求矩陣

| ID | 批評 | 來源 | 需要的文獻類型 | 現有 paper 覆蓋 | 本輪補調查 |
|----|------|------|----------------|-----------------|------------|
| **W1** | consistency ≠ audio-grounded correctness | R1, R2, Meta | agreement 極限 + audio-grounded confidence 對照 | 弱 | **R2 滿**：Ding + **BRACE** + CosyAudio + Wang |
| **W2** | hard filter 砍 53% data 不公平 | R1, Meta | filtering 傷 volume；condition 優於 discard | 半 | **R2 滿文獻**：Audiobox §5.6 一手 + QA-MDT Fig.1(b)；**實驗仍缺 size-matched** |
| **W3** | 單 backbone + 單 captioner | R2, Meta | multi-pipeline / multi-teacher | 無 | **R4**：Music Flamingo 作第二 captioner；TangoFlux 第二 backbone 候選；**仍需實驗** |
| **W4** | 缺 teacher distill / uncertainty / label-noise | R1 | 三條定位線 | 曾最大洞 | **R2 已填譜系**：MuLan/MU-LLaMA/Kong/CosyAudio + Ding/BRACE + QA-MDT/Audiobox |
| **W5** | 增益小、無 multi-seed、無 demo | R1–R3 | evidence 慣例 | 有 CMOS+CI | Audiobox bootstrap CI；demo 仍缺 |
| **W6** | w/o quantize 崩、Stage-2 only | Meta, R1 | quantize 先例；prepend 弱 | 半 | quantize：**R2 滿**；Stage-2：**R3 文獻無解**→empirical recipe only |
| **W7** | multi-valid captions = 音樂本質 | Meta | multi-caption / subjectivity | Intro 例子 | **R4 滿**：Lee MusicCaps 一手（annotator F1 0.76）+ SDD 25% + Manco + BRACE HH |

---

## 2. 文章品質評級（對 PromptCC 的幫助度）

評級尺度：

- **A（必 cite）**：直接定位 PromptCC 或直接回答 reviewer 一句話  
- **B（應 cite）**：重要對照族 / 方法祖先 / 實驗設計錨  
- **C（可 cite）**：背景或次要支援  
- **D（勿過度依賴）**：相關但易 overclaim、或與 claim 衝突  
- **品質** = 與一手來源一致度（本輪是否核對到摘要/方法段落）

### 2.1 核心文獻表

| 文獻 | 年 | arXiv / 出處 | 品質 | 幫助度 | 對論文怎麼用 | 風險 / 品質註記 |
|------|----|--------------|------|--------|--------------|-----------------|
| **CosyAudio** (Zhu et al.) | 2025 | [2501.16761](https://arxiv.org/abs/2501.16761) | **A 一手核對** | **A** | **最近鄰 related work**：audio-grounded confidence conditioning | 草稿 comment 已提但正文未 cite — **必補** |
| **QA-MDT** (Li et al.) | 2024/25 | [2405.15863](https://arxiv.org/abs/2405.15863)；IJCAI'25 | **A 一手** | **A** | quality-bin conditioning 典範；**filter 會傷 FAD** Fig.1(b) | 已 cite；可強化「p-MOS 是 audio quality 非 caption agreement」對比 |
| **MR-FlowDPO** | 2025 | [2512.10264](https://arxiv.org/abs/2512.10264) | **A 一手** | **A** | reward prompting；CLAP/AES/HuBERT 三軸 | 已 cite；**reward prompt = 自然語言描述分數**，≠ 裸 decimal prepend |
| **Self-Consistency** (Wang et al.) | 2022/23 | [2203.11171](https://arxiv.org/abs/2203.11171)；ICLR'23 | 標準引用 | **A** | PromptCC K=5 的 **方法祖先**（多次取樣→agreement） | 原論文用 majority 提高 *accuracy*；你們用 dispersion 當 *condition* — 必須寫清轉用 |
| **When LLMs Agree, Are They Right?** (Ding) | 2026 | [2607.08065](https://arxiv.org/abs/2607.08065) | **A 摘要核對** | **A** | **直接護航 W1**：agreement 是 regime-dependent proxy，不是 correctness | 領域是 LLM judge/GPQA/AIME，**跨域類比**；寫「同構假設被大規模審計削弱」勿寫「已在音樂 caption 證明」 |
| **LP-MusicCaps** (Doh et al.) | 2023 | ISMIR / [2307.16372](https://arxiv.org/abs/2307.16372) | 已用 | **A** | pseudo multi-caption 資料側 | 已 cite；R3 要 ambiguity claim 加 citation — 用它 + MusicCaps subjectivity |
| **Noise2Music** | 2023 | [2302.03917](https://arxiv.org/abs/2302.03917) | 已用 | **B** | LLM synthetic captions for TTM | 已 cite |
| **Audiobox Aesthetics** (Tjandra et al.) | 2025 | [2502.05139](https://arxiv.org/abs/2502.05139) | **A 一手 §5** | **A** | eval + **prompting ≫ filtering**（主觀全勝；filter 傷 CLAP/WER） | 詳見 §11.1；注入是 **rounded text prefix** |
| **MeanAudio** | 2025 | [2508.06098](https://arxiv.org/abs/2508.06098) | backbone | **A** | 兩階段 CFM→MeanFlow 必須寫清（R1 技術澄清） | 已 cite；Stage-1/2 + CFG 仍需展開 |
| **Resonate** (Li et al.) | 2026 | [2603.11661](https://arxiv.org/abs/2603.11661) | 內部筆記 | **B** | 同 backbone 族的 **post-train LALM reward**；與 PromptCC pretrain conditioning **互補** | double-blind 勿 self-cite 成「我們」；可當相關工作匿名句 |
| **Manco et al. Augment/Drop/Swap** | 2024 | [2409.11498](https://arxiv.org/abs/2409.11498) | **A 一手** | **B** | **對比學習**：curation 一階；**其上**用 10 partial views（Augmented View Dropout）增 diversity | ⚠️ Round-1 誤讀「diversity>curation」已 **R3 修正**；見 §13.1 |
| **Annotator Subjectivity in MusicCaps** (Lee et al.) | 2023 | [CEUR Vol-3528](https://ceur-ws.org/Vol-3528/paper6.pdf) | **A 一手 PDF** | **A** | caption 按 annotator 分群（BERT F1 0.76）；tag 偏好極端 | W7 硬證據；詳 §14.1 |
| **Music Flamingo** (Ghosh et al.) | 2025 | [2511.10289](https://arxiv.org/abs/2511.10289) | 摘要級 | **A(W3)** | 第二 captioner 候選；長 caption / theory-aware | 你們 MF 實驗線；cross-captioner q |
| **Song Describer Dataset** (Manco et al.) | 2023 | [2311.10057](https://arxiv.org/abs/2311.10057) | **A 一手** | **A** | **25% tracks 多 annotator multi-caption**；Jamendo 同源；W7+eval | 詳 §13.2；建議作 OOD human-caption eval |
| **PE-AV** (Vyas et al.) | 2025 | Meta | 內部筆記 | **C** | 比 LAION-CLAP 更強的 audio–text 評估 encoder | 長期 eval 升級；非 resubmit 必要 |
| **Make-An-Audio / WavCaps** | 2023–24 | 多篇 | 背景 | **C** | distill-then-reprogram / 弱標註 caption 精煉 | R1 teacher 線可一帶 |
| **Tango2 / MusicRL / TangoFlux** | 2024 | 各 arXiv | 背景 | **C** | preference / RL 對齊 TTA | 與 MR-FlowDPO 同族；非 PromptCC 直接祖先 |
| **「Hidden Semantic Bottleneck in Conditional Embeddings」** | 2026 | [2602.21596](https://arxiv.org/abs/2602.21596) | 摘要級 | **D→C** | AdaLN conditioning 冗餘分析；**不直接**解釋 w/o quantize 崩 | 勿硬扯成 quantize 必要；僅作 conditioning 機制背景 |
| **BRACE** (Guo et al.) | 2025 | [2512.10403](https://arxiv.org/abs/2512.10403)；NeurIPS'25 DB | **A 摘要+數字** | **A** | **CLAPScore 作 caption quality 有上限**（best ~70 F1） | W1：audio–text proxy 不可當 correctness 證書；詳 §11.2 |
| **Kong et al. Synthetic Captions** | 2024 | [2406.15487](https://arxiv.org/abs/2406.15487) | 摘要級 | **B** | ALM 合成 caption 改善 TTA | R1 teacher/pseudo-caption 線 |
| **MU-LLaMA** (Liu et al.) | 2023 | [2308.11276](https://arxiv.org/abs/2308.11276) | 摘要級 | **B** | 專為 TTM 產 caption 的 music understanding teacher | paper 草稿曾提；R1 distill 線 |
| **MusicLM / MuLan** | 2023 | [2301.11325](https://arxiv.org/abs/2301.11325) | 標準 | **B** | 無 caption、用 joint embedding 監督 | R1 soft semantic condition 先例 |
| **EzAudio** | 2024 | [2409.10819](https://arxiv.org/abs/2409.10819) | 摘要級 | **C** | 合成 caption pipeline + 主觀 OVL/REL | W5 evidence 慣例參考 |
| **TangoFlux + CRPO** | 2024/26 | [2412.21037](https://arxiv.org/abs/2412.21037) | 摘要級 | **C→B** | Flow matching + **CLAP-ranked preference** post-train | 與 PromptCC **正交**（post-train vs pretrain condition）；W3 第二 backbone 候選 |
| **Resonate Flow-GRPO** | 2026 | [2603.11661](https://arxiv.org/abs/2603.11661) | 摘要+表 | **B** | MeanAudio 族 + LALM AQAScore > CLAP reward | 解釋 PQ/CLAP-only conditioning 未必贏；double-blind 注意 |

### 2.2 品質審核：先前整理的問題（Round-1 修正）

上一輪（session `019f7df0`）整理**方向正確**，但有以下不足（本檔已修）：

| 問題 | 修正 |
|------|------|
| CosyAudio 只寫「audio-grounded confidence」口號，缺機制 | 補：confidence = AudioCapTeller 的 **query–text 最高相似度**（ATC 空間），caption+audio 同進模型；generator 把 **scalar-quantize confidence → embedding lookup → concat time embedding** |
| 暗示 CosyAudio 只 condition、不 filter | 修正：CosyAudio **同時** high-quality filter + DPO on low/high + confidence conditioning |
| MR-FlowDPO 只當「audio-grounded reward condition」 | 補：它是 **DPO post-train** + **NL reward prompting**（「Text alignment is 0.26…」），不是訓練期 AdaLN bin |
| QA-MDT 只當 quality condition | 補：Fig.1(b) **明確** 0/33/66/100% low-quality filter → FAD 持續變差 — W2 最硬彈藥 |
| Ding 2026 當「音樂 caption 已證 agreement≠accuracy」 | 降級為 **同構假設被 LLM 領域大規模打穿**；音樂側仍靠你們 diagnostics / human plan |
| 缺 Manco / MusicCaps subjectivity | 補 W7 與 caption diversity 線 |
| 未對照 paper.tex 已 cite / 僅 comment | 見 §3 與 §6 寫作 patch 清單 |
| 未落盤 | 本檔為正式 memo |

---

## 3. 一手機制對照（PromptCC vs 最近鄰）

### 3.1 信號從哪來？

| 系統 | 信號 | 是否看 audio？ | 是否多次取樣 captioner？ | 注入方式 |
|------|------|----------------|--------------------------|----------|
| **PromptCC（本論文）** | mean pairwise **text** embedding sim of K=5 captions | **否** | **是**（同模型不同 seed） | quantize → `pc_embed` → **AdaLN** |
| **CosyAudio** | AudioCapTeller **query–caption similarity** as confidence | **是** | 否（單次 caption + assess） | quantize → lookup → **concat time emb** in U-Net |
| **QA-MDT** | Pseudo-MOS（音訊品質） | 是（品質模型） | 否 | discretized quality **prefix tokens** |
| **MR-FlowDPO** | CLAP / AES-PQ / HuBERT-likelihood | 是（對 *generated* audio 的 reward） | 否 | **DPO** + **NL reward prompt** prepend |
| **CLAP/PQ conditioning baselines（本論文）** | 單 caption–audio CLAP 或 AES-PQ | 是 | 否 | 與 PromptCC 同式 condition（實驗表） |

**對 R1 的一句定位（建議直接進 related work）**：

> Prior work conditions TTM/TTA on *audio quality* (QA-MDT), *audio–text confidence of one caption* (CosyAudio), or *post-hoc multi-reward preference* (MR-FlowDPO). PromptCC instead conditions on the *stochastic self-agreement among multiple pseudo-captions in text space*, without an external quality model and without claiming audio-grounded correctness.

### 3.2 CosyAudio 細節（W1/W4/W6 高價值）

來源：arXiv:2501.16761 HTML 方法節（本輪閱讀）。

1. **AudioCapTeller**：Q-Former 風格 learnable queries + BEATs；多任務 ATM / ATC / AEC / AAC。  
2. **Confidence 定義**（關鍵）：生成 caption 後，再以 audio+caption 前向，取 query embedding 與 global text feature 的 **最高 pairwise similarity** 當 confidence。  
   → 這是 **audio–text matching 空間的 self-assessment**，不是「五個 caption 互相比」。  
3. **Generator**：Tango-style LDM；confidence **scalar-quantized** → embedding table → **concat with timestep embeddings**。  
4. **Self-evolving**：well-labeled 訓 → 弱標註用 μ−σ filter 高品質 → 再訓 → **DPO** 用高低品質偏好 → 全庫 recaption + confidence → 訓 generator。  

**對 PromptCC 的啟示**：

| 項目 | 啟示 |
|------|------|
| 注入 | 成功先例 = **quantize + dedicated embedding**，不是裸數字塞進 T5 文本 |
| 對照實驗 | 理想 P1：用 CLAP(audio, caption) 或 CosyAudio 式 confidence 當 **audio-grounded 條件**（你們已有 CLAP conditioning baseline，可在文中明確對齊 CosyAudio） |
| Filter | 他們 filter **高品質子集再訓練 captioner**，並對低品質做 DPO，**不是**「砍一半 TTM 資料就結束」— 可寫「naive hard discard of half the generator data is not the same as quality-aware corpus refinement」 |
| Teacher distill | R1 要的線：AudioCapTeller 是 **pretrained audio understanding teacher** 把 soft confidence 灌進 generator |

### 3.3 QA-MDT 細節（W2 高價值）

來源：arXiv:2405.15863v4。

- 問題拆兩軸：**waveform 品質低** + **text–audio 一致性低**。  
- **Fig.1(b)**：以 Pseudo-MOS < 4 當 low-quality，filter 0/33/66/100% → MusicCaps FAD **持續變差**。  
  → 與 PromptCC hard-filter 差、Audiobox「filtering 傷 volume」同方向。  
- 解法：quality-aware **condition** + caption refinement（LLM+CLAP），**不是**只砍資料。  
- Demo：https://qa-mdt.github.io/（W5 demo 慣例）。

**你們 hard-filter 論述建議改寫**：

1. Cite QA-MDT Fig.1(b) + Audiobox：volume 是一階項。  
2. 承認現有 hard-filter baseline **未控制 N**（R1/Meta 正確）。  
3. 補 **size-matched** random subsample of high-s **或** matched-N high-s filter — 才能說 condition ≫ filter。

### 3.4 MR-FlowDPO 細節（W6 prepend 論述）

來源：arXiv:2512.10264。

- **Reward prompting 範例**：`"Text alignment is 0.26, Audio quality is 8.24, Semantic Consistency is 0.31"` — **設計過的自然語言**，訓練時用 positive sample 的 reward，推論用 99-percentile。  
- 這與 PromptCC「**PromptCC w/o quantize = 直接把 float s prepend 進 prompt**」**不是同一類 continuous conditioning**。  
- R1 說 w/o quantize 是 weak implementation — **同意**；應用 MR-FlowDPO 對比時寫：  
  - fair continuous：AdaLN/FiLM 注入 float *s*，或  
  - fair textual：把 *s* 寫成 NL（「caption agreement is high/medium/low」）再 prepend，  
  - **不要**再只測裸 `"0.83 ...caption..."`。

### 3.5 Self-consistency 與 Ding 2026（W1 claim 護城河）

| 文獻 | 主張 | PromptCC 用法 |
|------|------|---------------|
| Wang et al. 2022 | 多次 CoT 取樣 + majority 提高推理正確率 | 你們借 **多次取樣** 與 **agreement 統計**，但 **不用 majority caption 當唯一標籤**，而是把 **dispersion 當 condition** |
| Ding 2026 | 53 runners × K=50；agreement 與 accuracy ρ 僅 0.20–0.59；frontier 高 agreement 仍可大量錯 | 主動寫：**我們不假設 high-q ⇒ correct caption**；high agreement 可為 stable bias（R1/R2 原話） |

**Claim 安全句（建議 Limitations / Method）**：

> Following evidence that self-agreement is only a *regime-dependent* proxy for correctness in language models (Wang et al., 2023; Ding, 2026), we treat prompt consistency strictly as the *stochastic self-agreement of one captioner in text-embedding space*. We do **not** claim that *q* measures audio-grounded caption correctness or trustworthiness.

### 3.6 Multi-validity / diversity（W7）

| 文獻 | 用法 |
|------|------|
| LP-MusicCaps | 一曲多 pseudo-caption；已是資料生成器 |
| MusicCaps annotator subjectivity (Lee 2023) | 不同 annotator 偏好不同音樂屬性 — **人類也無 singular GT** |
| Song Describer (~1.1k captions / 706 tracks) | 人類 multi-caption eval 集 |
| Manco et al. 2024 Augment/Drop/Swap | 在 music–text **contrastive** 中，**diversity 勝過 accuracy curation** |

**Framing（Meta 要求）**：把 one-to-many 當 **音樂描述的 fundamental property**，captioner error 是 **第二因子**；PromptCC 讓 generator **看見** variability，而非假裝每條 caption 同等確定。

---

## 4. 與 paper.tex 現況對齊（cite 缺口）

| 項目 | 正文狀態 | 建議 |
|------|----------|------|
| QA-MDT | 已 cite §2.3 | 加一句 filter→FAD 惡化 + demo |
| MR-FlowDPO | 已 cite | 澄清 reward prompt ≠ 裸 scalar；你們 CLAP/PQ baseline 對齊其 reward 軸但 **非 DPO** |
| CosyAudio | **僅 comment**（L144, L193） | **正文必 cite**；一 vs 多 caption + audio-grounded vs text agreement |
| Self-Consistency / Ding | **無** | Limitations 或 Related 新小節 |
| Manco Augment/Drop/Swap | **僅 comment**（L236–244） | 恢復 2–3 句：diversity as signal vs your conditioning |
| Resonate | 刻意移除（double-blind） | 維持；camera-ready 再考慮 |
| Noise2Music / LP-MC / AES / CLAP / PE-AV | 有 | 保留；Jamendo eval 目的拆開（R1） |

---

## 5. 文章對論文幫助度總評

### 5.1 高幫助（改寫/重投 ROI 最大）

1. **CosyAudio** — 填 W1/W4 最大洞；注入機制支援 W6 quantize。  
2. **Ding 2026 + Wang 2022** — 讓「我們不 claim correctness」變成有文獻的 *principled* 限制，而非認輸。  
3. **QA-MDT Fig.1(b) + Audiobox** — 撐 condition-over-filter 敘事；但 **實驗仍需 size-matched** 才過 R1。  
4. **MR-FlowDPO reward prompting 細節** — 修正 w/o quantize 敘事，避免被說「連續條件必然差」。  
5. **Manco + MusicCaps subjectivity** — 撐 W7 multi-validity。

### 5.2 中幫助

- Resonate：互補 post-train 路線；解釋 PQ-only conditioning 未必贏（你們表上 PQ conditioning 也弱）。  
- Song Describer / PE-AV：評估生態。  
- Tango2/MusicRL：preference 族背景。

### 5.3 低幫助 / 勿硬塞

- 通用 noisy-label co-teaching 經典（與 generative conditioning 距離遠）— 一句帶過即可。  
- Conditional embedding bottleneck 2026 — 不解釋你們 prepend 失敗。  
- 把 Ding 寫成「已在音樂 caption 證明 agreement=bias」— **過度**。

### 5.4 文獻 alone 解不了、必須實驗

| 缺口 | 為何文獻不夠 |
|------|----------------|
| Random-bin control | 唯一區分「consistency semantics」vs「extra embedding capacity」 |
| Size-matched filter | 唯一公平回應 W2 |
| Cross-captioner | 唯一實質回應 W3（MF 線） |
| Demo + multi-seed | 同儕慣例；文獻只證明「別人有做」 |
| Clean multicap 0.0650 | 歷史 cache 錯位 — **不可 cite**（見 CORRECTNESS plan） |

---

## 6. 可執行清單（寫作 / 實驗）

### 6.1 寫作（高 ROI，不需新 GPU）

| # | 動作 | 打哪個 critic |
|---|------|---------------|
| W-1 | Abstract 第一句定義 *prompt consistency* = mean pairwise caption agreement（Meta） | 可讀性 |
| W-2 | 全文搜 reliability / trust / correctness → 改成 self-agreement / ambiguity-aware condition | W1 |
| W-3 | Related work 加 CosyAudio 段（§3.1 表格濃縮成 4–6 句） | W1, W4 |
| W-4 | 加 Self-Consistency + Ding 2026 限制句（§3.5） | W1 |
| W-5 | 恢復 Manco diversity 2–3 句 | W7, W4 |
| W-6 | Hard-filter 段：承認未 size-match；cite QA-MDT Fig.1(b)；承諾/補 matched-N | W2 |
| W-7 | w/o quantize：承認是 weak continuous baseline；對照 CosyAudio quantize-embed 與 MR-FlowDPO NL prompt | W6, R1 |
| W-8 | Jamendo = in-domain pseudo-caption distribution；MusicCaps = human-prompt following 主戰場 | R1 |
| W-9 | MeanAudio：Stage-1 CFM / Stage-2 MeanFlow；condition dropout vs CFG scale；q 僅 Stage-2 | R1, Meta |
| W-10 | Multi-validity framing（音樂本質）+ R3 要的 ambiguity citation | W7, R3 |
| W-11 | Demo page + listener demographics | W5, R3 |
| W-12 | Limitations：single captioner/backbone；q 非 correctness；歷史 multicap 結果若曾提及需撤回 | 誠實 |

### 6.2 實驗優先序（對齊 CORRECTNESS plan + 本檔）

| 優先 | 實驗 | 打 critic | 狀態備註（2026-07-20） |
|------|------|-----------|------------------------|
| **P0** | Clean NoQ（修 q-null / CFG） | 所有增益可信度 | 進行中（phase8 clean） |
| **P0** | **Random-bin control**（同 bin 頻率） | W1 機制 | 等 clean NoQ Stage-1 |
| **P0** | **Size-matched hard filter** | W2 | 未做 |
| **P0** | Clean multicap / true-random（**勿用 0.0650**） | Meta 五 caption 建議 | tooling 已修；cache 未產 |
| **P1** | q vs CLAP/PE-AV + high-q/low-CLAP 反例 | W1 | 可無 GPU 部分做 |
| **P1** | Cross-captioner（Music Flamingo 等）重算 q | W3 | 實驗線已有 |
| **P2** | Fair continuous：AdaLN float *s* 或 NL agreement prompt | W6 | 未做 |
| **P2** | Multi-seed mean±std | W5 | 未做 |
| **P3** | Human groundedness 400–600 clips | W1 | Deferred（plan 已設計） |
| **P3** | 第二 TTM backbone | W3 | 算力門檻高 |

---

## 7. 建議 Related Work 段落骨架（可直接改寫）

```text
2.x Quality-aware and reward conditioning
  - QA-MDT: pseudo-MOS bins; filtering low-quality audio hurts FAD
  - CosyAudio: audio-grounded caption confidence → quantize → time-emb
  - MR-FlowDPO: multi-reward DPO + NL reward prompting (CLAP/AES/HuBERT)
  → PromptCC: text-space multi-caption agreement; no external MOS/reward model

2.y Pseudo-captions, noise, and diversity
  - Noise2Music, WavCaps, LP-MusicCaps
  - Manco et al.: diversity of LLM captions helps contrastive learning more than accuracy curation
  - Audiobox: prompting strategy > filtering at matched data volume

2.z Self-agreement as a signal—and its limits
  - Wang et al. self-consistency (sampling → agreement)
  - Ding 2026: agreement ≠ accuracy (regime-dependent proxy)
  → We use agreement as a training condition, not as a correctness certificate
```

---

## 8. 核心文獻速查（含連結）

| 論文 | 連結 | 與 PromptCC |
|------|------|-------------|
| CosyAudio | https://arxiv.org/abs/2501.16761 | Audio-grounded confidence；quantize-embed |
| QA-MDT | https://arxiv.org/abs/2405.15863 | Quality bins；filter 傷 volume |
| MR-FlowDPO | https://arxiv.org/abs/2512.10264 | Multi-reward + NL prompt；DPO |
| Self-Consistency | https://arxiv.org/abs/2203.11171 | 多次取樣 agreement 祖先 |
| Agreement ≠ accuracy | https://arxiv.org/abs/2607.08065 | Claim 護城河 |
| LP-MusicCaps | https://arxiv.org/abs/2307.16372 | Pseudo multi-caption |
| Noise2Music | https://arxiv.org/abs/2302.03917 | LLM captions for TTM |
| Manco Augment/Drop/Swap | https://arxiv.org/abs/2409.11498 | Caption diversity |
| MeanAudio | https://arxiv.org/abs/2508.06098 | Backbone |
| Resonate | https://arxiv.org/abs/2603.11661 | Post-train LALM reward 互補 |
| Audiobox Aesthetics | https://arxiv.org/abs/2502.05139 | Eval + **一手** prompting≫filtering + quantize text prefix |
| BRACE | https://arxiv.org/abs/2512.10403 | CLAPScore 作 caption quality 有上限 |
| Kong synthetic captions | https://arxiv.org/abs/2406.15487 | ALM synthetic captions for TTA |
| MU-LLaMA | https://arxiv.org/abs/2308.11276 | Music teacher captioner for TTM data |
| MusicLM / MuLan | https://arxiv.org/abs/2301.11325 | Soft embedding supervision w/o captions |
| 內部 correctness plan | `docs/reviews/ismir2026-487-promptcc/CORRECTNESS_VALIDATION_PLAN.md` | Claim 邊界 + multicap bug |
| 本檔 | `docs/literature/PromptCC_Literature_Quality_and_Gaps_2026_07_20.md` | 完整整理（R1+R2） |

---

## 9. 一句策略結論

Reviewer 真正卡的不是「沒有相關文獻可以引用」，而是：

1. **敘事**把 text-space self-agreement 講得像 audio-grounded label quality；  
2. **實驗** hard-filter 與 continuous ablation 不公平；  
3. **定位**沒接到 CosyAudio / self-consistency limits / diversity / BRACE 文獻。

文獻路徑已清楚：

| 步驟 | 內容 |
|------|------|
| 1 | 定位 = self-agreement uncertainty **proxy**（Wang, Ding），**不是** correctness；BRACE 再打 CLAP proxy |
| 2 | 對照 = CosyAudio confidence、QA-MDT quality、MR-FlowDPO rewards、Audiobox PQ prompt |
| 3 | 策略 = condition ≫ naive hard discard（QA-MDT + **Audiobox 一手**；**需 size-matched 實驗**） |
| 4 | 證據 = random-bin + cross-captioner + demo + multi-seed |

**文獻補齊 alone 可抬 related-work 分；accept 仍取決於 P0 實驗與 claim 降級。**

---

## 11. Round-2 大範圍補調查（2026-07-20 第二輪）

> 目標：針對 Round-1 標為「仍薄」的洞做一手/準一手補強——Audiobox 下游實驗、caption quality meta-eval、teacher/soft-label 線、synthetic caption 生態。

### 11.1 Audiobox Aesthetics 一手（arXiv:2502.05139 §5）— **W2/W6 升級彈藥**

**品質：A（HTML 全文核對） / 幫助度：A**

#### 方法事實（可寫進 paper）

1. **四軸**：PQ / PC / CE / CU；音樂上 CE 與 human OVL 相關最高（PAM-music OVL↔GT-CE **0.848**；PQ **0.778**）。  
2. **下游三情境**：Baseline（全資料）/ Filtering（丟 <p 百分位 PQ）/ Prompting（把 PQ 當 text prefix）。  
3. **Prompting 注入**（與 PromptCC w/o quantize **直接可比**）：  
   - 訓練：`"Audio quality: ŷ"` where `ŷ = round(y·r)/r`，**r ∈ {2, 5}**（**有意量化**）  
   - 推論：固定為訓練集 PQ 的 **p50 / p75 / p90**  
4. **Filtering**：p ∈ {25, 50} 百分位門檻。  
5. **結論原文要旨**（§5.6）：  
   - **Prompting 在主觀上全面勝 Filtering**  
   - Filtering 雖提 quality，但 **data volume 線性下降 → CLAP/WER 等 alignment 變差**  
   - Prompting **維持 alignment ≈ baseline 同時提 quality**

#### 對 PromptCC 的精確用法

| 論點 | 怎麼 cite |
|------|-----------|
| Condition ≫ hard discard | Audiobox + QA-MDT 雙錨；**Audiobox 有人耳 pairwise + CI** |
| 你們 hard-filter 不公平 | Audiobox 明確把「volume 下降」寫成 filter 的 **結構性代價** — 但他們 filter 用的是 **audio PQ**，你們是 **caption agreement**；類比要標清 |
| w/o quantize | Audiobox **也 quantize**（round factor 2/5）再寫進 text；MR-FlowDPO 用 NL 描述；CosyAudio 用 emb table。**三家成功做法都離散化**；你們 prepend 裸 float 是 outlier |
| PQ conditioning 為何你們表上弱 | Audiobox 用 **NL prefix + 推論設高百分位**；你們若用 AdaLN bin 餵 PQ，注入通路不同，需在文中區分 |

#### 品質註記 / 勿 overclaim

- Audiobox TTM 實驗用 **內部 18k 小時高品質音樂**，不是 Jamendo unsupervised。  
- 他們 filter/prompt 的是 **音訊美學 PQ**，不是 multi-caption agreement。  
- 可寫「quality-aware *conditioning* outperforms quality-based *discard* under matched generative recipes」；**不可**寫「Audiobox 證明 caption-consistency filter 無效」。

### 11.2 BRACE（arXiv:2512.10403, NeurIPS 2025 DB）— **W1 新 A 級彈藥**

**品質：A（數字核對） / 幫助度：A（先前完全漏）**

| 事實 | 含義 |
|------|------|
| BRACE-Main：best CLAP-based caption quality model **F1 ≈ 70.01**（LAION-CLAP） | 即使 **audio–text CLAP**，當 caption quality / preference 判官也遠非可靠 |
| Best open LALM on Main **F1 ≈ 63.19** | LALM 作 caption judge 也不夠 |
| BRACE-Hallucination 上部分模型更高，但 Main 仍難 | 細粒度 faithfulness ≠ 整體 caption ranking |
| 設計含 HH/HM/MM（human–human / human–machine / machine–machine） | 直接碰 **multi-valid human captions**（W7） |

**對 PromptCC 的寫法（建議 Limitations 或 Method）**：

> Even *audio-grounded* automatic caption-quality metrics (CLAPScore family) remain imperfect proxies of human judgments (BRACE; Guo et al., 2025). Our text-only self-agreement score is therefore *not* presented as a caption correctness certificate; it is an auxiliary training condition whose utility we evaluate empirically.

這比只 cite Ding 2026 **更貼 audio 領域**（Ding 是 LLM/math；BRACE 是 audio caption）。

### 11.3 Teacher / soft-label / pseudo-caption 線（R1 W4 補強）

| 文獻 | 機制 | PromptCC 定位句 |
|------|------|-----------------|
| **MusicLM + MuLan** | 不靠文字 caption，用 **joint embedding** 當 semantic teacher | soft semantic supervision without free-text labels |
| **MU-LLaMA** | MERT+LLaMA music understanding → **caption 整個 corpus 給 TTM** | explicit caption teacher；與 LP-MC 同族 |
| **Kong et al. 2024** synthetic captions | Audio LM 大規模合成 caption 改善 TTA | better captions 路線（vs 你們 keep noisy captions + condition） |
| **WavCaps** | ChatGPT 弱標註 → 400k captions | weakly-labeled caption ecology |
| **CosyAudio** | AudioCapTeller teacher + confidence | 最強「teacher confidence → generator」先例（Round-1） |
| **EzAudio** | 合成 caption 策略進 DiT pretrain | 同生態 |

**R1 要的三條線 — 填完版**：

```
Teacher → student:
  MuLan/MusicLM (embedding teacher)
  MU-LLaMA / LP-MusicCaps / Kong synthetic (caption teacher)
  CosyAudio (caption + confidence teacher)

Uncertainty / reliability:
  Self-consistency (Wang); Ding 2026 (agreement≠accuracy)
  CosyAudio confidence; BRACE (CLAP proxy limits)

Label-noise / quality-aware training:
  QA-MDT p-MOS condition; Audiobox PQ prompt vs filter
  CosyAudio filter+DPO+condition
  PromptCC: condition on caption self-agreement (not denoise labels)
```

### 11.4 同儕 evidence 慣例（W5）

| 論文 | 做法 | 可抄 |
|------|------|------|
| Audiobox §5.5 | 200 pairs × 3 listeners；bootstrap **95% CI**；net win rate | 你們已有 CMOS+CI；可加 **bootstrap 描述** + **demo** |
| QA-MDT / CosyAudio | 公開 demo page | 必做 |
| EzAudio | OVL/REL MOS + 95% CI 圖 | 可選第二主觀軸 |
| MR-FlowDPO | 專業 annotator；4 axes win rate | 你們 CMOS 協議已跟它對齊 |

### 11.5 Round-2 文章品質增量評級

| 文獻 | 相對 Round-1 | 判決 |
|------|--------------|------|
| Audiobox | 從「口號 prompting>filtering」→ **一手數字+注入機制** | **升級：必在 W2/W6 正文引用細節** |
| BRACE | 新發現 | **A 必 cite（W1）** |
| MU-LLaMA / Kong / MusicLM | 新補 | **B 應 cite（W4）** |
| EzAudio | 新補 | **C** |
| Round-1 CosyAudio/QA-MDT/MR-FlowDPO/Ding | 維持 | 仍成立；無推翻 |

### 11.6 Round-1 文件品質複審（本輪對 memo 自身）

| 項目 | 評分 | 說明 |
|------|------|------|
| Reviewer→文獻映射 | **優** | W1–W7 完整 |
| 一手深度（Round-1） | **中上** | CosyAudio/QA-MDT/MR-FlowDPO 夠；Audiobox 偏二手 |
| 一手深度（Round-2 後） | **優** | Audiobox §5 + BRACE 補上 |
| 與 paper.tex 對齊 | **優** | cite/comment 狀態清楚 |
| 可執行性 | **優** | 寫作+實驗清單可開工 |
| 仍缺（文獻 alone 解不了） | — | random-bin / size-matched filter / cross-captioner / demo |

### 11.7 寫作 patch 增量（疊加 §6.1）

| # | 動作 | 來源 |
|---|------|------|
| W-13 | W2 段同時 cite **Audiobox §5.6**（prompting wins filtering；filter 傷 alignment）+ QA-MDT Fig.1(b) | §11.1 |
| W-14 | w/o quantize 段列「成功系統皆離散化」：Audiobox round-r、CosyAudio emb、QA-MDT bins、MR-FlowDPO NL | §11.1 |
| W-15 | Limitations 加 **BRACE**：即使 CLAPScore 也非 human caption quality 的充分代理 | §11.2 |
| W-16 | Related work teacher 段：MuLan/MusicLM + MU-LLaMA + CosyAudio 一句譜系 | §11.3 |
| W-17 | 澄清：Audiobox/MR-FlowDPO 的 text-prefix quantize **≠** 你們 AdaLN `pc_embed`；成功模式是「離散化後可學習」，不是「必須用同一注入器」 | W6 |

### 11.8 仍不必追的文獻（節省噪音）

| 類型 | 理由 |
|------|------|
| 經典 co-teaching / mentor-net 影像 noisy label | 與 generative caption condition 距離過遠；一句「noisy-label literature」即可 |
| 通用 LLM hallucination 偵測長列表 | BRACE + Ding 已夠；勿把 related 做成 survey |
| Conditional embedding bottleneck 2026 | 仍不解釋 prepend float 失敗 |

---

## 12. 變更紀錄

| 日期 | 內容 |
|------|------|
| 2026-07-20 | Round-1：審核先前整理 → 一手 CosyAudio/QA-MDT/MR-FlowDPO/Ding → paper.tex 對齊 → 寫作/實驗清單落盤 |
| 2026-07-20 | **Round-2**：Audiobox 2502.05139 §5 一手（prompting≫filtering + round quantize prefix）；BRACE 2512.10403（CLAPScore 上限）；MU-LLaMA/Kong/MusicLM teacher 線；增量評級與 W-13–17；速查表擴充 |
| 2026-07-20 | **Round-3**：Manco 一手修正（curation 一階≠diversity 勝過 curation）；Song Describer 一手（25% multi-caption, Jamendo）；Stage-2 文獻弱支持；可貼 Related Work 草稿；W-18–22 |
| 2026-07-20 | **Round-4**：MusicCaps subjectivity 一手 PDF；Music Flamingo 作 W3 第二 captioner；Resonate/TangoFlux 正交定位；**文獻收斂宣告** + cite checklist |
| 2026-07-20 | **Round-5**：外部無新 A prior；**lit×phase_status 對齊表**；Qwen 已部分碰 W3；SonicVerse 標 C；**凍結廣域 survey** 建議 |
| 來源 | 排程任務 019f7df2a03b；前序文獻解法 session `019f7df0` |

---

## 13. Round-3 大範圍補調查 + 誤讀修正（2026-07-20 第三輪）

> 目標：(1) 修正 Round-1 對 Manco 的 overclaim；(2) Song Describer / multi-validity 一手；(3) Stage-2-only 文獻支持度誠實降級；(4) 產出可直接改寫進 paper 的 Related Work 草稿。

### 13.1 ⚠️ 修正：Manco Augment/Drop/Swap（arXiv:2409.11498）一手

**先前錯誤（Round-1）**：寫成「diversity > accuracy curation」。  
**一手事實**：

| 原文要旨 | 含義 |
|----------|------|
| §4–§4.2：**data curation is the single most important factor** in resource-constrained music–text contrastive training | 品質/curation **壓過** 盲目 scale |
| LLM tag→caption 是 **supplement not substitute** for curation；sparse tags → non-descript / **hallucinated** captions | 直接支援「pseudo-caption 可錯」 |
| **Augmented View Dropout**：每曲從 balanced tag 子集生成 **10 個 partial captions**，訓練時 random sample view | 與 PromptCC「K=5 隨機抽一條 caption」**同構於 input diversity**，但任務是 **contrastive retrieval** 不是 TTM generation |
| TextSwap：造 hard negatives；curriculum 提高 hard-neg 比例 | 非 PromptCC 直接對照 |
| 人耳 pairwise + 多 eval set（MTC/MC/SDD） | W5/W7 慣例 |

**對 PromptCC 正確定位句**：

> In music–text *contrastive* learning, Manco et al. show that high-quality curated text is primary, while *increasing the diversity of valid partial text views* (Augmented View Dropout) further helps. PromptCC operates in *generative* TTM training: rather than only diversifying caption inputs, we expose the model to a scalar summary of *inter-caption agreement* as an auxiliary condition.

**勿再寫**：Manco proves diversity beats accuracy（**錯**）。  
**可寫**：他們證明 (i) pseudo-captions 可幻覺；(ii) 多 valid text views 有用；(iii) curation 仍重要——故 **hard-filter 與 condition 的取捨需實驗**，不能只靠「diversity 好」口號。

### 13.2 Song Describer（arXiv:2311.10057）一手 — **W7 升級為 A**

| 事實 | 對 PromptCC |
|------|-------------|
| 1106 captions / 706 tracks；**~25% tracks 有 >1 annotator caption** | **人類 multi-validity 硬證據**（Table 2 同曲三句不同但合理） |
| 音源 = **MTG-Jamendo**（與你們 train 同源家族） | 可作 **Jamendo-domain human-caption eval** 補強 MusicCaps（R1 endogamy 疑慮） |
| 明確警告 synthetic eval 不可靠（LLM hallucination） | 與 claim 收斂一致 |
| 設計為 **evaluation-only**，促 cross-dataset | W5：勿只報 MusicCaps |
| 114–142 non-expert annotators | 多視角描述音樂 |

**建議實驗/寫作**（低成本高 ROI）：

1. Limitations/Intro cite SDD multi-caption 例（比 Brahms 更可引用）。  
2. 若算力允許：在 **SDD captions** 上跑 baseline vs PromptCC 的 CLAP/AES（OOD human prompt，且與 Jamendo 音源近）。  
3. 與 MusicCaps 並列時強調 distribution shift（SDD 論文自己強調 cross-dataset gap）。

### 13.3 Stage-2 only / MeanFlow — **文獻支持弱，誠實寫**

| 來源 | 能說什麼 | 不能說什麼 |
|------|----------|------------|
| MeanFlow (Geng et al., [2505.13447](https://arxiv.org/abs/2505.13447)) | 定義 average velocity；one-step generation | 原文強調 **self-contained、不必 pretrain/curriculum** — **不支持**「extra condition 只能 Stage-2」 |
| MeanAudio recipe | 你們用 CFM Stage-1 → MeanFlow Stage-2 | 這是 **pipeline 選擇**，非 MeanFlow 定理 |
| Meta/R1 問「為何 Stage-1 加 q 有害」 | 只能給 **假說**（Stage-1 需最大 caption diversity / 較弱 condition；過早 partition 妨礙 velocity 學習） | **沒有** 外部論文直接證明此假說 |

**寫作建議**：

```
We apply PromptCC only in Stage 2 of the MeanAudio recipe. Applying the
same conditioning in Stage 1 degraded validation metrics in preliminary
runs; we leave a full stage-ablation analysis to future work and treat
this as an empirical recipe choice rather than a general principle of
flow matching.
```

勿硬扯 MeanFlow / curriculum 文獻撐 Stage-2-only。

### 13.4 本輪其他掃描（低 ROI 或已覆蓋）

| 方向 | 結果 |
|------|------|
| random-bin / dummy condition 文獻先例 | 無直接對標 TTM 論文；**你們 CORRECTNESS plan 的 random-bin 仍是原創必要控制** |
| Tango2 / MusicRL | 已在 R1–2 表為 C；post-train preference 族，不重複深挖 |
| multi-caption **generative** training 專文 | 未找到「用 caption agreement 當 condition」的直接 prior → **novelty 仍可 defend**（在 claim 收斂後） |
| 「Random Conditioning」CVPR'25 | 影像 distillation 語境，**勿 cite 當 PromptCC prior** |

### 13.5 Round-3 品質判決（memo 自身 + 文獻）

| 項目 | 判決 |
|------|------|
| 文獻側 W1/W2/W4/W6/W7 | **已足夠寫 related work + limitations** |
| 文獻側 W3 | **不足**（需實驗 multi-captioner/backbone） |
| 文獻側 W5 multi-seed/demo | **慣例清楚，仍要做** |
| Stage-2 機制解釋 | **文獻無解**；實驗/假說即可 |
| Manco 品質 | 修正後 **B 可信**（先前 B 但論點錯） |
| Song Describer | **升 A**（W7 + 可選 eval） |
| 繼續每 20 分大掃 arXiv 的邊際收益 | **遞減**；下一優先應是 **寫作落地 + P0 實驗** 而非更多 survey |

### 13.6 寫作 patch 增量

| # | 動作 |
|---|------|
| W-18 | 刪/改任何「Manco: diversity > curation」；改為 §13.1 定位句 |
| W-19 | Intro/Limitations 加 Song Describer multi-annotator 例 + cite 2311.10057 |
| W-20 | Stage-2-only 改成 empirical recipe + 假說，不 cite MeanFlow 當證明 |
| W-21 | 可選：SDD 作第二 human-caption eval set |
| W-22 | Related work 直接用下方 §13.7 草稿改寫 |

### 13.7 可貼 Related Work 草稿（英文，~半頁）

```latex
\subsection{Quality-aware and confidence conditioning}
Recent TTM/TTA systems inject auxiliary signals beyond the text prompt.
QA-MDT conditions a diffusion transformer on discretized pseudo-MOS audio
quality scores and shows that hard-filtering low-quality audio can hurt
generation metrics as data volume drops.
CosyAudio estimates an \emph{audio-grounded} caption confidence from a
joint audio--text teacher and injects quantized confidence embeddings into
the generator, optionally refining weakly labeled corpora with filtering
and preference optimization.
MR-FlowDPO and Meta Audiobox Aesthetics explore multi-reward or aesthetic
prompting: continuous or rounded quality scores are written into natural-
language prefixes (or used in DPO), and Audiobox finds that
\emph{prompting} with quality scores outperforms \emph{filtering} training
data under matched generative recipes while better preserving text--audio
alignment.
Unlike these lines, PromptCC does not estimate audio quality or
audio--text confidence of a single caption; it conditions on the
\emph{stochastic self-agreement among multiple pseudo-captions in text
embedding space}.

\subsection{Pseudo-captions, diversity, and multi-validity}
Large-scale TTM training often relies on synthetic text
(Noise2Music; WavCaps; LP-MusicCaps; MU-LLaMA).
Human evaluation sets such as MusicCaps and Song Describer show that
multiple annotators produce different yet valid descriptions of the same
recording---about 25\% of Song Describer tracks carry multi-annotator
captions---indicating multi-validity is intrinsic to music description,
not only a captioner failure mode.
In contrastive music--text learning, Manco et al.\ show that curated text
quality is primary under limited data, while constructing diverse partial
text views (Augmented View Dropout) further improves retrieval; sparse
tag-to-caption generation can hallucinate.
We retain all unsupervised clips and treat inter-caption agreement as a
conditioning cue rather than discarding low-agreement audio.

\subsection{Self-agreement as a reliability proxy---and its limits}
Self-consistency sampling is a well-known decoding strategy in LLMs, but
large-scale audits show that agreement is only a regime-dependent proxy
for correctness, not a certificate of truth.
In audio caption evaluation, BRACE similarly finds that CLAPScore-style
metrics remain imperfectly aligned with human caption-quality judgments.
We therefore interpret PromptCC strictly as captioner self-agreement
metadata that is empirically useful for training, not as audio-grounded
caption correctness.
```

### 13.8 更新後優先序（給研究者，不是再掃文獻）

| 優先 | 動作 | 類型 |
|------|------|------|
| **1** | 把 §13.7 + claim 降級寫進 paper.tex | 寫作 |
| **2** | Random-bin + size-matched filter | 實驗 P0 |
| **3** | Demo page + multi-seed | 實驗/產物 |
| **4** | Cross-captioner（MF） | 實驗 W3 |
| **5** | 可選 SDD eval | 實驗低成本 |
| **停** | 無新 reviewer 問題前，**暫停** 廣域 arXiv survey | 邊際↓ |

---

## 14. Round-4 補查 + 文獻收斂（2026-07-20 第四輪）

> 掃描重點：仍摘要級的 W7 文獻、W3 第二 captioner/backbone 候選、2025–26 是否出現「caption agreement conditioning」直接 prior。  
> 結論：**無推翻 R1–3 的發現**；有三項升級；**廣域 survey 正式收斂**。

### 14.1 MusicCaps Annotator Subjectivity（Lee, Doh, Jeong, HCMIR@ISMIR 2023）一手

來源：https://ceur-ws.org/Vol-3528/paper6.pdf（PDF 全文）。

| 發現 | 數字 / 證據 | PromptCC 用法 |
|------|-------------|---------------|
| 不同 annotator **tag 類別偏好極端** | theme：annotator6 **94.4%** vs annotator1/7 **0–1%**；tempo：ann.7/1 **~95%** vs ann.4 **4.2%** | 人類描述 **本來就多視角**（W7 / Meta framing） |
| Caption embedding **按 annotator 分群** | BERT [CLS] UMAP 清晰 annotator clusters；RF 從 caption emb 預測 annotator **F1 0.76** | multi-valid ≠ noise；text-space 差異可來自 **描述風格** |
| Audio embedding **不**按 annotator 分群 | 同實驗 audio→annotator F1 **0.08**（隨機指派） | 排除「每人只標自己愛的音樂」混淆 |
| Annotator-specific 對比模型 **不泛化**到其他 annotator 的 caption | in-domain R@10 高、cross-annotator 崩 | 單一寫法/單一 captioner 的 train–test 分布風險 |
| 混合多 annotator 訓練對 **text→audio** retrieval 更好 | 相對 annotator-specific | 支援「保留多種描述」而非只留一種「正確」caption |
| 作者自陳 | TTM generation 上 caption style 影響 **尚待研究** | 你們 CMOS / PromptCC 正碰這條線 |

**Cite 級建議**：W7 與 Intro multi-validity 段 **A 必 cite**（與 Song Describer 並列）。

**可寫句**：

> Human music captions already encode strong annotator-specific style: in MusicCaps, caption embeddings cluster by annotator more than by audio content (Lee et al., 2023). Multi-validity is therefore not only a failure mode of automatic captioners.

### 14.2 Music Flamingo（arXiv:2511.10289）— W3 第二 captioner

| 要點 | 對 PromptCC |
|------|-------------|
| LALM for music understanding；MF-Skills ~2.1M **長 captions**（均長 ~452 words） | 與 LP-MusicCaps **短 pseudo-caption** 分布差很大 → **cross-captioner 壓力測試** 極佳 |
| Human SongCaps：MF 8.3 vs AF3 較低；LLM-judge correctness/coverage 高 | 更強 teacher 不保證 agreement 與 correctness 對齊 |
| 你們 repo 已有 MF ablation / caption 實驗線 | **P1 cross-captioner**：用 MF 對同 Jamendo clips 抽 K=5 → 重算 s/q → transfer 或輕量 S2 |

**幫助度**：**A 對 W3 實驗設計**；related work 一句即可（非 conditioning prior）。

### 14.3 Resonate / TangoFlux — 正交族，勿混進 PromptCC claim

| 系統 | 機制 | 與 PromptCC |
|------|------|-------------|
| **TangoFlux + CRPO** ([2412.21037](https://arxiv.org/abs/2412.21037)) | Flow matching + **CLAP-ranked preference** 迭代 DPO | **Post-train alignment**；第二 backbone 候選 |
| **Resonate Flow-GRPO** ([2603.11661](https://arxiv.org/abs/2603.11661)) | MeanAudio 架構族 + online GRPO；**LALM AQAScore reward > CLAP** | 解釋為何 paper 裡 **PQ/CLAP conditioning 未必贏**；pretrain condition（你們）vs RL post-train（他們）**互補** |
| 注意 | Resonate cite MeanAudio | camera-ready / double-blind 勿寫成「我們後續」 |

### 14.4 直接 prior 掃描結果

| 搜尋 | 結果 |
|------|------|
| "caption consistency/agreement conditioning" TTM/TTA 2025–26 | **未發現** 與 PromptCC 同構的 text-space multi-caption agreement → AdaLN condition |
| 最接近仍是 CosyAudio（單 caption audio-grounded confidence）+ QA-MDT/Audiobox（audio quality） | novelty 在 claim 收斂後仍可 defend |
| paper.tex / CORRECTNESS plan 自 R3 後 | **無實質變更**（tex 仍 2026-04-28） |

### 14.5 Round-4 文章品質增量

| 文獻 | 相對 R3 | 判決 |
|------|---------|------|
| Lee et al. MusicCaps subjectivity | 摘要→**一手 PDF** | **升 A** |
| Music Flamingo | 新 | **W3 實驗 A / RW B** |
| Resonate / TangoFlux | 加深 | **B 正交定位** |
| 「agreement-as-condition」直接 prior | 仍無 | novelty 未破 |

### 14.6 文獻收斂宣告（給排程 / 主 agent）

**文獻側 W1–W7 彈藥已足夠寫 related work + limitations + claim 降級。**  
繼續每 20 分鐘廣域 arXiv 掃描的 **預期新 A 級文獻 ≈ 0**（除非 reviewer 新問題或 camera-ready 窗口）。

| 應做 | 不應再做 |
|------|----------|
| 把 §13.7 + §14.1 句寫進 `paper.tex` | 無目標的「再掃一輪」 |
| P0：random-bin、size-matched filter、clean NoQ | 把 Resonate 當 PromptCC baseline 重訓 |
| P1：MF cross-captioner q | 再擴 noisy-label classic survey |
| Demo + multi-seed | 用 MeanFlow 論文硬解釋 Stage-2 |

### 14.7 Camera-ready / resubmit cite checklist（精簡）

**必加（正文目前缺或僅 comment）**

1. CosyAudio (2501.16761)  
2. Self-Consistency Wang (2203.11171) + Ding (2607.08065)  
3. BRACE (2512.10403)  
4. Audiobox §5.6 細節（2502.05139）— 已 cite AES，需 **prompting vs filtering** 句  
5. Song Describer (2311.10057) + Lee MusicCaps subjectivity (CEUR 2023)  
6. Manco Augment/Drop/Swap (2409.11498) — **用 R3 正確定位**  

**已有、強化對比句即可**

- QA-MDT、MR-FlowDPO、LP-MusicCaps、Noise2Music、MeanAudio  

**可選**

- Music Flamingo（W3 實驗時）  
- MU-LLaMA、Kong synthetic  
- Resonate / TangoFlux（related，非 baseline 表）  

### 14.8 寫作 patch 增量

| # | 動作 |
|---|------|
| W-23 | Intro/W7 加 Lee 2023：caption emb 按 annotator 分群 + tag 偏好表一句 |
| W-24 | Limitations 並列：人類 multi-validity（Lee + SDD）與 captioner self-agreement（PromptCC） |
| W-25 | Related 一句：Music Flamingo 等 stronger captioners 使 cross-captioner 泛化成為 open question |
| W-26 | 若提 Resonate：明確 **post-train LALM reward ≠ pretrain agreement condition** |

---

## 15. Round-5：外部再掃 + lit×內部實驗對齊 + survey 凍結（2026-07-20）

> 資料源：新 arXiv 掃描 + `docs/experiments/phase_status.md`（2026-07 含 multicap audit）+ R1–4 memo。  
> **核心產出**：文獻預測 vs 你們已跑 phase 的對照表；確認「缺的不是文獻而是實驗落地」。

### 15.1 外部文獻再掃（本輪）

| 候選 | 結果 | 幫助度 |
|------|------|--------|
| 「caption agreement / consistency conditioning TTM」直接 prior | **仍無** | — |
| **SonicVerse** multi-task feature-informed captioning ([2506.15154](https://arxiv.org/abs/2506.15154)) | 改進 **captioner**（key/feature 多任務），不注入 TTM | **C**（captioner 生態；非 PromptCC prior） |
| CLAP-Free / 其他 zero-shot caption eval | 評估側，不改 conditioning 敘事 | **D/C** |
| paper.tex mtime | 仍 **2026-04-28**（寫作尚未吸收 R1–4） | ⚠ 文獻 memo 超前 paper |
| 新 A 級必 cite | **0 篇** | 收斂確認 |

### 15.2 文獻預測 × 內部 phase 實證（高價值）

來源：`docs/experiments/phase_status.md`。

| Reviewer / 文獻主張 | 內部已有證據 | 狀態 | 論文可用度 |
|---------------------|--------------|------|------------|
| **W2**：hard filter 傷 volume（Audiobox、QA-MDT） | **P5 V1** hard filter 117K 退步；**P5 V2** random 117K ≈ V1 → **量是主因** | ✅ 已證 | **可寫**（已有 matched-N random 對照！） |
| **W1/W6**：audio-grounded 替代信號 | **P8 V2** Audiobox-PQ-Q **劣於** mean_sim Q；**P8 V3** CLAP-Q **全面退步** | ✅ 已證 | **可寫**：agreement condition ≠ quality/CLAP condition |
| **W6**：裸 text prefix / 弱 continuous | **P8 V4** `[consistency=X.XX]` 走 text encoder → natural-ref CLAP **崩到 ~0.057** | ✅ 已證 | **可寫**：與 CosyAudio/Audiobox「成功皆離散可學」一致；AdaLN q_embed 路線正確 |
| **W6**：Stage-2 only | **P6 V1** S2-only vs **P6 V2** S1+2；P6 效果受限、S1+2 亦試過 | 半 | 可寫 empirical；勿 overclaim 機制 |
| **Meta**：random one caption | **P7 V1** random > **P7 V2** CLAPBest >≈ **P7 V3** WorstConsensus | ✅ | random 選 caption + Q 是目前最佳配方 |
| **W3**：跨 captioner | **P8-Qwen / P7V1-Qwen / P4V2-Qwen** MC CLAP ~0.06 collapse；**不是**成功泛化 | ⚠ | Limitations：換 Qwen captioner 失敗；**MF 仍 open** |
| **Meta**：用全部 5 captions | P9 / P9.5 multi-cap **INVALID**（cache 錯配，CORRECTNESS plan） | ❌ 禁 cite | 必須 clean multicap rerun |
| **W1**：random-bin control | **未跑** | ❌ 缺口 | 機制仍欠一刀 |
| **W5**：multi-seed / demo | **未見完整 multi-seed 表 / public demo** | ❌ 缺口 | 同儕慣例 |

**關鍵修正（相對純文獻 memo）**：

1. **Size-matched filter 你們其實已有 P5 V1 vs V2**（hard vs random 同 N）— R1/Meta 批評 paper 裡 hard-filter 不公平，但 **內部已做 random matched-N**；resubmit 應 **把 P5 V2 寫進主文/補充**，不必從頭發明。  
2. **CLAP/PQ conditioning 已在內部輸給 PromptCC-style mean_sim Q** — 文獻要求的 audio-grounded 對照，實驗表已有。  
3. **W3 不是空白**：Qwen 軸已證明「換 captioner 可崩潰」；成功軸（MF）才是剩餘實驗。

### 15.3 文章品質總覽儀表板（R1–5 終局）

#### 外部文獻（對 paper 幫助）

| 等級 | 文獻 | 寫進 paper？ |
|------|------|--------------|
| **A 必 cite** | CosyAudio, QA-MDT, MR-FlowDPO, Audiobox(+§5.6), Self-Consistency, Ding 2026, BRACE, LP-MC, Song Describer, Lee MusicCaps subj., MeanAudio | 半數已 cite；**CosyAudio/Ding/BRACE/Lee/SDD 仍缺** |
| **B 應 cite** | Manco A/D/S（正確定位）, Noise2Music, MU-LLaMA, MusicLM/MuLan, Resonate, TangoFlux, Music Flamingo（實驗時） | 部分 comment only |
| **C** | Kong synthetic, EzAudio, SonicVerse, PE-AV | 可選 |
| **D / 勿 hard-cite** | 經典 co-teaching 長表；MeanFlow 撐 Stage-2；「diversity>curation」誤讀 Manco | — |

#### 內部實驗（對 claim 幫助）

| 等級 | 證據 | 注意 |
|------|------|------|
| **A 可用** | P5 filter vs random N；P7 V1 best；P8 NoQ ablation；P8 V2/V3 PQ/CLAP 輸；P8 V4 prefix 崩；CMOS | 確認 eval 協議與 paper Table 一致 |
| **禁引用** | P9/P9.5 multicap 0.065 等 | cache bug |
| **仍缺** | random-bin；clean multicap；MF cross-captioner；multi-seed；demo | P0/P1 |

### 15.4 品質審核：本 memo 自身（R5）

| 維度 | 評分 | 說明 |
|------|------|------|
| 覆蓋 reviewer W1–W7 | **優** | 文獻+實驗雙軌 |
| 一手深度 | **優** | CosyAudio/QA-MDT/Audiobox/Manco/SDD/Lee PDF |
| 與內部 phase 對齊 | **R5 補齊** | 先前 memo 低估 P5 V2 / P8 V2–V4 已有性 |
| 可執行性 | **優** | 但 paper.tex **未更新**＝最大執行缺口 |
| 邊際 survey 收益 | **近零** | R5 外部 0 新 A |

### 15.5 凍結建議（給排程任務）

| 動作 | 建議 |
|------|------|
| 廣域「再調查文章」每 20 min | **應暫停或降頻**（如 1d）；R5 確認外部無新 A |
| 本檔 `PromptCC_Literature_...md` | **凍結為 v1 定稿**；僅在 paper 改寫 / 新實驗結果時 append |
| 人力轉向 | (1) paper.tex 吸收 §13.7+§14.1+§15.2 (2) 把 **P5 V2** 寫進 hard-filter 公平對照 (3) random-bin + clean multicap (4) demo |

### 15.6 寫作 patch 增量（對齊內部結果）

| # | 動作 |
|---|------|
| W-27 | Hard-filter 段：**報告 P5 V1 vs V2 matched-N**（文獻 Audiobox/QA-MDT + 自家 random control） |
| W-28 | Related/實驗：CLAP-Q / PQ-Q **已劣於** mean_sim-Q（P8 V2/V3）— 直接回 R1「缺 audio-grounded 對照」 |
| W-29 | w/o quantize：除 paper 表外，可注 P8 V4 text-prefix 路線亦崩（若空間允許） |
| W-30 | Limitations：Qwen captioner collapse（W3 部分誠實）；MF 未測；multicap 待 clean rerun |
| W-31 | **禁止**任何 0.0650 / P9.5 multicap 敘事進入 paper |

### 15.7 一句終局

**文獻品質審核完成且對論文有幫助；不足處已大範圍補查。**  
剩餘 gap **不是「沒讀到某篇 paper」**，而是：

1. **paper.tex 未吸收** 已整理的 cite + claim 降級 + P5 V2 公平 filter；  
2. **random-bin / clean multicap / demo / multi-seed / MF** 仍欠跑。

外部 literature survey：**建議凍結**。
