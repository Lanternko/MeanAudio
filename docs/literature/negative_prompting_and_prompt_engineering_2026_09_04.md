# Negative prompting 與 prompt engineering 的文獻定位

日期：2026-09-04
用途：為 `negprompt_cfg3_content_interaction_2026_09_03.md` 的結果找文獻座標，
並判斷哪些部分是文獻空白（＝論文貢獻點）、哪些部分文獻已有更好的解釋。

> **引用可信度標記**：✅ 已取原文逐字核對數字；🟡 取自論文摘要／官方文件敘述，數字未逐字核對；
> ⚪ 僅為敘述性主張，本文不引數字。

---

## 0. 三句話總結

1. **negative prompting 在影像端是標準做法、在音訊端是官方 API 一級參數，但整個 TTA/TTM 文獻只有
   一張表做過量化 ablation**（QA-MDT Table A.3）。那張表的效果量是 p-MOS **+0.036**；
   我們量到的是 ΔPQ **+1.067**。效果量差兩個數量級，這件事本身需要解釋，而我們有解釋。
2. **「極性不被遵守」不是我們的異常，是文獻已知現象**，而且有四條互相獨立的機制文獻
   （embedding 未解耦、對比模型不懂否定、score space 幾何重疊、少步取樣下分支發散）。
   我們的 `loud` 反轉與 `reversed` 高增益都能對上其中具體一條。
3. **沒有任何一篇 TTA/TTM 論文做過 negative prompt 的內容階梯**（irrelevant / neutral /
   reversed / polarity control）。9 格內容階梯 + 三層拆解是文獻空白，這是最強的貢獻點；
   反過來說，把它寫成「negative prompting 有用」就是把貢獻寫小了。

---

## Q1：有沒有文獻支持 negative prompting 有用？

### 1.1 理論起源：negation 是 CFG 的合成運算子，不是提示詞技巧

Composable Diffusion（Liu et al., ECCV 2022, arXiv:2206.01714）把 diffusion model 用 EBM 觀點
重新詮釋，形式化出 **Conjunction (AND) 與 Negation (NOT) 兩個推論期運算子**，無需訓練。⚪
今天所有 UI 上的 "negative prompt" 欄位都是這個 NOT 運算子的工程化：把 CFG 裡的 unconditional
分支換成 `y_neg`。

我們的實作正是這個形式（`meanaudio/model/networks.py:593-602`、`eval.py:43-45`）：

```
cfg ≥ 1.0:  D = cfg · D(x, y) + (1 − cfg) · D(x, y_neg)
```

cfg 3.0 時 `y_neg` 的權重是 **−2**。這一點在解讀「任何文字」那 34% 時很重要：被減掉的不是
「低品質」這個概念，而是 `D(x, y_neg)` 這整個預測向量；只要 `y_neg` 不等於 stored null，
差向量就非零，模型就會沿著 `D(x,y) − D(x,y_neg)` 外推。**槽位裡放貓照片會有 +0.357，
在這個公式下是預期行為而非意外。**

### 1.2 影像端：普及但直到 2024 才有系統性研究

| 文獻 | 貢獻 | 對我們的意義 |
|---|---|---|
| Understanding the Impact of Negative Prompts（ECCV 2024, arXiv:2406.02965） | 自稱**第一篇**系統研究 negative prompt 機制 | 見 §2.3，直接解釋 `loud` 反轉 |
| Perp-Neg（arXiv:2304.04968） | 當 negative 與 main prompt 在 score space 重疊時樸素做法失效 | 我們量到 `reversed` vs `fidelity` 的 T5 cosine **0.814**，正是重疊 regime |
| NegToMe（arXiv:2412.01339, CVPR 2025） | 純文字的 adversarial guidance 不足以捕捉複雜概念，改用視覺特徵 | 支持「文字槽位的表達力有限」這條讀法 |

「直到 2024 才有第一篇機制研究」這件事值得寫進 related work：**negative prompting 的普及程度
遠超過它被理解的程度**，我們的內容階梯正好補在這個縫上。

### 1.3 音訊／音樂端：官方 default 多，量化 ablation 只有一張表

**工程慣例（皆為官方文件，非論文結論）**：

- **AudioLDM 2**（diffusers 官方文件）：明文建議 `negative_prompt="Low quality"`，理由是
  「可顯著改善產生的波形品質」。🟡 無 ablation。
- **Stable Audio Open**（diffusers 官方文件）：`negative_prompt` 是 pipeline 一級參數；
  `guidance_scale` 預設 **7.0**；文件明寫「更高的 guidance 讓音訊更貼合文字，
  **代價是音質下降**」。🟡 ← 這句話與我們 cfg sweep 的方向完全一致（CLAP 隨 cfg 升、
  PQ 在 3.0 之後翻轉），可以直接引用來說明我們不是特例。
- **Google Lyria**（Vertex AI 官方 prompt guide）：`negative_prompt` 是 API 參數，
  官方建議用法是排除 `vocals, excessive cymbal crashes, distorted guitar`。🟡
  注意這是**內容排除**（不要人聲），不是**保真度排除**（不要低品質）——與我們的用法不同類。

**唯一的量化 ablation：QA-MDT（arXiv:2405.15863, AAAI 2025）Table A.3**（MTT-FS）✅

| System | FAD ↓ | KL ↓ | p-MOS ↑ |
|---|---:|---:|---:|
| MDT（無品質引導） | 5.757 | 3.837 | 3.796 |
| MDT + Negative prompt（`y_neg` = "low quality"） | 5.641 | 3.461 | **3.832** |
| QA-MDT（訓練期品質前綴 + quality token） | **5.200** | **3.214** | **4.051** |

讀法：
- negative prompt **確實有效**（p-MOS +0.036、FAD −0.116），這是 TTM 文獻中唯一的直接證據。
- 但**訓練期注入品質資訊是它的 7 倍**（p-MOS +0.255）。
- QA-MDT 的結論句：*"any form of quality guidance improves the model's generative performance"*。✅

### 1.4 我們的效果量在文獻裡是異常值 —— 而這是可以解釋的

QA-MDT 對 negative prompt 效果小給了兩個理由，第一個直接指向我們的情況：✅（逐字）

> "previous attempts to improve quality relied on the **rare instances of "low quality" in the dataset**.
> This necessitated the careful design of numerous negative prompts to avoid generating low-quality results.
> Furthermore, the **text embedding of "low quality" might not be well disentangled during training**,
> leading to an suboptimal results."

**在我們的訓練語料裡，"low quality" 一點都不 rare**：

| 語料 | 保真度用語密度 | 出處 |
|---|---:|---|
| LP-MC boilerplate prefix（P8 系列） | ~45% | `phase_status.md:739`、`qwen_collapse_root_cause_2026_05_08.md:367` |
| c2p0 訓練 caption 提到 quality | **82.8%** | `fidelity_stripped_caption_arm_2026_08_30.md` |
| fulltrack 訓練 caption 提到 quality | 7.3% | 同上 |
| MusicCaps 測試提示詞含低保真語言 | 37% | 同上 |

於是形成一個乾淨的機制假說，且**已經有現成的對照組可以檢定**：

> **假說 N1**：negative prompt 的增益大小，取決於 `y_neg` 的詞彙是否命中訓練 caption 分布中
> 一個高密度、被模型學成銳利軸的模式。c2p0（82.8%）拿到 +1.067；
> 依此預測 **fulltrack（7.3%）的 negative prompt 增益應該顯著較小**，
> 而 **fidelity-stripped arm（設計上趨近 0%）應該幾乎沒有增益**。

> **2026-09-04 已檢定完成 → [negprompt_n1_density_2026_09_04.md](../experiments/results/phase8/negprompt_n1_density_2026_09_04.md)**
> **字面版 N1 被推翻**（負向詞彙自己的密度 ρ=−0.771，方向相反），
> **精煉版 N1′ 成立**：預測增益的是「語料談不談保真度」（`quality_rate` r=+0.965, R²=0.931, n=6 語料），
> 與極性無關。這同時修正了 QA-MDT 的 "rare instances" 歸因。

這個假說最漂亮的地方：`negprompt_reeval_cfg3.0/` 的 13-arm sweep 裡**已經有 fulltrack**，
第一個檢定不需要任何新的 GPU 時間，只要把 13 arm 的 ΔPQ 對各 arm 訓練語料的保真度詞彙密度
做散點圖。若成立，這條就從「我們的 negative prompt 很有效」升級成
**「negative prompt 的效果量由訓練 caption 分布決定」**——那是一個一般性的機制結論，
而且直接反駁「negative prompt 是通用免費午餐」的天真讀法。

---

## Q2：文獻用哪種 prompt？哪種最好？副作用？

### 2.1 文獻慣用的字串

| 來源 | 字串 |
|---|---|
| AudioLDM 2 官方 | `Low quality` |
| QA-MDT Table A.3 | `low quality` |
| QA-MDT 訓練期前綴 | `low quality` / `medium quality` / `high quality`（依 p-MOS 分佈：`s < μ−2σ` / `μ−σ ≤ s ≤ μ+σ` / `s > μ+2σ`）✅ |
| TTM 慣例（文獻與工具鏈常見） | `noise, distortion, low quality, static, hum, hiss, clipping, muffled, amateur recording` 🟡 |
| Lyria 官方（內容型） | `vocals, excessive cymbal crashes, distorted guitar` |
| **我們的 `fidelity`** | `low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi` |

我們的 8 詞版本幾乎就是文獻慣用集合的並集，**沒有 novelty，也不需要 novelty**——
這反而是好事：主結果用的是社群標準字串，不是我們調出來的魔法咒語。

**沒有任何文獻報告過「哪種 negative prompt 最好」的對照實驗。** 短版 vs 長版、
fidelity 詞彙 vs 無關文字、極性正確 vs 極性相反，這些格子在 TTA/TTM 文獻裡都是空的。

### 2.2 為什麼極性不被遵守：四條獨立文獻，各自對應我們的一個觀察

| 機制文獻 | 主張 | 對應我們哪個觀察 |
|---|---|---|
| **QA-MDT** ✅ | "low quality" 的 text embedding 在訓練期**可能沒有被 disentangle** | `reversed`（高品質詞）拿到 +0.722：若保真度軸未解耦，正反詞落在同一子空間 → 與我們量到的 T5 cosine 0.814 一致 |
| **NegBench / VLMs Do Not Understand Negation**（CVPR 2025, arXiv:2501.09425）✅ | CLIP 類對比模型在否定查詢上**接近隨機**；COCO 否定檢索 recall@5 掉 **6.8–7.7%**；用千萬級合成否定 caption 微調可 +10% recall / +28% MCQ | CLAP 是同族對比模型，同時是我們的 **conditioning encoder 之一與主要評估指標**。「模型不懂否定」在我們這裡是雙重繼承 |
| **Perp-Neg**（arXiv:2304.04968）⚪ | negative 與 main prompt **重疊時**樸素做法失效；是 score space 的幾何問題，提出取垂直分量 | `reversed` 與 `fidelity` cosine 0.814、且兩者都與 positive caption 重疊（82.8% caption 談 quality）→ 我們正坐在 Perp-Neg 描述的最壞情況上 |
| **ECCV 2024 arXiv:2406.02965** 🟡 | **Delayed effect**：negative 的作用發生在 positive 已經生成該內容之後（名詞移除的 critical step 約第 5 步、形容詞約第 10 步進入平台）；**Reverse activation**：在**早期**施加 negative 反而會**生成**該概念 | **直接解釋 `loud` 的 ΔRMS +1.10 dB 反轉**：要求「不要大聲」得到更大聲，正是 reverse activation 的定義 |

**額外一條，且對我們特別關鍵——少步取樣**：

- **NAG: Normalized Attention Guidance**（arXiv:2505.21179, NeurIPS 2025）🟡：
  *"CFG works well in standard settings, [but] fails under aggressive sampling step compression
  due to divergent predictions between positive and negative branches"* ——
  少步 regime 下兩個分支在早期就劇烈發散，結果是 **artifact 而非受控引導**。
- **VSF: Value Sign Flip**（arXiv:2508.10931）🟡：同一個問題的另一個解法，改在 attention value 上翻符號。

MeanAudio 是 MeanFlow **25 步**（甚至 1 步）模型。這條文獻是說：
**我們觀察到的「文字內容有影響但不按極性」，在少步 CFG 下是被預期的失效模式**，
不是我們的模型壞掉。這句話應該寫進論文的 limitation／機制段。

### 2.3 副作用：文獻與我們的資料互相印證

| 副作用 | 文獻 | 我們的量測 |
|---|---|---|
| **服從度／可控性下降** | CFG 以 diversity 換 fidelity（recall 下降是標準觀察）⚪；NegToMe 專門處理 negative guidance 的 diversity 損失 ⚪ | clean−lofi gap 從 +0.335 掉到 **+0.185**（縮 45%）：增益最大的設定最不服從「低保真」指令 |
| **過飽和／artifact** | **APG**（arXiv:2410.02416, ICLR 2025）✅ 診斷：CFG 更新項的**平行分量造成過飽和、正交分量提升品質**；只保留正交分量可在高 guidance 下消除過飽和，FID −10~50%、recall +20~50%、saturation −20~60% 🟡。CFG++ 走 manifold-constrained 插值 ⚪ | crest_min 崩到 1.76~2.75；`fidelity_loud` 加 5 個抑制詞救不回來（crest_min 反而更差 2.46） |
| **整體位移而非局部移除** | ECCV 2024：太早施加 negative「會顯著扭曲擴散過程、可能改變背景」🟡 | 頻譜重心位移 +72 ~ +706 Hz；`loud` 只有 +3 Hz（唯一只動響度不動音色的格） |
| **音質 vs 對齊的取捨** | Stable Audio 官方：guidance 越高越貼文字、音質越差 🟡 | cfg 3.0 是 PQ 最佳點；CLAP 仍隨 cfg 單調上升（0.2194 → 0.2392） |

**APG 是最值得直接試的一條**：它是 plug-and-play、不需重訓、與所有 sampler 相容，
而且它診斷的病（高 guidance 過飽和）與我們量到的 crest 崩塌是同構的。

> **2026-09-04 修訂（見 §7b④）**：交叉核對後找到兩條更貼近音訊的診斷。
> **Angle Domain Guidance**（ICML 2025, arXiv:2506.11039）把高 guidance 的失真歸因於
> **latent 樣本範數被放大**——在音訊就是波形幅度飽和，比 APG 的平行／正交分解更直接對應
> crest 崩塌。**LF-CFG**（arXiv:2506.21452）走頻域。建議嘗試順序改為 **ADG → APG → LF-CFG**。

---

## Q3：TTM 中 prompt engineering 的定位與可提升幅度

### 3.1 三層槓桿

| 層 | 做法 | 代表文獻 | 效果量 |
|---|---|---|---|
| **A. 訓練期 caption 工程** | 重寫／合成 caption、注入品質前綴 | AF-AudioSet（arXiv:2406.15487）、Sound-VECaps（arXiv:2407.04416）、QA-MDT | QA-MDT：FAD 5.757 → **5.200**、p-MOS 3.796 → **4.051** ✅ |
| **B. 推論期 positive prompt 改寫** | LLM 改寫使用者短提示詞為「audionese」 | Open Prompt Challenge（Meta, ICASSP 2024, arXiv:2311.00897）、In-Context Prompt Editing（arXiv:2311.00895） | 見 3.2 ✅ |
| **C. 推論期 negative / guidance** | negative prompt、guidance scale、APG/NAG | QA-MDT Table A.3、本專案 | p-MOS +0.036（QA-MDT）；ΔPQ +1.067（本專案） |

### 3.2 推論期 prompt 改寫能提升多少（有數字的兩篇）✅

**Open Prompt Challenge**（Meta；AudioLDM；CLAP 分數）—— Table 3，FLAN-T5-large 欄：

| 設定 | CLAP |
|---|---:|
| 使用者原始開放提示詞 | 0.0556 |
| Instruct（LLM 改寫） | 0.0701 |
| Instruct + 0-shot | 0.0763 |
| **Instruct + CLAP feedback** | **0.0809** |

相對提升 **+45%**。主觀／客觀評分（User 開放提示詞）：原始 OBJ 1.53 → Instruct **3.63**。
論文定義 **"audionese" = 能讓 TTA 模型產生最佳輸出的文字分布**，並指出開放提示詞與
專家提示詞在 CLAP 與資訊密度上有明顯落差、且資訊密度與 CLAP 單調正相關。

**In-Context Prompt Editing**（Meta；AudioLDM；1,525 筆真實使用者提示詞）—— Table 1 最佳列
（in-domain, K=100, closest exemplars），相對未改寫的使用者提示詞：

| 指標 | Δ |
|---|---:|
| ΔCLAP | **+0.047** |
| ΔFAD | **+3.068**（改善） |
| Δ多樣性比 | +0.594 |
| 主觀 SUB | 3.58 → **3.83** |
| 客觀 OBJ | 1.54 → **2.68** |

**校準句（可直接寫進論文）**：TTA 文獻中「純提示詞側介入」的典型效果量是
**ΔCLAP +0.03 ~ +0.05**。我們的 negative prompt 拿到 **ΔCLAP +0.0404**，
**正好落在這個區間內** —— 語意對齊的部分沒有超出文獻常態。
真正異常的是感知品質軸：**ΔPQ +1.02**，而文獻中最接近的可比數字是 QA-MDT 的 p-MOS +0.036。

### 3.3 prompt 是高槓桿但高變異的介面

**Evaluating Semantic Fragility in Text-to-Audio Generation Systems Under Controlled Prompt
Perturbations**（arXiv:2603.13824）✅：對 MusicGen 施加**語意等價**的擾動
（最小詞彙替換 MLS／強度位移 IS／結構改寫 SR），輸出之間的 CLAP cosine 只有
**0.59（small 模型，詞彙替換）~ 0.77（large 模型）**。模型越大越穩健，但都遠低於
「強等價」門檻。

意義：**改一個同義詞就能讓輸出實質改變**。這對我們有兩個直接後果：
1. 任何 negative prompt 的比較都必須**同 checkpoint、同 seed、逐檔配對**——我們已經是這樣做的。
2. 「短版 vs 長版差 0.096」這種量級的差異，在這個變異底下要非常小心。我們用訓練 seed 底線
   0.052 / 推論 seed 底線 0.142 來守門是對的做法，且比這篇文獻嚴謹。

### 3.4 prompt engineering 的上界：訓練期偏好對齊

- **Tango 2**（arXiv:2404.09956, ACM MM 2024）🟡：用合成偏好資料集 Audio-alpaca 做 DPO，
  CLAP 達 0.57，主觀 overall 3.99 / relevance 4.07，勝過 Tango 與 AudioLDM 2。
  其「負樣本」構造方式值得注意：**擾動描述以移除／打亂概念**＋**CLAP 低於門檻的對抗過濾**。
- **Improving Text-to-Music Generation with Human Preference Rewards**（arXiv:2606.21670）🟡：
  偏好獎勵模型一致優於純 CFG 基線（數字未逐字核對）。

**定位語句**：negative prompting 是「推論期、零訓練成本」的品質對齊近似；
DPO / 偏好獎勵是同一目標的訓練期上界。我們的貢獻不是宣稱前者比後者好，
而是**把前者的作用機制拆開**——這是 DPO 那條線沒有做的事。

---

## 4. 對本專案的可執行建議（按性價比排序）

1. **【零 GPU 成本，先做】保真度詞彙密度 × negative 增益的交叉檢定。**
   `negprompt_reeval_cfg3.0/` 的 13 arm 已經在檔，把各 arm 的 ΔPQ 對其訓練語料的
   quality-mention density（c2p0 82.8% / fulltrack 7.3% / 其他待統計）作圖。
   若呈單調關係，假說 N1 成立，QA-MDT 的 "rare instances" 論點就從別人的解釋
   變成我們的**因果證據**。這是整份文獻調查裡最高價值的一步。
2. **Step-window ablation：negative 只在中後段施加。**
   ECCV 2024 的 reverse activation 預測「早期施加會反向」。我們的 `loud` 反轉是現成的
   操作檢定訊號——若把 negative 限制在後 60% 的步數後 `loud` 的 ΔRMS 由 +1.10 轉負，
   reverse activation 就在音訊域被複現了一次。25 步夠切窗口。
3. **試 ADG／APG 取代樸素 CFG 外推**（arXiv:2506.11039 / 2410.02416）。針對 crest 崩塌，
   plug-and-play、不需重訓。**先試 ADG**（範數放大＝波形飽和，診斷最貼切），再試 APG。
   若 crest 回升而 ΔPQ 保持，`fidelity_loud` 那條死路就被繞開了。
4. **少步 regime 的 caveat 必須寫進論文。** NAG/VSF 已經說明 CFG 的 negative 分支在
   aggressive step compression 下會發散。我們是 MeanFlow few-step，這是 limitation 也是
   「為什麼極性不被遵守」的最強單一解釋。

### 論文寫法建議

- ❌ 不要寫 "negative prompting improves TTM quality" —— 文獻已知，且我們會被問「那 QA-MDT 呢」。
- ✅ 寫「negative slot 的貢獻可拆成 **any-text 34% / fidelity-domain vocabulary 34% /
  correct polarity 32%** 三份」，並指出**沒有任何前作做過這個拆解**。
- ✅ 用 QA-MDT 的 "rare instances" + "not well disentangled" 兩句話解釋我們的效果量
  為何比文獻大兩個數量級，並用 arm × 詞彙密度的交叉檢定支撐。
- ✅ 極性那 32% 必須帶著 T5 cosine 0.814 的 caveat 一起寫（可能是 embedding 距離而非語意極性），
  並引 NegBench 說明對比模型不懂否定。

---

## 5. 文獻缺口（＝我們的貢獻空間）

> 2026-09-04 第二輪修訂：§7 的交叉核對找到兩篇音訊域 negative guidance 論文，
> 缺口 1 與 3 已據此收窄。**收窄後仍然成立，但宣稱必須改寫**。

1. **TTM 沒有任何「文字 negative prompt 的內容 ablation」。** 只有 QA-MDT 一張三列表。
   ⚠️ 收窄：MM-Sonate（arXiv:2601.01568）對**聲學** negative 分支做過 8 種策略消融
   （zero vector vs 六級高斯噪聲 vs 自然白噪聲），所以不能寫「沒有人做過 negative 分支消融」，
   只能寫「**沒有人做過文字 negative prompt 的內容階梯**」。
2. **沒有人把 negative prompt 的效果量與訓練 caption 分布連起來。** QA-MDT 提了 "rare instances"
   當作解釋，但沒有做對照實驗。（此條未被任何新文獻推翻，仍是最強的貢獻點。）
3. **沒有 training-free、以保真度為目標的音訊 negative guidance 研究。**
   ⚠️ 收窄：Sony 的 Negative Audio Guidance（arXiv:2506.20995, ECCV 2026）已在 V2A 做過音訊域
   negative guidance，但 (a) 目標是**避免重複聲事件**不是提升保真度，(b) 需要 **finetune 一個
   guidance model**不是 training-free。原本「音訊域沒有 negative guidance 研究」的說法是錯的，
   必須改成上面這句。
4. **沒有 flow matching / MeanFlow 少步音訊模型的 negative guidance 研究。**
   NAG/VSF 都在影像／影片。（未被推翻。）
5. **沒有人在音訊域檢驗 reverse activation。** ECCV 2024 只做影像。（未被推翻。）

---

## 6. 引用清單

| 主題 | 文獻 | 連結 |
|---|---|---|
| negation 運算子起源 | Liu et al., Compositional Visual Generation with Composable Diffusion Models, ECCV 2022 | arXiv:2206.01714 |
| negative prompt 機制 | Ban et al., Understanding the Impact of Negative Prompts: When and How Do They Take Effect?, ECCV 2024 | arXiv:2406.02965 |
| negative 與 main prompt 重疊 | Armandpour et al., Re-imagine the Negative Prompt Algorithm (Perp-Neg) | arXiv:2304.04968 |
| 少步模型的 negative guidance | Chen et al., Normalized Attention Guidance, NeurIPS 2025 | arXiv:2505.21179 |
| 少步模型的 negative guidance | Value Sign Flip (VSF), 2025 | arXiv:2508.10931 |
| 高 guidance 過飽和 | Sadat et al., Eliminating Oversaturation and Artifacts of High Guidance Scales (APG), ICLR 2025 | arXiv:2410.02416 |
| 文字 negative 的表達力上限 | Singh et al., Negative Token Merging, CVPR 2025 | arXiv:2412.01339 |
| 對比模型不懂否定 | Alhamoud et al., Vision-Language Models Do Not Understand Negation, CVPR 2025 | arXiv:2501.09425 |
| **TTM 品質注入 + negative ablation** | Li et al., QA-MDT: Quality-aware Masked Diffusion Transformer, AAAI 2025 | arXiv:2405.15863 |
| TTA prompt 改寫 | Chang et al., On the Open Prompt Challenge in Conditional Audio Generation, ICASSP 2024 | arXiv:2311.00897 |
| TTA prompt 改寫 | Chang et al., In-Context Prompt Editing for Conditional Audio Generation | arXiv:2311.00895 |
| 合成 caption 提升 TTA | Kong et al., Improving Text-to-Audio Models with Synthetic Captions (AF-AudioSet) | arXiv:2406.15487 |
| 合成 caption 提升 TTA | Yuan et al., Sound-VECaps | arXiv:2407.04416 |
| prompt 擾動敏感度 | Evaluating Semantic Fragility in TTA under Controlled Prompt Perturbations, 2026 | arXiv:2603.13824 |
| 訓練期偏好對齊 | Majumder et al., Tango 2, ACM MM 2024 | arXiv:2404.09956 |
| 訓練期偏好對齊 | Improving Text-to-Music Generation with Human Preference Rewards, 2026 | arXiv:2606.21670 |

官方文件：AudioLDM 2 / Stable Audio Open（HuggingFace diffusers pipeline docs）、
Google Lyria music generation prompt guide（Vertex AI）。

---

## 7. NotebookLM 文獻地圖交叉核對（2026-09-04 第二輪）

來源：使用者提供的 NotebookLM 整理筆記 + 20 篇推薦清單。逐條查證結果如下。

### 7a. 需要修正的四處

| # | NotebookLM 的說法 | 實際 | 影響 |
|---|---|---|---|
| 1 | 給出雙權重 CFG 公式 `ε∅ + γ_pos(ε_c+ − ε∅) − γ_neg(ε_c− − ε∅)`，並說「在 CFG 3.0 的實作下…」 | **這不是我們的實作。** `networks.py:598-602` 是單權重替換式 `cfg·D(x,y) + (1−cfg)·D(x,y_neg)`：沒有獨立的 γ_neg，也沒有保留 ε∅ 項——負分支**取代**了 null 分支 | 把那條公式寫進論文會描述錯自己的系統。雙權重式屬 Composable／Perp-Neg 家族 |
| 2 | 第 14 條稱 Sony 的 V2A 論文是「音訊領域首篇引入 NAG 演算法」 | **兩個 NAG 被混為一談。** Normalized Attention Guidance（arXiv:2505.21179，影像／影片，training-free）與 Negative **Audio** Guidance（arXiv:2506.20995，Sony，V2A，需 finetune guidance model）是無關的兩篇 | 引用時務必分開，縮寫撞名 |
| 3 | Dynamic Negative Guidance 標為「NeurIPS Workshop 2024」 | **ICLR 2025 poster**（arXiv:2410.14398） | 送審引用錯誤 |
| 4 | 「5–15 個 Token 為黃金長度」，引 Stable Diffusion 教學文 | 部落格層級主張，且**與我們自己的資料相反**：長版 8 詞 +1.067 > 短版 2 詞 +0.971。FIGMA 量到的真正門檻是 **40–50 token**，兩版都遠低於它 | 不要引。正確讀法見 7b-③ |

另有一條**未能查證**：VL-DNP 的「ASR 0.958 → 0.084」在該論文摘要中找不到，未取得全文核對 → **標記為未驗證，先不要引用這個數字**。Instruct-MusicGen 的 ISMIR 2025 標註本輪未查證。

### 7b. 四條有價值的新增（皆已查證為真實論文）

**① Score-Aware Training for Text-to-Music Generation（arXiv:2606.07387, ICME 2026 ATTM Challenge）
—— 這是我們架構的孿生兄弟，本輪最重要的發現。** ✅

- **FluxAudio DiT 450M**（hidden 896 / depth 12）＋ **FLAN-T5（cross-attention）＋ CLAP（adaptive norm）雙文字條件**
  ＋ **MTG-Jamendo** —— 與 MeanAudio Stage 1 幾乎同一個設定。
- **Two-stage caption**：Stage 1 用 Qwen2-Audio ／ **Music Flamingo** 的資訊密集 caption（tempo/key/
  time signature/chord）；Stage 2 用 LLM 改寫剝除樂理細節、只留 genre/instrumentation/mood，
  **明確目的是弭平「冗長訓練 caption vs 簡短推論提示詞」的分布落差**。
  → 這正是我們 caption 2.0 與 fidelity-stripped arm 在處理的同一個問題，而且他們的解法是
  **兩階段用不同 caption 分布**，我們沒試過這個切法。
- **CLAP score 當監督訊號**：<0.20 丟棄、分層每檔留 6 段、並用 Beta 分布讓低分段集中在
  高噪聲 timestep（粗結構有用、細節不受污染）。→ 這是「品質信號」的第三種用法，
  與我們的 q_embed（條件）、QA-MDT 的 text prefix（提示詞）並列，值得寫進 related work 的對照。
- **REPA aux loss**：CLAP 分支 +0.018 CLAP、FAD 0.2856 → 0.2767（α=2.0, λ=0.2, 20K iter ablation）。
- 主結果：CLAP 0.295 / FAD 0.495 / MOS 3.119，客觀評比第 2。
- **可以提出的批評**：他們用 CLAP 過濾訓練資料，又用 CLAP 當主要評估指標 ——
  正是教授 2026-03-27 定下的 data leakage 禁區。這給了我們一個乾淨的 related-work 論點。

**② MM-Sonate（arXiv:2601.01568, Kuaishou）—— 「負分支該放什麼」的最佳外部對照。** ✅

明確指出：**zero vector 不是中性的**，因為在他們的訓練策略裡 zero vector 有一個
**已被訓練的語意**（代表 T2VA 任務），拿它當 CFG 負條件只會叫模型遠離「隨機音色生成」模式，
而不是遠離低品質音訊。因此改用 **Noise-based Negative Speaker Embedding**，並消融了 8 種策略
（zero-vector baseline、六級高斯白噪聲、自然採集白噪聲）。

→ 這與我們兩個既有發現直接對應：(a) `--negative_prompt ""` 會 fall back 到 stored null 的 trap；
(b) 「槽位有任何文字」就值 +0.360。**同一個機制在別人的系統上被獨立發現並消融過**，
可以引用來支撐我們對 34% 那一層的解釋，而不必只靠自己的資料。

**③ FIGMA（arXiv:2606.06615）—— token collapse 的定量門檻。** ✅

CLAP 類對比模型的長 caption 會塌陷成粗表徵，**第 40–50 個 token 之後幾乎不貢獻檢索效能**。
提出 frame-level ＋ token-wise 雙層對齊，並發布 FGMCaps（380K 訓練 / 10K 測試，標註 tempo/key/
chord/beat），相對提升最高 73.3%。

正確讀法（與 NotebookLM 的「5–15 token 黃金長度」相反）：
我們的長版 8 詞約 15 token、短版 2 詞約 4 token，**兩者都遠低於 40–50 的塌陷門檻**。
所以 token collapse **不能**解釋長短版的 0.096 差異，但它給出一個可檢定的預測：
**把 negative prompt 繼續加長到 40 token 以上，增益應該進入平台期。**
同時它也是我們 CLAP 評估的 caveat 來源（長 caption 的細節在 CLAP 裡本來就被丟掉）。

**④ 過飽和的第二、第三種解法（比 APG 更貼近 crest 崩塌）** ✅

| 論文 | 診斷 | 為什麼對音訊更貼切 |
|---|---|---|
| **Angle Domain Guidance**（ICML 2025, arXiv:2506.11039） | 高 guidance 下失真來自 **latent 空間的樣本範數被放大**；ADG 約束幅度變化、只優化角度對齊 | 「範數放大」在音訊就是**波形幅度飽和**——比 APG 的平行／正交分解更直接對應 crest_min 崩到 1.76 |
| **LF-CFG**（arXiv:2506.21452） | 低頻訊號中的冗餘資訊累積是過飽和主因，用自適應遮罩降權 | 頻域視角天然適合音訊；可與我們量到的頻譜重心位移 +72~+706 Hz 對照 |
| **Frequency-domain guidance**（arXiv:2506.19713） | 在頻域施加 guidance，讓低 CFG 也能高保真 | 同上；且我們的最佳點在 cfg 3.0 這個中低區間 |

**建議的嘗試順序改為：ADG → APG → LF-CFG**（原本只列 APG）。三者都是 training-free、採樣端介入。

### 7c. 其他已核對為真、但優先度較低

- **MusicRFM**（ICLR 2026, arXiv:2510.19127）✅：RFM 探針在 MusicGen hidden states 找出「概念方向向量」，
  推論時注入，目標音符準確率 **0.23 → 0.82**，prompt 對齊度波動 **±0.02** 內。
  這是 negative prompt 的**替代路線**（activation steering 而非 text conditioning），
  但 MusicGen 是自回歸模型，遷移到 MeanFlow 不是直接的。列為長期選項。
- **Dynamic Negative Guidance**（ICLR 2025, arXiv:2410.14398）✅：明確指出樸素 negative prompting
  「受限於固定 guidance scale 的假設，因逆過程的非平穩性與狀態相依性，可能導致高度次優甚至完全失效」。
  **這是我們 step-window ablation 最強的理論引用**（比 ECCV 2024 的 reverse activation 更一般）。
- **Academic Text-to-Music Grand Challenge**（arXiv:2605.21538, ICME 2026）：ATTM 標準資料集與評估方法，
  Score-Aware Training 就是投這個 challenge。若要對外比較，這是一個現成的共同基準。🟡

### 7d. 核對後不變的結論

- QA-MDT 仍是 **TTM 唯一的文字 negative prompt 量化 ablation**（p-MOS +0.036）。
- 假說 N1（增益 ∝ 訓練 caption 的保真度詞彙密度）沒有被任何新文獻推翻或搶先。
- 三層拆解（any-text 34% / fidelity vocabulary 34% / polarity 32%）仍是文獻空白。
