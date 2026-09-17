# 品質／美學指標的效度：文獻定位

日期：2026-09-18
用途：為我們長期使用 Audiobox-Aesthetics（PQ / CE / CU / PC）＋ CLAP 作為主指標一事找外部座標，
判斷 learned aesthetic predictor 的可用性，並釐清「指標上漲」在什麼條件下可以當成結論。

涵蓋兩篇：

| 代號 | 論文 | 一句話 |
|---|---|---|
| **[SongEval]** | SongEval: A Benchmark Dataset for Song Aesthetics Evaluation（arXiv:2505.10793v1, 2025-05-16；NWPU ASLP + 上海音樂學院 + Surrey + HKUST，通訊 Lei Xie） | 收 2,399 首**整首歌**的專家 5 維美學評分，訓 predictor；報出 **PC 與人類判斷相關性最低 0.408** |
| **[RF-Limits]** | The Limits of Reference-Free Speech Quality Metrics as Evaluators and Rewards on Modern Text-to-Speech（arXiv:2609.13150v1, 2026-07-02；AGIGO + ETH Zurich + Sapienza + UZH） | reference-free 指標是**瑕疵偵測器不是偏好預測器**；兩個 clip 都乾淨時全數掉到 chance，單一指標當 RL reward 必被 hack |

> **引用可信度標記**：✅ 已取 PDF 原文逐字核對數字；🟡 取自摘要／附錄敘述，未逐字核對；
> ⚪ 僅為本文的推論或定位主張，不引論文數字。

---

## 0. 合併後的四句話總結

1. **兩篇指向同一件事**：現有的 reference-free 品質／美學指標在「有明顯瑕疵」的比較上有效，
   在「都乾淨」的比較上失效。[SongEval] 從相關性側量（AES 對專家評分僅 0.61–0.66），
   [RF-Limits] 從成對偏好側量（乾淨語料上全數塌到 chance）。⚪
2. **PC（Production Complexity）是四個 AES 指標裡最弱的**，Pearson 0.408 vs CE 0.614 / PQ 0.630 /
   CU 0.662 ✅ — 我們把 PC 當次要指標的第一份外部證據。
3. **crest factor 當優化目標會災難性失敗**：[RF-Limits] 把 crest 當 GSPO reward，
   reward 從 −0.00 推到 7.49，同時獨立評審 UTMOS-v1 崩 4.51→1.23、人類 Elo 掉 516 分 ✅。
   這是對我們 061 結論的直接外部佐證，且證據強度高於 061 本身。
4. **可抄的是 protocol 不是模型**：報人類天花板、按品質分層、用 composite 而非單一指標。⚪

---

# 第一部分：[SongEval] — 音樂美學標註資料集

## 1. 論文在做什麼

不是方法論文，是**資料集 + 評分器**論文。動機：語音圈早有 MOS 資料集（DNSMOS / VoiceMOS / UTMOS）
把主觀品質變成可訓練的預測任務，音樂圈沒有對應物。

### 1.1 與既有主觀資料集的比較（Table 1）✅

| | MusicEval | AES-Natural (Audiobox) | **SongEval** |
|---|---|---|---|
| 語言 | – | EN | EN + ZH |
| 總時長 | 16.67 h | 29.44 h | **140.32 h** |
| 平均長度 | 0.36 min | 1.77 min | **3.51 min** |
| 組成 | 只有伴奏 | 伴奏 + 人聲 | 伴奏 + 人聲 |
| musicality / clarity / naturalness | ✔ / ✘ / ✘ | ✔ / ✔ / ✔ | ✔ / ✔ / ✔ |
| memorability / coherence | ✘ / ✘ | ✘ / ✘ | **✔ / ✔** |

賣點兩個：**整首歌**（不是 10s clip）、**含人聲**。

### 1.2 五個美學維度（1–5 分）✅

Overall coherence（段落間的音樂與情緒連貫）、Memorability（有沒有 hook）、
Naturalness of vocal breathing & phrasing（換氣／斷句）、Clarity of song structure（段落可辨識度）、
Overall musicality（整體聽覺享受）。

### 1.3 資料怎麼來的 ✅

- ChatGPT 生歌詞 + genre prompt，9 種曲風
- 5 個系統生成：**DiffRhythm 32.1%**、Suno 27.9%、Udio 12.8%、YuE 12.6%、Mureka 10.4%，
  另 4.2% "Others"（含少量無版權真歌與刻意保留的殘缺樣本）
- 共 **2,399 首 / 140.32 h**，ZH:EN 大致對半，男女聲時長比 60/40
- **每首 4 位標註者**（總 16 人，具正式音樂訓練），第三方公司管流程；
  每人每首 $5 USD，總成本約 $48,000 USD 🟡
- **分數分布高度偏斜** ✅：五維都有 41–53% 落在 5 分，1–4 分各僅 13–21%

## 2. 評分器實驗

2,199 train / 200 test（+50 首無版權真歌測泛化）。四個 baseline 全數自語音 MOS 移植：
MOSNet、LDNet、SSL-based（backbone 換 **MuQ**）、UTMOS-based。8×A6000。✅

**musicality 維度 LCC**（Table 3）✅：MOSNet 0.877 / LDNet 0.881 / SSL 0.908 / **UTMOS 0.916**
（utterance-level）；system-level 分別 0.934 / 0.944 / 0.965 / **0.966**。
五維趨勢一致：**有 SSL 預訓練特徵的兩個系統穩定勝出**。

### 2.1 最關鍵的一張表（Table 4）✅

與人類標註的 **Pearson correlation**：

| | CE | CU | PC | PQ | Vocal Range | **SongEval (UTMOS)** |
|---|---|---|---|---|---|---|
| Coherence | 0.631 | 0.679 | 0.433 | 0.636 | 0.657 | **0.917** |
| Memorability | 0.605 | 0.654 | 0.400 | 0.625 | 0.667 | **0.910** |
| Naturalness | 0.602 | 0.645 | 0.396 | 0.616 | 0.739 | **0.909** |
| Clarity | 0.574 | 0.627 | 0.394 | 0.603 | 0.694 | **0.908** |
| Musicality | 0.608 | 0.653 | 0.388 | 0.622 | 0.751 | **0.916** |
| **Average** | 0.614 | 0.662 | **0.408** | 0.630 | 0.702 | **0.912** |

兩個讀點：

- **PC 全維度墊底**（平均 0.408，musicality 只有 0.388）。
- **Vocal Range 這個極簡單的訊號**在 naturalness / musicality 上（0.739 / 0.751）**贏過所有
  Audiobox 指標**。論文的解讀是它主要在偵測「有沒有在唱歌」。⚪ 對我們的意義：在 AI 生成歌曲
  這個分布上，AES 與人類判斷的 0.6 相關性，有一部分可能只是在捕捉粗粒度的「像不像音樂」。

## 3. [SongEval] 的弱點

1. **沒有報 inter-rater agreement 的實際數字**。正文把 Pearson/Spearman/Kendall 列為工具，
   但沒給 rater 之間的 ICC。在 4 rater × ~47% 給 5 分的天花板效應下，LCC 0.91 有多少來自
   「大家都給高分所以容易預測」，論文沒有拆。⚪
   與 `memory/feedback_low_base_rate_spotcheck_needs_paired_design.md` 同型的坑。
2. **五維高度糾纏**，論文自己在 Limitations 承認（coherence vs clarity、musicality vs memorability）✅，
   但沒提供維度間相關矩陣。⚪
3. **利益相關未討論**：SongEval 由 DiffRhythm 同一團隊製作，而 **DiffRhythm 佔資料 32.1%**
   （最大單一來源）。用這個 predictor 排名 song generation 系統時，這是應揭露而未揭露的因素。⚪

---

# 第二部分：[RF-Limits] — 指標在乾淨音訊上失效

## 4. 核心實驗設計

六個人評語料，**按可聽見的合成瑕疵程度排序**（這條軸是整篇骨架）✅：

| 語料 | 品質 | pairs | 人類天花板 |
|---|---|---|---|
| BVCC | artifact-rich | 27.8k | 0.887 |
| SOMOS | artifact-rich | 89k | 0.836 |
| SingMOS | 歌聲 | 37.7k | – |
| SpeechJudge | defect-free | 7.6k | – |
| TTS-Arena | defect-free | 5.2k | – |
| TTS-HP（4 個商用系統） | defect-free | 2.7k | **0.764** |

**人類天花板**＝在有明確多數的 pair 上，個別聽眾同意多數決的比例。所有指標對讀這條線。
所有音訊先 normalize 到 **−23 LUFS** 才評分，讓音量無法驅動結果。✅

## 5. 主結果（Table II 節錄）✅

成對準確率，chance = 0.5：

| | BVCC | SOMOS | SingMOS | SpeechJudge | TTS-Arena | **TTS-HP** |
|---|---|---|---|---|---|---|
| **人類天花板** | 0.887 | 0.836 | – | – | – | **0.764** |
| argmax clip duration | 0.517 | 0.368 | 0.456 | 0.370 | 0.523 | **0.524** |
| **crest factor** | 0.396 | 0.501 | 0.450 | 0.478 | 0.501 | **0.495** |
| spectral flux | 0.440 | 0.559 | 0.567 | 0.573 | 0.500 | 0.468 |
| HF energy ratio | 0.460 | 0.529 | 0.466 | 0.492 | 0.485 | 0.516 |
| SCOREQ | **0.909** | 0.672 | 0.683 | 0.705 | 0.585 | 0.502 |
| UTMOS | 0.892 | 0.667 | 0.674 | 0.690 | 0.577 | 0.510 |
| UTMOSv2 | 0.899 | 0.654 | 0.653 | 0.621 | 0.540 | **0.528** |
| NISQA | 0.740 | 0.550 | 0.625 | 0.614 | 0.551 | 0.516 |
| DNSMOS | 0.678 | 0.521 | 0.573 | 0.614 | 0.507 | 0.516 |
| SQUIM-PESQ | 0.735 | 0.536 | 0.619 | 0.608 | 0.485 | 0.470 |
| SpeechJudge（專訓 pairwise judge） | 0.507 | 0.514 | 0.600 | – | 0.463 | **0.450** |

讀點：

- 最強的 UTMOSv2 在乾淨語料只有 **0.528**，天花板 0.764，**clip duration 0.524 幾乎打平**。
- SQUIM-PESQ 在 TTS-HP 是 **0.470**，反向相關。
- 連專門訓練的 pairwise judge 出域也塌到 0.45。
- Bootstrap 95% CI ≈ ±0.02（TTS-HP n=2666），指標之間差異不顯著。

## 6. 他們怎麼排除「指標壞了」這個解釋（Table III）✅

拿 500 個乾淨 clip **自己劣化自己**，看指標還能不能認出原版：

| 操作 | 偏好原版的比例 |
|---|---|
| 白噪 15 / 25 dB SNR | 0.998 / 0.983 |
| 帶限到 3 / 5 kHz | 0.863 / 0.771 |
| pitch shift +3 / +6 st | 0.950 / 0.977 |
| 插入停頓 3× / 6× | 0.927 / 0.947 |
| time-stretch +6 / +12 / +25% | 0.566 / 0.672 / 0.856 |

指標**沒瞎**，對注入的劣化仍然敏感。塌掉的是「排序兩個不同的乾淨 clip」這個**任務本身** ——
而且人類自己也塌：TTS-HP 的 15 個標註者隨機切兩半，兩半多數決只有 **0.53** 一致（BVCC 是 0.81）。
四個商用系統的 per-system vote share 是 0.49–0.51，在總體上無法區分。

這一節的方法論值得單獨記住：**「指標對介入敏感」與「指標能排序自然樣本」是兩回事**，
不能用前者證明後者。⚪ 這正是 061 crest 介入線踩到的同一個區分
（`docs/experiments/results/crest_intervention_cfg3_20260916_results.md`）。

## 7. Reward hacking 實驗（Table V）✅ — 最關鍵

policy = Qwen3-TTS-12Hz (0.6B)，**GSPO**，G=16，lr 1e-6，300 steps，**故意不加 KL penalty**。
獨立評審：held-out UTMOS-v1（與所有 reward 都不同的模型）＋ 30 人 × 300 次成對聽測的 Elo。
讀數取在 peak-reward checkpoint（區分 reward hacking 與 policy collapse）。

| Reward | 被優化的分數 | held-out UTMOS-v1 | 人類 Elo |
|---|---|---|---|
| baseline（真人錄音） | – | 4.51 | 2098 ±101 |
| **crest factor** | −0.00 → **7.49** | **1.23 (−3.28)** | **1582 ±123** |
| HF energy ratio | 0.49 → 0.96 | 1.30 (−3.21) | 1635 ±101 |
| SCOREQ | −4.71 → −0.19 | 1.26 (−3.25) | 1675 ±115 |
| SQUIM-PESQ | 4.11 → 4.37 | 3.54 (−0.97) | 2008 ±98 |
| UTMOSv2 | 3.61 → 4.07 | 4.30 (−0.21) | 1915 ±109 |
| DNSMOS | 3.51 → 3.71 | 4.41 (−0.10) | 2010 ±98 |
| Distill-MOS | 4.68 → 4.82 | 4.48 (−0.03) | 2115 ±94 |
| **equal-weight composite**（WER + Distill-MOS + UTMOSv2） | 0.60 → 0.69 | **4.56 (+0.05)** | **2125 ±95** |

- SCOREQ 那條**加了 KL penalty (β=0.1) 仍然 hack**：UTMOS-v1 照樣掉到 1.23，policy 照樣 padding 到上限。✅
- 「優化一個分數的後果，從那個分數本身看不出來」：UTMOSv2 上漲的同時，獨立的 UTMOS-v1 與人類 Elo 雙雙下降。✅
- equal-weight composite 的組合邏輯是**按防禦目標挑的**：WER 是可驗證的 intelligibility anchor
  （擋 padding/截斷），Distill-MOS 與 UTMOSv2 是**不同血統**的自然度模型（讓沒有單一模型成為唯一標的）；
  刻意排除單獨會崩的 DSP 統計量與 SCOREQ。✅

## 8. 一個掃興但重要的發現：公式不轉移 ✅

calibrated composite 在 artifact-rich 語料之間轉移 0.90，隨音訊變乾淨降到 0.61，
**兩個 defect-free 語料之間只有 0.53**（TTS-HP → SpeechJudge 是 **0.48**，低於 chance）。
leave-one-corpus-out 訓的 composite 在 TTS-HP 上等於沒用（0.52）。

**偏好是 corpus-local 的。** 連 cue 的**正負號**都是 corpus-specific：
clip duration 在 TTS-HP 有幫助，在 SpeechJudge 卻是反向的（那邊較長的 zero-shot clip 是在碎念）。

---

# 第三部分：對本專案的意義

## 9. 可以引用的

### 9.1 PC 是四個 AES 指標裡最弱的一個

我們的主指標一直是 **CLAP↑ / CE↑ / PQ↑**，PC 只作記錄（見 `CLAUDE.md` Eval 段）。
[SongEval] Table 4 給了這個選擇可引用的外部依據：PC 與五個專家美學維度的相關性全部落在
0.39–0.43，是 CE/CU/PQ 的三分之二。⚪ 安全措辭是
「PC 在 SongEval 的專家標註上與人類判斷相關性最低（0.408）」，**不要**寫成「PC 無效」。

### 9.2 crest 不可當優化目標 — 外部佐證比 061 本身更強

061 的結論是「crest 不可當訓練目標」（`memory/project_crest_intervention_061_result.md`），
但那條線的證據帶著處理劣化污染、主斜率反號不乾淨。[RF-Limits] 直接把
**crest factor 當成 GSPO reward 跑**：crest 從 −0.00 被推到 **7.49**，同時獨立評審崩 −3.28、
人類 Elo 掉 516 分（信賴區間完全不重疊）✅。這是「把 crest 當優化目標會災難性失敗」的**直接因果證據**。

**邊界條件（寫進論文時必須帶）**：
- 這是 **speech TTS**，不是音樂；跨域外推只能寫成 suggests。⚪
- crest 在 [RF-Limits] 的**評估**任務上是 0.396–0.501（chance 或反向），
  這與我們 051 量到的 **crest↔PQ +0.44 關聯**不衝突 —— 那是 crest 與**另一個代理指標**的關聯，
  不是與人類偏好的關聯。⚪ 不可寫成「crest 與音樂美學無關」。

### 9.3 「對介入敏感」≠「能排序自然樣本」

[RF-Limits] Table III 把這兩件事明確拆開，並用它來反駁「指標壞了」的替代解釋。✅
這個區分可以直接用來描述我們自己的處境：
- `memory/project_loudness_aes_sensitivity_2026_09_12.md`：同一檔案 −6 dB 就讓 PQ +0.097
  → 這證明 PQ **對介入敏感**
- 但這不保證 PQ 能在兩個都正常的 arm 之間排出對應人類偏好的順序

### 9.4 [RF-Limits] 的 −23 LUFS 前處理，正好對照出 AES 的缺口

[RF-Limits] 把**所有音訊先 normalize 到 −23 LUFS 才評分**，明說是「讓音量無法驅動結果」✅。
我們用的 `audiobox_aesthetics==0.0.4` **完全沒有響度／峰值正規化**（`normalize: False`，
原始絕對振幅直接進 WavLM conv feature extractor），見
`memory/reference_aes_no_loudness_norm_16khz.md`（source-code 驗證）。

這給了我們一個可引用的對照：**同期的評估文獻已把響度正規化當成基本衛生條件，而 AES 官方實作沒有做。**⚪
搭配 051 量到的「同一檔案 −6 dB → PQ +0.097」，這條論述是完整的（機制 + 實測 + 同期慣例）。

注意：[RF-Limits] 的 controlled intervention（Table III）測的是語音指標，**不包含 AES**，
所以 `reference_aes_no_loudness_norm_16khz.md` 裡「截至 2026-09-18 沒有論文對 AES 做
同一音訊只改增益／crest 的介入測試」這句仍然成立，051/061 補的空白沒有被搶走。⚪

## 10. 不可以引用的

### 10.1 不能把 [SongEval] 的 0.63 搬來質疑我們自己的 PQ

Table 4 的相關性是在**含人聲的整首 AI 生成歌曲**上量的。我們的評估域是 10s 無人聲
Jamendo / MusicCaps 片段。PQ 在那個域「只有 0.63」不能推論到我們的域。⚪

我們自己域內對 AES 可信度的證據另有來源，且比這篇更直接：
`project_loudness_aes_sensitivity_2026_09_12`（PQ 對音量敏感）、
`reference_inference_seed_noise_floor`（seed 造成 PQ 差 0.142）、
`reference_clap_batch_size_sensitivity`（CLAP 對 batch size 敏感且會翻排名）。

**引用紀律**：談「AES 指標的侷限」時用我們自己的數字，這兩篇只作為
「其他團隊在別的域也觀察到指標與人類判斷只有中等相關」的旁證。

### 10.2 [RF-Limits] 沒有實際測 Audiobox-Aesthetics

Audiobox-Aesthetics 被列在 related work 的同一家族裡（ref [35]）✅，但**沒有進 Table II**
（語料是語音）。所以這篇不能直接拿來論斷我們的 PQ/CE。⚪

### 10.3 SongEval predictor 不建議接進 `phase4_eval.py`

理由（皆為 ⚪ 推論）：

1. **域不符**：訓練資料平均 3.51 min 且全部含人聲；我們產 10s 無人聲片段。
2. **維度無定義**：`naturalness of vocal breathing`、`clarity of song structure` 在 10s
   instrumental 上沒有可評內容。
3. **天花板效應**：訓練標註有 ~47% 落在 5 分，predictor 很可能在 OOD 輸入上把分數全塞在同一格，
   於是在 arm 之間毫無鑑別力 —— 我們一再踩到的「判讀帶比 seed 雜訊還窄」型失敗
   （`memory/project_k3_rotation_line_retired_2026_09_03.md`）。
4. [RF-Limits] §8 的不轉移結果⚪ 進一步壓低期望：**連在同一語言域內，calibrated composite
   都無法跨語料轉移**（defect-free 之間 0.53），跨到音樂的先驗只會更差。

若日後仍要嘗試，**唯一有意義的維度是 `overall musicality`**，且必須先做 OOD sanity：
拿現成的 arm（slot0 / slot0nm / fulltrack）跑一遍，確認分數分布不是退化的單峰，再談當指標。
**這件事在跑之前不該預設會成功。**

## 11. 對 negprompt 線的直接處方

我們的 ΔPQ **+1.067** 是單一指標上的增益，而 `project_negprompt_hurts_fad_2026_09_04`
已經量到 **FAD 反而變差**。[RF-Limits] 提供了框架：
**「單一指標上漲 + 獨立評審下降」是 reward hacking 的標準簽名**。

我們沒有做 RL，所以**不是**嚴格意義的 reward hacking（沒有 policy 在優化 PQ）。⚪
但處方相同且直接適用：

1. **必須有獨立 held-out judge**。044/045 的 listening 至今未評
   （`memory/project_negprompt_content_terms_044_045.md`），這是目前最大的缺口。
2. **不要讓單一指標 headline**。[RF-Limits] 的結語原則：
   evaluations of clean systems should report the reliability ceiling, stratify by quality,
   and let no single predictor headline or reward them.✅
3. **composite 比單一指標安全**，而且 equal-weight 在沒有 in-domain 標註時就能用。

## 12. 可抄的 protocol 元素

| 元素 | 來源 | 我們缺什麼 |
|---|---|---|
| **報人類天花板**，指標對著它讀而非彼此相比 | [RF-Limits] | 我們完全沒有人類 ceiling 的估計 |
| **按品質分層**報告 | [RF-Limits] | arm 比較都是總體均值 |
| **utterance-level 與 system-level 分開報** | [SongEval] | 與 seed 雜訊框架相容：arm 排名本就該在 system-level 判定 |
| **四個互補指標** MSE + LCC + SRCC + KTAU | [SongEval] | 我們只報均值差，缺排序層統計 |
| **多位獨立標註者 + 第三方管流程** | [SongEval] | 與 `feedback_same_model_recheck_not_independent.md` 一致 |
| **equal-weight composite** 作 reward／評估 | [RF-Limits] | 組合邏輯：一個可驗證 anchor + 兩個不同血統的品質模型 |
| **controlled intervention** 證明敏感度，但**不當成排序能力的證據** | [RF-Limits] Table III | 061 正是踩在這個區分上 |

---

## 13. 一句話結論

**可引用 PC = 0.408 來正當化對 PC 的降權，可引用 crest-as-reward 的崩潰來加固 061 的結論；
可抄雙層評估與 composite 的 protocol；但不要把任一篇的 predictor 當成我們的新指標，
也不要拿它們的相關性數字去論斷我們自己域內的 PQ。**

---

## 相關文件

- `docs/literature/negative_prompting_and_prompt_engineering_2026_09_04.md` — negprompt 的文獻定位
- `docs/metrics/audiobox_aesthetics.md` — PQ/CE/CU/PC 的指標細節
- `docs/experiments/results/crest_intervention_cfg3_20260916_results.md` — 061 crest 介入結果
- `docs/experiments/crest_intervention_cfg3_20260916.md` — 061 的設計與預先登記
