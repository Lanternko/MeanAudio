# 2026-09-22 — 新方向盤點：CFG 幾何、caption 合寫、enc/dec、雜訊負樣本

**來源**：手寫/表格筆記（零散條目），本檔是解讀＋可執行化。**筆記原文逐條保留**，我的解讀與原文分開標示，避免把推論當成教授講過的話。

**背景時點**：caption 內容編輯線今天（2026-09-22）剛收線（`../experiments/caption_content_editing_line_retired.md`）；073 guidance 幾何線剛開（`../experiments/guidance_geometry_adg_apg_20260922.md`）；響度/limiter 五條階梯線（063/064/065/072/074）全部收線。新方向要接在這個狀態上讀。

---

## 筆記原文

```
合寫 / 重寫
基於某些原因 qwen 就是比 MF 強
讓 7B 的 model 重寫

動機 圍繞 cfg / aes 瑕疵 / 女聲音
normalize cfg
把正面和負面
先 normalize 再 -
或先 - 再 normalize

stable audio vision 上面？encoder 和 decoder
不同的 enc dec 對音樂生成的影響
latent diffusion model — 從雜訊去生成 latent feature

negative sample：給很多雜訊音檔，然後 caption 叫做雜訊
叫 MeanAudio 生成雜訊
low quality 的音檔有沒有共性
大聲的音量 / attribute
```

---

## A 線 — CFG normalize 的順序（接 073，最便宜、最該先做）

**解讀**：「先 normalize 再減」vs「先減再 normalize」＝ 對 `ε_pos`、`ε_neg` 做範數處理的**順序消融**：

| 變體 | 式子 | 對應文獻 |
|---|---|---|
| 樸素 CFG | `ε_neg + w·(ε_pos − ε_neg)` | baseline |
| 先 normalize 再減 | 各自投到同範數後才取差 | 沒有現成名字，073 尚未涵蓋 |
| 先減再 normalize | 取差後把結果範數拉回 `‖ε_pos‖` | ≈ ADG（範數保持） |
| 正交投影 | 差向量去掉平行分量 | APG |

073 現在只排了 ADG / APG 兩格，**「先 normalize 再減」是筆記新增的第三格**，值得補進去。

**接點**：
- cfg ≥ 2 的波形飽和（crest < 2.0）與 073 觀察到的 `crest_min 1.85` 都是**範數放大的指紋** → 筆記說的「aes 瑕疵」動機與 073 的 primary（純 CFG 的 headroom 能不能換成 PQ）是同一件事。
- 成本極低：推論期變更，不重訓，已有 `runpy + monkeypatch` harness（`memory/reference_eval_py_runpy_monkeypatch_harness.md`），不碰 `networks.py` / `eval.py`。
- **響度閘門必跑**：063/065 已證 CLAP 隨響度單調上升、PQ 反向 → 任何改變輸出範數的 guidance 變體，比較前必須先鎖 LUFS，否則量到的是響度不是 guidance。

**最小實驗**：073 的 arm grid 加一格「pre-normalize then subtract」，同 checkpoint、同 seed、MusicCaps 5521、CFG3+neg 與 CFG0 兩格照標準協定。

**未定**：「女聲音」是指主觀試聽時女聲的音色瑕疵，還是指 044/045 那條「negative prompt 加入 vocals/choir 反而降 PQ」的現象？兩者要做的事完全不同 — **待確認**。

---

## B 線 — caption 合寫 / 7B 重寫（有真東西，但踩在剛收線的線上）

**筆記主張**：「基於某些原因 qwen 就是比 MF 強」。

**現有證據對齊**（不是「某些原因」，已經量過）：
- `paired59k` captioner-only control：captioner 差異**只在 CLAP**（Qwen +0.0073 ＝ 24× seed 底線），四項 AES 全落在雜訊內；而且 Qwen 是在 **44.9% 截斷劣勢**下贏的。
- MF 全覆蓋 row-matched：MF 只在 CLAP 落後 ~0.012，AES 全在雜訊內。
- T5 77-token 窗口：MF 丟 60% token，CLAP 只掉到 0.908（`reference_caption_corpus_t5_truncation`）。

→ 所以「Qwen 比 MF 強」的已知成分是 **CLAP 一項、量級 ~0.007–0.012**，不是全面優勢。寫進任何文件都要帶這個量級。

**合寫（fusion）**：把 Qwen 與 MF 的內容融成一條，而不是 rotation（rotation 已在 046/047 測不出來）。
- 理由站得住：Qwen 在截斷劣勢下仍贏 → 兩者資訊可能互補。
- **主要風險先算再做（零 GPU）**：合寫必然更長 → T5 77-token 截斷更嚴重，可能把互補性吃光。先對合寫語料跑 token 長度與 CLAP-retention 分布，**通不過就不要進 GPU**。

**7B 重寫**：slot4v2 已證 Omni-3B 做不了句子重組、需要 ≥7B（slot0nmv2 實際用 Qwen3.6-27B）。技術上可行。
- **但**：純「重寫」屬於今天剛收線的 caption 內容編輯線（剝數字／去量測／rotation 四類在 MusicCaps 全測不出）。要啟動必須對照 `caption_content_editing_line_retired.md` 的重啟條件，並說清楚它為什麼不是第五類同型干預。
- **合寫不同**：換掉語料的資訊來源（接近「換 captioner」——目前唯一動得了 CLAP 的槓桿），不是編輯既有句子。這條可以獨立於收線論證成立。

---

## C 線 — 不同 encoder/decoder 對音樂生成的影響

**現況（已查證）**：
- `meanaudio/ext/autoencoder/autoencoder.py` — **mel-spectrogram VAE（MMAudio `v1-16.pth`）＋ BigVGAN 16k vocoder**，不是 Stable Audio 的波形 VAE。
- `sequence_config.py:26` — 實際跑 `CONFIG_16K`，duration 9.975 s、16 kHz、`latent_seq_len = 312`。

→ **整條 pipeline 的頻寬上限是 16 kHz**。這是一個從沒寫進任何實驗線的結構性限制。

**因此這條線的問題是可以被問對的**：換成 Stable Audio Open 的 44.1 kHz 波形 VAE（Oobleck），對音樂生成有多少影響？

**但評估先天有盲點（重要）**：
- AES 官方實作把輸入截到 16 kHz（`reference_aes_no_loudness_norm_16khz`）→ **換 44.1k VAE 後，AES 很可能量不到任何改善**，因為評分器本來就看不到 16 kHz 以上。
- CLAP 吃 48 kHz 輸入（`reference_clap_scoring_input_contract`），理論上看得到，但它對高頻細節的敏感度未知。
- → 這條線若要做，**必須先決定用什麼指標宣稱勝負**，否則會做出一個「聽起來更好但所有指標都平手」的結果。這件事要在動工前解決，不是事後補。

**成本**：換 enc/dec ＝ 全量重編 latent NPZ。現有 `npz_phase8v4` 87G、`npz_phase7_clean` 88G，44.1 kHz 只會更大；HDD 剩 ~103G。**磁碟是這條線的硬門檻**（Phase 9 rebuild 已經因為同樣理由卡住）。

**「latent diffusion model — 從雜訊生成 latent feature」**：這就是現行架構本身（flow matching / mean flow 在 VAE latent 上）。解讀為教授在對齊框架說法，不是新要求 — **待確認是否另有所指**。

---

## D 線 — 雜訊負樣本（我認為最有論文價值的一條）

**筆記**：給很多雜訊音檔、caption 標成「雜訊」；叫 MeanAudio 生成雜訊；low quality 音檔有沒有共性。

**為什麼這條特別有價值**：negprompt 消融的定論是「增益來自 fidelity **領域詞彙**，不是**缺陷極性**」（reversed 版本複製了 51% 增益、loud 探針反向）。一個從未被檢定的解釋是：**因為訓練語料裡根本沒有「壞音訊 ＋ 被標成壞」的監督訊號**，模型沒有缺陷方向可言。D 線正好直接補這個缺口 — 它能**反證或支持 031 的定論**，而不只是再測一個 arm。

拆成三個獨立可跑的東西，成本天差地遠：

### D0（零 GPU、今天就能做）— low quality 有沒有共性
對訓練語料的低品質子集盤點聲學特徵（LUFS / crest / 靜音率 / 頻譜斜率 / clipping）。工具鏈現成：`eval_metrics.py` 的 level 欄位。輸出 = 「缺陷」在我們語料裡到底長什麼樣，是 D2 造資料的前置。

### D1（P0 probe）— 叫 MeanAudio 生成雜訊 — ✅ **已跑完，2026-09-23 收線**
結果見 `../experiments/results/d1_defect_reachability_20260923_results.md`。三句話：

1. **模型做不出雜訊，它把音量關掉** — 要求 white noise 時頻譜平坦度只有 0.038（真雜訊 0.56），RMS −61 dBFS，cfg3 有 79.7% 的 clip 落在靜音門檻以下。
2. **缺陷詞壓低 AES，但壓出來的不是被點名的缺陷** — 方向與真劣化一致、量級不到一半，且波形簽名對不上（真加噪 crest 升、prompt 不動；真削波 crest −3.7、prompt −0.19）。
3. **CLAP 不能當缺陷驗收指標** — caption 固定時它確實隨劣化劑量單調上升，但**純數位靜音在缺陷 caption 上的得分高過真的被削波、被低通的音訊**，模型的近乎無聲產出因此在三條 caption 上都「贏過」真缺陷。

→ 支持「模型沒有缺陷方向」的解釋，031 的定論得到機制面的補強（supports，非 proves）。**D2 的驗收指標必須先定且不能是 CLAP。**

### D2（要重訓＋重編 latent）— 雜訊負樣本訓練
在訓練資料混入 N% 程式化劣化音訊（白雜訊、clipping、低位元率、極端 limiting），caption 標成明確缺陷描述。
- **劣化工具鏈我們已經有**：072/074 的 limiter ladder、063/065 的 gain ladder，degradation 是程式化的，**不需要重新 captioning**。
- 驗收指標：用同樣詞彙下 negative prompt 的 PQ 增益是否變大？**reversed 版本是否終於失效**（＝缺陷極性終於出現）？
- **D1 已排除 CLAP 作為驗收軸**（靜音刷分）。目前唯一可用的是 AES 的 PQ/CE（對真劣化單調且對 level 的方向已知）；PC 對加噪／bitcrush 反而上升，不可用。
- 成本門檻：劣化音訊要重編 latent NPZ → 回到磁碟問題。先跑 D0/D1 再決定要不要付這筆。

### 「大聲的音量 / attribute」
若是指把響度當成可控 attribute 來訓練 —— **先看已知的取捨**：CLAP 隨響度單調上升、PQ/CU 反向（063 全量、065 頂點 −21 dB），而 AES 對音量的敏感全來自 GroupNorm eps（不是音質判斷）。所以「學會生成大聲」＝ 用 PQ 換 CLAP，不是免費 attribute。要做必須先說清楚目標指標是哪個。

---

## 建議優先序

| 順位 | 做什麼 | 成本 | 為什麼現在 |
|---|---|---|---|
| ~~1~~ | ~~**D1** 叫 MeanAudio 生成雜訊（probe）~~ | ~~單次 eval~~ | ✅ 2026-09-23 收線，見上 |
| ~~2~~ | ~~**A** 073 補「先 normalize 再減」格~~ | ~~推論期~~ | ✅ 2026-09-23 收線（`results/guidance_geometry_prenorm_20260923_results.md`） |
| 3 | **D0** low-quality 共性盤點 | 零 GPU | D2 的前置，且可獨立成 caveat |
| 4 | **B-合寫** token 長度／截斷預檢 | 零 GPU | 通不過就不用進 GPU |
| 5 | **D2** 雜訊負樣本訓練 | 重訓＋重編 latent（實測僅 ~1.3 GB latent） | ✅ **075 2026-09-23 收線**：點名缺陷讓 negprompt 增益縮 0.37 PQ（`../experiments/results/d2_defect_negsample_075_results.md`）；D0 仍未做（零 GPU，可平行） |
| 6 | **C** enc/dec 替換 | 全量重編 latent，磁碟卡死 | 要先解決「用什麼指標宣稱勝負」 |

---

## 待確認（不影響前四項開工）

1. **「女聲音」**指主觀試聽的女聲音色瑕疵，還是 044/045 的「negprompt 加 vocals/choir 反而降 PQ」？
2. **「latent diffusion model — 從雜訊生成 latent」**是在描述現行架構，還是另有所指（例如換 diffusion 形式）？
3. **C 線的 Stable Audio** 是指換 VAE（Oobleck 波形 VAE），還是指參考 Stable Audio 的整體設計？
4. **B 線的「重寫」**是否要在 caption 內容編輯線已收線的前提下重啟？若是，重啟理由要寫進收線文件。
