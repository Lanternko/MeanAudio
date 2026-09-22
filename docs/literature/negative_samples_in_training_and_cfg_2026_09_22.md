# 負樣本進訓練有沒有用、怎麼加、跟 CFG 的關係

2026-09-22 文獻調查。起因：D1 probe 在問「模型有沒有缺陷方向」，而這個問題在文獻裡
被人講過、但沒有人量過。本檔把「負樣本」拆成四個互不相同的家族，標清楚哪些數字是論文
的、哪些是我們自己的推論。

相關既有檔：`negative_prompting_and_prompt_engineering_2026_09_04.md`（negprompt 的
文獻座標）、`quality_metric_validity_2026_09_18.md`（指標效度）。

---

## 0. 一句話結論

**有用，但幾乎所有證據都來自「偏好對 fine-tune」而不是「把壞音訊放進預訓練」。**
唯一把壞資料留下來、改用條件訊號標記的是 QA-MDT，而它同時給出目前唯一一組
「訓練期品質條件」vs「推論期負向 prompt」的正面對照 —— 前者贏四倍以上。
而 QA-MDT 對負向 prompt 為什麼弱的解釋（「low quality 的 text embedding 在訓練期
大概沒被 disentangle」）是**斷言，不是量測**。D1 就是在量這句話。

---

## 1. 四個家族（按成本由低到高）

### F1 — 不加任何負樣本，只在推論期用 CFG 的負向分支
把 CFG 的 unconditional 分支換成一段文字。零訓練成本，就是我們現在在用的。

- 機制與極限：ECCV 2024《Understanding the Impact of Negative Prompts》指出負向
  prompt 的作用出現在正向內容已經成形**之後**，是 latent 空間裡的互相抵消，不是一開始
  就避開。
- 常數 guidance scale 是次佳：ICLR 2025《Dynamic Negative Guidance》論證 reverse
  process 是非穩態、state-dependent 的，固定 scale 會明顯偏離最佳。
  → 對我們：031 定的 cfg 3.0 是「固定 scale 下的最佳點」，不是這條路的上界。
- 少步數會直接壞掉：NAG（Normalized Attention Guidance）與 VSF 都指出在強烈壓縮取樣
  步數時 CFG 的正負分支預測發散、負向 prompt 失效，需要改在 attention 空間做外插。
  → 對我們：MeanFlow 25 步還算不上 aggressive few-step，但如果之後要壓到 1–4 步，
  現有 negprompt 增益不能假設會跟著搬過去。
- 音訊側的負向 guidance 已有應用：《Step-by-Step Video-to-Audio Synthesis via Negative
  Audio Guidance》把「已經有的音軌」當負向條件，逐步疊出複合音景。

### F2 — 壞資料留著，改用品質/缺陷條件訊號標起來（訓練期）
- **QA-MDT**（IJCAI 2025）：pseudo-MOS 同時做兩層注入 —— coarse 是
  "low/medium/high quality" 文字前綴（門檻 μ−2σ / μ±σ / μ+2σ），fine 是把 p-MOS 量化成
  1–5 級的 quality token 接在 encoder/decoder 的 prefix。推論期的 guidance 寫成
  高品質條件 對 低品質條件（且無文字）的外插。
  - 關鍵對照（MTT-FS，論文 Table A.3）：
    | 系統 | FAD ↓ | p-MOS ↑ |
    |---|---|---|
    | MDT baseline | 5.757 | 3.796 |
    | MDT + negative prompt | 5.641 | 3.832 (+0.036) |
    | QA-MDT（quality token） | 5.200 | **4.051 (+0.255)** |
  - 論文明講：**直接濾掉低品質音樂會縮小資料集、model 表現一路下滑** —— 所以「留著 + 標記」
    才是他們的主張。
  - 論文對負向 prompt 弱的解釋是一句沒有實驗支撐的猜測（text embedding 未被 disentangle）。
- **IQA-Adapter**（影像）：把 IQA 分數當條件學進 diffusion。重點不是高品質端的 +10%，
  而是**低品質端真的會產生對應的劣化**（論文自述可當 degradation model 用，條件調低會
  出現 JPEG 類 artifact）。這是「訓練過的品質軸是雙向可走的」的存在證明。
  → 我們的 q_embed 是同一家族，但我們自己的稽核顯示 P7V1 的 Q 走的是 support-set
  gating 而不是品質軸（見 memory `reference_p7v1_q_support_gating_2026_04_21.md`），
  所以我們沒有這個存在證明。

### F3 — 偏好對 fine-tune（負樣本是生出來的，不是收來的）
- **Tango 2 / Audio-alpaca**（ACM MM 2024）：約 15k (prompt, audio_w, audio_l)。
  負樣本兩種造法：(a) 把 caption 擾動（刪概念、換順序）後餵回同一個模型生成；
  (b) adversarial filtering —— 同一 prompt 生多個，取 CLAP 低於門檻的當負樣本。
  然後用 DPO-diffusion loss fine-tune。
  ⚠️ 對我們最重要的一點：(b) 是**用 CLAP 挑負樣本、再用 CLAP 類指標報進步**，正是教授
  2026-03-27 那條 data leakage 原則禁止的形狀。要抄只能抄 (a)。
- 音樂側：**MusicRL**（~300k pairwise preference，RLHF 微調 MusicLM）、
  **DiffRhythm+ / LeVo**（multi-preference DPO）、**MR-FlowDPO**（多 reward：文字對齊、
  音訊品質、語義一致）。方向一致：負樣本來自自家模型的 rollout，由 reward model 分好壞。
- 共同前提：要有一個**不等於評估指標**的 reward model。我們目前沒有（AES 與 CLAP 都已經
  是評估端，而且 AES 的效度問題見 `quality_metric_validity_2026_09_18.md`）。

### F4 — 把負向分支從「一段文字」換成「一個模型」
這是「負樣本」與「CFG」接得最緊的一支，而且其中一種**完全不需要負樣本**。

- **Diffusion-NPO**（ICLR 2025）：拿同一批偏好資料**把偏好對反過來**訓一組權重
  （reward 取 1−R），推論時當 CFG 的負向分支：ε = (ω+1)ε_pos − ω·ε_neg，且 ε_neg 通常
  混入部分正向 offset 以免發散。不需要新資料或新策略，代價是多存一組權重（LoRA 下很小）。
  SDXL 上 PickScore 22.97→23.08、ImageReward 1.032→1.047、aesthetic 偏好 68.8%。
- **Autoguidance**（Karras et al., NeurIPS 2024，《Guiding a Diffusion Model with a Bad
  Version of Itself》）：負向分支用**同一個模型較小/訓練不足的版本**。不需要任何負樣本、
  不需要任何標註，並且把「提品質」與「殺多樣性」解耦（這正是 CFG 的老毛病）。
  ImageNet 64×64 FID 1.01、512×512 FID 1.25。
  → **對我們可立刻做**：同一條 run 的早期 checkpoint 現成就是「bad version of itself」。
  成本是 2× NFE、零訓練。這是目前這份調查裡 ROI 最高的一條。

### 旁支 — 對比學習裡的負樣本（改的是評估器不是生成器）
CompA 的 composition-aware hard negatives、T-CLAP 的 temporal negatives 是在**訓 CLAP**，
SLAP 則示範不用負樣本也能做 music-text 對齊。這條線只該用來討論「我們的 CLAP 量得準不準」，
不能當成「加負樣本能改善生成」的證據。

---

## 2. 跟 CFG 的關係（重點）

1. **CFG 與「訓練期負樣本」是同一個機制的兩端。** CFG 靠訓練期的 label dropout 造出
   unconditional 分支；負向 prompt 只是把那個分支的條件從 null 換成一段字。所以
   **訓練決定負向分支能到哪裡，cfg scale 只決定外插多遠**。prompt 工程動的是後者。
2. 因此負向 prompt 的效果上界，取決於那個負向條件有沒有對應到模型真的建模過的區域。
   QA-MDT 認為沒有（+0.036 vs +0.255）；IQA-Adapter 證明只要訓練過就有。**中間那格
   ——「沒訓練過的模型到底能不能被推進缺陷區」—— 文獻是空的，就是 D1。**
3. F4 的做法等於承認這件事：與其賭文字條件能指到缺陷區，不如讓負向分支變成一個**確實
   會產生壞輸出的模型**（NPO 用反向偏好訓、autoguidance 直接用沒訓好的自己）。
4. 我們自己的邊界條件必須一起寫：負向 prompt 的增益 +1.067 PQ 相對 QA-MDT 的 +0.036
   是異常值（memory `negprompt_literature_position`），而且 FAD 反而變差
   （`negprompt_hurts_fad`）、PQ/CLAP 對響度與 crest 敏感（063/065）。任何跨家族比較
   都要先鎖響度。

---

## 3. 對我們的三個可行動結論

| 選項 | 成本 | 為什麼值得 | 風險 |
|---|---|---|---|
| **autoguidance**：用同 run 早期 checkpoint 當 ε_neg | 零訓練、2× NFE、一個 eval 格 | 文獻上唯一「不需要負樣本」的負向分支；直接檢定「負向分支換成模型是否勝過換成文字」；我們 checkpoint 現成 | 要跑響度閘門；MeanFlow 上沒人做過，可能是新結果也可能直接失敗 |
| **quality/defect token 重訓**（QA-MDT 形狀） | 一條完整 S1+S2 | 目前唯一有正面證據的訓練期做法 | 我們的 Q 前科是 support gating 不是品質軸；且要先有可信的 per-clip 品質分數（AES 效度存疑） |
| **DPO with caption-perturbation negatives**（Tango 2 的 (a)） | fine-tune + 生負樣本 | 不需外部 reward model | 只能用 (a)；用 CLAP 挑負樣本就是 leakage |

D1 的結果決定前兩項的先後：若模型推不進缺陷區，負向 prompt 這條線的上界就被釘死，
重心要移到 F4（把負向分支換成模型）。

---

## 參考

- QA-MDT — https://arxiv.org/abs/2405.15863 ; IJCAI 2025 https://www.ijcai.org/proceedings/2025/1126
- IQA-Adapter — https://arxiv.org/abs/2412.01794
- Tango 2 / Audio-alpaca — https://arxiv.org/abs/2404.09956
- MusicRL — https://arxiv.org/abs/2402.04229
- LeVo（multi-preference alignment）— https://arxiv.org/abs/2506.07520
- MR-FlowDPO — https://arxiv.org/pdf/2512.10264
- Diffusion-NPO — https://arxiv.org/abs/2505.11245
- Autoguidance — https://arxiv.org/abs/2406.02507
- NAG — https://arxiv.org/abs/2505.21179 ; VSF — https://arxiv.org/abs/2508.10931
- Dynamic Negative Guidance — https://arxiv.org/abs/2410.14398
- Understanding the Impact of Negative Prompts（ECCV 2024）— https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12484.pdf
- Negative audio guidance（V2A）— https://arxiv.org/abs/2506.20995
- CompA — https://arxiv.org/abs/2310.08753 ; SLAP — https://arxiv.org/abs/2506.17815
