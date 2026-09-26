# 把 guidance（含負向）收進權重：MeanFlow CFG 目標、guidance 蒸餾與訓練期負向的文獻定位

2026-09-27。這份文件是為 084（NegMF）寫的。它回答三個問題：

1. 我們的 MeanFlow S2 訓練目標裡，本來就有什麼 guidance？
2. 文獻上有沒有人把**負向** guidance 收進訓練目標？
3. 本週的結果（073／075／080／081）在這張地圖上落在哪裡？

先讀：`negative_samples_in_training_and_cfg_2026_09_22.md`（四家族分類）、
`negative_prompting_and_prompt_engineering_2026_09_04.md`（推論期負向 prompt 的座標）。

## 1. MeanFlow 的 CFG-in-training（2505.13447，Geng et al.）

MeanFlow 學的是平均速度 u(z, r, t)。它的 1-NFE 取樣不能再做推論期 CFG，
因為推論期 CFG 要兩次前向。所以作者把 CFG 直接寫進訓練目標：

- 原始目標是 v = ε − x。
- 換成 ṽ = ω·v + κ·u(z,t,t|c) + (1−ω−κ)·u(z,t,t|∅)。
- u(·|c) 與 u(·|∅) 是模型自己在 r=t 時的輸出，並且 stop-gradient。
- 有效 guidance scale 為 ω/(1−κ)。論文 ablation 裡 κ=0.9 最好。

**這其實是一種自我蒸餾的 CFG**：模型在訓練時被要求直接輸出「已經加了 guidance 的場」。
推論時只要 cfg=0、一次前向，就拿到 guided 的結果。

**MeanAudio（2508.06098）** 沿用這個設計：ω=0.3、κ=0.9，有效 scale 3。
論文的 CFG ablation 顯示 CLAP 在 scale 3 最高（0.285），比無 guidance 高。

**我們的程式**（`meanaudio/model/mean_flow.py` `loss()`）是同一條式子：
`v_hat = w*v + k*u_t_c + (1-w-k)*u_t`。

- `u_t` 讀固定的空字串特徵 `weights/empty_string_t5.pth`（q=10）。
- `u_t_c` 讀未 drop 的 caption。
- 所以我們現在的「CFG0」格，其實是**已內建 scale 3 無條件 guidance 的模型**。
  CFG3+neg 格是在這之上，推論期再疊一次、並把 ∅ 換成 fidelity8。

固定點（訓練收斂時）：u(c) = 3·v(c) − 2·u(∅)。

- 對 drop 掉 caption 的 10% 樣本，我們的 target 仍用未 drop 的 caption 算 `u_t_c`。
- 所以 null 分支的固定點是 u(∅) = 0.3·v̄ + 0.9·ū(c) − 0.2·u(∅)，
  其中 v̄ 與 ū(c) 對 caption 取平均。
- 代入後 u(∅) = v̄。

**2026-09-27 補充事實（084 builder 量到）**：`empty_string_t5.pth` 不是 T5('') 的輸出。
它是**同一個常數向量（範數 1.32）重複 77 次**；T5('') 在三種 padding 下的 cos 分別是
−0.158／0.496／0.050。CLAP 端的 `empty_string_clap_c.pth` 則等於 CLAP('')（cos 1.0000）。
所以 null 的 T5 分支從來不是任何字串的編碼，而是上游留下的固定向量。

## 2. 把 guidance 收進權重的其他做法

| 方法 | 做法 | 對我們的意義 |
|---|---|---|
| **Model-guidance MG**（2502.12154） | 訓練目標 ε′ = ε + w·sg(ε(c) − ε(∅))，推論 1 次前向；報告約 6.5× 收斂加速、ImageNet FID 1.34 | 與 MeanFlow CFG 目標同一個家族：兩者都用模型自己的 c／∅ 差當 target 的修正項 |
| **GFT**（2501.15420） | 把 guidance 係數 β 當輸入，訓練一個模型同時代表 cond 與 guided | 需要新增 β 輸入（改網路架構）。我們禁改 MeanAudio 類別，**不可用** |
| **Guidance distillation**（Meng et al. 2210.03142） | 老師做 CFG，學生學 guided 輸出（w 當輸入） | 需要第二個模型與 w embedding，成本高；MeanFlow 目標已經是免老師版本 |
| **CFG-Zero\***（2503.18886） | 流模型早期速度估計不準，前幾步把 guided 速度歸零／縮放 | 與 075 分段結果相反的方向：我們的增益 93–100% 在前 9 步（高雜訊端） |

## 3. 訓練期或蒸餾期的「負向」guidance

| 方法 | 做法 | 關鍵發現 | 對我們的意義 |
|---|---|---|---|
| **NASA**（2412.02687，Negative-Away Steer Attention） | 單步蒸餾模型不能用 CFG 負向；改在 cross-attention 特徵空間減去負向 | 只把負向放在**老師**端再蒸餾，學生的負向效果**稍微變差**。學生必須自己在推論時看見負向 | 反證：「只要老師有負向、學生就學會」不成立。084 的 target 也是「老師＝自己」，所以必須量 CFG0 的增益能收回多少 |
| **ReNeg**（2412.19637） | 把負向 embedding 當可學參數，用 reward model 端到端訓練 | 學出來的負向 embedding 優於手寫負向 prompt，而且可跨模型轉移 | 另一條路：學 embedding 而不是蒸餾到權重。需要可微 reward，我們的 AES 可以當 reward，但有 reward hacking 風險（RF-Limits 的 crest-as-reward 崩潰） |
| **Dynamic Negative Guidance**（2410.14398） | 依後驗動態調整負向強度 | 固定強度的負向在低雜訊端傷多樣性 | 支持只在部分 t 區間施加負向 |
| **Guidance interval**（Kynkäänniemi et al. 2404.07724） | 只在中段雜訊水準施加 guidance | 高雜訊端的 guidance 主要傷多樣性、FID 變差；低雜訊端幾乎無效 | 對 075 的分段結果是**警告**：我們的增益 93–100% 集中在 t>2/3（高雜訊端），預測會付 FAD／多樣性代價。這也對得上 negprompt 線唯一的負帳 FAD +0.046 |

另外兩個家族在 2026-09-22 的文件已經整理過，這裡只摘要：

- **QA-MDT**：訓練期品質條件前綴，即 081–083 在測的設計。
- **DPO／偏好對**（Tango2、MusicRL）：需要成對樣本與偏好標籤，成本高。

## 4. 本週結果在這張地圖上的位置

| 本週結果 | 文獻對照 | 落點 |
|---|---|---|
| 073：ADG／APG 取代樸素 CFG 外插，純 CFG 只收回 0.022 PQ（fidelity8 的 2%）；N8 cfg4.5 ADG +0.096 PQ（對齊響度） | ADG／APG 論文都在影像上報告飽和改善 | 範數放大**不是**純 CFG 拿不到 PQ 的主因；幾何只能在「已經有負向文字」時補一點 |
| 075：點名缺陷的訓練資料讓 negprompt 增益縮小；增益 93–100% 來自前 9 步（t>2/3） | Guidance interval、CFG-Zero\*（兩者方向相反） | 我們的負向作用在高雜訊端，決定的是粗結構／能量分布，不是細節 |
| 080：負向分支換成較差的 EMA 快照（autoguidance）只拿到 +0.06 PQ，而且與文字負向不可疊加 | Autoguidance（Karras 2406.02507） | 負向槽要的是**文字**，不是「較差的模型」 |
| 081–083（跑中）：QA-MDT 式品質前綴 | QA-MDT | 訓練期條件化那一支 |
| 04–09 月：fidelity8 在 CFG3 帶來 +0.83～1.02 PQ，全部要推論期兩次前向 | MeanFlow 本身就把 ∅ 分支蒸餾進權重 | **沒人試過的空格**：MeanFlow 的 CFG 目標已經有一個 ∅ 分支，把它換成負向文字，就是免老師、免架構改動的負向蒸餾 |

## 5. 084 的設計依據（為什麼是這個方向）

1. **機制上最短的路徑**。我們的 S2 目標已經在做 guidance 蒸餾（scale 3、∅ 分支）。
   只要把 ∅ 分支的**文字**換成 fidelity8，就是把「CFG3+neg」的推論期效果寫進 target。
   不需要老師、不需要新輸入、不動 MeanAudio 類別。
2. **本週三條線都指向「負向槽要的是文字」**：
   - 080：較差模型不行。
   - 073：幾何不行。
   - 075：點名缺陷的訓練資料反而削弱增益。
   剩下還沒試過的，是把文字負向**收進權重**。
3. **有明確的失敗假說可以證偽**。NASA 發現只在老師端放負向，學生變差。
   如果 084 的 CFG0 增益 ≈ 0，就是 NASA 的結論在 MeanFlow 目標上複現。
   它也同時說明負向的效果需要推論期的兩次前向差分，不能被單一場吸收。
4. **Guidance interval 給了第二個 arm 的理由**。只在 t>2/3 施加負向（Nhi），
   同時測試「075 的分段結果能不能在訓練期複現」與「限制區間能不能減少多樣性代價（FAD）」。

## 6. 引用界線

| 可以寫 | 高可信推論 | 不能這樣寫 |
|---|---|---|
| MeanFlow／MeanAudio 的訓練目標內建 scale 3 的 CFG | 我們的 CFG0 格已經是 guided 模型，所以「CFG0 vs CFG3」比較的是**額外** guidance | 「MeanAudio 沒有 guidance」 |
| NASA 報告學生只靠老師端負向會稍變差 | 負向效果可能需要推論期差分 | 「負向不可能蒸餾」（NASA 是單步影像，不是 MeanFlow） |
| Guidance interval：高雜訊端 guidance 傷 FID | 我們的 FAD 負帳可能來自前 9 步 | 「我們的 FAD 變差是因為高雜訊端」（未量） |
| `empty_string_t5.pth` 是常數向量、不是 T5('') | — | 「null 分支等於空字串的語意」 |
