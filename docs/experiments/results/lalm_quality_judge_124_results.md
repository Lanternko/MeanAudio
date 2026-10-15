# p2 124：LALM 品質評審（Music Flamingo／Qwen2.5-Omni-3B）結果

收線 2026-10-15 10:37（terminal `~/gpu_queue/p2/done/124_lalm_quality_judge.terminal.json`）。
設計與預先登錄判讀：`../realpiece_control_lalm_judge_123_124_20261014.md`。
摘要 JSON：`lalm_quality_judge_124_summary.json`（sha256 `b7bb6ee3…4ada6a8`，與 terminal 記錄相同）。
原始逐段分數：`~/eval_output_nvme/lalm_judge_124/`。

**沒有人類評分**：下文的「不一致」都是指「與預先登錄的預期不一致」，不代表評審或指標錯了。

## 一句話

預先登錄的最低門檻（noise 組各來源要單調、clean > 純白噪音、真實 > 白噪音）**兩個評審都沒過**。依預先登錄的規則，後面不做解讀。

- Music Flamingo 對生成音訊加白噪音，分數**反而上升**。
- Qwen 的四個評分 prompt 都排得出「clean／真實 > 白噪音」，但噪音階梯仍不嚴格單調。

第 2–5 點只當描述性讀數。

## 執行紀錄

- 刺激 74,158 段 × 2 評審 × 6 任務，batch 1。分數是 1–5 數字 token 的期望值，數字機率質量 0.992–0.999。explain 的 `Rating: N` 解析率：Flamingo 0.914、Qwen 1.000。
- 第一次嘗試在 Qwen 評 ttm 組時崩潰（2026-10-15 00:18）。MusicGen 的 ttm 音檔是 32 kHz，driver 對 Qwen 要求 16 kHz，直接 raise。該次嘗試移到 `p2/failed/_superseded_124_attempt1/`。
- 修正（commit `8e3cb25`）：Qwen 分支在 driver 內用 soxr VHQ 把非 16 kHz 的檔重採樣到 16 kHz。Flamingo 的 processor 本來就自行重採樣，不受影響。00:27 續跑，已完成的 TSV 逐列跳過。
- **因此 Qwen 讀到的 MusicGen ttm 是 32→16 kHz 重採樣版**；其他來源原本就是 16 kHz。
- GPU 時間：Flamingo 約 15.4 h，Qwen 約 17.5 h（10-14 17:00 → 10-15 10:37，含 9 分鐘中斷）；加上 123 的約 1 h。

## Prompt 原文（`scripts/eval/lalm_quality_judge.py`，action sha 已綁 contract）

四個評分 prompt 都接同一個結尾 ` Answer with a single digit from 1 to 5.`：

| 任務 | prompt |
|---|---|
| overall | Rate the overall quality of this music recording on a scale from 1 (very bad) to 5 (excellent). |
| production | Rate the audio production quality of this recording (clarity, fidelity, absence of noise, distortion and glitches) on a scale from 1 (very bad) to 5 (excellent). |
| musical | Rate the musical quality of this recording (melody, harmony, rhythm, and how well the performance holds together) on a scale from 1 (very bad) to 5 (excellent). |
| professional | How likely is it that this is a professionally performed and produced music recording, on a scale from 1 (certainly not) to 5 (certainly)? |
| pair | This audio contains two music excerpts separated by a short silence. Which excerpt has the better overall quality, the first or the second? Answer with a single digit: 1 or 2. |
| explain | Listen to this music recording. In at most two short sentences (under 40 words in total), describe its most noticeable strengths or flaws in sound quality and musical performance. Then finish with "Rating: N", where N is a single digit from 1 (very bad) to 5 (excellent). |

格式細節：
- 單輪 user 訊息，不加 system prompt（Qwen 因此每次印出 system-prompt 警告）。
- Flamingo 的訊息是「文字在前、音訊在後」，Qwen 是「音訊在前、文字在後」，各自照官方範例的順序。
- 評分任務：取 assistant 回答第一個 token 在數字 1–5（pair 是 1–2）上的 softmax 期望值，不抽樣。
- explain：greedy 生成 128 token，取最後一個 `Rating: N`。
- pair：兩段各取 0.5–10.5 s，中間隔 0.5 s 靜音，兩種順序各評一次。

### explain 評語實例（說明失敗型態）

從 noise 組的 explain 輸出（`scores/{mf,qwen}__explain__noise.jsonl`）抽出的評語：

- **Flamingo**，ours CFG3＋neg 加 SNR 10 白噪音（Rating 4）：「The recording is clear and well-balanced, with a clean mix and **no noticeable distortion or background noise**.」
  同一來源的 clean 版本，有的評語反而寫「muffled and low-fidelity, with noticeable background noise」。
- **Flamingo**，純白噪音（Rating 1）：「extremely quiet and almost inaudible, with no discernible musical content」。它沒有說這是噪音，而是說幾乎聽不到。
- **Qwen**，純白噪音（Rating 4）：「The recording is clear and free of distortion, with a **balanced mix of instruments**.」
  ours 的 clean 與 SNR 10 版本拿到幾乎逐字相同的評語：「clear and well-balanced … the volume is slightly too high, causing some distortion」。

這些評語顯示，問題出在評審**是否聽得出噪音**，而不只是評分 prompt 的措辭：
- Flamingo 在 SNR 10 明確說「沒有背景噪音」。
- Qwen 的 explain 評語接近固定模板，不隨內容變化。

## 1. 最低門檻（預先登錄）：未過

noise 組：四個來源（ours CFG3＋neg、MusicCaps 真實、MusicGen-medium、MusicLDM）各 100 段，各有 clean 與白噪音 SNR 40／30／20／10／0 共 6 級，另有純白噪音 100 段。

| 評審 | prompt | 單調來源（/4） | clean > 純白噪音（/4） | ttm 真實 > 白噪音 | 純白噪音分數 |
|---|---|---|---|---|---|
| Flamingo | overall | 0 | 2 | 是（3.40 > 2.92） | 2.92 |
| Flamingo | production | 0 | 3 | 是 | 2.73 |
| Flamingo | musical | 0 | 4 | 是 | 1.88 |
| Flamingo | professional | 1 | 4 | 是 | 1.67 |
| Flamingo | explain | 0 | 4 | 是 | 1.00 |
| Qwen | overall | 0 | 4 | 是 | 1.73 |
| Qwen | production | 3 | 4 | 是 | 2.34 |
| Qwen | musical | 3 | 4 | 是 | 1.70 |
| Qwen | professional | 3 | 4 | 是 | 2.48 |
| Qwen | explain | 0 | 0 | **否**（3.54 < 4.00） | 4.00 |

### Flamingo：加噪音讓生成音訊的分數上升

Flamingo overall 的條件平均：

| 來源 | clean | SNR40 | SNR30 | SNR20 | SNR10 | SNR0 |
|---|---|---|---|---|---|---|
| ours CFG3＋neg | 2.77 | 2.88 | 2.94 | 3.31 | **3.75** | 3.55 |
| MusicLDM | 2.79 | 2.89 | 3.06 | 3.34 | **3.43** | 3.33 |
| MusicCaps 真實 | 3.37 | 3.41 | 3.45 | 3.47 | 3.41 | 3.25 |
| MusicGen-medium | 3.96 | 4.03 | 3.99 | 3.96 | 3.88 | 3.65 |

- 對兩個擴散系統（ours、MusicLDM），SNR 0 的分數比 clean 高 +0.79／+0.53，CI 不跨零。
- explain 任務更大：+1.46／+0.92。
- 五個任務都是同一型態。
- 對真實錄音與 MusicGen，分數只在 SNR 10–0 才下降。
- Flamingo overall 把純白噪音（2.92）評得比 ours clean（2.77）與 MusicLDM clean（2.79）高。

### Qwen：四個評分 prompt 排得出「clean > 白噪音」，但階梯不嚴格單調

- overall 0/4 單調：SNR 20–40 略高於 clean（最大 +0.05）。
- production／musical 各 3/4 單調。沒過的來源只差極小的上升（+0.007、+0.015），沒有任何一級的 CI 整段高於 clean。
- **事後（非預先登錄）的放寬讀法**：若把「沒有任何一級顯著高於 clean」當單調，Qwen production／musical 會過。這不是預先登錄的判準，只列出供參考。
- Qwen explain 對 clean 的分數 3.5–3.8，對純白噪音卻給 4.00。

**依預先登錄：寫「排不出 真實 > 白噪音」的嚴格版本——兩個評審在噪音階梯上都排不出單調的品質順序；以下各節不做解讀，只列數字。**

## 2. 人類效度（描述）：MusicEval／PAM

MusicEval（31 系統、2,748 段，專家 MOS impression）：

| 任務 | Flamingo clip ρ | Flamingo 系統 ρ | Qwen clip ρ | Qwen 系統 ρ |
|---|---|---|---|---|
| overall | 0.64 | 0.84 | 0.60 | 0.67 |
| production | 0.58 | 0.82 | 0.60 | 0.73 |
| musical | 0.65 | 0.82 | 0.61 | 0.69 |
| professional | 0.66 | 0.88 | 0.46 | 0.70 |
| explain | 0.30 | 0.85 | −0.18 | −0.36 |

- clip 層 Pearson：Flamingo 0.59–0.63，Qwen 0.46–0.61。
- 對照同資料的既有數字（`musiceval_full_20261003.md`）：AES PQ Spearman 0.656、CE 0.683、CU 0.692，MEva 0.865（MEva 可能看過 MusicEval 訓練資料）。
- 兩個 LALM 的評分 prompt 與 AES 同級，明顯低於 MEva。

PAM（5 系統、500 段，人類 OVL）：

| 任務 | Flamingo clip ρ | Flamingo 系統 ρ | Qwen clip ρ | Qwen 系統 ρ |
|---|---|---|---|---|
| overall | 0.14 | −0.10 | 0.47 | 1.00 |
| production | 0.13 | −0.10 | 0.32 | 0.50 |
| musical | 0.41 | −0.10 | 0.51 | 1.00 |
| professional | 0.21 | 0.00 | 0.27 | 0.30 |
| explain | −0.09 | −0.30 | −0.04 | 0.10 |

- 系統層只有 5 個點，ρ = 1.00 對應的是 5 個名次完全相同，信賴度很低。
- 對照：AES PQ 在 PAM 的 clip r 0.545。
- 兩個評審在兩份資料上的相對強弱相反：Flamingo 在 MusicEval 較好、在 PAM 很差；Qwen 在 PAM 較好。

## 3. MIDI 控制（描述）：與 123 同一套計數

- degrade 欄：25 個 degrade 中 CI < 0 的個數。
- control 欄：11 個 control 中「CI 不跨零且 |Δ| > 該指標 degrade 中位 |Δ|」的個數。
- floor 欄：12 個 floor 中 CI < 0 的個數。

| 任務 | Flamingo degrade／control／floor | Qwen degrade／control／floor |
|---|---|---|
| overall | 20／1／9 | 14／4／10 |
| production | 21／1／12 | 16／2／12 |
| musical | 20／1／9 | 15／4／11 |
| professional | 19／1／11 | 15／4／12 |
| explain | 4／0／12 | 8／6／7 |

123 的同一套數字：AES PQ 13／2／12、CE 11／1、CU 9／6、MEva 15／1／11、pulse_clarity 19／0、key_cnn_conf 14／1。

**Flamingo**
- 評分 prompt 的 degrade 命中 19–21/25，是目前所有指標裡最高的。control 只有 1/11 過門檻。
- 但量級小：degrade 中位 |Δ| 只有 0.04–0.07 分（clean 4.21）。全部錯音 −0.18、亂數音高 −0.30、掉 60% 音符 −0.21。
- floor 的「SNR 20 白噪音」**反而 +0.24**（overall），與第 1 節的加噪音上升一致。

**Qwen**
- degrade 14–16/25。control 在 overall／musical／professional 有 4/11 超過門檻，依預先登錄規則寫作「對不該影響品質的改動有反應」。主要是 hall 殘響 +0.12 與移調 +2 +0.09（overall）。
- Qwen professional 的 hard clip −1.34、音訊反轉 −1.23，是 floor 中反應最大的。

**explain 任務**：兩個評審的 degrade 命中都 < 13 →「對音符層介入不敏感」。

ambiguous 類（只報告）：
- 換 pad 音色：Flamingo +0.19、Qwen professional +0.68。
- 移調 −12：兩個評審 −0.19／−0.20。

## 4. pair 任務（描述）

| 評審 | p_first_mean | degrade CI > 0.5（/25） | control（/11） | floor（/12） | 最大 p_clean_better |
|---|---|---|---|---|---|
| Flamingo | 0.525 | 12 | 0 | 11 | 0.673 |
| Qwen | 0.560 | 8 | 1 | 9 | 0.637 |

- 位置偏誤：兩個評審都偏向前一段，Qwen 較強。
- 兩種順序平均後，音符層 degrade 的 p_clean_better 幾乎都在 0.50–0.52（全部錯音：Flamingo 0.507、Qwen 0.503）。只有純白噪音（0.63–0.64）與音訊反轉（Qwen 0.56）離 0.5 較遠。
- 能過 CI 的 degrade 多半是統計上可分但實際偏好極弱。

## 5. ttm／seeds（只當讀數，非品質結論）

overall，5,131 prompt 的平均：

| 系統 | Flamingo | Qwen |
|---|---|---|
| MusicCaps 真實 | 3.40 | 2.75 |
| MusicGen-medium | **3.98** | 2.73 |
| MusicLDM | 2.83 | 2.75 |
| MusicLDM＋fid8 | 3.00 | **2.86** |
| ours CFG3＋neg（3 seed） | 2.77／2.76／2.74 | 2.78／2.80／2.77 |
| ours CFG0（3 seed） | 2.59／2.60／2.57 | 2.62／2.64／2.62 |

- 兩個評審的系統排序不一致：
  - Flamingo 一律 MusicGen > 真實 > MusicLDM 系 > ours。professional 把 ours CFG0 評 1.90，真實是 2.83。
  - Qwen 的 overall／production／musical 全距只有約 0.2 分，fid8 與 ours CFG3＋neg 在前、真實居中。production 把 ours CFG3＋neg 評最高（3.15）。
- 兩個評審共同的一點：CFG3＋neg > CFG0（Qwen professional 例外，2.82–2.88 對 2.88–2.89）。
- 三個 seed 之間差 ≤ 0.03。
- 第 1 節顯示 Flamingo 會因白噪音給生成音訊加分，ttm 的系統差也可能混有同類訊號，不能讀成品質排名。

## 小結（只到預先登錄允許的層次）

- 兩個 LALM 評審都**沒過最低門檻**，不能當品質指標使用。
- Flamingo 最大的不一致是「對擴散系統加噪音，分數上升」。
- Qwen 的評分 prompt 至少把白噪音放在最底，但噪音階梯不單調；explain 任務完全排反。
- 描述性讀數中，Flamingo 的 MIDI degrade 命中率與 MusicEval 系統層 ρ 都高。由於門檻未過，這兩點不作為效度證據。
- 無人類評分。
- **本結果照原樣登錄，不等新 prompt。** 理由是 explain 評語顯示，Flamingo 在 SNR 10 說「沒有背景噪音」，Qwen 把白噪音描述成「樂器平衡的混音」。這是感知層的失敗，換評分措辭不太可能讓噪音階梯變單調。
- 若要再試，比較有資訊量的是一個**感知檢查**，而不是換評分措辭：直接問「這段錄音有沒有可聽見的嘶聲／白噪音」（yes／no 期望值），看偵測率隨 SNR 怎麼變。如果連偵測都做不到，品質評分就沒有基礎。這個實驗未排、未預先登錄。
