# 人類聽測 pilot：MeanAudio vs MusicCaps 真實錄音（2026-10-15）

狀態：2026-10-15 頁面已發布，等聽者。不需要 GPU，用的全是 115 已評分的音檔。

## 目的

- 2026-10-14 會議第 7 項：MeanAudio 對 MusicCaps 真實錄音的人類聽測。
- 同時是 TTM 比較表 E（人類欄）的 pilot，見 `ttm_quality_comparison_plan_20261007.md` §7.7。
- 我們手上所有「生成勝過原音」的說法都只靠 AES。AES 在生成音訊上和人類的 PQ 相關只有 0.55，人類差 < 0.5 的配對上擲硬幣。這是第一筆直接的人類判斷。

## 系統（全部取自 p2 115 的 lvl30 區塊）

| 代號 | 內容 |
|---|---|
| `meanaudio_cfg3_neg` | ours：slot0clean nmv2matched full s14159265（p2 114），CFG 3＋fidelity8 負向 |
| `meanaudio_cfg0` | 同一個 checkpoint，CFG0 |
| `musiccaps_real` | MusicCaps 原始錄音 |
| `musicgen_medium` | MusicGen-medium（p2 113） |

三組成對比較，同一組 20 個 prompt：

| 比較 | A／B | 為什麼 |
|---|---|---|
| `real_fid8` | ours CFG3＋neg vs 真實 | 會議要求的主比較 |
| `real_cfg0` | ours CFG0 vs 真實 | 沒有 PQ 導向手法時的對照 |
| `mg` | ours CFG3＋neg vs MusicGen-medium | 表 E 主比較的 pilot |

## 刺激與播放

- **選 prompt**
  - 母體：四個系統都有 lvl30 音檔的 MusicCaps id，共 5,131 個，依 TSV 順序。
  - `random.Random(20261015).sample` 抽 20 個，不過濾。
  - 生成靜音的 clip 留著，算失敗（ours CFG3＋neg 有 1 段）。
  - **這 20 個 prompt 與校準 clip 不可進正式表 E。**
- **校準 clip**
  - 集合外、status ok 的一段真實錄音：`4CkZqT26Vb0_30`。
  - 聽者開頭播一次、調好音量後鎖定。
- **播放檔**
  - 115 的 lvl30（−30 LUFS）FLAC PCM_24 轉成 WAV PCM_24。
  - prep 斷言逐樣本與評分用的檔案相同。
  - 原生取樣率：MusicGen 32 kHz，其餘 16 kHz。
- **檔名**
  - 每個（比較, prompt, 系統）各自一個不透明檔名（seed 20261016）。
  - 同一段音訊在不同比較中出現時，無法從檔名認出。
  - 共 121 檔、67.5 MB。
- **試次**
  - 60 個主試次：3 比較 × 20，A/B 位置隨機，順序打亂。
  - 6 個重複：每個比較 2 個。A/B 對調，放在後半，距原試次 ≥ 10 個。
  - 共 66 試次。
- **每個主試次的三題**，各答 A／一樣／B：
  - pq 技術音質（雜訊、失真、悶）
  - ovl 整體喜好
  - txt 和文字描述的吻合度
- **盲測檢查**
  - 放在最後，共 6 題，兩個 real 比較各 3 題，A/B 重抽。
  - 題目：「哪一段是真實錄音？」，答 A／不確定／B。
  - 頁面開頭**不**提到有真實錄音，以免影響前面的判斷。
- **熟悉度題**：問聽者是否熟悉 MeanAudio 或這個研究的生成音訊，答是／否。

## 預先登錄的指標預測

只用這 20 個 prompt、實際播放的 lvl30 檔。數字是 ours − other 的平均，括號是 ours 勝的 prompt 數 / 20。
預測的完整逐 prompt 差值在 `predictions.json`。

| 比較 | AES PQ | AES CE | MEva | CLAP | 靜音 |
|---|---|---|---|---|---|
| `real_fid8` | +0.525（13） | +0.730（15） | +0.148（12） | −0.075（3） | ours 1 |
| `real_cfg0` | −0.234（6） | −0.140（9） | −0.044（10） | −0.113（2） | 無 |
| `mg` | +0.027（10） | +0.166（12） | +0.200（13） | +0.016（11） | ours 1 |

指標的說法是：
- PQ 導向的 ours（CFG3＋neg）技術音質勝真實錄音，但 CFG0 輸。
- 文字吻合兩者都輸真實錄音。
- 對 MusicGen 大致打平。

## 讀數規則（開評前固定）

**排除聽者**
- 熟悉度答「是」。
- 重複試次的 pq 不一致（含平手轉為一方）有 ≥ 4/6 題。
- 排除的人另列，不靜默丟掉。

**主讀數**：每個比較、每一題，算 ours 勝率。
- 分母是非平手答案，做雙尾二項檢定（p = 0.5）。
- 平手比例另報。
- 以聽者為單位彙總，也以 prompt 為單位彙總。聽者少時以 prompt 層級為主。

**指標一致率**
- 每個（聽者, 試次）的人類偏好，與該 prompt 的指標差值符號比對，算一致率與 Wilson 95% CI。
- 配對：pq 題對 AES PQ／MEva，txt 題對 CLAP，ovl 題對 AES CE／MEva。
- 指標差值 |Δ| < 0.1（CLAP < 0.005）的 prompt 另列。

**重複一致性**：主試次與重複試次答案相同的比例（三題分開）。

**盲測檢查**：正確選出真實錄音的比例，兩個 real 比較分開報。
- ≈ 50%：聽者分不出來源，偏好結果不受「認出真實錄音」影響。
- 明顯高於 50%：偏好可能受來源辨認影響，報告時要並列。

**措辭**：與預測不一致時寫「與預先登錄的指標預測不一致」。pilot 不寫指標「錯」，也不寫 ours「勝過真實錄音」。

## 聽者

- 不可以是熟悉 MeanAudio 輸出的實驗室成員，包含作者本人。
- 由使用者招募。
- pilot 先求 3–5 人跑通流程，正式表 E 的規模（暫定 200 prompt × 5 人）另議。
- 每人約 25–35 分鐘：66 試次 × 2 段 10 s，加 6 題盲測。

## 頁面與資料

- 評分頁：https://claude.ai/artifact/Ssuv4tPZ2NF7FDCjwhWGYZ
- 頁面是私有的。要用 Share 選單分享給聽者，權限選 **Contributor**（interact），聽者才能存答案。
- db 規則：每人只能讀寫 `ratings/<自己的 uid>`，只有擁有者能讀全部。
- 答案格式：
  - `answers.t<trial> = {pq, ovl, txt, ms, at}`
  - `answers.b<blind> = {real, ms, at}`
  - 另有 `rater`、`familiar`、`calibrated`
- 答案檔：`~/nvme_experiment_artifacts/meanaudio/human_listen_20261015/`（repo 外）
  - `key.json`：檔名 → 系統／id／sha256。**不可給聽者**。
  - 另有 `selection.json`、`page_trials.json`、`predictions.json`、`deliver/`、`site/`。
- 產生腳本：`scripts/eval/human_listen_20261015.py {select|prep|predict}`。
- 頁面樣板：`scripts/eval/human_listen_20261015_page.html`，`__PAGE__` 由 `page_trials.json` 取代。

## 限制

- 20 prompt × 少數聽者，只能看方向與流程問題，不能當表 E 的結果。
- 只有一個 ours checkpoint（s14159265）。
- MusicCaps 真實錄音裡有大量業餘／現場錄音（AES 稽核：原音最好的 40% 生成仍輸），「勝過真實錄音」的意義依賴 prompt 抽樣。
- 聽者用自己的耳機與環境，只有校準 clip 控制音量。
