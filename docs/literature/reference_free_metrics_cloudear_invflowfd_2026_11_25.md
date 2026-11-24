# 不靠參考的音樂評分器文獻定位：CloudEar 與 InvFlowFD（2026-11-25）

背景：另一位研究員看過 197 的簡報後，推薦了幾份「不靠參考」的作品，並對攻擊的說服力提了建議（紀錄在 `../meetings/2026-11-25_peer_feedback_objective_distance_reference_free.md`）。
這份文件整理推薦的兩篇論文、相關作者，以及它們對我們指標線的意義。

來源限制：
- 兩篇都只讀了 arXiv 的 HTML 版，PDF 沒解出文字。圖裡的數字（InvFlowFD 的圖 2–4、6）沒有讀到。
- CloudEar 表 1 的欄位對應是從攤平的表頭推出來的，**引用前要對 PDF**。
- 第四個連結（MIREX 2026 CMI-RewardBench 結果頁）沒讀到：music-ir.org 的 TLS 憑證過期。見文末「未完成」。

## 先釐清「reference」指什麼

「reference」在這個領域有三個不同的意思，推薦的作品和我們攻擊的評分器落在不同格：

| 需要什麼 | 例子 | 我們碰過的 |
|---|---|---|
| 同一段的乾淨原版（成對參考） | MERT／MuQ 餘弦距離、ODAQ 類指標 | 133 |
| 一包乾淨音樂當背景集 | FAD、KAD、MAD | FAD（每條線必報） |
| 都不需要，但用人類評分訓練 | AES、MuQ-Eval、CMI-RM、PAM、SongEval、**CloudEar** | 126、179–198 的主角 |
| 都不需要，也沒用人類評分 | **InvFlowFD** | 尚未 |

所以我們攻擊的 AES／MuQ-Eval／CMI-RM 本來就不吃參考音檔。它們不客觀的地方在第三列：分數是從人類評分學來的，學到的是「那群標註者的喜好」。
研究員說的「不靠參考」比較接近第四列：不用背景集、不用人類標籤。

## CloudEar（arXiv 2610.11672，ISMIR 2026 LBD；Qiqi He、Anqi Huang）

題目：CloudEar: Integrating Perceptual, Diagnostic, and Cross-Modal Evidence for Music Evaluation。

**做法**
- 成對比較的評分模型，三個專家：
  - 感知：每首隨機取 30 秒，MuQ 特徵接 Transformer adapter；損失 = 0.4 序數迴歸 + 0.4 迴歸 + 0.2 成對排序。
  - 診斷：true peak、削波比例、crest、LUFS（BS.1770／R128），加 HEAR 分數與感知分數，過一層線性層。
  - 文字對齊：音訊端用 Discogs400 曲風與 MTG-Jamendo 樂器 tag；prompt 端用 Qwen3-30B 抽曲風／樂器；兩層 Transformer 出對齊分數。
- 訓練資料：AIME、Music Arena、MusicEval、CMI-Pref 訓練集，加約 7,000 小時自家人類評分（只留超過 200 人評過的歌）。

**結果**（表 1；P-ACC＝與人類成對偏好的一致率）

| 方法 | CMI-Test 音樂性 | CMI-Test 對齊 | Music Arena | PAM-Music SRCC |
|---|---|---|---|---|
| CloudEar | 0.818 | 0.764 | 0.766 | 0.726 |
| CMI-RM | 0.783 | 0.739 | 0.719 | 0.663 |
| HEAR | 0.788 | 0.697 | 0.577 | 0.618 |
| SongBench | 0.734 | 0.682 | 0.675 | 0.544 |
| MuQ-Eval（對齊欄用 MuQ-MuLan） | 0.594 | 0.643 | 0.648 | 0.317 |
| 只有感知專家 | 0.759 | 0.674 | 0.674 | 0.606 |

- Music Arena 測試集：1,340 對，2025 年 7–12 月生成，作者說與訓練不重疊。
- 沒有信賴區間、沒有顯著性檢定、沒有擾動或對抗分析、沒有 limitations 段、沒放 code 或模型。

**對我們的意義**
- 它明確是**人類喜好**那一側：目標函數就是預測成對偏好。
- 沒放模型，現在不能進 197 的攻擊表。
- 感知專家建在 MuQ 上。A1 的「頻譜下移加分」（193／197）會不會被繼承，值得測，但目前只是推測。
- 診斷專家直接吃 LUFS／crest。我們在 063／064 量到 AES 對響度敏感、limiter 推高響度會扣分；CloudEar 把這類特徵當輸入，調音量或過 limiter 很可能直接動到分數。同樣未測。
- 它報的 MuQ-Eval Arena 一致率 0.648，可對照我們 180 量到的 0.626（測試集不同）。
- 它在 Music Arena 上訓練又測，不符合我們 r2 海選要求的獨立資料集。

## InvFlowFD（arXiv 2608.04142，ISMIR 2026；Alon Ziv、Harel Pogoda、Yossi Adi）

題目：InvFlowFD: Reference-Free and Background-Set-Free Perceptual Music Quality Metric with Flow Matching Inversion。

**做法**
- 不訓練任何新模型。骨幹是公開的 JASCO-400M-chords-drums，只當無條件 flow matching 模型用；latent 來自 EnCodec 32 kHz（10 秒 = 500×128）。
- 每段音檔編碼後，用 100 步反向 Euler 從 t=1 推回 t=0.01。把所有反推後的 latent 匯在一起，估平均與 128×128 共變異數，算它到 N(0, I) 的 Fréchet 距離。越低代表越接近先驗，作者解讀為品質越高。
- **是分布層指標**，一包音檔出一個數字，不是逐段分數。
- 附帶的逐段指標 StabilityFlow：先反推一大步（s=0.7），再正向 10 步，量原 latent 與重建 latent 的餘弦相似度。

**驗證**
- 只跟 FAD（CLAP 嵌入，背景集 FMA-pop 與 MusicCaps）比。沒比 AES、MuQ-Eval、CMI-RM、PAM。
- 合成劣化（MTG-Jamendo 測試集 100 首，每格 100 段 10 秒）：白噪、低通（300 Hz–4 kHz）、高通、剪貼四類都單調。文字說 FAD 配 MusicCaps 背景集抓不到輕度白噪。
- 人類研究：8 人、400 次成對判斷，問「哪段聽起來比較乾淨、製作比較專業」。剪貼類 InvFlowFD r = 0.73；低通與白噪「與 FAD 相當」；高通「略差」。後兩者文字沒給數字。
- 生成模型：真實 > MusicGen-small > MAGNeT-small，與 FAD 趨勢相同，沒報相關係數。
- 細微濾波（低通 8–14 kHz、高通 50–300 Hz）InvFlowFD 不單調，例如把低通 10 kHz 排在 12 kHz 前面；StabilityFlow 在這段單調。
- 沒放 code。骨幹與 EnCodec 權重公開，配方短，可以自己重做。

**對我們的意義**
- 它的考題是自己改出來的劣化音檔，人類研究只有 8 人。照判例 M4，這不算證明判別力；要用之前得先在有真實標籤的資料上驗。
- 分布層指標進不了 126／197 的逐對勝／平／負框架，只能當 FAD 的替代讀數。逐段要用 StabilityFlow。
- 「低通越重分數越差」的方向它是對的，這點和 MuQ A1 相反，適合當 193 的對照欄。
- 骨幹用器樂訓練，對人聲、MIDI 單音色的行為未知。
- 它的弱點可以預測：分數量的是「像不像骨幹學到的音樂分布」。195 的低通鋸齒波琶音、188 的循環音這類簡單合成聲會落在哪裡，是直接可做的攻擊題。

## 相關作者：Yinghao Ma（QMUL C4DM）

- 博士生，指導教授 Emmanouil Benetos；MERT、MARBLE、CMI-Bench 的作者。
- CMI-RewardBench（arXiv 2603.00610）是他的，個人頁列為 ICML 2026 接受。CMI-RM 就是我們 192／194 攻擊的評分器。
- MIREX 2026 有一個用這個 benchmark 的任務（Music Evaluation via CMI-RewardBench）。

## 對論文定位的影響

研究員問的是：我們瞄準人類喜好，還是音樂好壞。從文獻看，兩側的作品都有，而且彼此不比：

- 喜好側（CloudEar、CMI-RM、MuQ-Eval、SongBench）：用成對偏好一致率驗證，沒有任何擾動分析。
- 不靠人類標籤側（InvFlowFD、FAD 家族）：用自製劣化驗證，人類證據很薄。

我們的攻擊表剛好補中間：對喜好側的評分器做擾動，並用「處理依構造只會帶進劣化」當方向的依據。
缺的一塊是**客觀改變量**：目前每列只有評分器的 Δ，沒有「這個處理把音檔改了多少」。這是會議紀錄裡的主要待辦。

## 未完成

- MIREX 2026 結果頁沒讀到。它會列出 CMI-RM 之外的參賽評分器，r3 海選該不該納入要看這頁。網址：`https://www.music-ir.org/mirex/wiki/2026:Music_Evaluation_via_CMI-RewardBench_Results`
- 研究員推薦的作品如果不只這兩篇，其餘尚未整理。
