# 除 AES／MEva 之外的音樂評估模型調查

查核日期：2026-10-20（Asia/Taipei）。本次依原始論文、作者官方 repo、模型卡與權重目錄查核；不是根據會議 ASR 補出模型名稱。只做文獻、程式及檔案目錄調查，未下載新權重、未執行評分、未修改實驗 queue。此為已查得候選清單，不宣稱涵蓋所有工作。

**對目前 MeanAudio 的 10 秒生成音訊，優先新增候選是 MuQ-Eval、CMI-RM、PAM；SongBench、HEAR 接續。** 這個順序是依任務接近程度、資源可取得性與方法互補性的研究建議，不是模型優劣排名，也不是 launch 授權。

## 1. 已確認可取得程式／評分資源的候選

「有權重」表示官方目錄或發佈入口可見，不表示已下載、成功推論或通過我們的效度測試。

| 模型／方法 | 評什麼 | 公開資源查核 | 對目前任務的判斷（本次推論） |
|---|---|---|---|
| **MuQ-Eval**（2026） | 單段生成音樂的 Musical Impression／品質 MOS；MuQ 特徵上訓練評分頭 | 官方 code；HF A1、A3a 權重目錄及設定存在 | **第一優先**：模型卡明確以 24 kHz、10 秒生成片段為輸入，最接近目前任務。 |
| **CMI-RM**（2026） | 音樂品質與指令對齊，支援文字、歌詞、參考音訊等條件 | 官方 code、HF `model.safetensors`／`config.yaml`，有獨立推論介面 | **第一優先**：可把 quality 與 alignment 分開觀察；須固定 prompt 與缺失模態處理。 |
| **PAM**（2024） | 單段音訊的清晰／乾淨程度與噪音／artifact 品質線索 | 官方 code，以 Microsoft CLAP 2023 權重實作；不需另訓練音樂 MOS 頭 | **第一優先的互補對照**：適合先看訊號劣化；不能預設它懂錯音、節奏或作曲品質。 |
| **SongBench 評分器**（2026） | Vocal、Instrument、Melody、Structure、Arrangement、Mixing、Musicality 七維 | 官方 code；repo 有 `ckpt/songbench.safetensors` | **第二優先**：有明確 Melody／Arrangement 維度，值得測；主要對象是生成歌曲，短純器樂的有效性未知。 |
| **HEAR**（2025 預印本／ICASSP 2026） | 整體 Musicality 或 SongEval 五維美學；結合 MuQ 與 MusicFM | 官方 code；HF `earlab/EAR_HEAR` 有 Track 1／2 權重 | **第二優先**：多來源、多尺度表示，方法上可補充現有評分頭；歌曲訓練與 30 秒視窗對短片段的影響需查。 |
| **SongEval toolkit**（2025） | Coherence、Memorability、Vocal Naturalness、Structure Clarity、Musicality 五維 | 官方 code 與 checkpoint | **既有候選，repo 已測過**，不是本次新發現。不能把整首歌曲美學頭直接視為 10 秒器樂品質真值。 |
| **多分軌美學模型 MSAF＋HiGIA**（2026） | 公開推論版輸出 Overall、Singing、Melody、Arrangement、Audio Quality | 作者 repo 有推論程式與 `ckpt/model.pt` 的 Git LFS 入口；未下載 payload | 適合帶人聲歌曲延伸；要準備 mixture／vocal／accompaniment，分軌處理增加干擾。 |
| **SingMOS／SingMOS-Pro predictor** | 歌聲自然度 MOS，歌聲 SSL-MOS 模型 | 官方 Torch Hub 介面、release 權重 URL；16 kHz | 歌聲專用，可納入未來帶人聲實驗；不適合當純器樂整體品質分數。 |
| **HAAQI-Net**（2024） | 音樂的技術音質／聽覺劣化，預測 HAAQI；含聽損條件 | 官方 repo 宣告提供預訓練模型，含推論入口；本次未驗證權重 payload | 邊界候選：可研究噪音／失真，任務來自助聽器與聽損，不是 TTM 音樂性。 |

上表的一手來源：

- MuQ-Eval：[論文](https://arxiv.org/abs/2603.22677)、[程式／模型卡](https://github.com/dgtql/MuQ-Eval)、[A1 權重](https://huggingface.co/zhudi2825/MuQ-Eval-A1)。
- CMI-RM：[論文 v3](https://arxiv.org/html/2603.00610v3)、[官方程式](https://github.com/Haiwen-Xia/CMI-RewardBench)、[權重](https://huggingface.co/HaiwenXia/CMI-RM)。CMI-RewardBench 是 benchmark，真正的模型叫 CMI-RM。
- PAM：[官方程式與用途](https://github.com/soham97/PAM)、[論文](https://arxiv.org/abs/2402.00282)。
- SongBench：[論文](https://arxiv.org/abs/2604.25937)、[官方評分程式／權重](https://github.com/Tencent/SongBench)。SongBench 同時指資料／評估框架及其公開評分器，不能只看名稱便當作只有資料集。
- HEAR：[論文](https://arxiv.org/abs/2511.18869)、[官方程式](https://github.com/Eps-Acoustic-Revolution-Lab/EAR_HEAR)、[權重](https://huggingface.co/earlab/EAR_HEAR)。這裡不是早年的 Holistic Evaluation of Audio Representations 同名 benchmark。
- SongEval：[官方 toolkit](https://github.com/ASLP-lab/SongEval)、[論文](https://arxiv.org/abs/2505.10793)。
- MSAF＋HiGIA：[論文](https://arxiv.org/abs/2601.12222)、[作者推論 repo](https://github.com/yisan33/song-aesthetics-evaluation)。
- SingMOS：[官方模型介面](https://github.com/South-Twilight/SingMOS)、[SingMOS-Pro 論文](https://arxiv.org/abs/2510.01812)。
- HAAQI-Net：[官方程式與任務說明](https://github.com/dyahayumgw/HAAQI-Net)、[論文](https://arxiv.org/abs/2401.01145)。

## 2. 幾個會影響選擇的重要細節

### MuQ-Eval：接近我們任務，但不能把 MusicEval 高相關當外部驗證

官方 A1 是 frozen MuQ-310M 加 pooling／MLP，模型卡報告的單段 MI SRCC 0.838 是 **MusicEval 上的五折交叉驗證**，不是我們資料的實測。其模型卡也明列跨資料泛化、受控劣化單調性及高品質系統間細微差異尚未充分驗證。雖然輸出包含 TA 頭，模型只看音訊，**不能把 TA 當作真正有讀取 prompt 的文字對齊分數**。[官方模型卡](https://github.com/dgtql/MuQ-Eval/blob/main/MODEL_CARD.md)

因此值得優先納入，但要另做未參與訓練的評分資料與受控介入測試；一般 MuQ embedding 距離和 MuQ-Eval 的已訓練品質頭是兩種方法。

### CMI-RM：有品質頭，但條件設定需要固定

官方推論回傳 `quality` 與 `alignment`；文字、歌詞及參考音訊是可選欄位。預設 final 模式按 30 秒分塊編碼，另有 standard 模式。正式測試前須確認短片段、空 prompt 和帶原 caption 的設定，避免在不同案例間改輸入條件。程式 Apache-2.0 與 HF 權重的 CC-BY-NC-4.0 宣告也要分開記錄。[官方推論 API](https://github.com/Haiwen-Xia/CMI-RewardBench/blob/main/baselines/inference.py)、[權重卡](https://huggingface.co/HaiwenXia/CMI-RM)

### PAM：模型方法與 PAM 人類評分資料是兩件事

我們已有的 PAM 人類評分資料，不代表已跑過 PAM metric。官方方法比較音訊對「清晰乾淨」與「噪音／artifact」兩類固定描述的匹配程度。它不需要對話式 LALM 生成理由，因此也不同於先前讓 Music Flamingo／Qwen 回答 1–5 分的流程。本次檢索未找到 repo 已跑 PAM metric 的明確紀錄，這不等於證明所有本機目錄都沒有。[官方實作](https://github.com/soham97/PAM/blob/main/PAM/PAM.py)

### 七個維度的名稱，不等於七種能力已成立

SongBench 的 Melody、Arrangement、Structure 很對題，但仍須實測是否對錯音、打亂節奏等介入有合理反應。HEAR、SongEval、多分軌美學模型也是如此：作者在人類評分資料上的相關性，不能直接代替我們的受控效度檢查。

多分軌模型論文涵蓋 SongEval 的五維及內部資料的另一套五維；**公開 repo 輸出的是 Overall／Singing／Melody／Arrangement／Audio Quality、0–100 分**。不能誤寫成公開 checkpoint 就輸出 SongEval 的那五維。[論文 §3](https://arxiv.org/html/2601.12222v1)、[公開版說明](https://github.com/yisan33/song-aesthetics-evaluation)

## 3. 新近論文／可追蹤方法，但即用權重尚未核實

| 工作 | 已查到的內容 | 尚缺什麼 |
|---|---|---|
| **CloudEar**，2026-10-08 | 感知評分＋HEAR／DSP 診斷＋細粒度文字對齊，做歌曲成對偏好；MuQ 30 秒片段，結合 ordinal／regression／pairwise 訓練 | 查得原論文，但未確認作者公開 code／checkpoint；不能列為已可部署。 |
| **MMMOS**，ASRU 2025 | 無參考、多領域四軸品質評分，維度與 AES 相同；作者 repo 有訓練程式 | README／目錄未找到即用的完整品質頭權重；它也不是人類標註來源完全獨立於 AES 的證據。 |
| **MoEvalSong**，ICASSP 2026 | 作者／會議頁確認的歌曲美學 MoE 模型 | 尚未核實作者公開 code／權重，先列文獻。 |
| **SongSQA**，2026-07 | 全曲歌聲品質與時間分數曲線，分段 predictor 加歌曲 aggregator | 查得論文與試聽頁，尚未核實公開可用 checkpoint。 |
| **MusicEval 原論文的 CLAP assessment baseline**，2025 | CLAP 基礎的品質與文字吻合評分模型；AudioMOS 2025 Track 1 官方列為 baseline | 此次原 repo 讀取失敗，未核實即用 checkpoint；不要把 MusicEval 資料集本身稱作評分模型。 |
| **ASTAR-NTU AudioMOS 2025 Track 1 solution** | 預測 Musical Impression／Text Alignment 的 challenge 方法 | 原論文可查，公開 checkpoint 尚未核實，不先算即用候選。 |

來源：[CloudEar 原論文](https://arxiv.org/html/2610.11672v1)、[MMMOS 論文](https://arxiv.org/abs/2507.04094)／[官方 repo](https://github.com/robert0518/MMMOS)、[MoEvalSong 官方會議頁](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=19004&bare=1)／[作者頁](https://lg-lg.github.io/shunlei.github.io/)、[SongSQA 論文](https://arxiv.org/abs/2607.16599)、[MusicEval 論文](https://arxiv.org/abs/2501.10811)／[AudioMOS 官方 baseline 清單](https://sites.google.com/view/voicemos-challenge/past-challenges/audiomos-challenge-2025)、[ASTAR-NTU 原論文](https://arxiv.org/abs/2507.09904)。

「未核實」是本次查核範圍的限制，不是宣稱作者沒有釋出。

## 4. 不應混成同一種品質模型的工具

| 類型 | 例子 | 用途與限制 |
|---|---|---|
| 音訊－文字對齊 | CLAP、MuQ-MuLan | 看是否符合文字，不能直接代表好聽／音質。PAM 雖使用 CLAP，卻另外定義了品質對比 prompt，不能和普通 caption cosine 混用。 |
| 有參考的音訊品質 | ViSQOL Audio | 需要 reference／degraded 對照，可補充受控劣化研究，不能直接給兩個無共同 clean reference 的 TTM 系統整體品質排名。 |
| 嵌入差異／分布差異 | MERT／MuQ cosine、FAD | 內容差異或分布偏離不是單段人類品質 MOS；MuQ-Eval 評分頭與 MuQ cosine 不能混稱。 |
| 語音品質 | NISQA 等 speech MOS predictors | 原定義是語音品質／TTS 自然度；不能直接套成音樂品質真值。 |
| 通用 LALM 評審 | Music Flamingo、Qwen-Omni | 可設計為候選評審，但會產生文字不等於能可靠聽出音樂劣化，且已有本地負面紀錄。 |

一手用途來源：[Microsoft CLAP](https://github.com/microsoft/CLAP)、[MuQ／MuQ-MuLan](https://github.com/tencent-ailab/MuQ)、[ViSQOL](https://github.com/google/visqol)、[NISQA](https://github.com/gabrielmittag/NISQA)。

## 5. MeanAudio 已有紀錄，避免重複算成新候選

- **SongEval 已跑過**：[`metric_win_tie_126d_results.md`](../experiments/results/metric_win_tie_126d_results.md)。該報告記載固定模型／協定未達多項預登錄門檻，不能概括為所有 SongEval 模型都無用。本次沒有重做其科學審核，也不沿用報告中的域外／非確定性「原因」為已證實因果。
- **MERT／MuQ reference cosine 已跑過**：[`metric_win_tie_133_results.md`](../experiments/results/metric_win_tie_133_results.md)。這不是 MuQ-Eval。
- **Music Flamingo／Qwen2.5-Omni-3B 已跑品質評審**：[`lalm_quality_judge_124_results.md`](../experiments/results/lalm_quality_judge_124_results.md)。固定 prompt／刺激下沒過門檻，不推廣為所有 LALM 都不行。

## 6. 建議下一步（設計建議，未啟動）

1. 先釐清 MuQ-Eval A1、CMI-RM、PAM 三者的固定輸入處理與輸出意義，保存版本與權重雜湊。
2. 設計一個共用、範圍受控的比較：已有獨立人類分數的資料＋訊號劣化＋音符層介入＋不變性對照。依各模型訓練資料排除重疊，不能把已見過的 MusicEval／SongEval 當完全獨立泛化資料。
3. SongBench、HEAR 用同一設計接續評估；對歌曲頭在短器樂片段上的結果保留域外限制。帶人聲線再加 SingMOS 與多分軌模型。
4. 後續訓練或修補前，先確認現有公開模型在哪些案例可靠；不能只看單一 benchmark 相關係數便換掉 AES／MEva。

資源查核細節、10 個官方 repo 的 commit ID，以及 4 個 HF 模型目錄版本保存於 [`music_eval_models_beyond_aes_meva_2026_10_20_sources.json`](music_eval_models_beyond_aes_meva_2026_10_20_sources.json)。
