# 2026-10-27 會議：指標海選延伸、MERT／MuQ、攻擊與修補方向

> **來源**：錄音 `~/Downloads/Recording at 4-52 PM.wav`（989 MB，md5 `0f4f4005e4480335c3236e373253c508`，57:15）。
> - **會議日期是推定的**：檔名只有時刻，檔案 mtime 是 10-28 15:30，比 16:52 早，所以錄音是前一天或更早。內容接續 10-27 的 full matrix 與 r3 Arena 閘門，因此推定為 10-27。
> - 前 0:00–7:56 是環境音：無 VAD 重跑也只產出幻覺字，不是遺漏。實際發言 07:56–57:15，約 49 分鐘。
>
> **逐字稿**：主檔是 Breeze-ASR-25 的 `~/tmp/asr_jobs/20261027/meeting_breeze.{txt,srt,json}`（faster-whisper，16 kHz mono、loudnorm、VAD、hotwords）。交叉核對用 large-v3 的 `meeting_v3.{txt,srt,json}`。
> - **ASR 錯字很多**，常見誤認：「深層音樂」= 生成音樂、「訊號列環」= 訊號劣化、「MILK／Muq」= MuQ、「Merton」= MERT、「cloud」= Claude、「分軌邊際／電子」= 分軌編輯。
> - 下文標 **〔ASR 不確定〕** 的地方沒有猜，引用原轉寫。
> - 時間戳是 `[HH:MM:SS]`，對 Breeze 檔。
>
> **出席**：教授、Lanternko（報告）。其他人無法從錄音辨識。
>
> **報告素材**：r2 海選延伸＋full matrix（`../experiments/results/benchmark_screening_full_matrix_20261027.md`）、兩條要求各線（126／133／134）、161–177 metric coverage 補格（`../experiments/metric_coverage_completion_20261027.md`）。

---

## 1. 報告內容（07:56–27:30）

- 海選延伸：新加兩個 2026 年的評分模型。
  - MuQ-Eval（A1／A3a）。
  - CMI-RM（「CMI reward bench」，11 次引用）。教授提醒不要簡稱成別的東西，CMI-Bench 是另一個東西〔08:19–08:32〕。
- 對照的公認人類評分 dataset 分三塊：生成音樂（6 個）、真實歌唱、訊號劣化。
- 判讀方式：
  - 分數是和人類評分的相關，表上標出人類天花板（`../experiments/results/benchmark_screening_r2_human_ceiling_20261027.md`）。
  - 灰格是評分模型自己的訓練集，不算分。
- 生成音樂：
  - MuQ-Eval 表現好，有三個綠格。
  - MEva 只在自己那包好。
  - 這幾個專做生成音樂的模型在其他區塊較不在行，**AES 反而比較全面**〔09:32–10:24〕。
- **Music Arena：所有指標都沒過**〔10:37〕。見下方指示 1。
- 兩條要求（分出好壞／說得出平手）的進度：
  - MERT-ref 表現不錯。
  - MuQ 抓得到錯音（音高被破壞處）。
  - CMI 還沒跑完。
  - MEva 較差。
  - AES 在旋律（錯音）上 PQ／CE 都抓不到〔19:08–20:05、26:21〕。
- 很多格還在等 GPU，還沒排到（NA = not available）〔16:14、24:32〕。

## 2. 教授的回饋與指示

1. **Arena 改看勝率／預測一致率，不看相關**〔11:03–13:19〕。
   - 原因：相關是以音檔為單位算的。Arena 裡每個音檔只跟 3–4 個對手比過，Elo 式分數不會收斂（教授用下棋比喻）。
   - 做法：對每個 battle，用指標分數預測哪邊贏，算預測對的比例。
   - Lanternko 提出事前登錄門檻 **55%**〔13:44〕。教授說沒問題，但**不會把 Arena 拿掉**，要看實際值多少〔13:53–14:08〕。
   - 對應 memory `project_bench_r3_arena_agreement_gate.md`（10-27 已定，x 依人類天花板）、`../experiments/results/benchmark_screening_r3_candidate_datasets_20261027.md`。
2. **投影片上的「勝率比例」分數調整不對**〔14:39–15:03〕。
   - 例：打三場贏兩場就把分數乘上去。這樣抽到爛對手的人會被高估，Arena 本來就是這樣。
3. **訓練重疊的模型不必一律排除**〔15:29–15:48〕。
   - MEva 雖然在那些 set 上訓練過，在那些 set 做得好仍可參考，後面的兩條要求才是重點。
   - **可以拿掉不重要的維度，例如 PC**〔15:55–15:58〕：「PC 就沒有用了」。
4. **GPU：新的 RTX 5080 電腦沒人在用，可以開給 Lanternko**〔16:18–16:43〕。
   - 要先想資料怎麼搬過去。
   - 環境由對面實驗室的負責人處理〔ASR 不確定：「對面他們的負責人」〕。
5. **S 的定義**〔16:50–18:57〕。
   - 現行 S 是 MusicCaps 上的全距，m = 0.10·S。MusicCaps 不是人類平均評分的 dataset、不是生成音樂、好壞難定義（`../experiments/metric_standard_S_definition_20261020.md`）。
   - 教授：現行做法**可以先接受、先試**，但不確定 0.1 合不合理。
   - 他**更偏向顯著性檢定**：確定這一包跟那一包有沒有差。
   - 或者每個 dataset ad hoc 定義、做調整後再算〔18:38–18:52〕。
   - Lanternko 補充：約三分之一的點沒落在範圍內〔18:57〕。
6. **MERT-ref 要講清楚是什麼**〔20:13–22:07〕。
   - 它不是評分指標，是類 BERT 的 SSL 音樂模型。
   - 要寫清楚：是否沒訓練、是誰提的（哪篇）、參考是什麼（「你的真實影響是誰」）。
   - 投影片上 chroma 怎麼算、key 模型是什麼等細節也不清楚。原話：「Claude 做得當然很漂亮，可是那怎麼做啊」〔21:50–22:07〕。
   - 對應 `../experiments/metric_win_tie_133_mert_20261020.md`、`../experiments/results/metric_win_tie_133_results.md`。
7. **如果 MERT 這種用法真的沒人做過，本身就是發現**〔22:13–23:00、28:16–30:14〕。
   - 發現內容：未訓練的 SSL embedding 跟真實音檔的相似度，robustness 反而好。
   - 合理的原因：預訓練見過大量真實音樂，錯音這種事不會出現在真實音檔裡。
   - Future work：與其在 task-specific data 上 fine-tune classification head，不如用 SSL 相似度／metric learning。
   - 理由：AES 只訓練在幾十萬筆上；人類是聽過大量音檔才有「這怪怪的」的知覺。
8. **需要實際數字，不只通過率**〔23:13–24:11〕。
   - 例：MuQ 對 mp3 壓縮判不了平手、給負向；SongEval 在某格顯示 0% 全沒過。
   - 這是二維的：哪個介入 × 是否通過，還要看變好或變差的方向。
   - 沒有實際數字，看不出 0.1 或 0.12 是否合適〔23:56–24:05、50:40–50:47〕。
9. **名詞要定義到別人同意**〔24:14–25:46〕。
   - 「分軌編輯」是什麼。
   - 「訊號小差／音樂小差」的「小」是多小。
   - SNR 階梯（40／30／20／10）是否本來就該判差。
   - 投影片要加註記。對應 `../experiments/results/metric_win_tie_126c_results.md`、`../experiments/results/wide_fail_list_full_20261022.md`。
10. **對「AI 做的 paper」的批評**〔27:30–28:11、34:13–35:15〕。
    - 2026 年很多 paper 長這樣：很漂亮、reviewer 掃過就放行，但仔細看就站不住，「我自己都不相信什麼叫做分軌編輯」。
    - 每一格都要能回答「為什麼這個應該打叉」。
11. **錯音不是訊號劣化，一定要判**〔ASR 不確定，約 26–27 分附近的討論〕。
    - 問題：新的 2026 模型在生成音樂 dataset 上很好，是不是表示訊號劣化不重要？
    - 回答：音樂層的錯（錯音、亂排）不屬於訊號劣化，仍要判。
    - 補充：人類對 5 級噪音階梯的敏感度約 0.99〔33:25「99.5」，ASR 不確定確切數字〕。

（35:40–48:20 主要是閒聊，未記錄：AI 證明 SAT、審稿義務、博士生要審稿、台大交通車、大學排名、中研院／南港通勤等。）

## 3. 新方向（48:20–55:30）

1. **先查每個模型是否還有明顯弱點，最強的 MuQ 先查**〔48:52–49:57〕。
   - **MuQ 全都過**：工作變成分析型。
     - 要解釋 MuQ 為什麼好（是否針對這類資料訓練）、MERT 為什麼好。
     - 結論：只看 dataset 相關不能保證可信；SSL／參考式方法在相關上不一定最好，但對人類重要的不變性與改變處理得好。
   - **還有弱點**：進入攻擊 → 修補。教授說「還不行我們才有事可以做」，目前都在做「把指標再算一次」的苦工〔50:00–50:15〕。
2. **可被攻擊是比「該變差沒變差」更強的結論**〔50:20–51:52〕。
   - 即之前說的三步：攻擊 → 揭露 → 修補。
   - 例：加一點噪音或調一下音量就讓分數變好、PQ 變好。
   - 能被攻擊，就代表訓練 TTM 時可以硬「刷」指標，而且刷得很好。
   - 現有例子：**CFG3 就能攻擊 PQ 和 CE**〔51:49〕，對應 NegMF／081 的 PQ 導向結果（`../experiments/results/negmf_full_120_results.md`、`../experiments/results/quality_label_prefix_081_results.md`）。
3. **修補提簡單方法**〔51:24–51:44〕。
   - data augmentation：該變差的就讓它變差。
   - pairwise／不變性訓練：不管絕對分數，這個要比那個好，或這兩個不准變。
4. **重新列介入清單**〔52:00–52:24〕。
   - 之前列了二十幾個，現在投影片只剩 7 類，要補回來，例如換鋼琴音色。
   - EQ（調音）很微妙，因為可能真的讓聲音變好。
5. **建議加低通（lowpass）**〔52:27–53:39〕〔ASR 不確定，轉寫成「Lopez」〕。
   - 截在 12k 或 8k，應該大幅降低品質。
   - 內部取樣率低的模型（AES 是 16 kHz）會被這個搞死。
   - 教授提到某個模型取樣率「應該是 10k」〔ASR 不確定：轉寫成「Word」，原意不明。MERT 實際是 24 kHz〕。
   - 已有的覆蓋：126 已有 lowpass 4k，PC 最強、CE 只在 1 kHz 才過（`../experiments/results/metric_win_tie_126_results.md`）。缺 8k／12k，以及「我們自己 16 kHz 生成本來就只到 8k」這個角度。
6. **讓 Claude 用觀察的方式找 MuQ 漏洞**〔54:21–55:12〕。
   - 教授說這是 **borderline OK**，可以做：套一些簡單變換（不該變的、該變的），觀察分數變化，例如「高頻很強分數就高」，跟人類觀察差不多。
   - **不可以**做 adversarial attack／adversarial training，那會把模型搞死。
7. **時程**〔55:25–55:52〕。
   - 若 MuQ 真的都過，就慢慢 polish。
   - **三月投稿**〔ASR 不確定：Breeze 寫「三月再去同意十一年」、large-v3 寫「3月再去投一十個月」，會議名稱聽不出〕。
   - 這段期間先做產學合作（**資策會**），賺點外快，也讓履歷有產學經驗〔55:40–56:07〕。

## 4. 決定

| # | 決定 | 依據 |
|---|---|---|
| D1 | Arena 保留，判定改看預測一致率（事前登錄 55% 或依人類天花板），不看逐檔相關 | 指示 1；memory `project_bench_r3_arena_agreement_gate.md` |
| D2 | S（MusicCaps 全距 × 0.10）先沿用試跑；最終傾向顯著性檢定或每 dataset 各自定義 | 指示 5 |
| D3 | 拿掉不重要的維度（PC） | 指示 3 |
| D4 | 下一步主線：查 MuQ（及其他模型）的弱點 → 有弱點就攻擊＋簡單修補，沒弱點就寫分析 | 方向 1–3 |
| D5 | 找漏洞只用觀察式 probe，不做對抗攻擊／訓練 | 方向 6 |
| D6 | 目標三月投稿（會議待確認） | 方向 7 |

## 5. 待辦

| 負責 | 事項 | 相關文件 |
|---|---|---|
| Lanternko | 申請並搬資料到 RTX 5080 機器，跑還在等 GPU 的格；環境請對面負責人處理 | `../experiments/metric_coverage_completion_20261027.md` |
| Lanternko | Arena 一致率實算（事前登錄門檻），報實際值 | `../experiments/results/benchmark_screening_r3_candidate_datasets_20261027.md` |
| Lanternko | 拿掉投影片的「勝率比例」分數調整 | 指示 2 |
| Lanternko | 寫清楚 MERT-ref：方法（未訓練、cosine 對參考均值）、提出者／出處、參考集是什麼；chroma／key 模型等細節補說明 | `../experiments/metric_win_tie_133_mert_20261020.md` |
| Lanternko | 文獻查核：「SSL embedding 對參考相似度」當評分是否已有人做 | 指示 7 |
| Lanternko | 投影片改報實際數字（介入 × 指標的分數差與方向），不只通過率 | `../experiments/results/wide_fail_list_full_20261022.md` |
| Lanternko | 定義分軌編輯、訊號小差／音樂小差的幅度、SNR 各級，並加註記 | `../experiments/results/metric_win_tie_126c_results.md` |
| Lanternko | 把介入清單補回二十幾項（含換音色、EQ 要註明可能變好），加 lowpass 8k／12k | `../experiments/results/metric_win_tie_126_results.md` |
| Lanternko（＋Claude） | 觀察式 MuQ 漏洞 probe（例：高頻增強、簡單不該變／該變的變換），不做對抗優化 | 新線，待設計 |
| 教授 | 資策會產學合作的安排 | 方向 7 |

## 6. 未解問題

- 投稿目標會議與確切 deadline（ASR 聽不出）。
- S／m 的最終定義：顯著性檢定要對什麼分布做？每個 dataset 各自定義時，怎麼避免事後挑門檻？
- Arena 門檻用 55% 還是依人類天花板換算（memory 傾向 ≥ 0.55）？
- MuQ 對 mp3 壓縮給負向，是 MuQ 的漏洞還是壓縮真的可聞？要不要等人類 JND pilot（`../experiments/human_jnd_pilot_20261019.md`）？
- 教授說的「sample rate 應該是 10k」指哪個模型（ASR 轉成「Word」）？
- 若 MuQ 全過，分析型論文的貢獻是否足夠（教授傾向夠）？
