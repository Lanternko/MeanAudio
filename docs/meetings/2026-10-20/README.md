# 本週會議錄音與 Whisper 轉錄

收件／整理日期：2026-10-20，Asia/Taipei。實際開會日期未另行確認。

**[主旨、討論重點與後續任務](../2026-10-20_metric_validity_and_next_tasks.md)**

## 保存的內容

| 檔案 | 用途 |
|---|---|
| [recording.flac](recording.flac) | 原 WAV 的無損備份，保留 48 kHz、雙聲道、24-bit；本機保存，依既有 `*.flac` 規則不進 Git。 |
| [transcript.txt](transcript.txt)／[transcript.srt](transcript.srt) | Whisper large-v3 原始帶時間逐字稿／字幕。 |
| [transcript.json](transcript.json) | large-v3 原始 segment、word timestamps 與辨識診斷資料。 |
| [breeze/transcript.txt](breeze/transcript.txt)／[breeze/transcript.srt](breeze/transcript.srt) | Whisper 系列 Breeze-ASR-25 的獨立中文複核轉錄，補查開頭及術語。 |
| [breeze/transcript.json](breeze/transcript.json) | Breeze 原始 segment、word timestamps 與辨識診斷資料。 |
| [transcription_state.json](transcription_state.json)／[breeze/transcription_state.json](breeze/transcription_state.json) | 來源／模型／輸出 SHA-256、設定、版本、執行狀態。 |
| [source_manifest.json](source_manifest.json) | 錄音格式、原始檔案雜湊與無損音訊一致性驗證。 |
| [transcribe.py](transcribe.py) | 本次使用的本機轉錄工具；預設 large-v3，`--breeze` 為中文複核。 |

## 來源與完整性

- 原始附件：`/home/kojiek/.codex/attachments/3c363788-e7ad-432d-b397-caef45510143/Recording at 5-02 PM.wav`
- 長度：**2780.4 秒（46 分 20.4 秒）**。
- 原始 WAV：800,769,286 bytes；SHA-256：`e25e77bee9c9c14af4aacd7bd94fe2a2587fc27950237aa3d4dc237ec324e6f8`。
- 無損 FLAC：445,372,923 bytes；SHA-256：`d296b41450aa24a138dd820cb12bd726a78bc6443668facb1ef64cf46be6032e`。
- 將兩檔解碼成相同 `pcm_s24le` 後，SHA-256 都是 `46ced6b1b9ce5126417a01bd9dc65ab51d3ede0562fc4fb1ee083140f930ed67`，確認備份保留相同音訊樣本。

## 轉錄與判讀方式

使用現有 `/home/kojiek/venvs/whisper`，`faster-whisper 1.2.1`／`CTranslate2 4.8.1`，本機快取的模型，不上傳錄音到外部服務。兩次都使用 beam size 5、VAD、word timestamps，停用前文 conditioning，語言自動辨識為中文。GPU 使用既有 `/home/kojiek/gpu_queue/gpu0.lock`；遇到持有者／既有 compute process 時使用 CPU，不更動實驗 queue。

模型版本：

- `Systran/faster-whisper-large-v3`：`edaa852ec7e145841d8ffdb056a99866b5f0a478`。
- `SoybeanMilk/faster-whisper-Breeze-ASR-25`：`85be11de5f67aaa6a92e931622f1c2b55cc1dd3a`。

兩份逐字稿保持 ASR 原文，**沒有逐句人工聽校**。專有名詞、較輕的發言、播放音樂時的對話與數字有錯字／漏字風險；large-v3 的第一個辨識片段始於 00:26，Breeze 補到 00:08。兩模型也可能有共同錯誤。會議重點以兩份轉錄的上下文整理；不清楚的軟體／模型名稱與數值保留待確認，不把示範數字當成已核定的超參數。

原音有停頓與音樂播放，VAD 不會替所有秒數產生文字。未做說話者 diarization，故沒有替句子加上可辨識人名。相關既有文件只用於確認研究術語與背景，不當作本次錄音的新結果。

本次產出是會議紀錄與待辦，不是新實驗的啟動合約；後續研究計算需照 repo 的正式預登錄與 queue 流程另行處理。
