# 響度、RMS、crest 與 AES：逐音檔分組及配對增益實驗

2026-09-11。Operator instruction:「設計實驗並加入 queue」。新實驗 append-only，P2 051，接在 050 後面；沒有訓練或自動 promotion。

## 問題與固定設計

1. 同一 checkpoint 的 eval 內，LUFS 較高的音檔，其 AES PQ 是否較高？RMS 與 crest 的關聯如何？
2. 同一音檔只改增益，AES 是否改變？這回答評分器對輸入音量的敏感度，不等於人類品質偏好。

Checkpoint：c2p0 slot0 / phase8_qwen_caption10s_multisent_noq_full_stage2_200000 EMA。選擇它是為延續現有對照，不依本次結果挑模型。
完整 MusicCaps 5,521，原 TSV 順序與 caption，MeanFlow 25，literal CFG3，fidelity8，seed42，NoMask，full precision，NoQ。
固定 fidelity8：`low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi`。
生成一份新且獨立的 canonical baseline；不假定歷史 audio 已保留，也不重標 CFG0/4.5。只 baseline 的原音訊可作 canonical 結果；−3/−6/−9 dB 是有獨立標籤的 secondary gain intervention。

## 訊號與評分

所有 5,521 首均計算 integrated LUFS（pyloudnorm K-weighting、400 ms gated meter）、RMS dBFS、crest_linear=peak/RMS、crest_db=20log10(peak/RMS)、sample peak dBFS、abs(sample)<0.001 的比例、abs(sample)>=0.999 的比例、spectral centroid。
`silence_fraction` 是樣本閾值統計，不是音樂停頓的聽覺標註；`clipped_fraction` 是接近滿刻度樣本比例，不足以證明硬削波。
舊版 `negprompt_ablation_matrix.py` 的 crest 是線性比值的逐音檔算術平均，部分 aggregate 只用檔名排序前 256 首；本實驗用全部樣本，CSV 保留兩種 crest。
零訊號 RMS/crest 與 gated LUFS 非有限值保存 null 與原因，不填 0。無法取得某個指標的樣本只排除該指標的分組/相關分析，其 AES 與配對增益仍保留；報告排除 ID。音檔缺失、重複、非有限波形或 AES 一律 fail closed，不做靜默完整案例刪除。

每首增益 0、−3、−6、−9 dB。全部從相同解碼原音檔乘固定純量後寫 FLOAT WAV（包括 0 dB），不 limiter、不壓縮、不正規化、不新增 PCM 量化。
增益與 crest 的數值驗證容忍度 1e-5 dB；0 dB 浮點轉檔須逐樣本相等。scorer 波形載入和 padding 後仍須保有倍率（atol1e-7、rtol1e-6）；模型 cfg.normalize 必須 false。
AES 使用固定 local snapshot、相同 batch size32、同一 TSV 順序，四項 CE/CU/PC/PQ 全記錄；baseline 額外 CLAP。單一 generation seed；bootstrap seed20260911，不解讀成 training/generation seed 穩健性。

## 預註冊分析

- 每個 LUFS、RMS dBFS、crest dB 各自算 20/40/60/80 百分位切點。相等數值留同組（落低側），組數固定五組、空組顯示；因此近似等人數，不強制拆開 ties。
- 每組報 n、範圍、四項 AES mean 與 pointwise95% percentile bootstrap CI，另報 Q5−Q1 與連續 Spearman rho。
- 觀察關聯 primary：LUFS 分組 PQ 的 Q5−Q1。固定觀察到的分組切點，各組內獨立重抽樣；CI 條件於此切點。
- 音量介入 primary：PQ(−6 dB)−PQ(0 dB)，以音檔 ID 配對重抽樣。−3/−9 dB 與其他 AES 為 secondary/exploratory。
- 每個 baseline LUFS 組內再分 crest 三組，報 high−low AES 與 CI。這是粗分層的描述性調整，組內仍有殘留響度與內容混雜，不宣稱 crest 因果效應。
- baseline LUFS 五組內亦報各增益的配對 AES delta，以固定原始分組避免介入後重分組。
- 全部 bootstrap 10,000 次，95% pointwise CI；不做全域顯著性/多重比較勝者宣稱。CI 跨零是證據不足，不是等效；未設等效界值，不宣稱「沒有影響」。不依結果變更 endpoint。
- 圖：LUFS vs 四項 AES（含五組 mean/CI）；增益 vs 配對 AES delta。CSV、JSON 保留全部分組與 CI。

不同曲風、瞬態、頻譜、製作品質可能同時影響 LUFS/crest 與 AES。只改增益能隔離 waveform level intervention；神經網路內部尺度不變性仍屬實驗結果的一部分。單一模型/seed 的結果不外推全部模型，不把衰減結果外推到放大或壓縮。

## Queue、容量與恢復

機器合約：`loudness_aes_cfg3_20260911_contract.json`；HARN bundle：`harn/loudness_aes_cfg3_20260911/`。
生成 5,521，AES 22,084 clip-condition scores，CLAP 5,521。預留 peak additional 6GB，保留 baseline FLAC、逐筆 JSON、CSV、圖與報告；增益 WAV 僅 batch32 暫存、逐筆成功落盤後移除。寫入 NVMe private0700 root 與自己的 logs/receipt state，不清理共享 cache 或 checkpoint。
磁碟 hard floor=max(50GiB,1.25*6GB)，warning80GB；每 batch/phase 與 independent supervisor 重測。估計 GPU 1.5–3 小時（baseline MF25 約半小時，再加四輪 AES 與 CLAP；是排程預估，非完成保證），CPU 分析預估 <30分鐘；排队/P1搶占時間另計。
不新增 scientific dependency；順序由 P2 檔名 051 保證。失敗/中斷由既有常駐 top-level host 送終態並掃描後續；storage wait 保持 pollable。啟動需 exact P2 seat、process identity、launcher/action/HARN/input hashes、schema、approval 與通知通過。
生成中斷且沒有完整音檔 manifest：只清本 run 的 partial FLAC，從原 TSV 順序 seed42 全部重生，避免跳過已存在檔案破壞 RNG。完整生成有 manifest：逐檔驗 hash 後重用。AES 每筆原子提交且綁 contract/source/gain；中斷只補未完成項目。CLAP 中斷重算 CLAP，report 可確定性重建。保留已送通知 receipts，不重送。

每個 generation/AES/CLAP/report gate 與 start/hold/interruption/failure/completion/handoff/storage/stall 由既有 receipt helper 冪等通知 Discord。通知結果不明 fail closed、不盲目重送。Independent parent 每2秒零模型輪詢；7200秒無音檔或逐批進度變化停止自有 child 並通知 hold。無自動修復權限。

舊 host terminal classifier 要求 `ema/cfg0_report`，因此沿用既有只作 scheduler compatibility 的 historical evidence；科學完成必須另外經本實驗 postflight 驗證全部 5,521×4 逐筆、canonical 協定、CLAP、報告與 retained audio hashes，絕不把 CFG0 作 comparator。

參考實作：[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)、[Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)。部署版本與程式 raw hashes 以本實驗合約為準。

## Acceptance 注意事項

舊 bare-guest queue fixture 有 idle flag 重設競態（極短工作可在兩次 poll 中間起迄）。051 supervisor 在驗證 exact seat 後明確重設 queue-owned idle marker；專屬 fixture 用同一 production function 驗證重新 idle 的通知，沒有修改或重啟共享 host。
測試另隔離 `GPU_NOTIFY` 與 `GPU_NOTIFY_QUEUE_STATUS` 兩個通知路徑。最初直接執行 legacy fixture 誤送了兩則 idle 通知；它們不是 live queue 狀態，正式註冊通知會更正。
