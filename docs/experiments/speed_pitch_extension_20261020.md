# 速度／音高延伸線（134，預登錄，2026-10-20；2026-10-20 依使用者決定改版）

狀態：**預登錄完成，action 已實作並通過小量 preflight，排 p2/pending/134**（見 §8）。
沿用 126 的 win/tie/lose 框架（Δ = 變體 − clean；|Δ| ≤ m 為平手；通過門檻 80%；m = 0.10 S，S 用現行 m₀ 讀法 = 126a 的 S（5 個 TTM 系統 clean 平均的 range）；MIR 的 S = 本 job 全部 clean 段的 SD）。D2/D3 版 S 未定，不在此套用；逐段分數存檔（`per_clip.tsv`），之後可用任何 S 重判。

## 1. 問題
126／133 的速度與移調只涵蓋 ±2% 速度、移調 ±1。延伸到大幅度：評分指標在多大的速度差內仍判平手、從多大開始判差（交叉點）？MEva 是否「快→高、慢→低」？

## 2. 刺激
- **tempo 倍率（45 格）**：0.10–2.00 每 0.05 一格（39 格）＋ 1 附近細格 0.97／0.98／0.99／1.01／1.02／1.03（6 格）。1.00 = rubberband 恆等對照（rb 實作）。
- **移調**：±1、±2、±5、±12 半音（沿用原設計）。
- **實作**：(a) `rb` 波形層：ffmpeg `rubberband` 濾鏡（tempo／pitch 各自獨立，formant 預設；移調用 `pitchq=quality`），所有來源；(b) `midi` MIDI 層：音符時間 warp／移調後重新合成（GeneralUser），僅 MIDI 來源。
- **來源**：
  1. `maestro`：123 的 150 段 MAESTRO（`midi_realpiece_123/excerpts.json`），每段合成 21 s 長源（rb 路徑變快時才有足夠內容填滿 10 s）；
  2. `piano126b`：126b 的 Mutopia 古典鋼琴 9 首窗（≤16 窗／首，條件同 126b），以固定 seed（crc32）抽 100 窗，窗起點移到第一個音符（避免 ×0.1 取到無音符的 1 s）；
  3. `ttm`：126a 的 5,131 個 prompt 中固定 seed 抽 60 個 × 5 系統（real／ours_cfg3neg／ours_cfg0／musicgen／musicldm）= 300 段，取 115 的 lvl30 clean。MusicGen 為 32 kHz，一律以 soxr VHQ 轉 16 kHz，**每個系統的 clean 底圖 = 寫出的 16 kHz 版**（避免重取樣混淆）。
- 一律 −30 LUFS、16 kHz、FLAC PCM_24。tempo 輸出取變速後訊號的前 10 s（×0.1／×0.25 = 原曲前 1 s／2.5 s 拉長）。**TTM 只有 10 s 源，tempo > 1 的輸出長 10/tempo s（×2 = 5 s）**；為拆開「長度」與「變速」，每個 tempo > 1 格加 `lenctl`：clean 直接截成同長度（無處理）、同樣 −30 LUFS，列為描述對照（`controls`）。MIDI 來源有長源，輸出恆 10 s。
- 規模：maestro 106 clips／段 × 150 = 15,900；piano126b 106 × 100 = 10,600；ttm 76 × 300 = 22,800；合計 **49,300 clips**。

## 3. 指標
AES（PQ／CE／CU／PC）、MEva、CLAP、MIR（pulse_clarity、ibi_cv，符號翻轉成越大越好；MIR worker 同時算 key_cnn_conf／chroma_entropy 並存檔但不報）。不跑 FAD。CLAP 逐檔 batch 1；MAESTRO／piano caption `solo piano music`，TTM 用原 MusicCaps caption。

## 4. 預先登錄的預測（結果前固定）
判定一律 126 的逐對勝／平／負：
| 區 | tempo 倍率 | 預測 | 判定 |
|---|---|---|---|
| 平手帶 | 0.95、0.97、0.98、0.99、1.01、1.02、1.03、1.05（及 1.00 恆等對照） | 小速度差內分數不變 | 平手率 ≥ 80% 為過 |
| 過渡區 | 0.85、0.90、1.10、1.15、1.20 | **不預測**；只報估計值 | 描述 |
| 變差帶 | ≤ 0.80（0.10–0.80，15 格）、≥ 1.25（1.25–2.00，16 格） | 大差異應變差 | 有方向指標（PQ／CE／CU／MEva／CLAP／pulse_clarity／ibi_cv）變差率 ≥ 80% 為過；PC 無品質方向，只讀平手 |

移調（原設計保留）：±1 預期平手；±2、±5、±12 該判差（`該判差` 標籤沿用原 §5；原 §4 預測 ±1／±2 的 PQ 變動 ≈ 0，±5／±12 PQ ≈ 0 至 −0.1，且預期 AES 類看不到——延續 126 音符層全盲；126d／133 顯示 MERT 對 ±1 判不對）。

**要報的量**
- 各指標的**交叉點**：從 1.0 向外走，平手率首次 < 80% 的速度（`tie_exit_low`／`tie_exit_high`，相鄰格線性內插）與變差率首次 ≥ 80% 的速度（`lose_entry_low`／`high`）；cluster bootstrap CI。找不到者報 none。
- 帶層級：平手帶／變差帶的合併率（cluster bootstrap CI）、通過格比例、是否全過。
- MEva 單調性（預登）：tempo 0.5–2.0（0.05 格，rb 31 格、midi 30 格）對 MEva 的逐段 Spearman ρ，群內平均後 cluster bootstrap；預登 ρ > 0 且 CI 排除零 = 「快高慢低」成立；不成立也照報。同樣的 ρ 對其他指標描述性報。
- rb vs midi 同速度的差（`controls` 的 `*_rb_*_vs_*_midi_*`）：rb 與 MIDI 層有差 → 歸處理偽影；`lenctl` 差 → 歸長度；`rb 1.00` 恆等 → 處理本身的底。

## 5. 統計
cluster bootstrap REP=400（群 = MAESTRO 段／Mutopia 曲／TTM prompt〔5 系統共群〕）；n 與 CI 每格報。`piano126b` 只有 9 個群，CI 會寬，如實報。

## 6. 實作與輸出
- action：`scripts/eval/speed_pitch_extension_134.py`（`--preflight`／`--validate-only`／`--analyze-only`／`--limit N --speeds … --semis …` 小量）；沿用 126a／126b 的 S、judge、cell、MEva／AES／CLAP 單元打分，MIR 每單元 CPU 子程序（10 madmom＋4 chroma workers）。
- 輸出：`~/eval_output_nvme/speed_pitch_extension_134_20261020/`（`stimuli/`、`units/`、`per_clip.tsv`、`summary.json`），報告 `docs/experiments/results/speed_pitch_extension_134_summary.json`。刺激（~15 GB）保留在 NVMe，結案後可刪（可由程式確定性重生）。

## 7. Preflight 與資源估算
- 小量 smoke（`--limit 2 --speeds 0.5,1.0,1.05,2.0 --semis 1,12`，138 clips）：輸出 `~/eval_output_nvme/speed_pitch_extension_134_20261020/smoke/`；43 s 全流程跑通；PQ 6.2–8.5、MEva 1.7–4.4、CLAP 0.01–0.53、MIR 皆有限；rb 恆等（tempo 1.00）MAESTRO 與 clean 完全相同（PQ 7.98＝7.98）；TTM 輸出長度 clean 9.98 s、tempo 1.05 → 9.46 s、tempo 2.0 → 4.97 s、lenctl 對應 9.51／4.99 s。
- 全量估算（49,300 clips）：刺激建置（CPU，16 workers）≈ 10 min；GPU 打分以 126b 的 ~1,000 clips/min 估 ≈ 50–60 min；MIR（CPU，與 GPU 並行，以 126b 的 ~290 clips/min@6 workers 按 workers 數放大至 ~450/min）≈ 110 min → **整體約 2–2.5 h（瓶頸是 MIR），上限 < 4 h**。NVMe 峰值 ~15 GB（stimuli）＋ 小量 units；需 free ≥ 30 GB。
- 走 queue 的理由：> 30 min 的全量打分。

## 8. 排隊狀態
- 佇列最大編號為 133 → 本線用 134。contract：`speed_pitch_extension_20261020_contract.json`（action／launcher／harness sha 已綁，`gpu_launch_allowed` true，使用者 2026-10-20 決定改版並要求上線）。wrapper：`scripts/queue_candidates/134_speed_pitch_extension.sh` → 複製到 `~/gpu_queue/p2/pending/134_speed_pitch_extension.sh`。
- 預登錄（本 doc、contract、action）先 commit＋push 再排隊；排隊後不得再改 action（sha 綁定）。


## 9. 10 s vs 5 s 切段（文字草稿，不排隊）
問題：評分以 10 s 段為單位，126 的平手界線 m 是否會因切 5 s 而改變？草稿設計：同一批 clean／變體段各切成前 5 s 與 10 s 兩版，各自算 S 與 m₀、對每介入比較判定標籤一致率；預期 5 s 的段內變異較大 → S 較大、m 較寬、tie 率升高，偵測力下降；若標籤一致率 < 90%，則 m 需依段長各自定義。未登錄門檻，待使用者決定是否立項。
