# 速度／音高延伸線（134，預登錄，2026-10-20）

狀態：**預登錄；action 腳本尚未實作，launcher 只在 `scripts/queue_candidates/`，未進 `~/gpu_queue/p2/pending/`**（見 §8）。
沿用 126 的 win/tie/lose 框架（Δ = 變體 − clean；|Δ| ≤ m 為平手；通過門檻 80%；m = 0.10 S，S 定義見 `metric_standard_S_definition_20261020.md`，同時報 D0 與 D3）。

## 1. 問題
126／133 的速度與移調只涵蓋 ±2% 速度、移調 ±1。延伸到大幅度：評分指標對「變快／變慢」與「升降調」的單調性與平手界線在哪？MEva 是否「快→高、慢→低」？

## 2. 刺激
- tempo 倍率：0.1、0.25、0.5、0.7、0.8、0.9、1.1、1.25、1.5、2.0。
- 移調：±1、±2、±5、±12 半音。
- 兩種實作對照：(a) 波形層 rubberband（tempo／pitch 各自獨立、formant 保持預設）；(b) MIDI 層（改音符時間／音高後重新合成，僅 MIDI 來源）。
- 來源：
  1. MAESTRO 150 段（沿用 123／124 的選段清單）；
  2. 126b 鋼琴 MIDI（沿用 126b `select()` 的固定 seed 抽樣，抽到的 piece 清單寫入 manifest）；
  3. TTM clean 段（5 系統 clean，沿用 126a 的 5,131 段子抽樣，固定 seed，抽樣清單寫入 manifest）。
- 一律 −30 LUFS、16 kHz，每段重新裁成 10 s（tempo 變更後長度改變，取前 10 s；×0.1／×0.25 為原曲前 1 s／2.5 s 拉長，已於 manifest 標記）。

## 3. 指標
AES（PQ／CE／CU／PC）、MEva、CLAP、MIR（pulse_clarity、ibi_cv）。不跑 FAD。

## 4. 預先登錄的預期方向（結果前固定）
| 介入 | PQ／CE／CU | MEva | CLAP | pulse_clarity | ibi_cv | 判定預期 |
|---|---|---|---|---|---|---|
| tempo 0.9／1.1 | ≈0（平手） | ≈0 | ≈0 | ≈0 | ≈0 | 應平手（±10% 屬可接受變異；126 以 ±2% 為平手，此處屬新增檔位，不預期必過） |
| tempo 0.7／0.8／1.25／1.5 | 小幅變動 | **快高慢低**（單調） | ≈0 | 隨速度上升 | 不變 | 此為音樂內容的合法差異，判定標籤見 §5 |
| tempo 2.0 | 不確定 | 高於 clean | ≈0 | 上升 | 不變 | 判差 |
| tempo 0.5 | 不確定 | 低於 clean | ≈0 | 下降 | 不變 | 判差 |
| tempo 0.25／0.1 | **預期 PQ 不下降，甚至上升**（理由：極慢拉長使音符變成持續音／質地，頻譜穩定、無瞬態與節奏、AES 對乾淨持續音偏高；006 單音 PQ 中位數 7.62 勝過 91% 真實錄音；rubberband 的拉長偽影可能反而讓 CE 下降） | 低 | CLAP 下降（音樂性描述失配） | 大幅下降 | 不穩定 | 真的是「音樂壞了」，若 PQ 判平手或上升，記為指標盲點 |
| 移調 ±1／±2 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | PQ 預期平手；126d／133 顯示 MERT 對 ±1 判不對 |
| 移調 ±5／±12 | PQ ≈ 0 至 −0.1 | ≈0 | ≈0 | ≈0 | ≈0 | 音高在整體功能上是大變，但預期 AES 類看不到（延續 126 音符層全盲） |
- MEva 單調性檢定：tempo 倍率（0.5→2.0 八檔）對 MEva 的 Spearman ρ（群 = 來源 clip），預登 ρ > 0 且 CI 排除零 = 「快高慢低」成立；不成立也照報。
- MIDI 層 vs rubberband 層的差：若 rubberband 偽影造成 PQ／CE 變化而 MIDI 層無，則歸「處理偽影」而非「內容」。

## 5. 判定標籤
tempo ≤ ±10% 與移調 ±1 預期「該平手」；其餘一律「該判差」（偵測類）。極端（×0.1、×0.25）若被判平手，列「明確問題」。

## 6. 統計
cluster bootstrap（群 = piece／clip）REP=400，與證據表相同；n 與 CI 每列報。

## 7. 預估資源
≈ (10 tempo + 8 pitch) × 2 實作 × (150 + 估 ~200 + ~200) 段 ≈ 2 萬段／指標，AES＋MEva＋CLAP＋MIR 約 3–4 h，< 30 min 的只有抽樣測試 → 走 queue（p2/pending/134）。

## 8. 排隊狀態與未完成
- 佇列最大編號為 133 → 本線用 134。
- `scripts/eval/speed_pitch_extension_134.py`（刺激產生＋評分 action）**尚未寫**；contract 的 action／launcher sha 因此為 `UNBOUND`。要排隊前須：實作 action、`--preflight`／`--validate-only`、填 sha、使用者核可，再 `cp scripts/queue_candidates/134_speed_pitch_extension.sh ~/gpu_queue/p2/pending/`。
- 不依賴 GPU 結果的文字：見 §9。

## 9. 10 s vs 5 s 切段（文字草稿，不排隊）
問題：評分以 10 s 段為單位，126 的平手界線 m 是否會因切 5 s 而改變？草稿設計：同一批 clean／變體段各切成前 5 s 與 10 s 兩版，各自算 S 與 m₀、對每介入比較判定標籤一致率；預期 5 s 的段內變異較大 → S 較大、m 較寬、tie 率升高，偵測力下降；若標籤一致率 < 90%，則 m 需依段長各自定義。未登錄門檻，待使用者決定是否立項。
