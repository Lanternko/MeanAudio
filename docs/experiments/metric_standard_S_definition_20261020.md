# 評分標準釐清：S 的定義、替代定義預登錄與敏感度（metric_std_20261020 · A）

> **狀態**：第 1 部分（現行定義、缺點、預登錄）在看任何重算結果之前存檔並 commit（git log 可驗）。第 2 部分（敏感度結果、建議）在重算後才補，不改動第 1 部分。
> 這是**提案**，不取代 126。來源：2026-10-20 會議（ASR 草稿，未聽校；口語數字不採用）。
> 相關：`metric_win_tie_126_20261016.md`、`results/metric_win_tie_126_results.md`、`126c`、`133`、`128`。
> 派工 `20261020-180329-b4b638`；程式 `research/eval/metric_std_20261020/`。

## 1. 現行 S 的精確定義（126／126b／126c 沿用）

- 逐對判定：Δ = score(介入版) − score(同一段 clean)；|Δ| ≤ m 判平手；Δ < −m 判變差；Δ > +m 判變好。通過＝「該平手」類平手率 ≥ 80%、「該判差」類判差率 ≥ 80%。
- **m = 0.10 · S**。
- **神經指標（PQ／CE／CU／PC／MEva／CLAP）**：S = 126a 五個 TTM 系統（MusicCaps real、ours CFG3+neg、ours CFG0、MusicGen、MusicLDM；各 5,131 段、lvl30）**clean 平均值的全距（max − min）**。數值：PQ 0.9189、CE 1.2005、CU 0.8452、PC 0.3447、MEva 0.1724、CLAP 0.1038。
- **MIR 指標（pulse_clarity／key_cnn_conf／ibi_cv／chroma_entropy）**：S = 126b 全部 clean 片段的 SD（0.233／0.247／0.0224／0.0998），↓ 指標存成反號。
- 126c 沿用 126a／126b 的 S；133（MERT／MuQ 餘弦距離）自己定 S = 5 個 TTM 系統平均 cosim 的全距。

## 2. 現行定義的缺點（只列事實性的結構問題，不含對結果的判斷）

1. **S 是 5 個點的全距**：只由 2 個極端系統決定，加／換一個系統就變；n = 5 沒有不確定度估計。
2. **系統間距 ≠ 該指標對人類有意義的差距**：PAM 上 PQ 的系統排序與人類只有 Spearman +0.20（n=5，描述性）；系統間距包含 AES 對 16 kHz 擴散系統的高估，不是人類可辨差。
3. **尺度單位不一致**：神經指標用「系統全距」，MIR 用「片段 SD」，兩者無法互比；MEva（1–5 尺度，逐段 |Δ| p90 0.4–1.0）的 S 只有 0.172，而它的逐段雜訊 > S。
4. **與雜訊底線無關**：m 可能小於同一輸入換推論 seed 就會出現的差（PQ seed 差 0.142 > m_PQ 0.0919），也可能遠大於人類可辨差。
5. **0.10 是任意的**：126 已報 m 曲線（0–0.5 S），但沒有原則說哪一格才對。
6. **對飽和／域外指標不穩**：CLAP 對靜音給高分、PQ 對純音給高分，S 取 TTM 系統範圍而不反映這些域外輸入；126c 的 S 取自 TTM（域外於 MUSDB 混音）。

## 3. 預登錄的替代定義（在看敏感度結果前登錄）

所有定義都輸出「每個指標一個 m」，判定規則（平／好／壞、80% 通過）完全不變，只換 m。為了可比，每個 m 另換算成「現行 S 的倍數」f = m / S_現行。

**D0（現行）**：m₀ = 0.10 · S_現行。

**D1（推論 seed 噪音底線）**：
- 資料：ours CFG3+neg 與 ours CFG0 各 3 個推論 seed（s14159265：p2 115；s16180339／s27182818：p2 118，lvl30，同 prompt id），逐 id 取 3 個 seed 兩兩的逐段差。
- **m₁ = 逐段 |seed 差| 的 80 百分位**（兩系統、三組 seed 對合併）。意義：若同一系統只換 seed 就有 80% 的機率落在 m 內，那 m 以內的差無法與「換個隨機結果」分開。
- 適用：PQ／CE／CU／PC／MEva／CLAP（逐段分數都有）。MIR 在 TTM seed 上沒有，D1 對 MIR 無定義（敏感度表中該欄留空，不用 D0 補）。
- 不使用 memory 中的 0.142（那是 5,521 首平均的 seed 差，不是逐段量）。

**D2（人類 JND 錨定）**：
- 人類 JND_h ＝ **0.5 分（AES 1–10 尺度）**，依據：PAM 上人類差 < 0.5 的配對 AES PQ 一致率 0.464 = 擲硬幣（`aes_human_corr_pam_20260929_results.md`）。MusicEval（1–5 分）按尺度折成 0.5 × 4/9 = 0.222 分。
- 指標差換算：用 **SD 比斜率** b = SD(metric)/SD(human)（total least squares 的對稱版，避免回歸稀釋偏誤）；**m₂ = JND_h · b**。
- 資料：PQ／CE／CU／PC：主要用 PAM 400 首生成音樂（raw）；次要用 MusicCaps 522 首真實。MEva：MusicEval 2,748 首的 MI 欄（對 OVL）。CLAP、MIR：無人類品質評分，D2 無定義（留空）。
- 先登錄預期：PAM 與 MusicCaps 的 b 不同；兩者並列，D2 的主值取 PAM（我們的研究範圍是生成音訊）。

**D3（系統間分布的有效範圍）**：
- 資料：MusicEval 31 個系統的系統平均（`runtime/musiceval_full_20261003/per_system.tsv`；PQ／CE／CU／MEva）。
- **m₃ = 0.10 · IQR（31 系統平均的四分位距）**。保留 0.10 這個比例，只換「5 個點全距」為「31 個點的穩健範圍」，隔離 S 定義的效果。
- 適用：PQ／CE／CU／MEva；其餘指標無 MusicEval 系統資料，留空。

**預登錄的預期（用來偵測程式錯誤）**：
- 因判定是單調的：m 變大 → 「該平手」的平手率只升不降；「該判差」的判差率只降不升。若敏感度表出現反向翻轉，先查程式。
- 翻轉定義：同一個（資料集 × 介入 × 系統 × 指標）格子，在 D₀ 與 Dₓ 下「通過／不通過」不同。
- 梯度級（ladder）翻轉：同一條階梯的「容忍範圍」（最後一個還通過的劑量）或「偵測門檻」（第一個通過的劑量）換了位置。
- 我預期（未看結果）：D1 的 m 對 PQ／CE 大於 D0（seed 底線 > 0.1 S），因此 tie 通過變多、lose 通過變少；D2 的 m 對 PQ 約在 0.1–0.3 之間（人類 0.5 分 ≈ 小於 1 分的 AES 尺度）；MEva 的 D1 m 遠大於 D0。若實際相反，照實報。
- 「建議採用哪個」在重算完後依：資料量（n）、是否有外部錨點、對 126 結論的穩健程度來寫；**不會因為某定義讓 126 的結論翻轉就偏好或排斥它**。

**126b／126c／133 的資料可得性（預先登錄的處理方式）**：
- 126a／126b／126c 有逐段分數（`units/*/scores.tsv`），可用任意 m 重算 → 全做。
- 133 的 MERT／MuQ cosim 只有 tsv 與 S，沒有 seed 重複與人類分數，D1／D2／D3 對它無定義；只報 126 已存的 m 曲線（0–0.5 S）隨 f 的變化。

## 4. 附：B 與 C 的預登錄（同一 commit）

### B. 音量回復對照（`results/volume_recovery_check_20261020_results.md`）

- 素材：各 200 段、固定 seed（`random.Random(20261020)`，從排序後的 id 抽樣）：MusicCaps 真實（126a 的 real lvl30）、ours CFG3+neg（同 id）、MIDI 渲染（126b GeneralUser clean，取自 `items.json` 的 ref）。
- 流程：x（float32 載入）→ 乘 10^(−g/20)，g ∈ {30, 60, 90} dB（float64 運算）→ 以儲存路徑 P 寫檔再讀回 → 乘 10^(+g/20)。P ∈ {float32 WAV、PCM_24、PCM_16 無 dither、PCM_16 + TPDF dither}。每個 g × P 都另存「降低後不回復」的版本（scorer 看到的是小聲音訊）。clean 也經 float32 WAV 讀寫，讓所有評分的檔案 I/O 一致。
- 讀數：波形 SNR（10 log₁₀ Σx²/Σ(x−y)²）與最大誤差；AES（PQ／CE／CU／PC）、MEva、CLAP（MIDI 沒有 caption，不算）；配對差（變體 − clean）的 bootstrap 95% CI（逐段 1,000 次）；以現行 m₀ 判平手率。
- 預期（寫在看結果前）：
  1. float32 路徑 SNR > 120 dB、所有分數差 ≈ 0（|Δ| 在 m₀ 內 ≥ 95%）。
  2. PCM_24 的 SNR：g=30 約 80 dB、60 約 50 dB、90 約 20 dB 量級（原訊號約 −30 dBFS RMS，量化底 −144 dBFS）；g ≤ 60 分數差 ≈ 0。
  3. PCM_16（無 dither）g=60 約 8 dB 量級、g=90 訊號低於 LSB（SNR ≤ 0 dB）→ 回復後明顯劣化；dither 版 g=90 回復後 SNR 也 ≤ 0 dB 但雜訊是白色的而非失真。
  4. 「降低後不回復」的 AES 分數在 float32 路徑也會變（063／065 已見 PQ 隨小聲上升至 −21 dB 後下降）；若回復版 ≈ clean、未回復版 ≠ clean，就歸因於評分器的音量依賴；若回復版也 ≠ clean，歸因於處理損失（量化）。
- 作法：先確認 `nvidia-smi` 與 `gpu0.lock`；若 GPU 被握或 > 30 分鐘，改排 queue。

### C. 統計與證據表（`results/evidence_table_20261020_results.md`）

- 對 126a／126b／126c 逐段分數：每格（資料 × 介入 × 指標）報 n（曲數／片段數／系統數）、配對 Δ 平均與逐段配對 bootstrap 95% CI（以**曲／excerpt 為叢集**重抽，避免同一首多視窗灌水）、跨曲符號一致比例、標準化效果量（Δ 平均 / SD(Δ)）、現行 m₀ 下的平／好／壞比例。
- 對照 ours vs MusicGen 的表 B（PQ +0.155、CLAP +0.0105）與 NegMF E1（+0.95）。
- 三類：**明確問題**＝該判差而 CI 顯示加分（Δ CI 下界 > 0），或應判差的介入在最大幅度仍判差率 < 20%；**爭議／證據不足**＝CI 跨零、n 小、或 CI 不跨零但 |Δ| < m₀；**做得好**＝該判差者判差 ≥ 80% 且 CI < 0，或該平手者平手 ≥ 80%。分類規則在看結果前固定如上。
- 126b 代表性：從 `items.json` 與腳本列出曲目數與選曲方式；缺口只列事實與建議，不下「無效」結論。

## 5. 取得資料的位置

- 126a：`~/eval_output_nvme/metric_win_tie_126a_20261016/units/<variant>/<system>/scores.tsv`（id, PQ, CE, CU, PC, meva, clap, over_fs）
- 126b／126c：`.../metric_win_tie_126{b_20261016,c_20261019}/{units/*/scores.tsv, stimuli/items.json, mir.json}`
- TTM seed：`ttm_metrics_115_20261009/meanaudio_{cfg0,cfg3_neg}/lvl30/{scores/*/per_clip.tsv, meva/meva_per_clip.tsv}`、`ttm_metrics_seeds_118_20261011/meanaudio_s{16180339,27182818}_{cfg0,cfg3_neg}/lvl30/...`
- 人類：`research/eval/output/aes_human_corr{,_pam}/per_clip.tsv`；MusicEval：`runtime/musiceval_full_20261003/{per_clip,per_system}.tsv`
