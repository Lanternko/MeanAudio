# 188：MuQ A1 喜歡什麼聲音——R 臂吸引子的拆解、手工合成與疊加（預登錄設計，2026-11-10）

狀態：設計登錄＋已實作（smoke 全階段通過，2026-11-10）；**尚未排入 p2 188（機器記憶體不足，等協調者排入）**；launcher 在 `scripts/queue_candidates/188_muq_attractor_probe.sh`。
授權：使用者 2026-11-10「把實驗步驟寫成一個 spec，然後交付給 Sonnet 去運行」。

## 0. 為什麼要做

183 的 R 臂（1-NFE、可微 MuQ A1 獎勵微調，λ 0.1、3,000 步）把 Jamendo val 的 A1 MI 從 2.8 推到 10.9。
2026-11-10 會議教授的要求：攻擊要**簡單、直白、講得出做了什麼**（例如後製砍高頻、疊一個單純的聲音），
不要聽不出差別的對抗擾動。使用者聽 R 臂音檔後發現：不論 prompt，輸出都變成同一個「噔噔噔」的中頻喇叭聲。

試聽包（`deliverables/listen_R183/`，64 對，seed 14159265，1-NFE CFG0）的初步量測（不是本實驗的結果，只是動機）：

| 特徵 | control | R |
|---|---|---|
| A1 MI（−30 LUFS） | 2.82 | 10.92（10.25～11.86） |
| A3a MI | — | 4.38 |
| 起伏週期（onset 包絡自相關） | 中位 0.38 s，分散 | 60/64 段 0.160 s、4 段 0.144 s |
| 週期強度 | 0.27 | 0.84 |
| 頻譜峰 | 55–62 Hz（各段不同） | 328–336、391–398、195–211、62 Hz |
| 段與段之間 mel 輪廓 cos（中位） | 0.83 | 0.9993 |
| 逐幀對平均頻譜 cos | 0.79 | 0.96 |
| RMS 變異係數 | 0.55 | 0.12 |
| 諧波比（HPSS） | 0.71 | 0.93 |

也就是 R 把所有 prompt 收斂到**同一個聲音**：約 6.25 Hz 的脈動、能量集中在約 199／332／395 Hz
（比例約 3:5:6，像是 66 Hz 基音的第 3、5、6 泛音；FFT 解析度 7.8 Hz，精確值待量）。

「超過 5 分」的原因已確認，不需實驗：A1 的頭是 `MLP → Linear(·,1)`，輸出沒有 sigmoid／clamp，
只是在 1–5 的 MOS 上做迴歸；A3a 是 5 個 logit 的 softmax 期望值，上限 5，同一批音檔只到 4.38。

本實驗要回答四題：

- **Q1（拆解）** 這個聲音的哪個屬性帶著分數：脈動、音高／頻段、音色、還是穩定不變？
- **Q2（手工合成）** 用一句話講得清楚的合成聲音（幾個正弦波＋固定週期的音量起伏）能重現多少？
- **Q3（疊加）** 把這個聲音當伴奏疊進正常音樂，A1 會不會漲？其他指標跟不跟？
- **Q4（真實音樂）** 真實音樂裡 A1 高的是不是也有這些特徵（R 是「偏好的誇大版」還是「分布外的對抗點」）？

## 1. 共同設定

- 評分器：MuQ-Eval A1（MI 主、TA 附）、A3a（MI、TA）；AES PQ／CE；CLAP（逐檔，對原 caption，僅有 caption 的刺激）。
  全部沿用 183 harness（`scripts/eval/muq_train_exploit_183.py`）的評分與 −30 LUFS 正規化函式，**import 不改檔**
  （183／187 的腳本被 contract 綁 sha）。主讀數 `A1_MI_lvl30`。
- 所有刺激：16 kHz 單聲道、約 10 s、float32，在記憶體裡建好、正規化到 −30 LUFS 再評分；不留全量音檔，只留逐段 TSV 與試聽包。
- 尺度（沿用 187）：S_A1 0.551、S_A3a 0.623、S_PQ 0.914、S_CE 1.19、S_CLAP 0.0714；m_A1 = 0.055；平手帶 = 0.10 S。
- bootstrap：逐段配對、2,000 次、seed 188。
- 資料切分（不可在測試集偷看評分器）：
  - **R 素材**：`deliverables/listen_R183/wav/NN_R.wav` 64 段與同編號 control（Jamendo val prompt）。
  - **dev 真實音樂**：`/mnt/HDD/kojiek/musiccaps_reference` 中**不在** Part B 1,000 的 clip，id 排序後以 seed 188 抽 200 段。
  - **test**：Part B（`H.part_b_ids()`）順序的前 500 個有參考音檔的 id。兩種 base：
    `real`（MusicCaps 真實錄音）與 `gen`（control `…nmv2pair_noq_quarter_s14159265`，標準 25 步 CFG0、同 500 prompt 重生）。
  - 所有參數選擇只看 R 素材、合成聲音本身與 dev；選完寫 `selection.json`（含時間戳與 sha256）後才准評 test，腳本要強制這個順序。

## 2. 實驗內容

### Part A：把吸引子量清楚（描述）

1. R 64 段：高解析度頻譜（≥ 4 s 窗、補零、拋物線內插）量前 24 個分音的頻率與相對振幅；包絡週期；
   以最強 onset 對齊後的平均單一週期包絡；段與段的波形／頻譜相似度。輸出 `attractor.json`（Part C 的合成參數從這裡取）。
2. 跨 seed／劑量軌跡：同 64 個 val prompt、同一份 noise（`scripts/training/run_listen_R183.py` 的做法，generator seed 183，1-NFE CFG0）重生：
   control；R step 3000 的 s14159265、s16180339；pilot λ0.1 的 step 500／1000／1500／2000／2500／3000；
   pilot λ0.3、λ1.0 的 step 3000；187 ladder s14159265 的 step 10／20／40／80／160 與 Z160。
   每格報：A1／A3a／PQ／CE、週期與週期強度、分音附近（±3%）能量占比、段間輪廓 cos、與 R 參考輪廓的 cos。
   回答：不同 seed／λ 是否收斂到同一個聲音；脈動與分音在第幾步出現。
3. 特徵空間：抓 A1 MI 頭的輸入（pooled embedding），比較 dev 真實 200、control 64、R 64 的範數、
   對真實平均的 cos、以真實分布為基準的 z 距離（PCA 前 32 維）。回答 10.9 是不是外插。

### Part B：拆解 R 音檔（64 段 × 約 50 個版本）

每個版本報平均 A1／A3a／PQ／CE 與**保留率** ρ =（A1_版本 − A1_control）／（A1_R − A1_control）。

- 時間結構：0.16 s 切段打亂；0.05 s 切段打亂；倒放；保音高變速 ×0.5／0.71／1.41／2.0；
  包絡壓平（除以 20 ms RMS 包絡）；單一週期鋪滿 10 s；以 R 的長時平均頻譜做隨機相位重合成（有音色、無脈動）；
  把 R 的包絡套到粉紅噪音與同編號 control（有脈動、無音色）。
- 頻譜：移調 ±2／±4／±7／±12 半音（另含 0 半音的處理對照）；低通 500／1k／2k／4k；高通 150／300／500／1k；
  逐倍頻帶挖掉（0–125、125–250、250–500、500–1k、1–2k、2–4k、4–8k）；只留單一倍頻帶（同七帶）；
  把 R 的長時平均頻譜當 EQ 套到同編號 control（只移植音色）。
- 時間占比：前 10／25／50／75% 是 R、其餘接同編號 control（看 attention pooling 是平均型還是極值型）。
- 音量：−40、−20 LUFS（其餘都是 −30）。

預登錄讀法：拿掉某屬性後 ρ < 0.5 → 該屬性**必要**；只移植某屬性到非 R 素材而 ρ > 0.5 → 該屬性**充分**。

### Part C：手工合成（約 190 段，全部是公式產生、可一句話描述）

- 音高組：P1 單音 f5（≈332 Hz）；P2 f3+f5+f6（Part A 量到的頻率與相對振幅）；P3 = P2＋低音 f_low（≈62）；
  P4 十二平均律 C4–E4–G4；P5 單音 1 kHz（教授舉的例子）；P6 = P2 上移 7 半音。
- 音色：純正弦；每個音是鋸齒波（泛音 1/n，到 7 kHz）；鋸齒波過 1.5 kHz 低通（銅管感）。
- 包絡：持續不變；正弦 AM（深度 1.0）週期 0.08／0.12／0.16／0.24／0.32／0.64 s；
  撥奏型（5 ms 起音、指數衰減 τ = 週期/3）週期 0.16／0.32 s；Part A 量到的平均週期包絡。
- 重合成橋接：取 R 的前 N = 3／6／12／24 個分音＋量到的包絡。
- 非音調對照：白噪、粉紅噪、粉紅噪 AM 0.16 s、0.16 s click、0.16 s 與 0.5 s 的 kick loop。

報：每段 A1／A3a／PQ／CE；主效果表（音高組／音色／包絡各自的邊際平均）；最高分的 10 段。
預登錄判定：最佳手工合成的 ρ ≥ 0.5 → `simple_recipe_reproduces`；
否則若高於 dev 真實音樂 A1 的第 90 百分位 → `simple_recipe_partial`；否則 `not_reproduced`。

### Part D：疊加到正常音樂（主實驗）

混音：base 正規化到 −30 LUFS，overlay 正規化到（−30 − r）LUFS，相加後整段再正規化到 −30 LUFS。
r（音樂比伴奏大幾 LU）∈ {+18, +12, +6, 0, −6}。配對差 Δ = 混音 − 同一段 base（−30 LUFS）。

- **選擇（只用 dev）**：Part C 單獨評分最高的 6 段，各自以 r = +6 疊到 dev 200 段，dev ΔA1 最大者 = `O_synth`。
  主檔位 r\* = `O_synth` 在 dev 上 ΔA1 ≥ 0.11（2m）的最大 r（伴奏最小聲的一檔）；都不到就 r\* = 0。寫入 `selection.json` 後凍結。
- **test（500 × 2 種 base）**：
  - 處理：`O_synth`、`O_R`（試聽包中 R_A1 為中位數的那一段）× 5 個 r。
  - 對照 overlay：粉紅噪、1 kHz 正弦、0.16 s click、另一段音樂（dev 第 i mod 200 段）× {r\*, 0}（r\* = 0 時用 {0, +6}）。
  - 指標：A1、A3a、PQ、CE、CLAP 全部；FAD 只算 `gen` base 在 r\* 的三格（無疊加、+`O_synth`、+`O_R`；
    ref `/mnt/HDD/kojiek/musiccaps_reference`，n = 500 只作描述）。
- **預登錄判定**（主格：`real` base、`O_synth`、r\*；`O_R` 與 `gen` base 為次要，同規則各判一次）：
  1. ΔA1 的 CI 下界 < m 0.055 → `no_effect`。
  2. 否則，ΔA1 − 同 r 四個對照 overlay 中最大的 ΔA1，CI 下界 ≤ 0 → `nonspecific`（疊什麼都漲）。
  3. 否則算轉移比 T_X =（ΔX／S_X）／（ΔA1／S_A1），X ∈ {A3a, PQ, CE, CLAP}：
     任一 T ≥ 0.5 → `general_shift`；全部 < 0.5 → `exploit`（A1 被推高而其他指標不跟或下降）。
  另報每個 r 的劑量曲線、`frac_gt5`、聽得出門檻以外的事一律不判（沒有人類評分）。

### Part F：真實音樂的對應（描述）

dev 池以外再抽到共 2,000 段非 Part B 的真實 MusicCaps：A1／A3a 與 Part A 同一組特徵
（0.16 s 附近的週期強度、任意週期的最大週期強度、200–500 Hz 能量占比、逐幀穩定度、諧波比、onset 率、RMS 變異、重心）。
報 Spearman、5-fold 線性迴歸 R²、A1 最高／最低 20 段的 caption 與特徵、R 64 段在每個特徵的真實分布百分位。
預登錄讀法：R 的招牌特徵（週期強度、逐幀穩定度、諧波比、200–500 Hz 占比）在真實音樂上與 A1 的 Spearman
至少三個 ≥ +0.10 → `exaggerated_preference`；否則 `off_manifold`（配合 Part A-3 的特徵距離一起讀）。

## 3. 產出

- `~/eval_output_nvme/muq_attractor_probe_188/`：`attractor.json`、`selection.json`、`per_clip/*.tsv`、`summary.json`、`progress.json`。
- `~/MeanAudio/deliverables/listen_attractor_188/`（gitignored）：`listen.html`＋wav：Part C 前 6 名與 6 個對照、
  2 段 R 的主要拆解版本、6 段 test `real` base × {`O_synth`, `O_R`} × 5 個 r ＋對照 overlay 於 r\*；頁面上標分數。
- 結果文件 `docs/experiments/results/muq_attractor_probe_188_results.md`（job 完成後寫）。

## 4. 執行與成本

- 一支 action（`scripts/eval/muq_attractor_probe_188.py`，分階段、每階段可續跑）＋ guest（照 187 guest，terminal 要帶 `eval_only`）
  ＋ contract ＋ launcher；先在 `--smoke`（每部分各幾段）量每秒評分段數並估總時間。
- 預估刺激數：A 約 1,100、B 約 3,200、C 約 190、D dev 約 1,400＋test 約 19,000、F 2,000，合計約 27,000 段；估 2–4 h GPU。
- 硬上限 6 h。smoke 估超過時依序縮減並記入偏差：(1) test 每種 base 500 → 300；(2) Part F 2,000 → 1,000；
  (3) `gen` base 的處理只留 {r\*, 0}。
- CPU 特徵用 process pool 時，worker 開頭要 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` 並移除 `MKL_NUM_THREADS`。

## 5. 不可寫

- 沒有人類評分：不可寫「疊加後的音樂比較差／一樣好」，只能寫「A1 漲、其他指標不跟（或下降）」。
- Part D 的選擇用了 dev 上的 A1 分數（攻擊者可以離線查評分器），但沒有碰 test；不可寫成「不需要評分器」。
- R 全劑量在 MusicCaps Part B 的正式評分（186）尚未完成（第三個 seed 沒有 checkpoint、186 在 held）；
  本實驗的 R 素材是 Jamendo val 上 seed 14159265 的 1-NFE 輸出，不能代替 186。
- Part B 的變速／移調帶有處理瑕疵，結論以 0 半音處理對照為基準。
- Part F 是相關，不是因果。

## 6. 實作偏差與說明

（2026-11-10 實作者填；判準與門檻未動。）

**D1　rubberband 不可用，用 librosa。** Part B 的變速／移調走 librosa（phase vocoder／resample 版本），處理瑕疵比 rubberband 大，結論一律以 0 半音處理對照為基準（§5 已述）。速度 > 1 的包絡處理以裁切、< 1 以循環拼接補足 10 s。
**D2　重生 float32 vs 磁碟 PCM16。** Part A-2 的 17 個 checkpoint 以 183 的 `R.sample_1nfe`／固定雜訊 seed 183 重生為 float32（不經 PCM16）；Part B／D 用的 64 段 R／control 則直接讀 `deliverables/listen_R183/wav` 的 PCM16 檔（與 183 試聽包同源）。兩者 A1 差異不大（scorer gate 對 `scores.json` 最大差 A1 0.008）。
**D3　R seed 只有兩個有 checkpoint。** `R_s14159265_3000`、`R_s16180339_3000`；第三 seed 沒有（186 在 held），A-2 該臂標 `missing`，不代替 186。
**D4　評分器實作。** MuQ-Eval A1／A3a 以 B=16 批次、GPU `torchaudio` 重取樣 16k→24k、中央裁切／補零到 240000（與 `score_muqeval` 逐檔比對最大差 0.0023）；AES 用常駐 predictor 吃記憶體 tensor（與 `eval_metrics.score_aes` 差 PQ 0.0005）；CLAP 逐檔 batch 1、文字 embedding 快取（與 `score_clap` 差 0.0004）。上線前有 scorer gate（容差 0.05／0.02／0.02／0.002），不過即中止。
**D5　FAD 混音存 PCM_24 flac**（FAD 子行程吃檔案）；其餘刺激全程 float32、不經 PCM16。
**D6　dev 選擇用了 5 檔 r 掃描。** O_synth 候選限在 Part C 的 184 個有調刺激，dev 先以 r=+6 取前 6 名、再對勝者做 r ∈ {+18,+12,+6,0,−6} 掃描決定 r*，dev 刺激約 2,200（spec 估 ~1,400）；dev 與 Part B 不重疊（Part B 前 500 筆之外的隨機排列前 200 筆），`selection.json` 含 sha256 與時間戳，D-test 缺檔或雜湊不符即拒跑。
**D7　`O_R` = 64 段 R clip 中 `ref_R` A1 為中位數的那一段。** P1 取三個主峰（199／332／395 Hz）的中間者。合成長度 159,600 samples（9.975 s）。
**D8　Part B 與 dev 不算 CLAP；AES 只用 PQ／CE（CU／PC 不報）。** CLAP 只在 Part D test 與 FAD。
**D9　週期／特徵定義。** 起伏週期＝onset 包絡自相關第一峰（0.05–1.0 s）；Part F 特徵為 per016（0.16 s 週期強度）、per_max、band200_500、frame_stab、harm_ratio、onset_rate 等；5 折 CV R²。
**D10　smoke 規模與實測速度。** smoke（n_test 24、n_dev 8、n_f 32、n_r 6、regen 4 臂 × 8 段）全階段 69 秒、1,558 個刺激（MuQ 32 s／AES 15 s／CLAP 11 s，約 21–30 刺激／秒含混音）；全量約 27.8k 個刺激 ≈ 25 min 評分，加 regen（17 臂 × 64）、gen base（eval.py 500 段）、FAD 3 格、Part F 特徵，估 1–1.5 h，遠低於 6 h，**§4 的縮減（test 500→300、Part F 2000→1000、gen 只做 {r*,0}）都沒有啟用**。
**D11　未排入 p2。** 2026-11-10 協調者通知機器 MemAvailable 約 8 GB、swap 滿、p2 185 兩度被 OOM 殺掉；規定 MemAvailable < 25,000 MB 不搬進 `~/gpu_queue/p2/pending/`。launcher 留在 `scripts/queue_candidates/`，contract 的 `bindings.launcher` 已指向 pending 路徑（搬入時 sha256 不變）。
**D12　Part A-2 的 regen worker 結束時有 NCCL 未 destroy 警告**，已在程式末尾補 `destroy_process_group()`；smoke 的 17 臂只跑 4 臂，其餘臂路徑相同但未在 smoke 實際跑過。
