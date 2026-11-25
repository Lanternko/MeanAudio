# 證據表結果（metric_std_20261020 §4C）

分類規則在結果前固定於 `docs/experiments/metric_standard_S_definition_20261020.md` §4C。
逐列資料：`research/eval/metric_std_20261020/evidence_rows.tsv`（4,354 列，同內容 `evidence_rows.json.gz`）。
欄位：ds、cell（來源|操作|類型）、step、system、kind（lose/tie）、metric、cat、why、n、ncl、mean（Δ = 變體 − clean）、lo/hi（cluster bootstrap 95% CI，REP=400，群 = 126a clip／126b、126c piece）、dz、tie（|Δ| ≤ m₀ 比例）、worse、better、sign、m0。
腳本：`evidence.py`（產列）、`evsum.py`／`evsum2.py`（彙總）。

## 1. 三分類總數

| 資料集 | 做得好 | 明確問題 | 爭議 |
|---|---|---|---|
| 126a（訊號層，5 TTM 系統） | 484 | 42 | 884 |
| 126b（鋼琴 MIDI／真人／童謠） | 781 | 206 | 1175 |
| 126c（MUSDB 分軌） | 84 | 12 | 298 |

依指標（做得好／明確問題／爭議）：
- 126a：PQ 142/0/118、CE 123/0/137、CU 133/0/127、PC 31/0/79、CLAP 49/13/198、MEva 6/29/225
- 126b：CE 185/53/104、CU 177/53/112、PQ 183/41/118、PC 53/0/71、CLAP 121/19/202、MEva 9/21/312、pulse_clarity 30/6/46、ibi_cv 13/1/68、key_cnn_conf 3/2/77、chroma_entropy 7/10/65
- 126c：PQ 15/0/38、CE 17/0/36、CU 18/0/35、PC 11/0/22、CLAP 13/5/35、MEva 0/2/51、pulse_clarity 4/3/17

「爭議」佔多數，主因是 CI 不跨零但 |Δ| < m₀、或判定率介於 20%～80%。

## 2. 旗標

### 2.1「加雜訊但分數上升」（lose 類、CI 排除零且方向相反）：406 列
- 126a：PC 36、CLAP 13、MEva 29
- 126b：CE 51、CU 52、PC 136、PQ 31、chroma_entropy 8、CLAP 14、ibi_cv 1、key_cnn_conf 2、MEva 21
- 126c：PC 5、CLAP 5、MEva 2
- 最大者全是噪音階梯上的 PC：SNR 0–10 時 csd_real／pro／poor／midi 的 PC +1.0～+1.57（噪音被 PC 當成內容複雜度）。
- 非 PC 的最大例（不含 pulse_clarity）：126b `poor_sing` clicks20ps／5ps／1ps MEva +0.20／+0.19／+0.12（CI 下界 +0.09／+0.08／+0.06，m₀ 0.017）；`midi_fluidr3` extra30pct CE +0.170 [+0.134, +0.207]；`midi_generaluser` blastnote_x1 PQ +0.155 [+0.143, +0.166]、blastnote_x3 PQ +0.143；posshuffle100pct CU +0.13；`poor_sing` noise_snr25／30 CE +0.118／+0.103。
- 這些都是「該判差、分數卻變高」，已全數歸「明確問題」。

### 2.2「CI 不跨零但 |Δ| < m₀」：262 列
- 126a：CE 20、CU 11、PQ 11、CLAP 40、MEva 7
- 126b：CE 24、CU 14、PQ 16、chroma_entropy 33、CLAP 33、ibi_cv 4、key_cnn_conf 6、pulse_clarity 4
- 126c：CE 8、CU 5、PQ 5、chroma_entropy 9、CLAP 7、ibi_cv 3、pulse_clarity 2
- 解讀：統計上可測、實質上低於 m₀ 的偏移，不應被算成偵測成功也不應算成平手失敗，列「爭議」。

### 2.3「最大幅度判差率 < 20%」的明確問題（偵測不到的介入，126b）
- 錯音與亂音高是完全分開的兩類 cell，沒有合併：
  - 錯音（wrong100pct）：fluidr3 PQ Δ +0.004（worse 6.7%）、pulse_clarity −0.001（10.4%）；generaluser PQ −0.009（14.9%）、CLAP +0.002（18.4%）、pulse_clarity 0.000（11.9%）。
  - 大幅隨機音高（randpitch100pct）：fluidr3 PQ −0.007（11.7%）、pulse_clarity −0.002（16.9%）；generaluser pulse_clarity −0.003（19.9%）。
  - pitchshuffle_all：PQ +0.008／+0.007（3.2%／9.0%）、CLAP 0.000（18.9%）。
  - 另有 jitter200ms PQ +0.007（7.7%）、unsteady30pct PQ／CU +0.002／+0.001、extra30pct PQ −0.007（6.0%）。
- 126c：vocals_detune100c 的 pulse_clarity／chroma_entropy、vocals_lag250ms chroma_entropy 同屬此類（worse 6–17%）。
- 這與 126／133／126d 先前的結論一致：音符層音高錯誤，AES 類全盲。

## 3. 與論文主結果的對照

這些數字與表 B（ours vs MusicGen：PQ +0.155、CLAP +0.0105）和 NegMF E1（+0.95）是**同尺度比較**，用來說明 PQ 增益相對於指標已知盲點的大小：
- 表 B PQ +0.155 ≈ 1.7 × m₀(PQ=0.0919)，大於 m₀；但與下列「不可靠」偏移同量級：126b blastnote_x1 讓 PQ **上升** +0.155（同為 +0.155）、PCM_16 儲存損傷 −0.14～−0.18（B 結果）。即 PQ +0.155 的差距，與「在乾淨 MIDI 加一堆爆音」所得的 PQ 變化一樣大。→ 表 B 的 PQ 優勢不能被解讀為音樂品質提升，只能當 PQ 讀數差。
- 表 B CLAP +0.0105 ≈ m₀(CLAP=0.0104)，剛好在平手界線上；126a CLAP 有 13 列「該判差卻上升」、126c 5 列，且 CLAP 對靜音／正弦曾高過 39% 音樂（126）。→ CLAP 優勢屬邊界，不下結論。
- NegMF E1 +0.95 PQ ≈ 10 × m₀，遠大於所有上述偏移與 B 的 PCM_16 損傷（−0.14～−0.18），也大於 126a 的噪音階梯 PQ 判差線（SNR ≤ 25–30）；因此**幅度上**可信地大於指標已知的噪音。但 FAD 三 seed 都更差（CFG0 3.8→6.2），且 081 系列 PQ 被優化（FAD 同時惡化），故為「PQ 讀數大幅上升」而非品質證據。
- 仍沒有人類評分，以上都不是對人類偏好的宣稱。

## 4. 126b 鋼琴代表性與資料缺口

- 126b items 87,653 筆；來源與 piece 數：midi_fluidr3 59、midi_generaluser 59（兩者相同 59 首，含 alla_turca、bach_prelude_c、clair_de_lune 等古典鋼琴，及 CSD 童謠 en001…）；csd_real 50（真人童謠演唱）；poor_sing 8；poor 8；pro 9；floor 11 種（binary_01、binary_pm1、dc…）。
- piece 為 cluster 單位，故 CI 以 59／50／8／9 個群計；poor／pro 僅 8–9 首，CI 很寬，列結論時「爭議」比例偏高屬樣本小。
- 代表性限制：
  1. 僅鋼琴（SoundFont 兩種）＋童謠演唱，沒有多樂器、人聲＋伴奏混音、電子／打擊為主的曲目；MusicCaps／TTM 實際輸出的分布遠比這寬。
  2. MIDI 合成音乾淨，介入的是音符層；真人演奏（pro／poor）只有 8–9 首、其「poor」標籤為 presumed，非人類評分。
  3. 錯音（wrongN）與大幅隨機音高（randpitch／pitchshuffle）分列，結論不可互相代換：大幅隨機音高（100%）使旋律面目全非，卻仍被 PQ 判不出（見 2.3）。
  4. 缺：多樂器 MIDI、節奏型主導的曲目、以人類標註的好／壞演奏為地面真值的刺激。
- 資料缺口對應的後續：人類 JND pilot（`docs/experiments/human_jnd_pilot_20261019.md`，待使用者授權）與 D 的速度／音高延伸（`docs/experiments/speed_pitch_extension_20261020.md`）。

## 5. 限制
- `evidence_rows.tsv` 的每列有 Δ、CI、n、tie／worse／better；**clean 與變體的原始分數平均未逐列存入**（只存 Δ）。原始逐 clip 分數在各 126 資料集的 scores.tsv／`items.json`，`cells_rates.json.gz` 為判定率快取。若要逐列原始值，需以 `evidence.py` 加一欄重跑（約 576 MB RAM、數分鐘）。
- 126b／126c 以 piece 為群，群數少（8–59）時 bootstrap CI 偏窄的風險未校正。
- 分類為預先登錄的機械規則；「爭議」不代表指標好或壞。

## 勘誤（2026-10-22）

- 第 25 行（§2.1 標題）：`409 列` → `406 列`。依據 `research/eval/metric_std_20261020/evidence_rows.tsv` 重算 `kind=lose & lo>0 & metric≠pulse_clarity`：126a 78／126b 316／126c 12，合計 406；若含 `pulse_clarity` 為 408。
- 第 27 行（126b 分項）：`CLAP 17` → `CLAP 14`。依據同一檔：126b `clap` 且 `kind=lose`、`lo>0` 為 14 列；修正後 126b 分項合計 316，與上列總數 406 一致。
