# 真實曲 MIDI 控制實驗（p2 123）與 LALM 品質評審（p2 124）

2026-10-14 排入。來源：`docs/meetings/2026-10-14_metric_controls_real_pieces_fad_lalm.md`。
兩個 job 都只評分，不訓練、不用我們的模型生成。沒有人類評分，結論只能寫「指標對已知介入的反應」。

## 刺激（123 與 124 共用）

`scripts/eval/midi_realpiece_stimuli.py`，CPU，已跑完（直接跑：純 CPU 渲染，90 秒）。

- 來源：MAESTRO v3 的真實鋼琴演奏 MIDI（比賽錄音，力度與踏板都是真人演奏）。150 段 30 s excerpt，
  14 位作曲家（Bach／Beethoven／Chopin／Debussy／Liszt／Schubert 各 14，Haydn／Mozart 12，
  Rachmaninoff 11，Scarlatti／Schumann 10，Mendelssohn／Brahms 5，Tchaikovsky 1）。
- 2 個 SoundFont（GeneralUser GS、FluidR3），鋼琴 patch，mono 16 kHz，−30 LUFS，peak cap 0.999，FLAC PCM_24。
- 每段 57 個版本，共 17,100 段。輸出 `~/eval_output_nvme/midi_realpiece_123/`（16 GB；`items.json`、`checks.json`）。

| 類別 | 個數 | 內容 | 預先登錄的預期 |
|---|---|---|---|
| base | 1 | clean | — |
| degrade | 25 | 錯音 10／30／100%、亂數音高、音高打亂（2 s／全段）、兩手 detune 30／50 cent；和弦 jitter 30／80／200 ms、單音 jitter 80 ms、速度跳變 ×1.15／1.5／2.0、局部加速 ×2／減速 ×0.6；掉音 30／60%、多音 30%；力度抹平／亂數；區塊打亂 0.5／2 s、逆行 | 分數**下降**（配對 Δ 的 95% CI 整段低於 0） |
| control | 11 | 移調 ±1／±2 半音、速度 ±5%、力度 ×0.85、room／hall 殘響、高頻 −3 dB、亮音色鋼琴 | **不變**（CI 跨零，或 \|Δ\| 小於同一指標在 degrade 類的中位 \|Δ\|） |
| ambiguous | 8 | 換音色（長笛／尼龍吉他／小提琴／pad）、移調 ±12、速度 ×0.75／×1.33 | 不預設方向，只報告 |
| floor | 12 | 純白噪音、SNR 0／20 白噪、3-bit bitcrush、hard clip 30 dB、click 5／50 次每秒、20 ms dropout、音訊反轉、取樣區塊打亂 20／200 ms、每秒插入別首 | 分數**下降**，且是紅綠燈的紅燈候選 |

操弄檢查（`checks.json`）：clean 300 段、音符數中位 249、無 peak cap；各介入的音高／IOI 統計與設計一致。

## p2 123：AES＋MEva＋FAD＋MIR

`scripts/eval/midi_realpiece_control_123.py`，輸出 `~/eval_output_nvme/midi_realpiece_control_123/`。

- 兩個視圖 × 兩個 SoundFont（4 個單位，各 8,550 段）：`full30`（30 s 原檔）與 `crop10`（0.5–10.5 s，重新對齊 −30 LUFS）。
  crop10 是為了與 10 s 的 TTM 系統同長度比較（MEva 隨長度變）。
- 讀數：AES 四軸、MEva、VGGish FAD；MIR 四項只算 full30。
- 配對分析：每個版本減同段同 SoundFont 的 clean，cluster＝excerpt（150），bootstrap 10,000，seed 20261014。合併與分 SoundFont 都報。
- 紅綠燈：clean＝1、某個 floor 刺激＝0，把 p2 115 六個系統（真實、MusicGen、MusicLDM、MusicLDM＋fid8、ours CFG0、ours CFG3＋neg）的 lvl30 平均放上去。綠減紅 < 0.25 的指標不算位置（尺太短）。
- FAD：各版本對 MusicCaps 真實 reference 與對同 SoundFont 的 clean；兩個 SoundFont 的 clean 互算；clean 對半底線。
- 會議要的 FAD 檢查（用 p2 115／118 的快取嵌入，1,924 個共同 prompt）：真實對半、CFG0 對 CFG3＋neg（同 seed 與跨 seed）、seed 對 seed、各系統對真實。

判讀（預先登錄）：

- 每個指標報「degrade 類 25 個裡 CI 低於 0 的個數」與「control 類 11 個裡 CI 跨零的個數」。
  一個指標若 degrade 命中 < 13／25，寫成「對音符層介入不敏感」；control 有 > 3／11 的 CI 不跨零且 \|Δ\| 超過 degrade 類中位 \|Δ\|，寫成「對不該影響品質的改動有反應」。
- 紅綠燈位置只描述，不排名；系統位置 > 1（高過乾淨真實曲）或 < 0 都照實報。
- 沒有人類評分：不可寫「指標錯了」，只能寫「與預先登錄的預期不一致」。預期方向本身（尤其 control 類）是我們訂的，待使用者確認。

## p2 124：LALM 評審

`scripts/eval/lalm_quality_judge.py`，輸出 `~/eval_output_nvme/lalm_judge_124/`。

- 評審：Music Flamingo（`~/venvs/music_flamingo`）、Qwen2.5-Omni-3B thinker（`~/venvs/dac`）。一律 batch 1（bs1 與 bs16 分數不同）。
- 分數不靠抽樣：固定評分 prompt 後，取下一個 token 在 1–5 五個數字上的期望值。每個（評審, prompt, clip）一個決定性分數。smoke 的數字機率質量 0.98–0.999。
- 四個 prompt：overall、production、musical、professional。
- 另有兩個任務：`pair`（MIDI clean 與介入版各 10 s 接成一檔，兩種順序，問哪段較好）與 `explain`（兩句評語＋`Rating: N`，greedy，128 token）。
- 刺激 74,158 段（`items.tsv` sha256 `83bc3f43…be3ade`）：

| 組 | n | 已知答案 |
|---|---|---|
| musiceval | 2,748 | 專家 MOS（效度錨點） |
| pam | 500 | 人類 OVL／REL |
| noise | 2,500 | 白噪音階梯（clean > SNR 40 > … > 純噪音） |
| midi | 17,100 | 123 的刺激與預期 |
| ttm | 30,786 | 六個系統 × 5,131 prompt（無已知答案，讀數） |
| seeds | 20,524 | ours 另兩個 seed |

判讀（預先登錄）：

1. **最低門檻**（會議指示）：noise 組各來源的條件平均要單調，clean > 純白噪音，且 ttm 的真實錄音 > 純白噪音。沒過就寫「排不出 真實 > 白噪音」，後面不解讀。
2. **人類效度**：musiceval 的 clip 層 Spearman 與系統層 Spearman。對照 AES／MEva 在同一資料的既有數字（MIR 增量效度結果）。
3. **MIDI 控制**：與 123 同一套 degrade／control 計數。
4. **pair 任務**：smoke 顯示兩個評審都強烈偏向某個位置（期望值幾乎不隨內容變），先報位置偏誤 `p_first_mean`；兩種順序平均後才讀 `p_clean_better`。
5. ttm／seeds 只當讀數。系統差異不可寫成品質結論。

smoke（n=12）讀數：Flamingo overall midi 4.21／noise 3.32／ttm 3.37；Qwen overall midi 2.51／noise 2.73。Qwen 把噪音評得比 MIDI 高，門檻 1 可能不過。

## 資源

- 123：約 1.5–2 h（AES 約 5 min／單位、MEva 約 13 min／單位、分析的 FAD bootstrap 約 30–60 min）。
- 124：約 26 h（Flamingo 14–15 h、Qwen 10–11 h）。兩者合計 ≥ 24 h GPU。
- 輸出都在 NVMe：2026-10-14 量到 Seagate 寫入只有約 4 MB/s。123 的 crop10 音檔約 5.6 GB，可重生，收線後可刪。
- 續跑：123 以單位為粒度（完成的單位跳過）；124 以（評審, 任務, 組）的 TSV 為粒度，逐列續寫。
