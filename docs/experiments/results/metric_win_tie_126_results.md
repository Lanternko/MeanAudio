# 126 評分模型兩條要求：結果（126a 訊號層 × 5 TTM、126b 名曲 MIDI／真人／底線）

> 收線 2026-10-18（GPU：p2 126 於 2026-10-16 23:17 完成、p2 127 於 2026-10-17 00:33 完成，exit 0）。
> 設計與預登錄：`../metric_win_tie_126_20261016.md`。數字來源：`metric_win_tie_126a_summary.json`、`metric_win_tie_126b_summary.json`。
> **無人類評分**。「應判差／應平手」是設計者預先指定的方向，不是聽者判斷。

## 0. 一句話

**沒有任何一個評分模型同時滿足兩條要求。**

- AES PQ／CE／CU 在小幅增益（±1 dB）、PCM_16、極性上能判平手。但 **10 ms 前置靜音**在所有系統上都判不了平手（45–76%）。
- 音符層的錯音、亂音高、音高打亂，**全部指標都偵測不到**。
- MEva 逐段幾乎沒有平手能力，−0.5 dB 增益只有 5–8% 判平手。
- 底線方面，−30 LUFS 的 1 kHz 正弦波、掃頻、方波，PQ 高過 34–38% 的 126b 音樂片段。

## 1. 協定

- 逐對比較 (clean, 介入) 判勝／平／負，平手邊界 m = 0.10·S。
  - AES／MEva／CLAP 的 S = 126a 五個系統 clean 平均的全距：

    | 指標 | S |
    |---|---|
    | PQ | 0.9189 |
    | CE | 1.2005 |
    | CU | 0.8452 |
    | PC | 0.3447 |
    | MEva | 0.1724 |
    | CLAP | 0.1038 |

  - MIR 的 S = 126b 全部 clean 片段的 SD：

    | 指標 | S |
    |---|---|
    | pulse_clarity | 0.233 |
    | key_cnn_conf | 0.247 |
    | ibi_cv | 0.0224 |
    | chroma_entropy | 0.0998 |

    越低越好的 MIR 指標已反號存檔。
- 通過條件：「應判差」的介入要有 ≥ 80% 判差；「應平手」的介入要有 ≥ 80% 平手。
- 126a 每格的系統數為 5（real、ours CFG3+neg、ours CFG0、MusicGen、MusicLDM），每個系統 5,131 段 MusicCaps，共 52 個訊號層介入。
- 126b 共 87,653 段，涵蓋 9 首古典＋10 首童謠的鋼琴 MIDI × 2 個 SoundFont（FluidR3、GeneralUser）× 42 個音符層介入，再加上真人錄音（csd_real、pro、poor、poor_sing）與底線刺激。

## 2. 126a：該平手的介入

格內為「通過的系統數／5」，括號內是五個系統平手率的最小–最大。

| 介入 | PQ | CE | CU | PC | MEva | CLAP |
|---|---|---|---|---|---|---|
| clean_shuf（同檔重評） | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| gain −0.5／±1 dB | 5/5 | 5/5 | 5/5 | 多數 5/5（80–94） | **0/5（5–8）** | 5/5（83–97） |
| gain −2 dB | 5/5（83–96） | 5/5（81–96） | 5/5（86–98） | 0/5（49–77） | 0/5 | 1/5（67–82） |
| gain −3 dB | 3/5（65–90） | 1/5 | 2/5 | 0/5 | 0/5 | 0/5（53–66） |
| gain ≤ −6 dB、+6 dB | 全不過 | | | | | |
| polarity | 4/5（78–91） | 5/5 | 3/5 | 0/5（54–66） | 0/5（2–4） | 5/5（100） |
| pcm16 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5（46–60） | 1/5（68–86） |
| resample 44k | 4/5（58–100） | 4/5 | 4/5 | 4/5 | 0/5 | 5/5 |
| **lead 10／50／200 ms** | **0/5（45–73）** | 0/5 | 0/5 | 0/5 | 0/5 | **0/5（21–38）** |
| mp3 160k | 3/5（32–100） | 3/5 | 3/5 | 2/5 | 0/5 | 1/5 |
| mp3 32k | 全不過 | | | | | |

增益向下的容忍度如下：

| 指標 | 容忍到 |
|---|---|
| PQ | −2～−3 dB |
| CE、CU | −2 dB |
| PC、CLAP | −1 dB |
| MEva | 無 |

## 3. 126a：該判差的介入（偵測門檻）

- **白噪音**：

  | 指標 | 首次通過 |
  |---|---|
  | PQ | SNR 30 dB（4/5）、SNR 25 dB（5/5）；SNR 40 dB 只有 48–68% |
  | CU | SNR 25 dB |
  | CE | SNR 10 dB |
  | CLAP | 只在 SNR 0 dB 通過 |
  | MEva | 任何 SNR 都不過（判差率 40–79%） |

- **削波**：clip 3／6 dB 全指標都不過；clip 12 dB 時 PQ 5/5。
- **bitcrush**：8 bit 不過；6 bit 時 PQ 5/5。
- **爆音（clicks）**：每秒 1 個時 3/5 不過；cfg0、MusicLDM 的 PQ 已通過（84／81%）；每秒 5 個時 PQ／CE／CU 5/5。
- **dropout**：20 ms × 30 時 PQ／CE／CU 5/5。
- **低通**：PC 是最強的偵測者（4 kHz 低通 5/5），CE 只在 1 kHz 才過。
- **單發 80 ms blast**：×1 僅 MusicGen 的 CE 偵測到（81%）；×3 只有 CE 偵測到。
- **reverse、0.25 s shuffle**：CE／CU 5/5。

## 4. 126a：MEva 與 FAD

**MEva** 本身是決定性的（clean_shuf 100% 平手），但逐段變動極大：

| 介入 | 平均 Δ | 逐段 \|Δ\| p90 |
|---|---|---|
| gain −0.5 dB | 0.001–0.009 | 0.38–0.43 |
| polarity | — | 0.77–1.0 |

在 S = 0.17 的尺度下，逐段 MEva 等同擲硬幣，只能當群體平均用。

**FAD**（對各系統自己的 clean）：

| 介入 | FAD |
|---|---|
| polarity、lead、pcm16、clean_shuf | ≈ 0 |
| gain −3 dB | 0.08–0.10 |
| gain −30 dB | 3.1–6.1 |
| mp3 32k | 0.4–1.3 |
| noise SNR 40／20／0 | 0.34–0.79／2.6–5.1／12.7–16.9 |
| clip 6／20 dB | ≈0.07／4.4–5.6 |
| lowpass 4k | 2.2–3.2 |
| reverse | 2.1–2.9 |
| shuffle 1 s | ≈0.2 |
| blast ×1、dropout ×3 | 0.02–0.03 |

- FAD 對時間結構破壞（shuffle）與短暫缺陷幾乎無感，對頻寬與響度很敏感。
- 各系統 clean 對 real 的 FAD：ours CFG3+neg 4.34、ours CFG0 2.54、MusicGen 4.28、MusicLDM 2.70。

## 5. 126b：名曲 MIDI 的音符層介入

格內為判差率 %（soft = 移調／速度／力度，報平手率 %）。

| 介入（FluidR3） | PQ | CE | CU | PC | MEva | CLAP | pulse | key | ibi | chroma |
|---|---|---|---|---|---|---|---|---|---|---|
| wrong 100% | 7 | 15 | 4 | 15 | 51 | 26 | 10 | 79 | 26 | 42 |
| wrong 1 semitone | 0 | 1 | 0 | 2 | 46 | 2 | 0 | 44 | 7 | 5 |
| randpitch 100% | 12 | 19 | 9 | 19 | 49 | 30 | 17 | 77 | 29 | 43 |
| pitchshuffle all | 3 | 16 | 3 | 20 | 49 | 25 | 13 | 48 | 30 | 36 |
| posshuffle 100% | 4 | 50 | 4 | 46 | 78 | 63 | **95** | 48 | **88** | 61 |
| jitter 30 ms | 0 | 9 | 2 | 11 | 59 | 20 | **89** | 30 | **82** | 2 |
| unsteady 15% | 0 | 9 | 3 | 17 | 59 | 22 | **91** | 41 | **88** | 4 |
| drop 60% | 8 | **93** | 66 | **98** | **86** | 37 | **90** | 61 | 71 | 22 |
| blastnote ×1 | 2 | **85** | 32 | 58 | 61 | 13 | 49 | 42 | 35 | 10 |
| blastnote ×10 | 79 | **94** | 66 | 61 | 67 | 23 | 71 | 49 | 43 | 31 |
| tempo ×1.02（soft，平手%） | 98 | 78 | 95 | 66 | 3 | 54 | 86 | 26 | 30 | 95 |
| transpose +1（soft） | 97 | 47 | 89 | 51 | 4 | 46 | 83 | 15 | 49 | 43 |
| vel ×0.85（soft） | 100 | 100 | 100 | 96 | 15 | 85 | 100 | 88 | 77 | 98 |

- **音高錯誤偵測是空的**：錯音、亂音高、音高打亂在兩個 SoundFont 上沒有任何指標通過。最接近的是 key_cnn_conf，在 100% 錯音時判差 77–80%。
- **節奏類介入只有 MIR 拍點指標偵測得到**：
  - pulse_clarity：jitter 30 ms、unsteady 15%、posshuffle 30% 起就通過。
  - ibi_cv：jitter 30 ms 起通過，posshuffle 要到 100%。
  - AES 對 jitter 30 ms、unsteady 15% 判差 ≤ 12–17%，posshuffle 可達 28–50%；仍遠低於 80% 門檻。
- **blastnote ×1 的 PQ 判差率只有 2%**：m = 0 時有 78.6% 判「變好」，在 m 之下有 18.9%。CE 則判差 85%。預登錄的 Q3 預測未成立（見 128a）。
- **GeneralUser** 方向相同。訊號層介入的偵測門檻比 126a 低，例如 noise SNR 40 dB 時 PQ 100%、clip 6 dB 時 PQ 92%：乾淨合成音訊上的缺陷比較顯眼。
- **soundfont_swap**（FluidR3 ↔ GeneralUser，應平手）：PQ 38%、CE 0%、CLAP 0%，全部不過。換音色在每個指標上都被當成「大差異」。

## 6. 126b：真人錄音與交叉比較

真人錄音上的訊號層平手／偵測和 126a 同向。差別在 poor_sing（幼兒演唱，n=63）：noise SNR 30 dB 的 CE 判差只有 6%、CLAP 只有 35%，clip 6 dB 的 PQ 只有 17%。本身就差的錄音，再加缺陷也看不出來。

交叉比較（以 alla_turca 為例，better/worse %）：

| 比較 | PQ | CE | CU | PC | CLAP | pulse |
|---|---|---|---|---|---|---|
| pro vs poor | 84/10 | 86/9 | 94/4 | 49/48 | 90/6 | 48/45 |
| pro vs MIDI（GeneralUser） | **0/95** | 60/28 | 16/58 | 100/0 | 33/47 | 0/96 |
| FluidR3 vs GeneralUser | 38/18 | 0/98 | 14/62 | 60/28 | 0/100 | 46/27 |

- pro > poor 在 PQ／CE／CU／CLAP 大多成立，但依曲目而變：Für Elise 的 pro 反而在 PQ 判差 59%，bach_prelude 是 46/32。
- **乾淨 MIDI 的 PQ 高過專業真人演奏**（alla_turca 8.00 vs 7.53；minuet_g、ode_to_joy 0/100），與 006 的「乾淨合成 PQ 高」一致。

## 7. 126b：底線

表中為 PQ 平均值（vs 白噪音判較好的 %）與「126b 音樂片段低於此底線」的 %。

| 底線 | PQ 平均 | PQ 勝白噪% | 音樂 PQ 低於它% | CE 低於% | CLAP 低於% |
|---|---|---|---|---|---|
| silence | 6.74 | 100 | 7.4 | 0.2 | **39.4** |
| DC | 6.73 | 100 | 7.4 | 0.2 | 1.9 |
| sine 1 kHz −30 LUFS | **7.69** | 100 | **34.2** | 0.0 | 12.6 |
| sweep −30 LUFS | 7.78 | 100 | **37.7** | 0.1 | 1.3 |
| square 1 kHz −30 LUFS | 7.67 | 100 | 34.1 | 0.0 | 8.9 |
| white −30 LUFS | 5.17 | — | 0.6 | 0.0 | 1.9 |
| 0/1 二值 full scale | 4.94 | 71 | 0.1 | 0.0 | 0.6 |

- 靜音／DC 的 PQ 6.735 是 GroupNorm eps 的全零收斂值（見 memory `reference_aes_scale_invariance_groupnorm_eps.md`）。
- **PQ 與 CU 把純音、掃頻排在三分之一以上的真實音樂之上**。CE 是唯一把這些底線全部排在幾乎所有音樂之下的 AES 軸。
- CLAP 對靜音給 0.23，高於 39% 的音樂片段（D1 已見過的現象）。
- 0/1 二值波形在 PQ 上勝白噪音 71%，其餘五個軸都把它排在白噪音之下，符合「比白噪更差」的設計。

## 8. 判讀（可寫／不可寫）

**可寫（觀察）：**

1. AES PQ 對 ±1 dB、PCM_16、極性的平手能力合格。但 10 ms 前置靜音就會打破，−3 dB 只剩部分系統合格。所以 arm 間的 PQ 差若 < 0.1·S，必須先排除對齊與響度。
2. 音高錯誤是所有指標的盲點；節奏錯誤只有 pulse_clarity／ibi_cv 看得到。
3. MEva 只能當群體平均，不能做逐段或小樣本比較。
4. PQ／CU 會把純音底線排在 34–38% 的音樂之上，CE 不會。

**不可寫：**

- 「AES 偏好合成音」：沒有人類評分。
- 「指標不能分辨我們的 arm」：126 沒有在 arm 對上測試。
- 「任何一個指標通過兩條要求」。

**後續：**

- 128 已用 126 的 S 與子集測試不變性修補（C1–C3）。結果是無方案通過，見 `metric_invariance_128_results.md`。
- MUSDB18-HQ 的去樂器／人聲 job 尚未排。

## 勘誤（2026-10-22）

- 第 58 行（PQ 欄，lead 10／50／200 ms）：`0/5（45–76）` → `0/5（45–73）`。依據 `metric_win_tie_126a_summary.json` `cells.lead{10,50,200}ms.*.metrics.PQ.at_m.tie`：最小 0.4522、最大 0.7264；76.26 是 CU lead200 的值。
- 第 85 行（clicks 1/s）：`每秒 1 個不過` → `每秒 1 個時 3/5 不過；cfg0、MusicLDM 的 PQ 已通過（84／81%）`。依據 `cells.clicks1ps.ours_cfg0.metrics.PQ.at_m.pass=true`（worse 0.8433）、`cells.clicks1ps.musicldm.metrics.PQ.at_m.pass=true`（worse 0.8119）。
- 第 88 行（單發 80 ms blast ×1）：`沒有指標偵測到` → `僅 MusicGen 的 CE 偵測到（81%）`。依據 `cells.blast80ms_x1.musicgen.metrics.CE.at_m.pass=true`、`worse=0.8141`；其餘系統與指標皆 false。
- 第 136 行（tempo ×1.02，FluidR3）：PQ `100` → `98`（0.9801）、CE `77` → `78`（0.7836）、PC `62` → `66`（0.6567）、CLAP `43` → `54`（0.5373）、pulse `81` → `86`（0.8607）、chroma `96` → `95`（0.9478）。依據 `metric_win_tie_126b_summary.json` `cells["midi/midi_fluidr3"].tempo_x1.02.metrics.<metric>.at_m.tie`。
- 第 144 行（AES 對節奏介入）：`全部 ≤ 12% 判差` → `jitter 30 ms、unsteady 15% 判差 ≤ 12–17%，posshuffle 可達 28–50%；仍遠低於 80% 門檻`。依據 `cells["midi/midi_fluidr3"]`：`unsteady15pct.PC.at_m.worse=0.1716`、`posshuffle30pct.CE/PC=0.2736/0.2811`、`posshuffle100pct.CE/PC=0.5025/0.4627`。原句只列 jitter 30 ms／unsteady 15%／posshuffle 30% 起，故未寫成「jitter／unsteady 一律 ≤ 17%」（jitter 200 ms 的 CE 可達 47.0%、unsteady 30% 的 PC 21.6%）。
