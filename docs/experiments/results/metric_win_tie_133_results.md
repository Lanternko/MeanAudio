# 133 MERT／MuQ 餘弦距離：兩條要求結果（2026-10-20 收線）

p2 133 r2：rc=0，VALIDATE OK，時間 13:11 → 16:12。設計見 `../metric_win_tie_133_mert_20261020.md`（r1 失敗與 r2 修正見 §9），數字見 `metric_win_tie_133_summary.json`。

**判定方式**：
- d = cosim(case, 同一首 clean) − 1；平手條件 |d| ≤ m，判「較差」條件 d < −m；m = 0.10 S，通過門檻 80%。
- MERT：S = 0.0393，m = 0.0039，也就是 cosim ≥ 0.9961 才算平手。
- MuQ：S = 0.1087，m = 0.0109，也就是 cosim ≥ 0.9891 才算平手。

## 總結

**沒有指標同時通過兩條要求。** 但 MERT 是 126 系列裡第一個**偵測得到音符層錯誤**的指標，在 126b 的平手格也是 22/22 全過。

| 集合 | MERT lose | MERT tie | MuQ lose | MuQ tie |
|---|---|---|---|---|
| 126b（鋼琴 MIDI，音符層＋訊號層） | 39/64 | **22/22** | 47/64 | 16/22 |
| 126c（MUSDB 分軌編輯） | 7/20 | 14/16 | 9/20 | 14/16 |
| 126a（5 TTM 系統 × 訊號層，只跑 MERT） | 見下 | 見下 | — | — |

## 126b：音符層

AES、MEva、SongEval 在這一層全部偵測不到。

**MERT 通過的介入**：
- wrong30pct、wrong100pct
- randpitch30／100pct
- posshuffle30／100pct
- pitchshuffle_all、retrograde
- extra30pct、drop30／60pct
- jitter200ms、vel_random
- blastnote ×1／×3／×10

**MERT 沒通過的是小幅錯誤**：
- wrong1_semi 0.07、wrong1_tritone 0.17
- wrong5pct 0.13、wrong10pct 0.34
- jitter10–80ms 0.16–0.76
- unsteady 0.14–0.61
- randpitch10pct 0.39

**MuQ**：小幅錯誤同樣全部沒過。jitter30／80ms、unsteady15／30pct、blockshuffle 這幾個 MuQ 過了，MERT 沒過。

**應判平手的變化**：MERT 與 MuQ 都**把 transpose ±1 判成大差**，cosim 中位數 0.962／0.952，這符合預登錄的預測失敗。tempo ±2% 判為平手（0.9987），tempo ±5% 是 0.997，落在平手門檻附近。

## 126b、126a、126c：訊號層

**MERT**：
- 126b 的平手格（gain −30～+6、mp3、lead、極性）全過。
- 126a 不是這樣，**polarity 只有 1/5 系統過、lead 50／200ms 只有 1–2/5 過**：TTM 生成音訊上，MERT 對極性與平移不具不變性（polarity 平均 cosim 0.9956，剛好落在門檻上）。
- 白噪音要 SNR ≤ 15 才 5/5 偵測到；SNR 25 是 3/5、SNR 30 是 0/5。clip3／6dB、bitcrush8、dropout ×3 都偵測不到。

**MuQ**：對大幅衰減（gain −15／−20／−30）和 mp3 都**不具不變性**。mp3_32k 平手率只有 0.01，126c 的 mp3_64k 和 dmx_resum 都是 0.00。換句話說，MuQ 把編碼瑕疵看成內容變化。

## 126c：分軌編輯

- **鼓延遲**：MuQ 在 100ms 以上偵測到；MERT 全部沒過，最高是 250ms 的 0.77。
- **人聲錯位**：兩個指標都沒過。
- **走音**：只有 MERT 偵測到 100c 的走音。
- **換軌**：swap_drums／swap_vocals 兩個指標都過。
- **單軌 ±1 dB**：兩個指標都判平手。
- **MERT 的平手失敗**：polarity 0.34、lead10ms 0.79。

## 可寫／不可寫

- **可寫**：在本刺激集上，MERT／MuQ 這種以參考為基準的嵌入距離，是唯一偵測得到大比例音符層錯誤的指標，錯音達 30% 以上就抓得到。
- **可寫**：兩者都把移調 ±1 判為大差。MuQ 對 mp3 與大幅衰減不具不變性；MERT 在 TTM 音訊上對極性不具不變性。
- **不可寫**：「MERT 通過兩條要求」。音符層的小錯（單一錯音、10% 以下）和訊號層的輕度劣化都偵測不到。
- **限制**：這是以參考為基準的指標，需要同一首的 clean 版本，所以不能直接拿來評 TTM 系統之間的排名。沒有人類評分。S_mert 的重算值與 r1 差 3.2e-4，超過預定容差 1e-4，判定沿用預登錄的 r1 值。
