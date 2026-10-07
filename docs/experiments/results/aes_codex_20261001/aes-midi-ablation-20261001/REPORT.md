# MIDI 音樂 property ablation 實測

8 個配對模板 × 2 個 SoundFont；2160 個 10 秒音訊，4320 次四軸評分。

## 主要發現

本批控制合成中，AES PQ 對音域、音色與噪聲的變化，比對音階種類的變化敏感。這是介入敏感性，不是人類效度或不合理偏誤的證明。

- 同音階跨八度：−1 octave 平均 −0.337、+1 octave +0.100；再升 +2 octaves 反而 −0.144，並非越高越好。方向在兩個音色庫的平均結果相同，但各樂器可不同。
- 音階類型：相對 major 的 PQ 平均差都在約 ±0.02 內；不能据此認定 AES 看懂／完全忽略音階。
- 三音音簇相對大三和弦 −0.387；開放間距 +0.096。這測局部聲學和聲，不是完整功能和聲。
- 四樂器輪流分配固定 32 音，平均 +0.161；兩樂器 +0.026 且 CI 跨零。不是越多樂器必然越好。
- 單音 C4 連發相對原旋律 −0.215；同一音高 8Hz 相對 4Hz −0.183。更多 attack 不保證更高 PQ。
- 鋼琴加噪設定 SNR 20dB：white −0.982、pink −1.059。實際 AES mono16 SNR 另存 independent_validation.json，不把設定 SNR 當 AES 實際 SNR。
- 樂器排名會隨音域和音色庫改變，不能把跨五個音域平均的排名當成普遍樂器偏好。

中央音域、相同 MIDI，相對鋼琴的音色庫分層差值：

| 樂器 | GeneralUser ΔPQ | FluidR3 ΔPQ |
|---|---:|---:|
| nylon_guitar | +0.3328 | +0.0650 |
| acoustic_bass | +0.2214 | -0.0506 |
| violin | +0.2169 | -0.0115 |
| flute | +0.0020 | -0.1007 |
| warm_pad | -0.2014 | -0.3615 |

## 完整配對主分析

每個音訊都有可重跑 MIDI。主分析在 16k mono 對齊 −23 LUFS；原始響度另列。SoundFont 與旋律情境先在模板內平均，再對 8 個模板 bootstrap 10,000 次。CI 未校正多重比較，只描述本批模板，不代表歌曲族群。

| 因子 | 處理 | ΔPQ | 95% exploratory CI |
|---|---|---:|---|
| chord_timing | arpeggio | +0.0696 | [+0.0598, +0.0802] |
| gain_control | minus6dB | +0.0000 | [-0.0001, +0.0001] |
| harmony | cluster | -0.3872 | [-0.3930, -0.3816] |
| harmony | minor | +0.0054 | [+0.0005, +0.0103] |
| harmony | open | +0.0963 | [+0.0882, +0.1051] |
| instrument | acoustic_bass | -0.0518 | [-0.0572, -0.0466] |
| instrument | flute | -0.3295 | [-0.3384, -0.3209] |
| instrument | nylon_guitar | +0.0589 | [+0.0488, +0.0688] |
| instrument | violin | -0.3243 | [-0.3346, -0.3158] |
| instrument | warm_pad | -0.5420 | [-0.5772, -0.5060] |
| multiple_instruments | 2_split | +0.0260 | [-0.0208, +0.0694] |
| multiple_instruments | 2_unison | -0.1014 | [-0.1191, -0.0839] |
| multiple_instruments | 4_split | +0.1609 | [+0.1414, +0.1817] |
| multiple_instruments | 4_unison | +0.1331 | [+0.1198, +0.1465] |
| noise | pink_20 | -1.0588 | [-1.0846, -1.0335] |
| noise | pink_30 | -0.4710 | [-0.4864, -0.4567] |
| noise | pink_40 | -0.2108 | [-0.2260, -0.1965] |
| noise | white_20 | -0.9818 | [-1.0116, -0.9483] |
| noise | white_30 | -0.4077 | [-0.4213, -0.3950] |
| noise | white_40 | -0.1364 | [-0.1523, -0.1226] |
| register | -12 | -0.3369 | [-0.3456, -0.3278] |
| register | -24 | -0.5392 | [-0.5556, -0.5231] |
| register | 12 | +0.1002 | [+0.0825, +0.1189] |
| register | 24 | -0.1440 | [-0.1592, -0.1301] |
| scale | dorian | -0.0121 | [-0.0173, -0.0071] |
| scale | lydian | -0.0027 | [-0.0078, +0.0024] |
| scale | minor | -0.0045 | [-0.0089, -0.0001] |
| scale | pentatonic | -0.0187 | [-0.0213, -0.0157] |
| scale | phrygian | -0.0055 | [-0.0119, -0.0000] |
| scale | whole_tone | +0.0029 | [-0.0021, +0.0082] |
| single_note_pitch | 48 | -0.4890 | [-0.4909, -0.4866] |
| single_note_pitch | 72 | +0.0391 | [+0.0381, +0.0403] |
| single_note_rate | 1hz | -0.0096 | [-0.0110, -0.0081] |
| single_note_rate | 8hz | -0.1834 | [-0.1851, -0.1818] |
| single_note_vs_melody | single_C4 | -0.2150 | [-0.2219, -0.2084] |
| tonic_transposition | 2 | +0.0286 | [+0.0173, +0.0397] |
| tonic_transposition | 5 | +0.0246 | [+0.0146, +0.0347] |
| tonic_transposition | 7 | +0.0633 | [+0.0542, +0.0721] |

## 判讀限制

- 樂器差异包含音色、包絡、取樣錄音與 SoundFont 程式差異，不能單憑 PQ 宣稱某樂器真正較好／AES 存在不合理偏誤。
- register 是整段跨八度，保留音階 pitch class；tonic_transposition 保留音程、改變調高與絕對音高。
- scale 會同時改變音程與 pitch distribution；不能聲稱完全固定聲學音高後只改音樂理論標籤。
- harmony 固定每個時點三音，但本輪是每個旋律音上的和弦類型，尚未包含完整功能和聲進行。
- chord_timing 的分解和弦另減短每音 duration，總 held-note time 從 16.8s 變成 5.6s；屬次要複合介入，不能當成只改 note-on 順序的證據。
- 單音連發固定音高與 5.6 秒總 MIDI note duration；1/4/8Hz 改變音符與 attack 數，並非固定聲學包絡。
- 多樂器 split 固定總 note events；unison 則與同數量鋼琴疊層配對，避免把單純多音符當多音色。
- 八個模板的 velocity／melodic pattern 有變化，但共享合成設計；不是八個訓練 seed 或八首真實歌曲。
- 合成器與 SoundFont 只控制產生流程，不能保證各條件的人類 PQ 相同；尚無人類盲聽。
- 全部為聊天 sidecar，未改 karaoke-jp canonical，未 commit。

完整設計 design.json；個別音訊／MIDI stimuli/；所有四軸與音色庫分層結果 summary.csv。
檢查 validation.json、independent_validation.json、synth_validation.json；音色×音域圖 instrument_register.png；試聽 listen.html。
稀疏低能量 release 的 BS1770 gate 會在增益後改變；本輪對每個標準化波形重新量測／修正 gain，最終 LUFS 誤差已檢查。
