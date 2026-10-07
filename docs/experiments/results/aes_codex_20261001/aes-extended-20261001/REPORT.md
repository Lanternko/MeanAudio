# AES 延伸實驗結果

已完成 19,531 個音訊、39,062 次四軸評分。MIDI：64 個 30 秒模板 × 2 個音色庫 × 149 條件；karaoke：12 首歌的 27 個既有片段 × 17 條件。

CUDA event 記錄的 forward 合計 10.75 分鐘；各 score session wall 合計 15.85 分鐘。這是實測推論時間，不等於含 CPU／資料準備的 GPU 佔用時間。

## 對原始問題的答案

**這批結果支持 AES PQ 偏好較少的已知白噪／粉紅噪聲，但不支持「任何背景變安靜，分數都會提高」。** 噪聲劑量效應在合成音樂與真實歌曲都重現，幅度強烈依賴素材、樂器與音域。
- SNR 20 dB 白噪：合成 MIDI ΔPQ −2.1305，真實歌曲 −0.6352；不能把合成音樂的噪聲敏感度直接套在生成音樂上。
- MIDI 移除已知 pad 背景：PQ +0.0652、CE −0.1589；兩音色庫的 PQ 方向相同，32 個驗證模板亦重現。PQ 上升與內容享受下降可同時發生；兩個都是模型分數，尚非人耳結論。
- 真實歌曲壓低低能量片段：對齊響度後 frame p10 真正降低約 5.13／10.70 dB，但 PQ 反而下降 −0.252／−0.443。此操作也改變弱音、尾音與接縫，不能當成純粹移除底噪。
- 保持音符數／pitch histogram 的旋律 shuffle：平均 ΔPQ −0.0068；驗證集為 −0.0048，28 項 Holm 校正後 p≈0.091，沒有通過主要驗證門檻。伴奏錯位仍有小幅但可重現的 PQ 懲罰（約 −0.01）。
- 音高 × 聲部不可用單一排名概括：上一輪單線旋律 +1 octave 得分上升，本輪三聲部整體 +1 octave 平均 PQ −0.035。前後設計不同，顯示情境依賴，並非一個實驗推翻另一個。
- 旋律 release trimming 在 GeneralUser PQ +0.129、FluidR3 −0.037，方向翻轉；不能宣稱 AES 一律偏好短尾音。噪聲的樂器交互作用也很大：同為 pink20，相對鋼琴，吉他前景的 PQ 懲罰平均少約 1.075 分。
- 三個原封不動的 10 秒區塊只換順序，raw 四軸差最大 0.000002；響度對齊後的極小差值是整段 loudness/gating 變化，不能當作 AES 看懂跨窗結構。

因此，這輪找到了噪聲、音色／音域與部分背景聲部對 PQ 的可重現介入反應，也找到 PQ 與 CE 的分歧。尚不能由此證明 NegMF 迎合評分器或人類品質真的變差。

## 可以回答什麼

噪聲、聲學形狀、旋律與和聲介入均有分開的配對對照。以下是模型敏感性，不是人類品質效度的證明。不同樂器並不是只改抽象標籤：取樣、頻譜與衰減也會改變。

## 配對主結果（MIDI）

| 因子 | 條件 | ΔPQ | 95% template bootstrap CI | 驗證集 ΔPQ |
|---|---|---:|---|---:|
| block_order | block_reverse | -0.0002 | [-0.0003, -0.0001] | -0.0002 |
| block_order | block_rotate | -0.0000 | [-0.0001, +0.0001] | -0.0000 |
| chord_timing | arpeggio_fixed_duration | +0.0359 | [+0.0281, +0.0436] | +0.0455 |
| functional_harmony | chord_rotate1 | -0.0128 | [-0.0157, -0.0100] | -0.0098 |
| functional_harmony | chord_rotatehalf | -0.0097 | [-0.0123, -0.0070] | -0.0077 |
| functional_harmony | chord_shift1 | -0.0365 | [-0.0419, -0.0310] | -0.0341 |
| functional_harmony | chord_shift6 | -0.0462 | [-0.0558, -0.0361] | -0.0276 |
| functional_harmony | chord_shuffle | -0.0128 | [-0.0152, -0.0105] | -0.0119 |
| gain_control | gain_m6 | -0.0000 | [-0.0000, +0.0000] | -0.0000 |
| instrument | 24 | +0.1121 | [+0.1065, +0.1177] | +0.1039 |
| instrument | 32 | +0.0383 | [+0.0347, +0.0418] | +0.0357 |
| instrument | 40 | -0.1208 | [-0.1317, -0.1101] | -0.1282 |
| instrument | 73 | -0.0660 | [-0.0718, -0.0601] | -0.0710 |
| instrument | 89 | -0.1921 | [-0.1999, -0.1845] | -0.2031 |
| lead_duration | 0.3 | -0.0325 | [-0.0360, -0.0291] | -0.0315 |
| lead_duration | 1.7 | +0.0235 | [+0.0213, +0.0257] | +0.0231 |
| lead_envelope | attack150ms | -0.0913 | [-0.1017, -0.0818] | -0.0804 |
| lead_envelope | attack50ms | -0.0880 | [-0.0967, -0.0799] | -0.0763 |
| lead_envelope | release_trim | +0.0461 | [+0.0399, +0.0521] | +0.0452 |
| melody | pitch_reverse | -0.0062 | [-0.0089, -0.0035] | -0.0037 |
| melody | pitch_shuffle | -0.0068 | [-0.0096, -0.0040] | -0.0048 |
| melody | single_note | -0.1116 | [-0.1195, -0.1038] | -0.1198 |
| multiple_instruments | split2 | +0.0535 | [+0.0485, +0.0585] | +0.0457 |
| multiple_instruments | split4 | +0.1678 | [+0.1634, +0.1724] | +0.1670 |
| noise | pink20 | -2.0614 | [-2.0931, -2.0284] | -2.0489 |
| noise | pink30 | -1.0023 | [-1.0252, -0.9786] | -0.9925 |
| noise | pink40 | -0.3596 | [-0.3709, -0.3480] | -0.3488 |
| noise | white20 | -2.1305 | [-2.1623, -2.0973] | -2.1458 |
| noise | white30 | -1.2942 | [-1.3201, -1.2676] | -1.2826 |
| noise | white40 | -0.5592 | [-0.5741, -0.5441] | -0.5477 |
| register | -12 | -0.1344 | [-0.1492, -0.1185] | -0.0947 |
| register | 12 | -0.0354 | [-0.0490, -0.0223] | -0.0540 |
| sustained_bus | pad_m12 | +0.0710 | [+0.0643, +0.0779] | +0.0589 |
| sustained_bus | pad_m6 | +0.0571 | [+0.0524, +0.0618] | +0.0484 |
| sustained_bus | pad_off | +0.0652 | [+0.0561, +0.0739] | +0.0461 |

## PQ 與 CE 對音樂結構的反應

| 條件 | ΔPQ | ΔCE |
|---|---:|---:|
| block_reverse | -0.0002 | -0.0006 |
| block_rotate | -0.0000 | -0.0000 |
| arpeggio_fixed_duration | +0.0359 | -0.0510 |
| chord_rotate1 | -0.0128 | -0.0024 |
| chord_rotatehalf | -0.0097 | -0.0061 |
| chord_shift1 | -0.0365 | -0.0776 |
| chord_shift6 | -0.0462 | -0.2770 |
| chord_shuffle | -0.0128 | -0.0304 |
| attack150ms | -0.0913 | -0.4045 |
| attack50ms | -0.0880 | -0.1271 |
| release_trim | +0.0461 | -0.1458 |
| pitch_reverse | -0.0062 | -0.0021 |
| pitch_shuffle | -0.0068 | -0.0038 |
| single_note | -0.1116 | -0.5723 |
| pad_m12 | +0.0710 | -0.0597 |
| pad_m6 | +0.0571 | -0.0136 |
| pad_off | +0.0652 | -0.1589 |

## 真實歌曲轉移

| 因子 | 條件 | ΔPQ | 95% song bootstrap CI |
|---|---|---:|---|
| gain_control | gain_m6 | +0.0000 | [-0.0000, +0.0000] |
| local_order | 0.25 | -0.5197 | [-0.6712, -0.3706] |
| local_order | 1 | -0.0735 | [-0.0994, -0.0497] |
| noise | pink10 | -1.6024 | [-1.9625, -1.2606] |
| noise | pink20 | -0.3851 | [-0.5442, -0.2451] |
| noise | pink30 | -0.0825 | [-0.1231, -0.0468] |
| noise | pink40 | -0.0179 | [-0.0294, -0.0085] |
| noise | pink50 | -0.0028 | [-0.0044, -0.0013] |
| noise | white10 | -2.1364 | [-2.4907, -1.8044] |
| noise | white20 | -0.6352 | [-0.8386, -0.4555] |
| noise | white30 | -0.1741 | [-0.2358, -0.1163] |
| noise | white40 | -0.0477 | [-0.0722, -0.0270] |
| noise | white50 | -0.0108 | [-0.0172, -0.0054] |
| quiet_frames | -12 | -0.4429 | [-0.6801, -0.2471] |
| quiet_frames | -6 | -0.2521 | [-0.4036, -0.1284] |
| spectral_balance | dark_m6 | -0.0015 | [-0.0133, +0.0116] |

## 驗證、統計與缺口

- MIDI 音符、時間、音符庫、增益與音訊 hash 檢查見 preflight_validation.json。評分 coverage、響度對齊與四軸控制檢查見 final_validation.json。
- discovery/validation 為預先固定的 32+32 模板；模板內先平均音色庫與情境。不是把音訊／窗數當獨立樣本。12 首歌亦先在歌曲內平均主副歌；30 秒重疊片段不進主要歌曲 CI。
- 七個事先指定的主要條件 × 四軸，共 28 個驗證集 sign-flip tests，Holm 校正另見 primary_validation_tests.csv。CI 是探索性區間，並未同時校正。
- raw block-order control 的不變性來自官方 10 秒窗平均架構。這不能單獨宣稱 PQ 有不合理偏誤，也不能測到整曲結構品質。
- 噪聲主效應與 noise × instrument/register 交互作用見 interactions.csv。合成測試不能證明 NegMF 的實際中介機制。
- NegMF/control 三 seed 原始生成 FLAC 不在原評估目錄，轉移支線沒有執行；缺失明列於 design.json。舊報告分數未當作新介入分數。
- 尚無人類盲聽評分。blind_listening.html 是預先抽樣的配對介面，供主觀驗證，不代表驗證已完成。
- 音訊輸入有效模型精度為 float32、autocast=false；predictor.precision 的 bf16 只是未參與 forward 的標籤。前輪 provenance 只存該標籤，不能由此聲稱當時的 forward 真使用 bf16。

完整資料：summary.csv 四軸／bank／split／context；midi_paired.csv、karaoke_paired.csv；所有 MIDI 及音訊 audio/。

參考：[官方程式](https://github.com/facebookresearch/audiobox-aesthetics)、[AES 論文](https://arxiv.org/html/2502.05139v1)。

樂器代碼：0=piano、24=nylon guitar、32=acoustic bass、40=violin、73=flute、89=warm pad。它們指前景音色；伴奏 guitar/bass 固定。

額外判讀限制：lead_duration1.7 出現同 pitch/channel 重疊，合成器 note-off 可能一起釋放重疊 voice；它只能作延音／重疊的複合介入，不可當成獨立 sustain-duration 證據。段內 sample-block shuffle 也會在接縫產生邊界改變，不能把全部降分歸因於音樂順序。midi_manipulation_checks.csv 記錄和弦音匹配率；quiet_frame_manipulation_checks.csv 記錄響度對齊後實際的低能量變化，避免把設定增益當成達成的 noise-floor 下降。

歌曲主／副歌標籤沿用成熟 All-In-One 模型的自動預測，尚未人工確認；本輪整體歌曲介入效果不依賴主／副歌分組正確，但不能把分組差異視為已完成的段落效度結論。
