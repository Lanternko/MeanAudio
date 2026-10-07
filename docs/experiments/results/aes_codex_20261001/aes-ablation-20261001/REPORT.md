# AES 歌曲 property ablation：實測報告

12 首歌曲；891 個處理音訊；1782 次四軸評分。

## 主要發現

本批歌曲支持 AES PQ 對白噪聲敏感，但不支持「只要背景更安靜、音色更暗、音樂更稀疏就普遍加分」。

- 人工白噪聲設定 SNR 40/30/20 dB，ΔPQ 分別 −0.0138/−0.0638/−0.2382，呈劑量反應。設定以 44.1k stereo context 計算；實際 AES mono16 窗口 SNR 另存 independent_validation.json。
- 衰減估計的寬頻頻譜底層 +0.0434，高頻 hiss 候選 +0.0241；這些不是已確認的真實噪聲，仍需檢查 removed residual。
- quiet_frames_m12 實際使等響度低能量 p10 約降低 1 dB，PQ 約 +0.0047，CI 跨零；不是 NegMF 已觀察的約 −3.8 dB 操作。
- dark_shelf_m6 明確降低頻譜重心，PQ −0.0054，CI 跨零；單獨降低重心在本操作下沒有穩定加分。
- 去鼓 −0.2943；大量移除 HPSS harmonic 成分 −1.1572。HPSS 會改變音色與產生處理缺陷，因此不能泛化成所有 sustain/pad 操作。
- 伴奏降低 6 dB +0.0561，完全移除伴奏 −0.4131；內容密度與 PQ 關係不是簡單單調。
- 原曲 −23 LUFS 的歌曲等權平均 PQ 約 8.198，屬高分素材；本結果不能直接外推到生成音樂或 NegMF。

30 秒只有 4 首歌曲／4 個窗口。加噪、估計底層衰減、去鼓、harmonic 移除的方向與 10 秒相容；樣本小且窗口重疊，不能當獨立重現。

## 完整主分析

主分析：10 秒、16 kHz mono 後以純增益對齊 −23 LUFS。配對差值的 baseline 依處理來源選原曲或完整 stems 重組；每首先平均段落，再以歌曲 bootstrap 10,000 次。

主歌／副歌為 Harmonix-all 自動預測，尚未人工確認。30 秒只納入段落 >=32 秒者，與 10 秒重疊，不能視為獨立重現。CI 為探索性、未校正多重比較。

| 處理 | 基準 | ΔPQ | 95% bootstrap CI | 歌曲數 |
|---|---|---:|---|---:|
| demucs_recompose | original | +0.0697 | [+0.0413, +0.1035] | 12 |
| mel_accompaniment_m6 | mel_recompose | +0.0561 | [+0.0235, +0.0888] | 12 |
| broad_floor_m9 | original | +0.0434 | [+0.0279, +0.0624] | 12 |
| demucs_guitar_m6 | demucs_recompose | +0.0360 | [+0.0131, +0.0704] | 12 |
| hiss_m9 | original | +0.0241 | [+0.0160, +0.0346] | 12 |
| demucs_guitar_off | demucs_recompose | +0.0232 | [-0.0273, +0.0721] | 12 |
| broad_floor_m3 | original | +0.0204 | [+0.0131, +0.0295] | 12 |
| demucs_other_off | demucs_recompose | +0.0189 | [-0.0024, +0.0404] | 12 |
| demucs_other_m6 | demucs_recompose | +0.0165 | [+0.0021, +0.0314] | 12 |
| hiss_m3 | original | +0.0109 | [+0.0074, +0.0153] | 12 |
| quiet_frames_m12 | original | +0.0047 | [-0.0017, +0.0112] | 12 |
| dark_shelf_m3 | original | +0.0043 | [-0.0024, +0.0105] | 12 |
| demucs_piano_m6 | demucs_recompose | +0.0036 | [+0.0001, +0.0069] | 12 |
| quiet_frames_m6 | original | +0.0032 | [-0.0011, +0.0072] | 12 |
| mel_recompose | original | +0.0000 | [-0.0000, +0.0000] | 12 |
| gain_m6_control | original | -0.0000 | [-0.0000, +0.0000] | 12 |
| dark_shelf_m6 | original | -0.0054 | [-0.0179, +0.0059] | 12 |
| white_noise_snr40 | original | -0.0138 | [-0.0227, -0.0067] | 12 |
| demucs_piano_off | demucs_recompose | -0.0245 | [-0.0597, -0.0022] | 12 |
| demucs_bass_m6 | demucs_recompose | -0.0266 | [-0.0486, -0.0093] | 12 |
| demucs_bass_off | demucs_recompose | -0.0476 | [-0.0754, -0.0213] | 12 |
| demucs_drums_m6 | demucs_recompose | -0.0476 | [-0.0717, -0.0251] | 12 |
| harmonic_m6 | original | -0.0629 | [-0.1090, -0.0201] | 12 |
| white_noise_snr30 | original | -0.0638 | [-0.0983, -0.0357] | 12 |
| demucs_vocals_m6 | demucs_recompose | -0.1225 | [-0.1834, -0.0619] | 12 |
| mel_vocals_m6 | mel_recompose | -0.1794 | [-0.2675, -0.0936] | 12 |
| mel_vocals_off | mel_recompose | -0.1879 | [-0.3648, -0.0303] | 12 |
| white_noise_snr20 | original | -0.2382 | [-0.3182, -0.1630] | 12 |
| demucs_drums_off | demucs_recompose | -0.2943 | [-0.4111, -0.1807] | 12 |
| demucs_vocals_off | demucs_recompose | -0.4065 | [-0.7405, -0.1392] | 12 |
| mel_accompaniment_off | mel_recompose | -0.4131 | [-0.7041, -0.2007] | 12 |
| harmonic_off | original | -1.1572 | [-1.4423, -0.8376] | 12 |

## 解讀邊界

- hiss/broad_floor 使用估計頻譜底層，沒有 ground-truth 噪聲；必須聽 contexts/*/*_removed.wav，不能直接宣稱真正去噪。
- harmonic 組是 HPSS harmonic 成分衰減；會移除有意義的旋律與和聲，不是純 pad 或純殘響。
- quiet_frames 只控制低能量 frame，不能把數值當成真實噪聲底。
- stems 是模型估計，去樂器仍可能有 bleed/artifacts；完整重組與原曲差值另列。
- AES CE/CU 是同一家預測器的其他輸出，不能當成人類盲聽驗證。
- 本實驗可以測 AES 的處理敏感性，不能單獨證明 NegMF 機制或人類品質偏誤。

完整資料：design.json、manifest.json、features.csv、paired_results.csv、summary.csv、scores.json。

檢查：validation.json、independent_validation.json、smoke_validation.json。
主歌／副歌抽查：section_review.html。盲聽：blind_listening.html（230 個預先固定配對，作答可匯出；尚未收集人類評分）。
聽測使用 44.1k stereo 響度對齊，與 AES mono16 對齊分開記錄。答案鍵 listening_answer_key.json 不顯示在盲聽頁。

全部為本聊天獨立 sidecar，未修改 karaoke-jp canonical，未 commit。
