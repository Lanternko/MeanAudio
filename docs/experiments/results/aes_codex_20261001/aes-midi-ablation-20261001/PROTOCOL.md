# MIDI 精確介入 AES 實驗 — 2026-10-01

目的：不用分離器，直接控制音符與合成音色，測 AES 四軸敏感性。
本輪是合成音訊 pilot，不能直接等同人類音樂品質或 NegMF 的機制。

## 固定條件

- 8 個原創 MIDI 模板，SEED 20261001。每個模板使用 8 音 phrase 重複 4 次，
  permutation 與 velocity 在模板間變化。不是 8 首自然歌曲或 8 個模型訓練 seed。
- 120 BPM、32 個八分音符 onsets、velocity 約 74–94；標準 note duration 175ms。
- MIDI 真實保存並由 mido 重新讀入後合成；sample schedule 由 MIDI 的 tick 時間決定。
- 10 秒、44.1k stereo float WAV；初始 0.5 秒、末段保留 release。
- TinySoundFont 0.3.7、mido 1.3.3；renderer gain −6 dB，无額外 reverb/chorus。
- 每段 reset 並丟棄 100ms release ramp，音符前 490ms 必須為零。
  初版發現 reset 殘留後已全部覆寫；AES 尚未評分時完成修復。
- 兩個音色庫：GeneralUser GS、FluidR3 GM。音色庫實際 SHA256 固定在 design.json。
  音色不是跨音色庫的同一錄音，結果必須分音色庫檢查。

音色库來源：
<https://github.com/mrbumpy409/GeneralUser-GS>；Ubuntu fluid-soundfont-gm 3.1-5.3。
合成器來源：<https://github.com/nwhitehead/tinysoundfont-pybind>。

## 配對因子

1. 樂器：piano／nylon guitar／acoustic bass／violin／flute／warm pad。
   在每一音域內比較完全相同 MIDI notes、時序、velocity，對鋼琴作配對。
2. 音高／同音階不同音高：−24、−12、0、+12、+24 semitones，保留 pitch class。
   每個樂器分別與自身中央音域比較，不假設極端音域可由真實樂器演奏。
3. 調高：+2、+5、+7 semitones，保留音程；此時 pitch class 也改變。
4. 音階：major 對 minor、dorian、lydian、phrygian、whole tone、pentatonic。
   固定時序、velocity、音符數與 degree pattern。whole tone 的第七個 rank
   是八度根音；pentatonic 以重複 scale degrees 保留 32 音，並非相同音高分布。
5. 和聲：固定三音 polyphony，major/minor/open/cluster 局部和弦。
   每一旋律音作局部和弦根音。另比較相同三音音高的同時和弦與分解和弦。
   這不是完整的 I–vi–IV–V 功能和聲，也尚未測旋律與伴奏的和聲錯配。
6. 複數樂器：2／4 instruments 輪流分配原有 32 音，固定總音符數。
   另測 2／4 unison instruments，分別與 2／4 同音鋼琴層比較，固定疊層數。
7. 單音連發：MIDI 48／60／72，1／4／8 Hz，單音 duration 為間隔的 70%。
   三個 rate 的總 MIDI note duration 均為 5.6 秒；attack 數、release 與聲學包絡不同。
   同為 4 Hz 的 C4 連發另與原旋律比較，固定時序與 velocity。
8. 加噪：鋼琴基準，加 white/pink noise，44.1k stereo 全段 SNR 40／30／20 dB。
   同模板使用固定 noise realization；實際 AES mono16 SNR 另外量測。
9. 純增益 −6 dB 控制組，響度對齊後應近乎歸零。

## 評分與統計

與前輪使用同一官方 AES、相同模型 revision、bf16、batch 8。
先 scipy polyphase resample 到 16k mono，純增益對齊 −23 LUFS。
每段另外評 raw；不壓縮、不限制峰值、不量化 PCM16。

每個基準配對、所有四軸均保存。每個模板先平均音色庫與樂器／音域情境，
再對 8 個模板 bootstrap 10,000 次。分層結果同樣列出。
這些是未校正多重比較的探索性 CI，不是 2,160 個獨立音樂樣本。
保留 pitch range、centroid、crest、harmonic share 等，不能把音階差異
直接解讀為 AES 看懂了音樂理論；單音源的錄音、頻譜、包絡也有可能解釋分數。

每個 score 與 MIDI/WAV 雜湊固定；檢查 pure gain、合成器重置、MIDI note counts、
配對 note inventory、固定 polyphony、加噪真實 SNR 与長度／finite。
完整資料在此聊天 outputs，依賴／原音色庫在 work；未修改 karaoke-jp，未 commit。
