# AES 音樂 property ablation — 2026-10-01

## 問題與判讀

測量 AES PQ 對背景能量、聲音組成、頻譜與持續諧波的介入敏感性。
這是 12 首既有歌曲的探索性配對實驗，不能單獨證明人類偏誤，也不能
把真實歌曲的結果直接歸因到 NegMF。人類盲聽需另收集。

## 素材與選片

素材僅讀取 `/home/kojiek/karaoke-jp`，不改 canonical 音軌。
選取 12 首已有 source/vocals/instrumental 的不同歌曲，屬便利樣本。
使用 All-In-One 的 Harmonix-all 八折 ensemble，透過 all-in-one-infer 3.1.0
分析完整歌曲。來源：<https://github.com/mir-aidj/all-in-one>、
<https://github.com/openmirlab/all-in-one-infer>。

每首選最長的預測 verse 和 chorus（並列時選最早；至少 12 秒）。
取中央 10 秒，離預測邊界至少 1 秒。段落至少 32 秒才另取中央 30 秒。
所有切點在 AES 評分前寫入 manifest.json 並固定。段落自動預測不等於人工真值；
section_review.csv 留有人工確認欄。模型 softmax 平均值只是 activation，非校準信心。
30 秒與 10 秒重疊，當成不同窗口長度的穩健性分析，不稱獨立驗證。

## 處理（共 33 arms）

- 原曲、只減少 6 dB 增益控制、Mel-Band 完整重組、Demucs 完整重組。
- Mel-Band 人聲與伴奏分別衰減 6 dB、移除（120 dB 近似）。
- htdemucs_6s 的 vocals/drums/bass/guitar/piano/other，各衰減 6 dB、移除。
- hiss：估計 STFT 各頻點第 10 百分位底層，只在 3.5–4.5 kHz 漸進啟用；
  最大衰減 3/9 dB。不是已確認的 hiss 真值。
- broad_floor：同一估計底層，作用在所有頻率，最大衰減 3/9 dB。
- quiet_frames：20 ms frame RMS 的第 25 百分位門檻，下方柔性衰減，
  最大 6/12 dB，gain 以 30 ms 平滑。這測低能量背景，不等同去噪。
- dark_shelf：二階 2 kHz 零相位低通與原波形混合，高頻漸近衰減 3/6 dB。
- harmonic：STFT 17-frame/17-bin 中值濾波的 HPSS soft mask，harmonic
  成分衰減 6 dB／近似移除。不是 pure pad、pure sustain 或 pure reverb。
- 白噪聲控制：固定 random seed 的獨立雙聲道白噪聲，context 全段 RMS
  SNR 40/30/20 dB；窗口內實際能量另有 features 可查。

Demucs 分離以選片前後各 5 秒上下文執行，shifts=0、overlap=0.25。
DSP 處理同樣先作用於上下文，再抽 10/30 秒，減少切片邊界影響。
Mel-Band 使用既有完整歌曲 stems，保留聲道與原比例，沒有獨立 peak normalize。
所有模型 stems 屬估計，有 bleed、缺失和 artifact；完整重組作為 matched baseline。
Mel-Band 舊 stems 的當初模型設定若無紀錄，不能視為可重跑同一分離結果；
此輪以實際輸入 stem SHA256 固定素材。

## 評分與檢查

輸出音訊為 44.1 kHz stereo float WAV，沒有 clipping、limiting 或重新壓縮。
AES 前以 scipy polyphase 轉 16 kHz mono。每個版本評 raw 和以純增益
對齊 −23 LUFS 的 lufs23；响度計在 AES 實際看到的 16 kHz mono 上量測。
AES 使用官方預測器、bf16、batch 8，記錄版本、模型與素材雜湊。

主要比較為 10 秒 lufs23。去 stem 對完整重組比較；其他 DSP 對原曲比较。
PQ/CE/CU/PC 全部記錄。檢查長度、finite、響度、只改增益控制歸零、
重組 residual、spectral centroid、crest、harmonic share、low-energy proxy。
保留 hiss/broad_floor/quiet_frames/harmonic 的 removed residual 供耳測。

每首先平均兩個段落，以歌曲 bootstrap 10,000 次，seed=20261001。
主副歌另列；所有 CI 都是未校正多重比較的探索性區間。
沒有用不同 arm 或重疊窗口膨脹歌曲樣本數，沒有事後挑最佳片段。
不將 AES CE 當人類評分，不宣稱「去噪證實」「NegMF 機制證實」。

## 重跑

在本聊天工作目錄，以 `/home/kojiek/venvs/dac/bin/python` 執行。
`PYTHONPATH=work/structure-packages` 提供本輪隔離安裝的結構辨識依賴。
先 `run_structure.py`，再 `run_ablation.py plan/separate/generate/score/report`。
score 有 cache 恢復，並比對音訊雜湊；manifest 已存在時不覆寫選片。

尚未 commit。所有正式輸出在本目錄，scratch 和套件在聊天的 work/。
