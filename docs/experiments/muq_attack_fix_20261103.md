# MuQ-Eval 低通攻擊 → 前處理修補 → TTM 16k 重評（2026-11-03）

> 設計文件；判準在跑評分前寫定，之後不改。前情：`docs/experiments/results/muq_lowpass_probe_20261028_results.md`（寬頻真實音樂 lp4–6 kHz 讓 A1 MI +0.38～+0.46，ODAQ LP 人類 MUSHRA 98→18 時 MuQ 不降、LP50 最高）。候選清單：`docs/experiments/muq_weakness_probe_candidates_20261028.md` §4。

## 目的

1. 問「評分前套一個簡單前處理能不能修掉低通攻擊」——不訓練、不改權重、不對抗最佳化，只改送進評分器的輸入。
2. 問「我們的 TTM 表（115，lvl30）能不能用 MuQ-Eval 當指標」——6 個系統裡只有 musicgen_medium 是 32 kHz，其餘 16 kHz；若 MuQ 偏袒窄頻，跨系統比較就被取樣率左右。把 musicgen 降到 16 kHz 重評，看排名與成對差是否改變。
3. 跑候選 3／4／5（dither、增益階梯、分離重組），補齊 MuQ 前端機制的觀察證據。

## 腳本與輸出

- 腳本：`scripts/eval/muq_attack_fix_20261103.py`（import `muq_lowpass_probe_20261028.py` 的素材載入與工具函式，不修改既有腳本）
- 輸出：`~/nvme_experiment_artifacts/meanaudio/muq_attack_fix_20261103/{per_clip.tsv, summary.json, run.log}`
- 模型：MuQ-Eval A1、A3a；MI 與 TA 都存，**判讀以 MI 為主**。
- 增量寫入 `per_clip.tsv`，重跑時跳過已完成的 (part, set, sys, item, fix, transform)。

## Part A：修補方案

對每個輸入在評分前套用 F：

| 代號 | 前處理 |
|---|---|
| F0 | 不修（現狀對照） |
| F1 | 頻寬統一：soxr 降到 16 kHz 再交給評分器（評分器自己會重採到 24 kHz） |
| F2 | 更嚴：Butterworth 8 階 `sosfiltfilt` 低通 4 kHz 再評分（與既有 `lp()` 相同） |

先 mono 平均（與評分器內部一致），再套 F，避免 F0／F1／F2 的聲道處理不同。

### A1 攻擊是否消失

素材 `musdb`（MUSDB18-HQ 150 首中央 10 s，44.1 kHz）與 `odaqref`（ODAQ 30 個 reference，48 kHz），條件 clean／lp4k／lp5k／lp6k／lp8k／sr16k（沿用既有 `transform()`）。
量 Δ = score(F(lpX)) − score(F(clean))，逐首配對；bootstrap 95% CI（`boot()`，2,000 次），up/down 以 ±0.05 為界。

### A2 人類錨點

`odaqlp`：ODAQ LP 4 trial × 8 條件（reference＋LP35…LP150），用官方釋出的音檔與 `ODAQ_results.csv` 的 MUSHRA。修補後算 pooled Spearman（MuQ vs MUSHRA）與 trial 內 Spearman；另報每個條件的平均分，檢查 LP50／LP70 是否仍高於 reference。

### A3 效度不受損

- `mc522`：MusicCaps-522 的 clean 分數與人類 PQ 的 Spearman，F0／F1／F2 各算一次。素材是 16 kHz，所以 F1 應與 F0 幾乎相同（sanity）。
- `me`：MusicEval manifest（`/mnt/seagate/benchmarks_r2/manifests/me.tsv`，附 `human_OVL`）的 clip 層 Spearman，F0／F1／F2 各一次。**只評 dev＋test（825 段）**，因為 train（1,923 段）在 MuQ-Eval 的訓練來源裡，in-sample 相關沒有意義；summary 同時附 all（含 train）當註記。

## Part B：TTM 表 16k 重評

- 素材：`/home/kojiek/eval_output_nvme/ttm_metrics_115_20261009/<sys>/lvl30/audio/*.flac`，6 系統：meanaudio_cfg0、meanaudio_cfg3_neg、musiccaps_real、musicgen_medium、musicldm、musicldm_fid8。
- 6 系統共同檔名 5,131 個；固定 seed 0（`np.random.default_rng(0)`）不置回抽 1,000 個。「prompt」以檔名 stem 計。
- 全部系統在 F0 評分；musicgen_medium 另加 F1（32 kHz → 16 kHz）。musiccaps_real 本來就是 16 kHz，不加 F1。
- 報：每系統每讀數（A1／A3a × MI／TA）平均＋bootstrap 95% CI；F0 排名 vs「musicgen 換成 F1」的排名；musicgen 逐檔 Δ(F1−F0) 與 up/down；musicgen 對 meanaudio_cfg3_neg 的同檔成對差在 F0／F1 下各多少（paired bootstrap，2,000 次）。

## Part C：候選 3／4／5（MUSDB 150＋ODAQ ref 30，F0）

- **候選 3 dither**：20 ms 幀，只在 mono RMS < −50 dBFS 的幀加高斯白噪，振幅 std = 10^(L/20)，L = −90／−80／−70 dBFS。逐檔 Δ（對 F0 clean）＋CI，並報單段最大 |Δ|。
- **候選 4 增益**：−30…+6 dB 每 3 dB（0 dB 用 clean 不重算），float32 相乘、**不截幅**。逐檔 Δ＋CI、單段最大 |Δ|。
- **候選 5 分離重組**：用 126c 已快取的 htdemucs stems（`/home/kojiek/eval_output_nvme/metric_win_tie_126c_20261019/stimuli/cache/<win>_dmx.npz`，16 kHz mono，551 個 11 s 窗、覆蓋 145／150 首）。每個窗算 mixture（基準）、true stemsum（真四軌相加）、dmx_resum（htdemucs 四軌相加），Δ = 後兩者 − mixture。未覆蓋的 5 首不跑：`~/venvs/dac` 沒有 demucs，規定不得 pip install（`~/venvs/karaoke-jp-align` 有，但換 venv 跑分離再回 dac 評分超出這次範圍）。
- noisereduce 不在 `~/venvs/dac` → 跳過「降噪」並在結果註明。

## 預登錄判準（跑分前寫定）

1. **修補消除攻擊**：musdb 與 odaqref 的 **A1 MI** lp4k／lp5k／lp6k Δ 在修補後 **CI 上界 ≤ +0.05**（不再加分）。
2. **修補方向對齊人類**：ODAQ LP 修補後 **pooled Spearman（MuQ vs MUSHRA）> 0.3**，且 **LP50 不高於 reference**。
3. **修補不損效度**：mc522 人類 PQ 的 Spearman 下降 **≤ 0.05**（相對 F0）；MusicEval（dev＋test）同條件。
- 一個修補方案三條都過才算「可用修補」；**只過第一條**要寫成「只消除跨頻寬偏袒，沒有讓指標學會低通是劣化」。
- **TTM**：musicgen 改 16 kHz 後與 meanaudio_cfg3_neg 的成對差若**改變正負號**，或 **CI 從不跨零變成跨零**，判為「MuQ 的 TTM 比較受取樣率左右」。

## 實作註記（跑分前）

- **F2 不是逐位退化**：F2(lp4k(x)) 與 F2(clean) 是「4 kHz 低通兩次」對「一次」，級聯後響應更陡，smoke run 上已看到 lp4k 有 ±0.1～0.26 的差。判準 1 照量、不假設為 0；F2 的成敗主要仍由判準 2、3 決定。
- MusicEval 的 `human_OVL` 是 1–5 分的主觀整體分數，不是 MOS；只用 clip 層 Spearman。
- ODAQ LP 每個 trial 只有 8 個點、trial 內 Spearman 的變異很大，判準 2 只看 pooled 與 LP50 vs reference。
- Part C 的 stems 素材是 16 kHz mono（126c 快取），與 Part A 的 44.1／48 kHz 素材不同；候選 5 只判「分離重組的分數變化」，不與 Part A 的 Δ 直接比大小。

## 執行方式與預估

- 評分總量約 3.7 萬次 forward（Part A ≈ 7.4k 次／模型、Part B 7k 次／模型、Part C 8.7k 次／模型，已扣掉重複的 clean）。
- 依 2026-10-28 probe 的實測速率估約 1.5–2.5 小時。GPU 目前空閒、p2 queue 無 job，直接在 tmux 跑並掛 `notify_on_exit`；腳本可續跑（per_clip.tsv 增量、重跑跳過已完成列），中途掛掉不用整個重來。
- 先用 `--part A --limit 5` 之類的 smoke run 確認流程與數值合理，再全量。
