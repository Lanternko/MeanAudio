# 126d：SongEval 在 126 刺激集上的兩條要求測試

設計登錄日期：2026-10-20
設計登錄人：Claude Code（派工）
GPU 任務：p2 132（`132_metric_win_tie_songeval.sh`）
收線依據：`docs/experiments/results/metric_win_tie_126d_results.md`

---

## 0. 背景與動機

126（訊號層）、126b（音符層 + 錄音）、126c（多軌真實歌曲）三輪測試得出：**沒有任何指標同時通過兩條要求**（偵測明顯差距 + 判平相近差距）。神經指標（AES PQ/CE/CU/PC、MEva、CLAP）對音符層錯誤（錯音、亂音高、亂節奏）以及多軌錯位/走音/換軌完全偵測不到。

開放問題：**受過音樂性判斷訓練的指標能不能做得更好？** 候選指標：**SongEval**（arXiv 2505.10793，ASLP-lab/SongEval，MuQ 骨幹）。SongEval 在 2,399 首含人聲 AI 生成歌曲上訓練五個美學維度，與人類評分 Pearson 0.91（見 `docs/literature/quality_metric_validity_2026_09_18.md` §2.1）。

**重要的域警告（登錄）**：SongEval 訓練在整首含人聲歌曲（平均 3.51 分鐘），而 126 刺激集主要是 10 秒片段，其中大量是純器樂（126a TTM 系統輸出）或 MIDI。維度「人聲換氣自然度」和「歌曲段落清晰度」在無人聲材料上定義不明。故**同一域外警告已預先寫入**，不在事後解釋失敗時使用此點。

---

## 1. 判定協定（沿用 126）

- **d** = score(case) − score(clean base，同一 clip 或同一窗)
- **tie（平手）**：|d| ≤ m，m = 0.10 × S
- **S 定義（預登錄）**：
  - 對 SongEval 五個維度各自定義 S：S_i = max(μ_{sys,i}) − min(μ_{sys,i})，μ_{sys,i} = 5 個 TTM 系統（real/ours_cfg3neg/ours_cfg0/musicgen/musicldm）在 5,131 個 MusicCaps clean clips 上的第 i 維 SongEval 均值
  - 這與 126a/126b/126c 對神經指標的 S 定義完全平行
  - S 值在評分完成後計算，但定義在開始評分前固定
- **通過門檻**：≥ 80% 的 pair 判對
- **PC（AES Production Complexity）無方向**，SongEval 五維均有方向：分數越高代表越好

---

## 2. 預登錄方向預測（評分前）

### 2A. 應偵測到差距（lose / should-be-worse）的介入

| 介入 | 預期通過的 SongEval 維度 | 理由 |
|---|---|---|
| wrong30pct+（30% 以上錯音） | se_mus、se_coh | 大量音符錯誤應破壞音樂性與連貫性 |
| randpitch（100%） | se_mus、se_coh | 隨機音高應明顯破壞音樂性 |
| posshuffle（100%） | se_mus、se_coh | 隨機位置亂序接近噪音 |
| pitchshuffle_all | se_mus | 所有音符音高重排 |
| blockshuffle0.5s | se_mus、se_coh | 0.5 秒塊亂序破壞時序連貫 |
| drums_lag ≥ 100 ms | se_mus、se_coh | 鼓錯位應破壞節奏連貫性 |
| vocals_detune 100c | se_nat（人聲走音） | 1 個半音走音明顯 |
| noise_snr ≤ 20 | 全部 | 明顯訊號失真 |

預測「SongEval 能偵測」定義：≥ 80% 判差，與 AES 失敗的格（< 80%）在同類介入中存在

### 2B. 應判平手（tie / soft）的介入

| 介入 | 預期 | 理由 |
|---|---|---|
| gain ±1 dB | 全部判平 | 不可聽見的音量差異 |
| polarity | 全部判平 | 相位翻轉不影響音樂性 |
| PCM_16 | 全部判平 | CD 品質格式轉換 |
| transpose ±1 半音（MIDI） | soft（可接受不完全通過） | 移調不改變音樂性 |
| tempo ±2%（MIDI） | soft | 速度微調 |

### 2C. 判定「SongEval 不優於 AES」的準則（預登錄）

以下任一條件成立即判「SongEval 未優於 AES」：
1. 所有 SongEval 五個維度，在 126a 中 AES 失敗的 lose 格（錯音系列 wrong/randpitch/posshuffle/pitchshuffle/blockshuffle），通過率均 < 80%
2. 在 126c 的 drums_lag100ms、vocals_detune100c，所有 SongEval 維度均 < 80%
3. 在 126a/126c 的 gain ±1dB 和 polarity，任何 SongEval 維度通過率 < 80%（即也沒有平手能力）

反面情況（SongEval 優於 AES）：在 AES 失敗的≥ 1 個 lose 格，SongEval ≥ 1 個維度通過，且同時通過 gain ±1dB/polarity 平手要求。

---

## 3. 刺激集規格

### Part A-1：126c（MUSDB18-HQ 分軌編輯）
- 36,366 個 FLAC 已存在：`/home/kojiek/eval_output_nvme/metric_win_tie_126c_20261019/stimuli/audio/**/*.flac`
- 對每個 FLAC 直接評分 SongEval（24 kHz 重採樣）
- SongEval 分數儲存到 `{OUT_126D}/126c_songeval.tsv`（key, se_coh, se_mem, se_nat, se_cla, se_mus）

### Part A-2：126b（MIDI + 錄音 + 底線）
- 87,653 個 FLAC 仍在：`/home/kojiek/eval_output_nvme/metric_win_tie_126b_20261016/stimuli/audio/**/*.flac`
- 同上；儲存到 `{OUT_126D}/126b_songeval.tsv`
- **身分驗證閘門（預登錄）**：重新評分約 200 個 clip 的 AES PQ，與儲存的 126b 每 clip 分數（units/ 的 scores.tsv）比對，允許差異 ≤ 1e-3（AES batch composition drift）；任何超過門檻的差異需調查

### Part A-3：126a（TTM 系統訊號層介入）
- N = 5,131 clips per system（MusicCaps clean set 的完整集合）
- **固定 N 預登錄**：5,131（與 126a 相同的 clip list）
- 53 variants × 5 systems × 5,131 clips = 1,334,060 clip-scores
- 每個 (vname, system) unit 的 IDs 從 `units/{vname}/{system}/scores.tsv`（已完成的 126a 結果）讀取
- 生成：從 `R115/{system}/lvl30/audio/{id}.flac` 載入 → 套用 126a.transform() → SongEval 評分
- 儲存到 `{OUT_126D}/126a/{vname}/{system}/songeval.tsv`

### Part A-4：S 值計算
- 從 Part A-3 的 `clean/{system}/songeval.tsv` 讀取 5 個系統的 clean 均值
- 計算 S_i = max_sys(μ_i) - min_sys(μ_i) for i in {coh, mem, nat, cla, mus}
- 結果存入 `{OUT_126D}/s_values.json`

---

## 4. Part B 設計

### Part B-1：走音乾淨控制（rubberband 處理次數相同）
- **動機**：126c 的 vocal_pv0 (-50 到 -100c 走音的扣分大過走音本身，而 pv0 是兩次 rubberband)。需要一個「同樣兩次處理，但音高不完全歸零」的控制。
- **設計**：
  - A：+50c 再 −50c（等於 vocal_pv0，純處理，期望接近 d≈0）
  - B：+100c 再 −50c（淨 +50c，兩次處理）
  - C：+100c 再 −100c（淨 0c，兩次處理，更大的 artifact）
  - 差 B−A = 淨 +50c 效果（兩次處理時）；C−A = 處理增加效果
- **素材**：126c MUSDB windows，直接從 `stimuli/cache/{win}.npz` 讀取 vocals 軌
- **正規化**：每個 case 重新正規化到 −30 LUFS（與 126c 一致）
- **評分**：AES + MEva + SongEval + MIR（與 126c 相同套件）
- **儲存**：`{OUT_126D}/partb1_detune_control_songeval.tsv`
- **直接跑**（非 GPU 主 lock）：此任務主要是 CPU 生成 + GPU 評分，可在 126d queue job 內一起跑，或以帶 notify trap 的直接腳本跑；選擇在 queue job 內。

### Part B-2：Karaoke 12 首去軌 LUFS 比較
- **動機**：Codex 10/01 的去鼓 PQ −0.29 / 去人聲 −0.41 在 126c 的 MUSDB −30 LUFS 上沒重現（反而 +0.01/+0.26）。是素材效應還是響度設定？
- **設計**：
  - 條件 1（−30 LUFS）：`ttm_midi_karaoke_meva_20261012/audio/` 的現有 FLAC（已正規化）
  - 條件 2（原始響度）：把每個 FLAC 乘以 10^(gain_db/20)（用 manifest.tsv 的 gain_db 逆轉正規化），回到原始 LUFS（約 −8 到 −15 LUFS）
  - 比對：demucs_drums_off - original（d_drums），demucs_vocals_off - original（d_vocals）
  - 若兩個響度條件下 d 符號不同 → 響度是主因；符號相同 → 素材差異可能更重要
- **評分**：AES + SongEval（只需這兩個，MIR 非核心）
- **儲存**：`{OUT_126D}/partb2_karaoke_lufs.tsv`
- **執行位置**：在 queue job 內（輕量）

---

## 5. SongEval 安裝細節

**決策**：重用 `~/venvs/dac/`（不建新 venv）。所有依賴（muq==0.1.0、hydra-core==1.3.2、safetensors、librosa==0.11.0、einops、omegaconf）均已安裝；torch 版本（2.11.0.dev）與 SongEval 要求（2.7.0）不同，但實測功能正常。

**固定版本（預登錄）**：
- SongEval repo commit：`848fb2ff3a2a9d64dcb20d46f07238c28abd7add`
- SongEval repo 路徑：`/home/kojiek/MeanAudio/scripts/eval/songeval_src/`
- SongEval ckpt：`songeval_src/ckpt/model.safetensors`
- SongEval ckpt sha256：`0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd`
- MuQ backbone：`OpenMuQ/MuQ-large-msd-iter`
- MuQ HF revision：`0562a57814f6f8bbd9fdea0a25921a2fce1a841a`

**批次評分**：batch_size=16，輸入 24 kHz（16→24 kHz via librosa.resample）
- 決定論驗證：同一 clip 評分兩次，diff < 1e-6（實測 0.00e+00）
- 批次敏感度：bs=1 vs bs=16 的分數差 < 2e-4（實測 1.41e-4）；判定為可接受的數值雜訊，不影響排名

---

## 6. 磁碟規劃

- NVMe 現有空間：46 GB（99% 使用率）
- 126c 評分結果：~2 MB（36,366 clips × 5 dims × float32）
- 126b 評分結果：~4 MB
- 126a 評分結果：53 variants × 5 systems × 5,131 clips × 5 dims × 4B = ~54 MB
- 總新增：~60 MB
- 126a 臨時音訊：每個 unit 最多 ~5,131 × 50 kB × 1 system（一次一個 unit）= 256 MB，評分後刪除
- Part B：可忽略不計
- **峰值 NVMe 用量：< 500 MB（遠低於 20 GB 限制）**

**輸出目錄**：`/home/kojiek/eval_output_nvme/metric_win_tie_126d_20261020/`

---

## 7. 時間預算（基於煙霧測試）

- SongEval 吞吐量（bs=16，dac venv，RTX 5090）：~125 clips/s（24 kHz input）
- 126c scoring：36,366 / 125 = 291s ≈ 5 min
- 126b scoring：87,653 / 125 = 701s ≈ 12 min
- 126a scoring（1,334,060 clips）：10,672s ≈ 178 min ≈ 3.0 h（含 transform 生成）
- Part B：< 30 min
- 分析：< 15 min
- **總計：~3.5 h，遠低於 10.5 h 上限**

---

## 8. 停止規則

- 任何單一 unit 失敗（nonfinite score、CUDA OOM）：停止並回報，不繼續跑後續 units
- NVMe free < 5 GB：停止
- SongEval 全部五維在 clean clips 的 score 分布出現 >50% 落在 1.0 或 5.0（退化分布）：不報告 win/tie/lose

---

## 9. 結果判讀規則（預登錄）

- **可寫**：SongEval 在某個介入 ≥ 80% 判對或平手（與判准一致）
- **不可寫**：SongEval「已驗證不變性」（未設計不變性測試）
- **不可寫**：SongEval 的分數映射到音樂性以外的物理意涵
- **不可寫**：「SongEval 訓練在含人聲歌曲，因此結果不可信」——這是已知限制，不在事後解釋失敗時使用

---

## 10. 相關文件

- `docs/experiments/metric_win_tie_126_20261016.md` — 原始設計
- `docs/experiments/results/metric_win_tie_126_results.md` — 126 結果
- `docs/experiments/results/metric_win_tie_126c_results.md` — 126c 結果
- `docs/literature/quality_metric_validity_2026_09_18.md` — SongEval 文獻定位
- `scripts/eval/metric_win_tie_126d.py` — 主驅動腳本
- `docs/experiments/metric_win_tie_126d_20261020_contract.json` — 合約
