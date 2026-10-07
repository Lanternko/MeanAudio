# AES 合成音／單音 × VAE 重建 probe（006，2026-10-07）

## 問題

生成音訊的 AES PQ 能高過原音，而且 081 HQ＋LQ 在乾淨 caption 上仍勝原音 +1.10（見 `results/aes_gen_exceeds_ref_audit_20260930_results.md`）。稽核已排除兩件事：響度不是主因，曲風／節奏偏好也沒有證據。目前仍缺兩個直接的量：

1. **AES 會給「極簡、極乾淨」的訊號多高分？** 一個合成單音、一段 MIDI 旋律，能不能超過真實錄音？
2. **7.06 這個 VAE＋BigVGAN「天花板」，是 VAE 的硬上限，還是只是真實錄音內容的上限？** 如果乾淨合成音經過 VAE 重建後仍有 PQ ≈ 8，那麼生成器要超過 7.06，並不需要超越 VAE 的保真度，只需要產生「比訓練語料更乾淨」的內容。

## 先前做過的事（Codex，2026-10-01～03，未入 repo）

10/01 有一個 Codex session 跑過三輪 AES ablation，產出都在 repo 外：`~/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/`（約 77 GB，未 commit，沒有實驗 doc，所以先前沒找到）。三輪如下：

- `aes-ablation-20261001`：karaoke 12 首、23 片段、33 種處理。去鼓 PQ −0.29、去伴奏 −0.41、白噪 SNR 20 −0.24；壓低安靜段、降低頻譜重心都不加分。
- `aes-midi-ablation-20261001`：8 模板 × 2 音色庫（GeneralUser-GS、FluidR3），共 2,160 段，−23 LUFS。降八度 −0.34、音簇和聲 −0.39、四樂器 +0.16、單音連發 −0.22、白噪 SNR 20 −0.98；樂器排名隨音色庫改變。
- `aes-extended-20261001`：64 模板，共 19,531 段。**乾淨 MIDI 旋律（−23 LUFS）PQ 平均 8.26（p10 8.04、最高 8.51），2,304 段全部 > 7.55**。白噪 SNR 20 −2.13；移除 pad PQ +0.07、CE −0.16。

本 probe 實測以下兩項：先前**沒有做過真正的單一音符**，也**沒有經過我們的 VAE＋BigVGAN**。延伸那輪的乾淨旋律直接重用，音色庫和 tinysoundfont 0.3.7 已複製並以 sha 釘住，放在 `~/nvme_experiment_artifacts/meanaudio/aes_midi_assets/`。

## 刺激（全部 16 kHz mono、10 s、peak-norm 0.95；與稽核 stage1 測 7.55／7.06 的條件相同）

| 組 | 內容 | n |
|---|---|---:|
| **S 單音** | 2 音色庫 × 6 樂器（GM 0 鋼琴、24 尼龍吉他、32 原聲貝斯、40 小提琴、73 長笛、89 warm pad）× 5 音高（MIDI 36/48/60/72/84）× 2 包絡（held：0.5→9.5 s 持續；short：0.5→1.0 s 後自然衰減），velocity 100，tinysoundfont gain −6 dB、44.1 kHz 合成後重取樣 | 120 |
| **C 對照** | 正弦波（5 音高 × 2 包絡，10 ms 淡入淡出）、數位靜音、白噪、粉紅噪（全長） | 13 |
| **M 旋律** | 從 Codex extended 的 2,304 段 `*_clean` 中以 seed 20261007 抽 256 段，取前 10 s（原檔已是 16 kHz mono 30 s） | 256 |
| **J 錨點** | 稽核 stage1 `jam/audio` 中以 seed 20261007 抽 64 段（已 peak-norm），重新評分並重新過 VAE | 64 |

每段都產生四個版本：`pk`（原樣）、`vae`（mel → VAE encode mode → decode → BigVGAN，與稽核相同的 `RoundTrip`）、`pk_lvl30`、`vae_lvl30`（−30 LUFS 純量增益，peak cap 0.999，與 stage2 相同）。AES 用 `eval_metrics.score_aes`（batch 32，cuDNN TF32 預設開，與 7.06／7.55 的計分條件相同；批次漂移上限約 0.03，見 negmf 099/100）。另外對 pk 與 vae 版本計算稽核 stage1 的 14 個聲學特徵。

## 參考線

- 真實 MusicCaps 6.90
- Jamendo 語料 7.55（lvl30 數字讀 stage2）
- Jamendo 經 VAE 7.06（lvl30 7.33）
- 081 HQ＋LQ lvl30 8.20
- 全零輸入 6.735

## 驗收閘（不過則結果無效，不解讀）

- G1：數位靜音的 PQ 落在 6.735 ± 0.01。
- G2：J 錨點的 `pk` 重評分與 stage1 `jam` 逐段 |ΔPQ| 平均 ≤ 0.03；`vae` 與 stage1 `jam_vae` 平均 ≤ 0.05。
- G3：沒有 NaN；所有刺激的 RMS > 1e-5（靜音除外）；四個版本齊全。

## 預登錄讀數（主讀 `pk`，bootstrap 以段為單位，10,000 次，seed 20261007）

- **R1 單音**：S 的 PQ 中位數 ≥ 7.55，且平均的 95% CI 下界 > 6.90 → 「一個乾淨單音就勝過真實音樂語料」。另報超過 6.90／7.55／8.20 的比例，以及按樂器、音高、包絡的分層數字。
- **R2 VAE 天花板**：讀 M 的 `vae` 平均。
  - 95% CI 下界 > 7.36（= 7.06 + 0.30）→ **天花板取決於內容**：VAE 不會把乾淨內容壓到 7.06。
  - CI 上界 < 7.36 → **VAE 本身封頂**。
  - 介於兩者之間 → 不確定。
  - 另報逐段配對 Δvae（vae − pk），比較 M、S、J 三組；J 預期約 −0.49。
- **R3 樂器音色是否必要**：正弦單音的 PQ 對上同音高、同包絡的樂器單音中位數。這組只有 10 段，只描述，不做判定。
- **探索性**：哪些聲學特徵在 VAE 前後移動最多（M vs J），以及特徵與 PQ 的相關。不做判定。

## 不能回答的事

- 合成音高分不等於 AES 有「偏誤」：人類可能也覺得乾淨合成鋼琴的「製作品質」高。要判斷偏誤需要人類評分，見盲聽線 `blind_listen_20261006.md`。
- 本 probe 不碰模型生成。若 R2 判定「天花板取決於內容」，下一步是把 081／NegMF 生成音訊的特徵分布放到 M／J 之間比較；那批音訊有一部分已經缺失（Codex extended 的 NegMF 支線就是因此沒跑），要先盤點。
- 兩個 GM 音色庫加一個合成器，不代表所有合成音。

## 執行

走 p2 queue，編號 006 插隊（排在 107 之後、108 之前）。預計 GPU 約 5～10 分鐘，連 CPU 特徵約 15 分鐘。
不直接跑的原因：10/01 那三輪直接跑、產出放在 repo 外，結果一週後找不到。queue 會強制要求 contract、報告 sha 和 Discord 通知。

- action：`scripts/eval/aes_midi_note_vae_probe_20261007.py`（`--preflight`／`--validate-only`）
- 輸出：`~/nvme_experiment_artifacts/meanaudio/aes_midi_note_vae_probe_20261007/`（`scores.tsv`、`summary.json`、`audio/`）
- 結果：`docs/experiments/results/aes_midi_note_vae_probe_20261007_results.md`
