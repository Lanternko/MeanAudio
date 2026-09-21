# slot0nmv2 vs slot0clean — 3-seed paired quarter ablation (queue 066–071)

**狀態**：收線（2026-09-21）。6 個 run 全部完成，12 格 eval 全部 5,521/5,521。
**判準來源**：`docs/experiments/caption2p0_slot0nmv2_nmv2pair_quarter_s*_contract.json` 的 `decision_rule`（launch 前預先登錄）。
**數字檔**：`nmv2pair_three_seed_readout.json`（由 `scripts/analysis/nmv2pair_three_seed_ci.py` 產生，可重跑）。

## 設定

- arm：`slot0nmv2`（只在 slot0clean 上移除量測詞：BPM / 拍號 / 調性 / Hz / dB / 時長；decades、808、8-bit、12-bar 逐字保留）
- 對照：`slot0clean` 本身，限制到同一組 251,596 id、同順序 → 單變數成對比較
- seeds：14159265 / 27182818 / 16180339，recipe 同 057/058/059（quarter：S1 100k + S2 50k，NoQ、NoMask、lr 1e-4、batch 8）
- eval：`scripts/eval/mc_mf25_eval.sh`（MusicCaps 5521、MeanFlow 25 步、CFG0 與 CFG3+neg 兩格、CLAP batch 1）

## 結果：delta = slot0nmv2 − slot0clean

### CFG0

| metric | s14159265 | s27182818 | s16180339 | mean | 95% CI (df=2) | seed floor | 判讀 |
|---|---|---|---|---|---|---|---|
| CLAP | +0.0054 | −0.0034 | +0.0032 | **+0.0017** | [−0.0097, +0.0132] | 0.0042 | inconclusive；**非劣性 FAIL**（CI 下界 −0.0097 < −0.0084） |
| PQ | +0.1017 | −0.0064 | +0.0415 | +0.0456 | [−0.0890, +0.1801] | 0.0523 | inconclusive（符號不一致） |
| CU | +0.1146 | −0.0164 | +0.0284 | +0.0422 | [−0.1232, +0.2076] | 0.0520 | inconclusive（符號不一致） |
| CE | +0.1515 | −0.1145 | +0.0419 | +0.0263 | [−0.3059, +0.3585] | 0.1343 | inconclusive（符號不一致） |
| PC | −0.0330 | −0.1146 | +0.0653 | −0.0275 | [−0.2512, +0.1963] | 0.0554 | inconclusive（符號不一致） |

### CFG3+neg

| metric | s14159265 | s27182818 | s16180339 | mean | 95% CI (df=2) | seed floor | 判讀 |
|---|---|---|---|---|---|---|---|
| CLAP | +0.0052 | −0.0042 | −0.0007 | **+0.0001** | [−0.0118, +0.0120] | 0.0042 | inconclusive（符號不一致） |
| PQ | +0.1038 | −0.1047 | +0.1460 | +0.0484 | [−0.2850, +0.3818] | 0.0523 | inconclusive（符號不一致） |
| CU | +0.1367 | −0.1025 | +0.1233 | +0.0525 | [−0.2814, +0.3864] | 0.0520 | inconclusive（符號不一致） |
| CE | +0.1319 | −0.1178 | +0.0902 | +0.0348 | [−0.2975, +0.3670] | 0.1343 | inconclusive（符號不一致） |
| PC | +0.1675 | +0.0352 | +0.1416 | +0.1148 | [−0.0594, +0.2890] | 0.0554 | 三 seed 同號但 CI 跨 0 → 不可宣稱 |

## 結論

1. **兩格 CLAP 的點估計都貼著 0**（+0.0017 / +0.0001），而且逐 seed 翻號。去量測在 MusicCaps CLAP 上**沒有可測效果**。
2. **預先登錄的非劣性閘沒過**：CFG0 CLAP 的 CI 下界 −0.0097 略低於 −0.0084（= 2× CFG0 seed floor）。依 contract，這要報成 **inconclusive，不能報成「打平」**。n=3 的 t 區間本來就寬（df=2，t=4.30），單靠三個 seed 還不足以把這個效果框到 ±0.0084 以內。
3. **四項 AES 全部 inconclusive**。唯一三 seed 同號的是 CFG3+neg 的 PC（+0.115），但 CI 跨 0，且 PC 是 SongEval 相關性最低的一項（0.408，見 `reference_quality_metric_validity_literature`），不當結論用。
4. **arm-level 的系統性差異在響度側，不在語義側**：
   - `level_crest_mean`：nmv2 在 **6/6 格**都較高（CFG0 +0.24、CFG3+neg +0.36）。依 061，crest 不可當因果目標，但這是本次唯一完全一致的差異。
   - `level_lufs_mean`：nmv2 在 6 格中 5 格較小聲（CFG0 −0.36 dB、CFG3+neg −0.75 dB 平均）。依 063/065，小聲會抬 PQ/CU、壓 CLAP → 那些偶爾出現的 +0.10 PQ 帶響度污染，不算乾淨增益。
   - `level_silent_n`：nmv2 在 6 格中 5 格較多（CFG0 43.3→61.3、CFG3+neg 77.0→92.3），唯一例外是 CFG3+neg s16180339（49→29）。比 057 的 slot0nm 溫和，但**沒有完全消失**，與 `project_slot0nm_silence_mode` 同一簽名。

## 可寫 / 不可寫

- 可寫：「在 3 個匹配訓練 seed 下，移除量測詞在 MusicCaps CFG0/CFG3+neg 的 CLAP 與四項 AES 上都沒有可測效果；預先登錄的非劣性界線（±0.0084）在 n=3 下未能判定。」
- 可寫：「去量測 arm 的輸出一致地 crest 較高、平均略小聲、靜音 clip 略多。」
- **不可寫**：「去量測沒有傷害」／「打平」／任何 PQ 或 PC 的方向性宣稱。
- 提醒：MusicCaps CLAP/AES **量不到** tempo/key/meter 的遵循度（contract D4）；本實驗對「該不該保留量測詞」這個問題只回答了「在這兩個指標上看不出來」。

## 若要把非劣性判到底

需要縮小 CI，可行順序：(a) 再加 2–3 個 seed（CI 半寬 ∝ t/√n，n=6 的 t 掉到 2.57）；(b) 改用 per-clip paired 分析當次要證據（但依 `reference_training_seed_pq_noise_floor`，paired t 不能替代 seed 對照）；(c) 先鎖 LUFS 再比 AES，把響度通道從比較裡拿掉。
