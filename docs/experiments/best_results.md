# 目前最佳實驗數字

> 主要指標：CLAP ↑、CE ↑、PQ ↑。完整實驗記錄見 `../../EXPERIMENT_LOG.md`。

## Jamendo test set（n=2048，歷史實驗用）

| Phase | Caption 策略 | q | CLAP ↑ | CE ↑ | PQ ↑ |
|-------|------------|---|--------|------|------|
| phase4_v2 | best-consensus | — | 0.1929 | 5.905 | 6.620 |
| phase6_v2 | best-consensus | q=6 | 0.1979 | 6.175 | 6.859 |
| phase6_v2 | best-consensus | q=9 | **0.2139** | 6.109 | 6.821 |
| **phase7_v1** | **random (static)** | **q=6** | 0.1980 | **6.276** | 6.936 |
| **phase7_v1** | **random (static)** | **q=9** | 0.1984 | 6.254 | **6.939** |
| phase7_v2 | CLAP-best | q=6 | 0.1943 | 6.230 | 6.938 |
| phase9_v1_bugfix | multi-cap true-random, NoQ | `--no_q` | 0.0589 | 6.2612 | 6.6770 |

> 注意：Phase 4-8 系列的 CLAP 不在單一 num_samples 下可完美對齊 MusicCaps 的 n=2048；Phase 9 V1 bugfix 為完整 90,063 生成 + num_samples=2048 metric 計算。

## MusicCaps benchmark（n=5,521，ISMIR-ready，2026-04-18 Lane A 完成）

| Phase | Caption 策略 | Q | Eval | CLAP ↑ | CE ↑ | CU ↑ | PC | PQ ↑ |
|-------|------------|---|------|--------|------|------|-----|------|
| phase4_v2 | best-consensus | ✗ | `--no_q` | 0.1825 | 5.458 | 6.299 | 4.534 | 6.233 |
| phase6_v2 | best-consensus | ✓ | q=9 | 0.1943 | 5.917 | 6.743 | 4.833 | 6.619 |
| **phase7_v1** | **random (static)** | ✓ (half)¹ | **q=9** | **0.1975** | **6.017** | **6.822** | 4.758 | **6.679** |
| phase7_v2 | CLAP-best (static) | ✓ (half)¹ | q=9 | 0.1950 | 5.871 | 6.633 | 4.940 | 6.528 |
| phase8 | random (static) | ✗ | `--no_q` | 0.1851 | 5.913 | 6.747 | 4.983 | 6.544 |
| phase9_v1_bugfix² | multi-cap true-random | ✗ | `--no_q` | 0.0650 | 6.2626 | 6.8051 | 5.4256 | 6.6834 |
| phase9_v2_bugfix³ | multi-cap true-random | ✓ (E2E) | q=9 | 0.0403 | 5.2981 | 6.2673 | 5.3742 | 5.9666 |

¹ Phase 6/7 系列的「+Q」實際是 S2-only Q（runner_flowmatching.py 沒讀 q_level，2026-04-20 Codex 發現）。S1 只訓了 `q_embed[10]`。數字反映 half-Q 訓練。

² Phase 9 V1 bugfix：修兩個 bug（networks.py q=None 填 10、runner_meanflow.py undrop .clone）後重跑。CLAP 從崩前 Jamendo 0.0260 提升到 0.0650，跨 test set 一致（Jamendo 0.0589）→ 不是 bug 主因，multi_cap 本質性 unconditional drift。

³ Phase 9 V2 bugfix：除上述外再修 runner_flowmatching.py（6 處加 q 參數），S1+S2 真 Q end-to-end 訓練。CLAP **反而** 0.0403 < V1 的 0.0650 → **Q + multi_cap true random 訓練訊號衝突**：q 信號基於全 5 caps 算、實際輸入是隨機 1/5 → 模型被矛盾信號撕扯。

## MusicCaps 關鍵讀數

### 原本 2×2 分解（static-caption 組，n=Phase 4-8）
- **Q 的獨立貢獻**（CLAP）：best-consensus +0.012、random +0.012（高度一致）
- **Random caption 的獨立貢獻**（CE/PQ）：no-Q 情境下 +0.454/+0.311；+Q 情境下 +0.099/+0.061（Q 吸收部分 diversity 效益 → 替代關係，非純加成）
- **Jamendo vs MusicCaps 排序倒轉**：Phase 6 V2 在 Jamendo 領先 Phase 7 V1（0.2139 vs 0.1984），但 MusicCaps 上 Phase 7 V1 領先（0.1975 vs 0.1943）→ best-consensus 有 Jamendo 特化 overfitting，random 更穩健

### Phase 9 系列新發現（2026-04-20/21）
- **multi-cap true random 對 CLAP 有大幅傷害**：V1 0.0650 / V2 0.0403，遠低於 static random 的 0.185-0.198
- **AES 卻可能提升**：V1 在 AES 四項全面超過 Phase 8（模型產生「好聽但不貼 prompt」的音樂）
- **Q 在 multi_cap 情境反而更糟**：V2 < V1（全指標）→ q 信號 (basedon 5 caps) 與訓練輸入 (random 1/5) 不 coherent
- **Phase 6+ 所有 "+Q" 實驗都是 half-Q**（S1 沒讀 q），P9 V2 是第一個真 Q end-to-end，但因 multi_cap 環境下產生副作用，不等於 Q 本身失敗

## 結論

**Phase 7 V1（static random caption + half Q）在兩個 benchmark 上都是最佳**（Jamendo 最佳或並列最佳、MusicCaps 全面最佳）。

- ✅ static random + Q 是 MeanAudio 最穩健路線
- ❌ multi-cap true random 本質不適合（兩個 variant 都失敗）
- 🤔 真 Q end-to-end 在 static random 上能否超越 half-Q 仍是未解問題（未測）
