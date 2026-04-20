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

### Phase 9 系列新觀察（2026-04-20/21，需更多 control 才能下定論）
- **觀察**：multi-cap true random 下 CLAP 降低（V1 0.0650 / V2 0.0403，vs static 0.185-0.198）
- **觀察**：V1 AES 四項超過 Phase 8，V2 反而 AES 下降 → V1 呈「好聽但不貼 prompt」，V2 特性未明
- **觀察**：V2 (q=9) < V1（全指標）→ 假說是 aggregate-q 與 random-1/5 caption mismatch，但未對 q sweep 驗證
- **方法論注意**：Phase 6+ 所有 "+Q" 實驗都是 half-Q（S1 runner 沒傳 q，Codex 2026-04-20 發現）。P9 V2 是**第一個真 full Q E2E**。V2 變差可能來自 (a) multi_cap 本身、(b) full Q 可能就不如 half Q、(c) q=9 不是最適 — **三個 confound 尚未拆開**

## 結論（截至 2026-04-21，Codex review 後 conservative 版本）

**Phase 7 V1（static random + half Q）目前仍是已測中最佳**（Jamendo + MusicCaps 雙料）。

### 可以下的結論
- ✅ **static random + half Q** 是目前最穩健已驗證路線
- ✅ P9 V1 bugfix 跨 test set 一致（Jamendo 0.0589 / MusicCaps 0.0650），**不是 Jamendo overfit**
- ⚠️ P9 V1/V2 在 **current setup** 下 underperform Phase 7 V1
- ⚠️ multi-cap route 目前 **not promising**（≠ intrinsically incompatible）

### 尚不能下的結論（缺 control 實驗）
- ❓「**multi_cap 本質不適合 MeanAudio**」— 需先做 **Phase 7 V1 static-random + full Q E2E** 作 control 才能分離「multi_cap 效應」vs「full Q vs half Q 效應」。目前 V2 vs 舊 Phase 7 V1 混了兩個變量
- ❓「**Q 在 multi_cap 有害**」— P9 V2 僅測 `q=9`，但訓練分布眾數在 q=7/8，**最適 q 未知**。必須 sweep q=6/7/8 才算完整
- ❓「**aggregate-q + random-1/5 caption 是 mismatch 主因**」— 此為目前最合理假說但未證，需更多實驗支持

### 建議下一步
1. **P9 V2 q sweep**（~2 hr 總計）：q=6/7/8，看是否有 q_level 讓 V2 追上或超過 V1
2. **Phase 7 V1 full-Q rerun**（~20 hr）：乾淨 control，分離 multi_cap 與 full Q 獨立效應
3. 完成以上才能下定論
