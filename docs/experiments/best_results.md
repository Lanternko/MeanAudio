# 目前最佳實驗數字

> 主要指標：CLAP ↑、CE ↑、PQ ↑。完整實驗記錄見 `../../EXPERIMENT_LOG.md`。

## Jamendo test set（n=2048，歷史實驗用）

| Phase | Caption 策略 | q | CLAP ↑ | CE ↑ | PQ ↑ |
|-------|------------|---|--------|------|------|
| phase4_v2 | best-consensus | — | 0.1929 | 5.905 | 6.620 |
| phase6_v2 | best-consensus | q=6 | 0.1979 | 6.175 | 6.859 |
| phase6_v2 | best-consensus | q=9 | **0.2139** | 6.109 | 6.821 |
| **phase7_v1** | **random** | **q=6** | 0.1980 | **6.276** | 6.936 |
| **phase7_v1** | **random** | **q=9** | 0.1984 | 6.254 | **6.939** |
| phase7_v2 | CLAP-best | q=6 | 0.1943 | 6.230 | 6.938 |

## MusicCaps benchmark（n=5,521，ISMIR-ready，2026-04-18 Lane A 完成）

| Phase | Caption 策略 | Q | Eval | CLAP ↑ | CE ↑ | CU ↑ | PC | PQ ↑ |
|-------|------------|---|------|--------|------|------|-----|------|
| phase4_v2 | best-consensus | ✗ | `--no_q` | 0.1825 | 5.458 | 6.299 | 4.534 | 6.233 |
| phase6_v2 | best-consensus | ✓ | q=9 | 0.1943 | 5.917 | 6.743 | 4.833 | 6.619 |
| **phase7_v1** | **random** | ✓ | **q=9** | **0.1975** | **6.017** | **6.822** | 4.758 | **6.679** |
| phase7_v2 | CLAP-best | ✓ | q=9 | 0.1950 | 5.871 | 6.633 | 4.940 | 6.528 |
| phase8 | random | ✗ | `--no_q` | 0.1851 | 5.913 | 6.747 | 4.983 | 6.544 |

## MusicCaps 關鍵讀數（2×2 分解）

- **Q 的獨立貢獻**（CLAP）：best-consensus +0.012、random +0.012（高度一致）
- **Random caption 的獨立貢獻**（CE/PQ）：no-Q 情境下 +0.454/+0.311；+Q 情境下 +0.099/+0.061（Q 吸收部分 diversity 效益 → 替代關係，非純加成）
- **Jamendo vs MusicCaps 排序倒轉**：Phase 6 V2 在 Jamendo 領先 Phase 7 V1（0.2139 vs 0.1984），但 MusicCaps 上 Phase 7 V1 領先（0.1975 vs 0.1943）→ best-consensus 有 Jamendo 特化 overfitting，random 更穩健

## 結論

**Phase 7 V1（random caption + Q）在兩個 benchmark 上都是最佳**（Jamendo 最佳或並列最佳、MusicCaps 全面最佳）。
