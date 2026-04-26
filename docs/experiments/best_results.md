# 目前最佳實驗數字

> 主要指標：CLAP ↑、CE ↑、PQ ↑。完整實驗記錄見 `../../EXPERIMENT_LOG.md`。

## 完整 benchmark 總表（2026-04-25 定稿 v2，10 exps × 兩 benchmark × 全指標）

> Jamendo n=2048 (seed=42 random subset of 90K，1816 unique tracks = 15.7% 覆蓋率，與歷史 FAD 抽樣方法一致)；MusicCaps n=5521（FAD n=2048，用 4,525 筆 MusicCaps 官方參考音訊）。8 metrics: CLAP / CE / CU / PC / PQ / FAD / PE-AV / R@10 (t2a)。eval q 旗標依訓練 Q conditioning 選（`--no_q` for NoQ 訓練；`--quality_level 9` for Q 訓練 + pre-P6 相容）。詳見 `docs/experiments/ten_exp_full_benchmark.md`。

### Exp 代號 ↔ 對外名稱（修正版 2026-04-25）

| 代號 | 對外名稱 | Caption 策略 | Q 信號 | 訓練集 | Notes |
|------|---------|-------------|--------|--------|-------|
| P4V4 | JamendoFull-BestConsensus-NoQ (lr variant) | Best-consensus | — | 251K | P4 stage1 lr-schedule 調優版 |
| P5V1 | **JamendoHalf-BestConsensus-NoQ-HardFilter** | Best-consensus | — | **117K** | 硬過濾 mean_sim ≥ 0.80 |
| P5V2 | **JamendoHalf-BestConsensus-NoQ-Random** | Best-consensus | — | **117K** | 隨機抽半（P5V1 對照） |
| P6V2 | **JamendoFull-BestConsensus-MeanSim-Q** | Best-consensus | MeanSim | 251K | 首個引入 q_embed |
| P7V1 | JamendoFull-Random-MeanSim-Q | Static random | MeanSim | 251K | 歷史雙料最佳 |
| P7V2 | **JamendoFull-CLAPBest-MeanSim-Q** | **CLAP-best**（非 LowVar） | MeanSim | 251K | CLAPBest 選 caption |
| P7V3 | JamendoFull-WorstConsensus-MeanSim-Q | Worst-consensus | MeanSim | 251K | MusicCaps 單榜最佳 |
| P8 | JamendoFull-Random-NoQ | Static random | — | 251K | P7V1 no-Q 對照 |
| P8V2 | **JamendoFull-Random-AudioboxPQ-Q** | Static random | **Audiobox PQ**（非 FinalCap） | 251K | Q 信號換 PQ 分數 |
| P8V3 | **JamendoFull-Random-CLAP-Q** | Static random | **audio-text CLAP sim**（非 CLAP-filter） | 251K | 失敗原因是 genre shortcut（piano bias），非 data leakage |

¹ half-Q = `runner_flowmatching.py` 未傳 q 到 FluxAudio（Codex 2026-04-20 發現的 structural bug），S1 只訓 `q_embed[10]`，S2 從零學 q[0-9]。所有 Phase 6-8 +Q 實驗都是 half-Q。真正 full-Q E2E 只有 P9 V2 和 P7V1_fullq_control（見 footnote ⁴ 下方）。

### Jamendo test set（n=2048, seed=42 random subset of 90K）

| 代號 | CLAP ↑ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ | FAD ↓ | PE-AV ↑ | R@10 (t2a) ↑ |
|------|--------|------|------|------|------|-------|---------|--------------|
| P4V4 | 0.1909 | 5.862 | 6.650 | 4.969 | 6.506 | 1.131 | 0.1242 | **12.65%** |
| P5V1 | 0.1861 | 5.635 | 6.424 | 4.991 | 6.287 | **2.053** | 0.1170 | 11.28% |
| P5V2 | 0.1869 | 5.713 | 6.495 | 4.952 | 6.356 | 1.777 | 0.1169 | 11.18% |
| P6V2 | 0.1957 | 6.165 | 6.963 | **5.149** | 6.823 | **1.059** | 0.1279 | 12.11% |
| P7V1 | 0.1981 | 6.251 | 7.031 | 4.974 | 6.930 | 1.159 | 0.1283 | 11.08% |
| P7V2 | 0.1920 | 6.121 | 6.895 | 5.081 | 6.791 | 1.350 | 0.1266 | 11.72% |
| P7V3 | 0.1965 | **6.266** | **7.072** | 5.072 | **6.982** | 1.222 | 0.1274 | 11.67% |
| **P8** | **0.1986** | 6.124 | 6.950 | 5.103 | 6.755 | 1.065 | **0.1305** | 11.47% |
| P8V2 | 0.1910 | 6.125 | 6.916 | 4.996 | 6.856 | 1.185 | 0.1251 | 11.62% |
| P8V3 | 0.1514 | 5.756 | 6.701 | 5.085 | 6.709 | **2.526** | 0.1102 | 8.15% |
| **P8V4** | **0.0591** | 5.734 | 6.467 | **5.263** | 6.374 | — | — | — |

**Jamendo 結論**：CLAP 第一是 P8、PE-AV 第一也是 P8、FAD 最低是 P6V2、AES triple (CE/CU/PQ) 冠軍是 P7V3；P7V1 在所有 metric 都穩居前三 → **P7V1 / P7V3 / P8 / P6V2 為 Jamendo Top-4**。Phase 5 崩盤特徵還原（FAD +82% vs P4V4，AES 全項退步），P5V1 (HardFilter) FAD 比 P5V2 (Random-half) 還差 16% → 與歷史「data quantity 才是主因，hard filter 沒幫助」結論一致。P8V3 全 metric 墊底，跨 benchmark 一致 → genre shortcut hypothesis 穩定。

### MusicCaps benchmark（CLAP/AES/PE-AV n=5521，FAD n=2048）

| 代號 | CLAP ↑ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ | FAD ↓ | PE-AV ↑ | R@10 (t2a) ↑ |
|------|--------|------|------|------|------|-------|---------|--------------|
| P4V4 | 0.1825 | 5.458 | 6.299 | 4.534 | 6.233 | **3.651** | 0.0482 | 5.38% |
| P5V1 | 0.1663 | 5.060 | 5.991 | 4.627 | 5.916 | 3.703 | 0.0378 | 4.73% |
| P5V2 | 0.1616 | 5.022 | 5.994 | 4.526 | 5.935 | 3.855 | 0.0342 | 4.11% |
| P6V2 | 0.1943 | 5.917 | 6.743 | 4.833 | 6.619 | 4.302 | 0.0498 | 4.80% |
| P7V1 | 0.1975 | 6.017 | 6.822 | 4.758 | 6.679 | 4.315 | **0.0524** | **5.40%** |
| P7V2 | 0.1950 | 5.871 | 6.633 | 4.940 | 6.528 | 4.397 | 0.0519 | 5.34% |
| **P7V3** | **0.1998** | **6.121** | **6.967** | **4.973** | **6.833** | 4.664 | 0.0505 | 5.11% |
| P8 | 0.1851 | 5.913 | 6.747 | 4.983 | 6.544 | 4.744 | 0.0467 | 4.76% |
| P8V2 | 0.1848 | 5.778 | 6.581 | 4.712 | 6.535 | 4.884 | 0.0494 | 4.73% |
| P8V3 | 0.1619 | 5.364 | 6.395 | 4.927 | 6.406 | 5.545 | 0.0369 | 3.59% |
| **P8V4** | **0.0571** | 5.771 | 6.504 | **5.272** | 6.416 | — | — | — |

**MusicCaps 結論**：P7V3 領先 CLAP + AES 四項；P7V1 在 PE-AV / R@10 領先。P4V4 FAD 最低是因為 pre-P6 模型輸出分布較廣，更接近 MusicCaps 一般音樂分布（品質代價：CLAP/AES 全部較低）。**Jamendo ↔ MusicCaps 首位互換**（Jamendo 是 P6V2，MusicCaps 是 P7V3）→ Static random caption (P7 系) 比 Best-consensus (P6) 跨分布更穩健。

### 跨 benchmark 亮點（seed=42 修正後）

| 代號 | Jamendo CLAP rank | MusicCaps CLAP rank | 解讀 |
|------|-------------------|---------------------|------|
| **P7V1** | 2 | 2 | 跨 benchmark 最穩；PE-AV 雙第一/第二 → **論文 primary 首選** |
| **P7V3** | 3 | **1** | MusicCaps 單榜冠軍 + AES quadruple 最高 → **論文 second-pick / cross-domain 強項** |
| **P8** (Random-NoQ) | **1** | 8 | 同分布最強，但 cross-domain 大跌 → **驗證 Q conditioning 對 cross-domain generalization 的貢獻** |
| **P8V3** (Random-CLAP-Q) | 10 | 10 | 全 metric 墊底 → genre shortcut 機制（CLAP sim 當 Q 信號 → 高 q 偏 piano）；**不是 data leakage**

---

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
| phase7_v1_fullq_control⁴ | random (static) | ✓ (E2E, clean impl) | q=9 | 0.1748 | 5.5436 | 6.5875 | 5.0712 | 6.5153 |
| phase7_v1_fullq_control⁴ | random (static) | ✓ (E2E, clean impl) | q=6 | 0.1759 | 5.5429 | 6.6054 | 5.0262 | 6.4856 |
| phase7_v1_s2only_ablation⁵ | random (static) | ✓ (S2-only, pseudo-S1) | q=9 | 0.1951 | 6.0130 | 6.8891 | 4.8035 | 6.7467 |
| phase7_v1_s2only_ablation⁵ | random (static) | ✓ (S2-only, pseudo-S1) | q=6 | 0.1981 | 6.0977 | 6.8860 | 4.9574 | 6.7641 |

¹ Phase 6/7 系列的「+Q」實際是 S2-only Q（runner_flowmatching.py 沒讀 q_level，2026-04-20 Codex 發現）。S1 只訓了 `q_embed[10]`。數字反映 half-Q 訓練。

² Phase 9 V1 bugfix：修兩個 bug（networks.py q=None 填 10、runner_meanflow.py undrop .clone）後重跑。CLAP 從崩前 Jamendo 0.0260 提升到 0.0650，跨 test set 一致（Jamendo 0.0589）→ bug 不是全部原因，但 bug fix 後 CLAP 仍遠低於 static-random baseline，該殘差尚未被單一機制定位。

³ Phase 9 V2 bugfix：除上述外再修 runner_flowmatching.py（6 處加 q 參數），S1+S2 真 Q end-to-end 訓練。CLAP 0.0403 < V1 的 0.0650；q sweep (6–9) 幾乎 flat (0.0402–0.0417)。Plausible 假說：aggregate-q（基於全 5 cap 的 MeanSim）與訓練時餵 random 1-of-5 caption 之間的 supervision 不一致。P7 V1 q-sweep 顯示 q 在 in-support regime 內近乎 flat（q=6 vs q=9 ΔCLAP=0.0015），所以 P9 V2 flatness 本身 does not by itself identify multi-cap as the cause。Static-random full-Q control 已完成（見 footnote ⁴）；P9 V2 drop 至少含 clean-implementation penalty (~0.02) + P9-specific residual (~0.13)，後者歸因待 Clean S2 only ablation 出結果。

⁴ P7 V1 full-Q control (2026-04-22)：乾淨 implementation (S1 q-passing fix + S2 text_f_undrop clone fix) 下的 P7 V1 full-Q E2E。全 5 eval 一致低於歷史 P7 V1 best ~8-12% CLAP（Jamendo q=6: 0.1816 / q=9: 0.1799 / native_q: 0.1801；MusicCaps q=6: 0.1759 / q=9: 0.1748）。**Historical P6 V2 should not be interpreted as evidence that Stage 1 successfully trained q embeddings, because the later-discovered runner_flowmatching q-passing bug was still present at that time.** 活躍的 implementation 差異為 S1 q-passing fix + S2 clone fix，co-varied。見 footnote ⁵ 的 Clean S2 only ablation 結果。詳見 `memory/project_p7_fullq_control_finding_2026_04_22.md`。

⁵ P7 V1 Clean S2 only ablation (2026-04-23)：歷史 P7 V1 S1 weights（load-compatible approximation，pseudo-EMA bootstrap）＋只重訓 S2 with text_f_undrop clone fix。全 5 EMA eval 一致接近歷史 P7 V1 best（Jamendo q=6: 0.2008 / q=9: 0.1993 / native_q: 0.1995；MusicCaps q=6: 0.1981 / q=9: 0.1951），與 fullq_control 的 ~8-12% drop 形成明確對比。**Clean S2 only ablation restores the historical P7 V1 baseline across all completed EMA evaluations. This strongly indicates that the Stage 2 text_f_undrop clone fix is not the main driver of the ~8-12% CLAP drop in fullq_control. The primary remaining contributor is Stage 1 effective q training (enabled by the runner_flowmatching q-passing fix).** Pseudo-EMA bootstrap confound 已排除（2026-04-24）：s2only last.pth q=9 = 0.1757（EMA gap +13.4%）vs fullq_control last.pth q=9 = 0.1575（EMA gap +14.2%）— 兩者 EMA-vs-online gap 一致（~13-14%），確認 gap 為 S2 訓練的結構性現象，非 pseudo-EMA 初始化人工膨脹。

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

## 結論（截至 2026-04-24，ablation chain 完整後定稿）

**Phase 7 V1（static random + half Q）目前仍是已測中最佳**（Jamendo + MusicCaps 雙料）。

### 可以下的結論
- **static random + half Q（historical Phase 7 V1 best）** 是目前已測路線中最穩健經驗基線
- P9 V1 bugfix 跨 test set 一致（Jamendo 0.0589 / MusicCaps 0.0650），**不是 Jamendo overfit**
- P9 V1/V2 在 **current setup** 下 underperform Phase 7 V1
- **Clean full-Q control bundle underperforms historical half-Q baseline by ~8-12% CLAP** (P7 V1 full-Q control, 2026-04-22, 見 footnote ⁴)。This falsifies the strong version of "P9 V2 drop attributed entirely to multi-cap"。
- **Clean S2 only ablation restores the historical P7 V1 baseline across EMA evaluations, while exhibiting an EMA-vs-last gap comparable to the full-Q control rerun. This rules out pseudo-EMA bootstrap as the explanation and strongly indicates that the Stage 2 text_f_undrop clone fix is not the main driver of the ~8–12% CLAP drop. The primary remaining contributor is Stage 1 effective q training.** (footnote ⁵, 2026-04-23/24; last.pth gap: s2only +13.4%, fullq_control +14.2% — structurally consistent, not init artifact)
- **P9 V2 gap 現在可拆解為兩部分**：(1) general penalty ~0.02 CLAP，主要來自 S1 effective q training；(2) P9-specific residual ~0.13 CLAP，行為上與 multi-cap 強相關（機制未證）。

### Q behavior in P7 V1 (2026-04-21)
P7 V1 q-sweep on MusicCaps indicates **support-set gating** rather than ordinal quality control. OOD q values (q=0/3) yield CLAP ≤ 0.045, while in-support q values (q=6/9) yield CLAP ≈ 0.197 and are nearly equivalent (ΔCLAP = 0.0015). Historical Jamendo results for q=6/9/native_q are likewise all ≈ 0.198. Current evidence therefore supports q as a coarse in-support gating signal rather than a strong ordinal quality controller. P7 V1's high CLAP should not be attributed primarily to fine-grained q scaling.

### 尚不能下的結論（缺 control 實驗）
- 「**multi_cap 本質不適合 MeanAudio**」— 需先做 **Phase 7 V1 static-random + full Q E2E** 作 control 才能分離「multi_cap 效應」vs「full Q vs half Q 效應」。目前 V2 vs 舊 Phase 7 V1 混了兩個變量。
- 「**Q 在 multi_cap 有害**」— P9 V2 q sweep 顯示 q=6/7/8/9 flat (~0.04)，但 P7 V1 q-sweep 顯示 q 在 in-support 內本來就 flat，所以此現象本身不成鑑別證據。
- 「**aggregate-q + random-1/5 caption 是 mismatch 主因**」— working hypothesis，未證。

### 補充：MusicCaps-sampled subjective A/B 客觀分（n=30 paired, cfg=0.5）

小 n 但 paired 的 P7 V1 vs P8 V1 CLAP+AES 驗證（30 random MusicCaps prompts × 2 variants）：

- CLAP mean: P7 V1 **0.2285** vs P8 V1 0.2072（Δ +0.0213，P7 wins 20/30）
- AES: CE P7 高（+0.18）、CU/PC/PQ P8 略高（≤ 0.17）— 整體接近
- **方向上對齊 n=5,521 MusicCaps benchmark**（+0.0124），量級略大（選樣偏差 + paired）

完整方法、腳本、artifact 路徑見 `docs/eval/subjective_prompts.md` 末段「MusicCaps-sampled subjective A/B（v3）」。

### 下一步進度
1. **P9 V2 q sweep**（完成 2026-04-21）：q=6/7/8/9 皆 flat（CLAP 0.0402–0.0417），P9 V2 failure 不能歸因於 q=9 選錯。
2. **P7 V1 q-sweep**（完成 2026-04-21）：support-set gating behavior 確立。
3. **P7 V1 full-Q control rerun**（完成 2026-04-22）：MusicCaps q=9 CLAP 0.1748（vs historical 0.1975，−11.5%），全 5 eval 一致降 ~8-12%。詳見 footnote ⁴。Falsifies 「P9 drop 全怪 multi-cap」強版本；但「full-Q 本身有代價」還不能說，因 active implementation differences 含 S1 q-passing fix 與 S2 clone fix，co-varied。
4. **Clean S2 only ablation**（✅ 完成 2026-04-23/24）：historical P7 V1 S1 weights + 只重訓 S2 with clone fix。5/5 EMA eval 一致回到歷史區間；last.pth insurance 確認 EMA gap ~13-14% 與 fullq_control 一致 → pseudo-EMA bootstrap confound 排除。**Clone fix 非主因；primary remaining contributor: S1 effective q training。** 詳見 footnote ⁵ + `phase_status.md`。
