# 10-exp 完整 benchmark（2026-04-25 定稿）

> 10 個實驗 × 8 個指標 × 兩個 benchmark (Jamendo / MusicCaps) 的完整紀錄。
>
> 所有數字在 2026-04-24 ~ 25 之間以最新腳本重新生成 + 評測，確保一致性。

---

## 1. 範圍與 metadata

| 項目 | 值 |
|------|----|
| 生成工具 | `eval.py --variant meanaudio_s --use_meanflow --num_steps 1 --cfg_strength 0.5 --full_precision` |
| CLAP/AES/FAD 腳本 | `~/research/meanaudio_eval/phase4_eval.py --fad` |
| PE-AV 腳本 | `~/research/meanaudio_eval/peav_eval.py --batch_size 8` |
| Jamendo TSV | `/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test_2048.tsv` (n=2048) |
| MusicCaps TSV | `/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv` (n=5521) |
| MusicCaps FAD reference | `/mnt/HDD/kojiek/musiccaps_reference/` (5,131 wavs, 93% coverage) |
| Metrics 存放 | `~/MeanAudio/eval_output/metrics/<exp_dir>/metrics.txt` + `peav_metrics.json` |
| Jamendo FAD 抽樣 | n=2048 |
| MusicCaps FAD 抽樣 | n=2048（CLAP/AES/PE-AV 用全 n=5521） |

### Eval q 旗標政策（避免啟用未訓練的 q embedding 污染結果）

| 訓練時 Q conditioning | 代表 Phase | Eval 旗標 |
|---|---|---|
| 有（q 0~9 被訓練） | P6V2 / P7V1/V2/V3 / P8V2/V3 | `--quality_level 9` |
| 無（永遠餵 null token） | P8 | `--no_q` |
| pre-P6（模型無 q_embed 層） | P4V4 / P5V1 / P5V2 | 兩者等價；用 `--quality_level 9`（auto-zeroed） |

---

## 2. Exp 代號 ↔ 對外名稱

| 代號 | 對外名稱 | Caption 策略 | Q conditioning | 備註 |
|------|---------|-------------|----------------|-------|
| P4V4 | JamendoFull-BestConsensus-NoQ (lr variant) | Best-consensus (final) | — (pre-P6) | P4 stage1 lr-schedule 調優變體；doc baseline 為 P4V2，但只存活 P4V4 ckpt |
| P5V1 | JamendoFull-Best-NoQ | Per-clip best | — (pre-P6) | |
| P5V2 | JamendoFull-Worst-NoQ | Per-clip worst | — (pre-P6) | |
| P6V2 | JamendoFull-Best-Q | Per-clip best | ✓ (half-Q)¹ | 首個引入 q_embed 層 |
| P7V1 | JamendoFull-Random-MeanSim-Q | Static random + MeanSim Q | ✓ (half-Q)¹ | 歷史 Jamendo + MusicCaps 雙料最佳 |
| P7V2 | JamendoFull-Random-LowVar-Q | Static random + LowVar Q | ✓ (half-Q)¹ | |
| P7V3 | JamendoFull-WorstCons-MeanSim-Q | Worst-consensus + MeanSim Q | ✓ (half-Q)¹ | **本次 MusicCaps 單榜最佳 CLAP + AES** |
| P8 | JamendoFull-Random-NoQ | Static random | — (NoQ E2E) | P7V1 的 no-Q 對照 |
| P8V2 | JamendoFull-Random-FinalCap-Q | Static random + FinalCap Q | ✓ (half-Q)¹ | |
| P8V3 | JamendoFull-CLAPFiltered-Q | CLAP-filtered + Q | ✓ (half-Q)¹ | **資料洩漏警示，勿作為論文 primary** |

¹ **half-Q**：`runner_flowmatching.py` 未傳 `q` 到 FluxAudio（Codex 2026-04-20 發現的 structural bug），S1 只訓 `q_embed[10]`（null token），S2 從零學 `q_embed[0-9]`。Phase 6-8 所有 "+Q" 實驗均為 half-Q。真正的 full-Q E2E 另有 P9 V2 和 P7V1_fullq_control（不在本表中）。

---

## 3. Jamendo test set（n=2048）

| 代號 | CLAP ↑ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ | FAD ↓ | PE-AV ↑ | R@10 (t2a) ↑ |
|------|--------|------|------|------|------|-------|---------|--------------|
| P4V4 | 0.1928 | 6.241 | 6.894 | 5.270 | 6.688 | 0.908 | 0.1385 | 11.08% |
| P5V1 | 0.1862 | 5.991 | 6.656 | 5.251 | 6.431 | 1.533 | 0.1318 | 9.77% |
| P5V2 | 0.1873 | 6.095 | 6.749 | 5.244 | 6.519 | 1.405 | 0.1324 | 10.21% |
| **P6V2** | **0.1983** | 6.479 | 7.138 | **5.375** | 6.926 | **0.865** | **0.1421** | **11.57%** |
| P7V1 | 0.1971 | 6.546 | 7.207 | 5.186 | 7.050 | 1.019 | 0.1414 | 10.74% |
| P7V2 | 0.1857 | 6.381 | 7.040 | 5.302 | 6.867 | 1.158 | 0.1388 | 9.72% |
| P7V3 | 0.1948 | **6.557** | **7.241** | 5.264 | **7.078** | 1.023 | 0.1409 | 10.69% |
| P8 | 0.1934 | 6.404 | 7.108 | 5.290 | 6.864 | 1.031 | 0.1423 | 9.96% |
| P8V2 | 0.1879 | 6.435 | 7.080 | 5.197 | 6.941 | 1.052 | 0.1368 | 10.55% |
| P8V3 | 0.1518 | 6.095 | 6.880 | 5.271 | 6.797 | 1.965 | 0.1205 | 6.20% |

**Per-metric 排名（僅列前 3）**

| 指標 | 第 1 | 第 2 | 第 3 |
|------|------|------|------|
| CLAP | P6V2 (0.1983) | P7V1 (0.1971) | P7V3 (0.1948) |
| CE | P7V3 (6.557) | P7V1 (6.546) | P6V2 (6.479) |
| CU | P7V3 (7.241) | P7V1 (7.207) | P6V2 (7.138) |
| PC | P6V2 (5.375) | P7V2 (5.302) | P8 (5.290) |
| PQ | P7V3 (7.078) | P7V1 (7.050) | P8V2 (6.941) |
| FAD ↓ | P6V2 (0.865) | P4V4 (0.908) | P7V1 (1.019) |
| PE-AV | P8 (0.1423) | P6V2 (0.1421) | P7V1 (0.1414) |
| R@10 | P6V2 (11.57%) | P4V4 (11.08%) | P7V1 (10.74%) |

---

## 4. MusicCaps benchmark（CLAP/AES/PE-AV n=5521, FAD n=2048）

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

**Per-metric 排名（僅列前 3）**

| 指標 | 第 1 | 第 2 | 第 3 |
|------|------|------|------|
| CLAP | P7V3 (0.1998) | P7V1 (0.1975) | P7V2 (0.1950) |
| CE | P7V3 (6.121) | P7V1 (6.017) | P6V2 (5.917) |
| CU | P7V3 (6.967) | P7V1 (6.822) | P8 (6.747) |
| PC | P8 (4.983) | P7V3 (4.973) | P7V2 (4.940) |
| PQ | P7V3 (6.833) | P7V1 (6.679) | P6V2 (6.619) |
| FAD ↓ | P4V4 (3.651) | P5V1 (3.703) | P5V2 (3.855) |
| PE-AV | P7V1 (0.0524) | P7V2 (0.0519) | P7V3 (0.0505) |
| R@10 | P7V1 (5.40%) | P7V2 (5.34%) | P4V4 (5.38%) |

---

## 5. 跨 benchmark 亮點

### 5.1 首位互換
| | Jamendo CLAP 第 1 | MusicCaps CLAP 第 1 |
|---|---|---|
| 名字 | P6V2 (Best-Q) | P7V3 (WorstCons-MeanSim-Q) |
| 另一邊排名 | MusicCaps 第 4（0.1943） | Jamendo 第 3（0.1948） |

Best-consensus (P6V2) 在 Jamendo 領先但 MusicCaps 失去優勢 → 顯示 Jamendo 分布特化 / overfitting。
Random-style caption (P7 系) 在 MusicCaps 上更穩健 → 支持「Static random 訓練法有更好 cross-domain generalization」的一般假說（但 n=10 實驗、未做 significance test，僅經驗性觀察）。

### 5.2 P7V1 是雙料前段的穩定答案
| Benchmark | CLAP | 排名 |
|---|---|---|
| Jamendo | 0.1971 | 2 |
| MusicCaps | 0.1975 | 2 |

PE-AV 亦在 MusicCaps 單榜第一 (0.0524) + Jamendo 前三。**仍是論文 primary model 首選。**

### 5.3 pre-P6 的 FAD anomaly
P4V4 / P5V1 / P5V2 的 MusicCaps FAD (3.65 / 3.70 / 3.85) 顯著低於所有 P6+ 實驗 (≥ 4.30)。

**Plausible 解釋**：pre-P6 模型 caption 品質較粗、Q conditioning 尚未引入 → 輸出分布「更平均 / 更通用」→ 偶然更貼近 MusicCaps 這種泛音樂分布。**代價是 CLAP / AES / PE-AV 全部顯著較低**。不是「pre-P6 比較好」，是**單一 FAD 不足以判斷模型品質**的反例。

### 5.4 P8V3 — CLAP-filtered training + CLAP eval 的 data leakage 教科書證據
| | Jamendo | MusicCaps |
|---|---|---|
| CLAP | 0.1518（10/10 末） | 0.1619（10/10 末） |
| FAD | 1.965（10/10 末） | 5.545（10/10 末） |
| PE-AV | 0.1205（10/10 末） | 0.0369（10/10 末） |
| R@10 | 6.20%（10/10 末） | 3.59%（10/10 末） |

CLAP 閾值過濾訓練資料 + CLAP 評分 → 看起來應該提升 CLAP，結果**全部指標皆為 10/10 最末**。符合教授 2026-03-27 的「禁用同指標過濾 + 同指標 eval」原則。

---

## 6. 生成與評測路徑

### Jamendo dirs
```
~/MeanAudio/eval_output/phase4_jamendo_v4_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase5_v1_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase5_v2_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase6_v2_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase7_v1_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase7_v2_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase7_v3_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase8_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase8_v2_stage2_200000_jamendo2048/
~/MeanAudio/eval_output/phase8_v3_stage2_200000_jamendo2048/
```

### MusicCaps dirs
```
~/MeanAudio/eval_output/phase4_jamendo_v4_stage2_200000_musiccaps_q9/
~/MeanAudio/eval_output/phase5_v1_musiccaps_q9/
~/MeanAudio/eval_output/phase5_v2_musiccaps_q9/
~/MeanAudio/eval_output/phase6_v2_musiccaps_q9/
~/MeanAudio/eval_output/phase7_v1_musiccaps_q9/
~/MeanAudio/eval_output/phase7_v2_musiccaps_q9/
~/MeanAudio/eval_output/phase7_v3_musiccaps_q9/
~/MeanAudio/eval_output/phase8_musiccaps_noq/
~/MeanAudio/eval_output/phase8_v2_musiccaps_q9/
~/MeanAudio/eval_output/phase8_v3_musiccaps_q9/
```

### 重跑腳本
- Jamendo (7 P6+ exps): `~/MeanAudio/run_jamendo_regen.sh`
- Jamendo (3 pre-P6 exps): `~/MeanAudio/run_jamendo_regen_early.sh`
- MusicCaps FAD (10 exps): `~/MeanAudio/run_musiccaps_fad.sh`
- MusicCaps reference download: `~/research/meanaudio_eval/download_musiccaps_reference.py`（rate-limited 版）

### 機器可讀
平行的 TSV 版本：`ten_exp_metrics.tsv`（同目錄）。

---

## 7. 已知限制 / 尚未做的事

1. **n=10 實驗**，跨 benchmark 排名觀察屬經驗性、未做 significance test。
2. **本表全用 half-Q 訓練模型**；真正的 full-Q E2E（P9 V2 / P7V1_fullq_control）不在本表，獨立紀錄於 `best_results.md` 及相關 memory file。
3. MusicCaps FAD 用 5,131 wavs（從 5,521 官方清單拿到 93% 有效下載），n=2048 抽樣對 FAD 穩定性足夠，但 full-5521 FAD 未做。
4. 所有指標都是 single-seed；生成採 `cfg_strength 0.5 + num_steps 1`（MeanFlow 推理設定），未掃 cfg / steps。
5. P8V3 的資料洩漏結論是基於「全指標皆為末」的經驗觀察；嚴格說應有對照 unleaked 版本實驗，但這不是本次目標。
