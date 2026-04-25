# 10-exp 完整 benchmark（2026-04-25 定稿）

> ⚠️ **Jamendo 表格數字 rerun 中（tmux `jamendo_s42`）**：初版 Jamendo 數字用了 `head -n 2049` 取的窄 subset（只涵蓋 6.9% track），FAD/AES/CLAP 絕對數字與歷史不可比。新版改用 `seed=42 random 2048 from 90K`（涵蓋 15.7% track, 1816 tracks）與歷史 FAD 計算一致。2026-04-25 07:00 啟動，ETA ~2h。
>
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

| 代號 | 對外名稱 | Caption 策略 | Q conditioning | 訓練集 | 備註 |
|------|---------|-------------|----------------|--------|-------|
| P4V4 | JamendoFull-BestConsensus-NoQ (lr variant) | Best-consensus | — (pre-P6) | 251K | P4 stage1 lr-schedule 調優變體；doc baseline 為 P4V2，但只存活 P4V4 ckpt |
| P5V1 | **JamendoHalf-BestConsensus-NoQ-HardFilter** | Best-consensus | — (pre-P6) | **117K** | 硬過濾 mean_sim ≥ 0.80 後保留半數 |
| P5V2 | **JamendoHalf-BestConsensus-NoQ-Random** | Best-consensus | — (pre-P6) | **117K** | 隨機抽半數 → P5V1 的對照（驗證 data quantity 才是主因） |
| P6V2 | **JamendoFull-BestConsensus-MeanSim-Q** | Best-consensus | ✓ (half-Q)¹ | 251K | 首個引入 q_embed 層；Q 信號為 MeanSim |
| P7V1 | JamendoFull-Random-MeanSim-Q | Static random | ✓ (half-Q)¹ | 251K | 歷史 Jamendo + MusicCaps 雙料最佳 |
| P7V2 | **JamendoFull-CLAPBest-MeanSim-Q** | **CLAP-best**（每 clip 5 candidates 選 audio-text CLAP 最高） | ✓ (half-Q)¹ | 251K | 原假設：CLAP-best > random，結果反而被 random 打敗 |
| P7V3 | JamendoFull-WorstConsensus-MeanSim-Q | Worst-consensus（選 inter-text sim 最低的 caption） | ✓ (half-Q)¹ | 251K | 本次 MusicCaps 單榜最佳 CLAP + AES |
| P8 | JamendoFull-Random-NoQ | Static random | — (NoQ E2E) | 251K | P7V1 的 no-Q 對照 |
| P8V2 | **JamendoFull-Random-AudioboxPQ-Q** | Static random | ✓ (half-Q)¹ | 251K | **Q 信號改用 Audiobox PQ**（不是 MeanSim） |
| P8V3 | **JamendoFull-Random-CLAP-Q** | Static random | ✓ (half-Q)¹ | 251K | **Q 信號改用 audio-text CLAP sim**。失敗原因是**genre shortcut**（高 q_level 系統性偏向 piano/acoustic，因 LP-MusicCaps 對這類音樂描述更精確），**非「CLAP-filter 訓練 + CLAP eval」資料洩漏** |

¹ **half-Q**：`runner_flowmatching.py` 未傳 `q` 到 FluxAudio（Codex 2026-04-20 發現的 structural bug），S1 只訓 `q_embed[10]`（null token），S2 從零學 `q_embed[0-9]`。Phase 6-8 所有 "+Q" 實驗均為 half-Q。真正的 full-Q E2E 另有 P9 V2 和 P7V1_fullq_control（不在本表中）。

---

## 3. Jamendo test set（n=2048）

> ⚠️ **本節數字 rerun 中（2026-04-25 07:00 啟動 `tmux jamendo_s42`）**。下表用 **窄 subset**（`phase4_test_2048.tsv` = `head -n 2049 phase4_test.tsv`，只 794 unique tracks = 6.9% 覆蓋率）計算，FAD / AES / CLAP 絕對數字**與歷史不可比**，也**不適合作為 10 exp 互比的定版**。Rerun 採 `seed=42 random 2048 from 90K`（1816 unique tracks = 15.7% 覆蓋率，與歷史 FAD 抽樣方法一致）。Rerun 完成後本表將被替換。

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

> ⚠️ 涉及 Jamendo 的比較（5.1 / 5.2 部分）暫緩 — 等 `jamendo_s42` rerun 完成後重寫。

### 5.1 首位互換（待 Jamendo rerun 後確認）
| | Jamendo CLAP 第 1 | MusicCaps CLAP 第 1 |
|---|---|---|
| 名字 | P6V2 (Best-Q) ⚠️窄 subset | P7V3 (WorstCons-MeanSim-Q) |
| 另一邊排名 | MusicCaps 第 4（0.1943） | Jamendo 第 3（0.1948）⚠️窄 subset |

Best-consensus (P6V2) 在 Jamendo 領先但 MusicCaps 失去優勢 → 顯示 Jamendo 分布特化 / overfitting 的**假說**，待 rerun 確認。
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

### 5.4 P8V3 — CLAP-sim 當 Q 信號造成 genre shortcut 的範例
| | Jamendo | MusicCaps |
|---|---|---|
| CLAP | 0.1518（10/10 末） | 0.1619（10/10 末） |
| FAD | 1.965（10/10 末） | 5.545（10/10 末） |
| PE-AV | 0.1205（10/10 末） | 0.0369（10/10 末） |
| R@10 | 6.20%（10/10 末） | 3.59%（10/10 末） |

用 audio-text CLAP sim 當 q_embed 信號（10 bin 等頻分箱）→ 高 q_level 系統性偏向 piano/acoustic（LP-MusicCaps 對這類更精確 → 更高 CLAP sim）→ q=9 inference 時帶 genre shortcut → 音訊偏移 reference distribution → **全部指標皆為 10/10 最末**。

**與「CLAP-filter 訓練 + CLAP eval」資料洩漏不同**：P8V3 不是用 CLAP 閾值過濾訓練資料，而是把 CLAP sim 轉成 ordinal q 信號。失敗根因是「Q 信號與 genre 相關 → q_embed 學到 genre shortcut」，非「訓練-評測同指標」。詳見 `docs/experiments/Phase4_to_Phase8_Complete_Summary.md` 第十一節。

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
5. P8V3 的 genre-shortcut 結論來自 piano 分箱實測（q=0~1: 10.4% piano, q=8~9: 21.2%，見 `Phase4_to_Phase8_Complete_Summary.md`）。本表只呈現 benchmark 結果，機制分析在該 doc。
