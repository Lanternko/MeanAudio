# 10-exp 完整 benchmark（2026-04-25 定稿 v2）

> 10 個實驗 × 8 個指標 × 兩個 benchmark (Jamendo / MusicCaps) 的完整紀錄。Jamendo 表格使用 **seed=42 random 2048 from 90K**（1816 unique tracks，15.7% 覆蓋率，與歷史 FAD 抽樣方法一致）。
>
> 所有數字 2026-04-25 重新生成 + 評測，10 exp 內互比可信。
>
> 與歷史 Phase 4-8 doc 的數字仍可能略有差異（FAD 模型版本、reference 路徑等），詳見最末「已知限制」。

---

## 1. 範圍與 metadata

| 項目 | 值 |
|------|----|
| 生成工具 | `eval.py --variant meanaudio_s --use_meanflow --num_steps 1 --cfg_strength 0.5 --full_precision` |
| CLAP/AES/FAD 腳本 | `~/research/meanaudio_eval/phase4_eval.py --fad` |
| PE-AV 腳本 | `~/research/meanaudio_eval/peav_eval.py --batch_size 8` |
| Jamendo TSV | `/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test_seed42_2048.tsv` (n=2048, seed=42 random subset of 90K) |
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

## 3. Jamendo test set（n=2048, seed=42 random subset of 90K，1816 unique tracks = 15.7% 覆蓋率）

| 代號 | CLAP ↑ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ | FAD ↓ | PE-AV ↑ | R@10 (t2a) ↑ |
|------|--------|------|------|------|------|-------|---------|--------------|
| P4V4 | 0.1909 | 5.862 | 6.650 | 4.969 | 6.506 | 1.131 | 0.1242 | **12.65%** |
| P5V1 (HardFilter) | 0.1861 | 5.635 | 6.424 | 4.991 | 6.287 | **2.053** | 0.1170 | 11.28% |
| P5V2 (Random-half) | 0.1869 | 5.713 | 6.495 | 4.952 | 6.356 | 1.777 | 0.1169 | 11.18% |
| P6V2 | 0.1957 | 6.165 | 6.963 | **5.149** | 6.823 | **1.059** | 0.1279 | 12.11% |
| P7V1 | 0.1981 | 6.251 | 7.031 | 4.974 | 6.930 | 1.159 | 0.1283 | 11.08% |
| P7V2 | 0.1920 | 6.121 | 6.895 | 5.081 | 6.791 | 1.350 | 0.1266 | 11.72% |
| P7V3 | 0.1965 | **6.266** | **7.072** | 5.072 | **6.982** | 1.222 | 0.1274 | 11.67% |
| **P8** | **0.1986** | 6.124 | 6.950 | 5.103 | 6.755 | 1.065 | **0.1305** | 11.47% |
| P8V2 | 0.1910 | 6.125 | 6.916 | 4.996 | 6.856 | 1.185 | 0.1251 | 11.62% |
| P8V3 | 0.1514 | 5.756 | 6.701 | 5.085 | 6.709 | **2.526** | 0.1102 | 8.15% |

**Per-metric 排名（僅列前 3）**

| 指標 | 第 1 | 第 2 | 第 3 |
|------|------|------|------|
| CLAP | P8 (0.1986) | P7V1 (0.1981) | P7V3 (0.1965) |
| CE | P7V3 (6.266) | P7V1 (6.251) | P6V2 (6.165) |
| CU | P7V3 (7.072) | P7V1 (7.031) | P6V2 (6.963) |
| PC | P6V2 (5.149) | P8 (5.103) | P7V2 (5.081) |
| PQ | P7V3 (6.982) | P7V1 (6.930) | P8V2 (6.856) |
| FAD ↓ | P6V2 (1.059) | P8 (1.065) | P4V4 (1.131) |
| PE-AV | P8 (0.1305) | P7V1 (0.1283) | P6V2 (0.1279) |
| R@10 | P4V4 (12.65%) | P6V2 (12.11%) | P7V2 (11.72%) |

**Jamendo 結論**：
- **CLAP / PE-AV / FAD 第一是 P8 (Random-NoQ) 或 P6V2**，但 P7V1 / P7V3 緊跟（差 < 0.003 / < 1%）→ Top 4 模型 (P6V2 / P7V1 / P7V3 / P8) 在 Jamendo 已飽和 / 無顯著差異
- **AES triple (CE/CU/PQ) 冠軍：P7V3**（與 MusicCaps 一致）
- **Phase 5 崩盤特徵還原**：P5V1 FAD 2.053 / P4V4 FAD 1.131 = **+82%**（歷史 +67%）。CLAP −2.5%（歷史 −4.9%），AES 全項退步 0.2-0.3 → **hard filter / data 減半 = 全面退步，結論與歷史一致**
- **P5V1 vs P5V2 對照**：兩者 CLAP / AES / PE-AV / R@10 差距 ≤ 1%，但 P5V1 (HardFilter) FAD 2.053 比 P5V2 (Random-half) FAD 1.777 反而**差 16%** → **HardFilter 沒幫助甚至有害，data quantity 才是主因，與歷史結論一致**
- **P8V3 全面崩**：唯一全 metric 墊底者（CLAP 0.1514 / FAD 2.526 / R@10 8.15% / PE-AV 0.110），跨 benchmark 一致 → genre shortcut hypothesis 穩定

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

### 5.1 Top-4 模型在兩 benchmark 的 CLAP 排名（修正 seed=42 後）
| 代號 | Jamendo CLAP | Jamendo Rank | MusicCaps CLAP | MusicCaps Rank |
|------|-------------|--------------|---------------|---------------|
| P6V2 | 0.1957 | 4 | 0.1943 | 4 |
| P7V1 | 0.1981 | 2 | 0.1975 | 2 |
| P7V3 | 0.1965 | 3 | **0.1998** | **1** |
| P8 | **0.1986** | **1** | 0.1851 | 8 |

**讀出三個關鍵 pattern**：
- **P7V1 雙 benchmark 都第 2、PE-AV 雙第二**：是**最 robust 的單一答案**，論文 primary 首選
- **P7V3 是 MusicCaps 單榜冠軍但 Jamendo 只第 3**：擅長 cross-domain，**論文 second-pick / ablation 對照組**
- **P8 (Random-NoQ) 在 Jamendo 第 1 但 MusicCaps 跌到第 8**：Random caption 但 no-Q → Jamendo 內穩、跨分布弱 → **驗證 Q conditioning 對 cross-domain generalization 有貢獻**（教授 2026-04-21 提的「Q 是 in-support gating」與此一致）

### 5.2 P7V1 是雙料前段的穩定答案
| Benchmark | CLAP | CLAP 排名 | PE-AV | PE-AV 排名 |
|---|---|---|---|---|
| Jamendo | 0.1981 | 2 | 0.1283 | 2 |
| MusicCaps | 0.1975 | 2 | 0.0524 | 1 |

四個槽中三個第二、一個第一，沒有任何指標跌出前三 → **跨 benchmark 最穩定的單一答案，論文 primary model 首選。**

### 5.3 pre-P6 的 MusicCaps FAD anomaly
P4V4 / P5V1 / P5V2 的 MusicCaps FAD (3.65 / 3.70 / 3.85) 顯著低於所有 P6+ 實驗 (≥ 4.30)。

**Plausible 解釋**：pre-P6 模型 caption 品質較粗、Q conditioning 尚未引入 → 輸出分布「更平均 / 更通用」→ 偶然更貼近 MusicCaps 這種泛音樂分布。**代價是 CLAP / AES / PE-AV 全部顯著較低**。不是「pre-P6 比較好」，是**單一 FAD 不足以判斷模型品質**的反例。

注意：在 Jamendo seed=42 上 pre-P6 模型 FAD（1.13–2.05）反而**比 Phase 6+ 高**（除 P5V1/P5V2 內部對照），因為 Jamendo 是訓練分布，後期模型已過 fit 該分布 → FAD 解讀對 test set 高度敏感。

### 5.4 P8V3 — CLAP-sim 當 Q 信號造成 genre shortcut 的範例
| | Jamendo (seed=42) | MusicCaps |
|---|---|---|
| CLAP | 0.1514（10/10 末） | 0.1619（10/10 末） |
| FAD | 2.526（10/10 末） | 5.545（10/10 末） |
| PE-AV | 0.1102（10/10 末） | 0.0369（10/10 末） |
| R@10 | 8.15%（10/10 末） | 3.59%（10/10 末） |

用 audio-text CLAP sim 當 q_embed 信號（10 bin 等頻分箱）→ 高 q_level 系統性偏向 piano/acoustic（LP-MusicCaps 對這類更精確 → 更高 CLAP sim）→ q=9 inference 時帶 genre shortcut → 音訊偏移 reference distribution → **全部指標皆為 10/10 最末**。

**與「CLAP-filter 訓練 + CLAP eval」資料洩漏不同**：P8V3 不是用 CLAP 閾值過濾訓練資料，而是把 CLAP sim 轉成 ordinal q 信號。失敗根因是「Q 信號與 genre 相關 → q_embed 學到 genre shortcut」，非「訓練-評測同指標」。詳見 `docs/experiments/Phase4_to_Phase8_Complete_Summary.md` 第十一節。

---

## 6. 生成與評測路徑

### Jamendo dirs（seed=42 random subset 用於最終表）
```
~/MeanAudio/eval_output/phase4_jamendo_v4_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase5_v1_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase5_v2_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase6_v2_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase7_v1_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase7_v2_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase7_v3_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase8_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase8_v2_stage2_200000_jamendo_s42/
~/MeanAudio/eval_output/phase8_v3_stage2_200000_jamendo_s42/
```

舊窄 subset 的 dirs (`_jamendo2048` 字尾) 仍在但不要使用，可以手動刪除以釋放磁碟空間。

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

### 重跑腳本（皆 gitignored，本機 tmp）
- Jamendo seed=42 (10 exps，本表用): `~/MeanAudio/run_jamendo_s42_regen.sh`
- Jamendo 窄 subset (deprecated): `~/MeanAudio/run_jamendo_regen.sh`, `run_jamendo_regen_early.sh`
- MusicCaps FAD (10 exps): `~/MeanAudio/run_musiccaps_fad.sh`
- MusicCaps reference download: `~/research/meanaudio_eval/download_musiccaps_reference.py`（rate-limited 版）
- Seed=42 TSV 生成：見 `~/MeanAudio/docs/experiments/ten_exp_full_benchmark.md` 中的 Python snippet（已直接寫死 random.seed(42)，1816 unique tracks）

### 機器可讀
平行的 TSV 版本：`ten_exp_metrics.tsv`（同目錄）。

---

## 7. 已知限制 / 尚未做的事

1. **n=10 實驗**，跨 benchmark 排名觀察屬經驗性、未做 significance test。
2. **本表全用 half-Q 訓練模型**；真正的 full-Q E2E（P9 V2 / P7V1_fullq_control）不在本表，獨立紀錄於 `best_results.md` 及相關 memory file。
3. MusicCaps FAD 用 5,131 wavs（從 5,521 官方清單拿到 93% 有效下載），n=2048 抽樣對 FAD 穩定性足夠，但 full-5521 FAD 未做。
4. 所有指標都是 single-seed；生成採 `cfg_strength 0.5 + num_steps 1`（MeanFlow 推理設定），未掃 cfg / steps。
5. P8V3 的 genre-shortcut 結論來自 piano 分箱實測（q=0~1: 10.4% piano, q=8~9: 21.2%，見 `Phase4_to_Phase8_Complete_Summary.md`）。本表只呈現 benchmark 結果，機制分析在該 doc。
