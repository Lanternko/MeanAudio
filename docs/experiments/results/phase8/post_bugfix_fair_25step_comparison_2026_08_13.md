# Phase 8 bug-fix 後 25-step 公平比較總表

> **2026-08-20 supersession notice:** 本文件記錄的是歷史
> `25 steps / CFG 4.5` guided-evaluation protocol。它在 2026-08-13 曾被誤設為
> canonical，但 operator 並未指定 4.5；新的 canonical protocol 是
> `MusicCaps 5,521 / MeanFlow 25 / CFG 0 / seed 42 / NoMask / full precision`。
> 下列數字保留作 provenance，不再作 primary comparator；不得改標成 CFG0。

更新日期：2026-08-13（Asia/Taipei）

## 口徑

- 主比較集固定為 MusicCaps 5,521、generation seed 42、25 steps、CFG 4.5、NoMask、full precision。
- Stage 1 使用 Flow Matching 25 steps（FM25）；Stage 2 使用 MeanFlow 25 steps（MF25）。
- `Stage 1 NoQ baseline` 是科學名稱。歷史實驗 ID 或路徑中的 `official` 只為 provenance，不代表人工官方標註，也不作為結果名稱。
- Full scale 為 Stage 1 400k updates、Stage 2 再訓練 200k updates；quarter scale 為 100k + 50k。
- 不把 MF1 / CFG 0.5、MusicCaps 512 pilot 或其他 protocol 的數字混入 25-step 排名。

## Full-scale 25-step 主表

下表各列皆為 MusicCaps 5,521、25 steps、CFG 4.5、NoMask、full precision。`Q / K` 欄的 `NoQ` 表示推論與該次訓練皆不使用 q conditioning；因此 Stage 1 NoQ baseline 不需要附加 K 值。

| 報告名稱 | 完整實驗 ID | Caption / training data | Stage | Q / K | Solver | CLAP | CE | CU | PC | PQ | 定位 |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| Historical Phase 7 Stage 1 reference | `phase7_v1_stage1_400000_musiccaps_fm25_noq_nomask` | LP-MC random-caption historical checkpoint | S1 400k | NoQ eval | FM25 | 0.1799 | 5.3512 | 6.5058 | 4.1161 | 6.3406 | 歷史 checkpoint；只作共同 eval protocol 參考 |
| Catalog-matched clean NoQ Stage 1 | `phase8_catalog_matched_noq_stage1_400000_musiccaps_fm25_noq_nomask` | LP-MC catalog-matched caption | S1 400k | NoQ | FM25 | 0.1909 | 5.4926 | 6.6257 | 3.9484 | 6.5045 | bug-fix 後 clean-NoQ control |
| Stage 1 NoQ baseline | `phase8_qwen_official_noq_full_stage1_400000_musiccaps_fm25_noq_nomask` | upstream track-level Qwen caption | S1 400k | NoQ | FM25 | 0.2003 | 6.5038 | 7.0474 | 4.6308 | 6.8943 | caption/stage 主表 baseline |
| Caption 1.0 Stage 2 | `caption_granularity_s1_s2_fair_ablation_caption1p0_s2_mf25_cfg4p5` | local first-10s Qwen, one sentence | S2 +200k | NoQ | MF25 | 0.2123 | 5.3443 | 6.4768 | 4.0105 | 6.3913 | local 10s concise caption |
| Caption 2.0 Stage 1 | `caption_granularity_s1_s2_fair_ablation_caption2p0_s1_fm25_cfg4p5` | local first-10s Qwen, multi-sentence | S1 400k | NoQ | FM25 | 0.2287 | 6.1257 | 6.8474 | 4.3176 | 6.7082 | Caption 2.0 的 Stage 1 control |
| Caption 2.0 Stage 2 | `caption_granularity_s1_s2_fair_ablation_caption2p0_s2_mf25_cfg4p5` | local first-10s Qwen, multi-sentence | S2 +200k | NoQ | MF25 | **0.2419** | 6.2105 | 6.6855 | 4.6891 | 6.5823 | Caption 2.0 的 Stage 2 result |

### 主表可直接回答的比較

- Caption 2.0 的 Stage effect：S2 - S1 CLAP = `+0.0132`。這是目前最乾淨的同 caption、同規模、同 seed、同 25-step/CFG 比較。
- Caption 2.0 S1 相對 Stage 1 NoQ baseline：CLAP = `+0.0284`。這同時包含 caption provenance、時間粒度與文字風格改變，不應寫成純 caption-style effect。
- Caption 2.0 S2 相對 Stage 1 NoQ baseline：CLAP = `+0.0416`。這同時包含 caption/corpus 與 Stage 2 effect。
- Caption 1.0 S2 相對 Stage 1 NoQ baseline：CLAP = `+0.0120`，但同樣同時改變 caption provenance 與 Stage。

## 四個主實驗：1-step / 25-step 詳細橫向表

四個報告指定實驗的 25-step 數字如下。AES 分別為 Content Enjoyment（CE）、Content Usefulness（CU）、Production Complexity（PC）與 Production Quality（PQ）；各欄都是分數，方向為越高越好。

| 實驗 | Scale / Stage | Caption | Steps | CFG | Q | CLAP | CE | CU | PC | PQ | 證據狀態 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| Stage 1 NoQ baseline | full S1 400k | upstream track-level Qwen | 25 | 4.5 | NoQ | 0.2003 | 6.5038 | 7.0474 | 4.6308 | 6.8943 | 完整 5,521 筆 |
| Caption 1.0 Stage 2 | full S2 +200k | local first-10s, one sentence | 1 | 0.5 | NoQ | 0.1927 | 5.7088 | 6.4092 | 4.9280 | 6.3793 | 完整 5,521 筆；舊 protocol |
| Caption 1.0 Stage 2 | full S2 +200k | local first-10s, one sentence | 25 | 4.5 | NoQ | 0.2123 | 5.3443 | 6.4768 | 4.0105 | 6.3913 | 完整 5,521 筆 |
| Caption 2.0 Stage 1 | full S1 400k | local first-10s, multi-sentence | 25 | 4.5 | NoQ | 0.2287 | 6.1257 | 6.8474 | 4.3176 | 6.7082 | 完整 5,521 筆 |
| Caption 2.0 Stage 2 | full S2 +200k | local first-10s, multi-sentence | 1 | 0.5 | NoQ | 0.2100 | 6.1519 | 6.5419 | 5.2592 | 6.5297 | 完整 5,521 筆；舊 protocol |
| Caption 2.0 Stage 2 | full S2 +200k | local first-10s, multi-sentence | 25 | 4.5 | NoQ | 0.2419 | 6.2105 | 6.6855 | 4.6891 | 6.5823 | 完整 5,521 筆 |

Stage 1 NoQ baseline 與 Caption 2.0 Stage 1 目前沒有找到相同 5,521-prompt protocol 的 1-step 完整 metric，因此不以 512 pilot、其他 checkpoint 或 Stage 2 數字代填。

### 由 1-step 到 25-step 的觀察差值

| 實驗 | protocol 變化 | ΔCLAP | ΔCE | ΔCU | ΔPC | ΔPQ |
|---|---|---:|---:|---:|---:|---:|
| Caption 1.0 Stage 2 | MF1/CFG0.5 → MF25/CFG4.5 | **+0.0196** | -0.3645 | +0.0676 | -0.9175 | +0.0120 |
| Caption 2.0 Stage 2 | MF1/CFG0.5 → MF25/CFG4.5 | **+0.0319** | +0.0586 | +0.1436 | -0.5701 | +0.0526 |

這兩列顯示 CLAP 分別增加 `0.0196` 與 `0.0319`，但它們同時把 steps 從 1 增至 25、CFG 從 0.5 增至 4.5，所以只能稱為「MF1/CFG0.5 到 MF25/CFG4.5 的 protocol gain」，不能歸因為純步數效果。若要隔離 steps，仍需補同一 CFG 下的 MF1 與 MF25。

### Steps / CFG matrix 狀態

完整的 Caption 2.0 `Stage 1 / Stage 2 × steps 1/25 × CFG 0.5/4.5` 八格矩陣尚未完成。原序列 `rmatched_s1_s2_steps_cfg_matrix_seed14159265` 在第一格 `s2_mf25_cfg0p5` 生成到 708/5,521 後中斷，沒有 metric，也沒有通過的 matrix report，因此不列入結果表。

不過矩陣中報告最需要的兩個 CFG4.5 / 25-step cells，已由後續完整公平評測獨立完成，對應上表的 Caption 2.0 Stage 1（FM25 / CFG4.5）與 Caption 2.0 Stage 2（MF25 / CFG4.5）。

## Quarter-scale Stage 1 K / Q 25-step 表

這一組固定為 upstream track-level Qwen caption、251,599 training rows、S1 100k、MusicCaps 5,521、FM25、CFG 4.5。Balanced arms 的低端為 q0；fixed arms 因實際 support 不足，以有支持的 q5 作低端。這張表可比較 Stage 1 的 Q/K 設定，但不可和 full-scale 主表直接當成同規模排名。

| 報告名稱 | 完整實驗 ID | Strategy | Eval Q | CLAP | CE | CU | PC | PQ | 角色 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Quarter Stage 1 NoQ | `phase8_qwen_bucket_quarter_noq_stage1_100000_musiccaps_n5521_fm25_noq` | NoQ | NoQ | **0.1873** | 6.2360 | 6.9328 | 4.4796 | 6.7655 | primary NoQ control |
| K=2 balanced low | `phase8_qwen_quarter_e2e_halfq_stage1_100000_musiccaps_fm25_q0` | balanced | 0 | 0.1654 | 6.0148 | 6.4406 | 4.6112 | 6.3160 | primary K-resolution |
| K=2 balanced high | `phase8_qwen_quarter_e2e_halfq_stage1_100000_musiccaps_fm25_q9` | balanced | 9 | 0.1670 | 6.0944 | 6.4878 | 4.7275 | 6.3449 | primary K-resolution |
| K=3 balanced low | `phase8_qwen_bucket_quarter_k3_balanced_stage1_100000_musiccaps_n5521_fm25_q0` | balanced | 0 | 0.1604 | 5.9557 | 6.5688 | 4.5382 | 6.4108 | backup K-resolution |
| K=3 balanced high | `phase8_qwen_bucket_quarter_k3_balanced_stage1_100000_musiccaps_n5521_fm25_q9` | balanced | 9 | 0.1627 | 6.0453 | 6.6277 | 4.6336 | 6.4520 | backup K-resolution |
| K=5 balanced low | `phase8_qwen_bucket_quarter_k5_balanced_stage1_100000_musiccaps_n5521_fm25_q0` | balanced | 0 | 0.1600 | 6.0504 | 6.8004 | 4.3059 | 6.6320 | primary K-resolution |
| K=5 balanced high | `phase8_qwen_bucket_quarter_k5_balanced_stage1_100000_musiccaps_n5521_fm25_q9` | balanced | 9 | 0.1609 | 6.1746 | 6.7899 | 4.3790 | 6.5782 | primary K-resolution |
| K=10 balanced low | `phase8_qwen_bucket_quarter_k10_balanced_stage1_100000_musiccaps_n5521_fm25_q0` | balanced | 0 | 0.1644 | 6.0843 | 6.7250 | 4.6310 | 6.5615 | primary K-resolution |
| K=10 balanced high | `phase8_qwen_bucket_quarter_k10_balanced_stage1_100000_musiccaps_n5521_fm25_q9` | balanced | 9 | 0.1625 | 6.1315 | 6.5558 | 4.7645 | 6.3513 | primary K-resolution |
| K=5 fixed low | `phase8_qwen_bucket_quarter_k5_fixed_stage1_100000_musiccaps_n5521_fm25_q5` | fixed | 5 | 0.1558 | 5.9256 | 6.2014 | 4.5200 | 6.0979 | diagnostic only |
| K=5 fixed high | `phase8_qwen_bucket_quarter_k5_fixed_stage1_100000_musiccaps_n5521_fm25_q9` | fixed | 9 | 0.1616 | 5.9032 | 6.1604 | 4.5846 | 6.0037 | diagnostic only |
| K=10 fixed low | `phase8_qwen_quarter_e2e_fullq_stage1_100000_musiccaps_fm25_q5` | fixed | 5 | 0.1651 | 6.2248 | 6.6986 | 4.6947 | 6.4875 | historical reference |
| K=10 fixed high | `phase8_qwen_quarter_e2e_fullq_stage1_100000_musiccaps_fm25_q9` | fixed | 9 | 0.1600 | 6.0393 | 6.4218 | 4.7256 | 6.1658 | historical reference |

Quarter-scale 的直接結果是：Stage 1 NoQ CLAP `0.1873` 高於所有 Q-conditioned K arms；各 balanced arm 的 q9 - q0 CLAP 只有 `-0.0019` 到 `+0.0023`，沒有顯示穩定、單調的高 q 優勢。

## Full-scale、有 Q、Stage 2 的 25-step 缺口

下列四個 full-scale checkpoint 已完成，且都從同一個 Stage 1 NoQ 400k checkpoint 啟動，只在 Stage 2 開啟 balanced K-bucket Q conditioning。目前只有 MusicCaps 5,521、MF1、CFG 0.5 的結果；MusicCaps 5,521、MF25、CFG 4.5 公平版本已依 K=2 → 3 → 5 → 10 排入串行 queue，固定 q9、seed 42、NoMask、full precision。

| 報告名稱 | 完整實驗 ID | K / strategy | 現有 MF1 CFG0.5 CLAP（僅參考） | MF25 CFG4.5 公平版 |
|---|---|---|---:|---|
| Full-scale Stage 2 Q, K=2 balanced | `phase8_qwen_s2q_from_noq_full_k2_balanced` | K=2 balanced, q9 | 0.1741 | **queue #1** |
| Full-scale Stage 2 Q, K=3 balanced | `phase8_qwen_s2q_from_noq_full_k3_balanced` | K=3 balanced, q9 | 0.1775 | **queue #2** |
| Full-scale Stage 2 Q, K=5 balanced | `phase8_qwen_s2q_from_noq_full_k5_balanced` | K=5 balanced, q9 | 0.1779 | **queue #3** |
| Full-scale Stage 2 Q, K=10 balanced | `phase8_qwen_s2q_from_noq_full_k10_balanced` | K=10 balanced, q9 | 0.1754 | **queue #4** |

曾有 MusicCaps head-512 的 K=5 MF25 / CFG0.5 protocol probe，但不是 CFG4.5、不是完整 5,521 prompts，因此不算完成上述公平表。

## 證據來源

- Full-scale caption/stage 四格：`/home/kojiek/logs/caption_granularity_s1_s2_fair_ablation_REPORT.json`
- Stage 1 NoQ baseline：`/home/kojiek/logs/phase8_qwen_official_noq_full_STAGE1_METRICS.json`
- Quarter K/Q 七臂：`/home/kojiek/logs/phase8_qwen_bucket_quarter_backlog_FINAL_METRICS.json`
- Full-scale Stage 2 K=2/3/5/10：`/home/kojiek/logs/phase8_qwen_s2q_from_noq_full_k{2,3,5,10}_balanced_FINAL_METRICS.json`
- Caption provenance：`/home/kojiek/MeanAudio/docs/experiments/caption_provenance_granularity_and_aes_controls.md`
- Caption 1.0 Stage 2 MF1/CFG0.5：`/home/kojiek/logs/phase8_qwen_caption10s_noq_full_FINAL_METRICS.json`
- Caption 2.0 Stage 2 MF1/CFG0.5：`/home/kojiek/MeanAudio/eval_output/metrics/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_musiccaps/metrics.txt`
