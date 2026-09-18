# 062 negprompt 核心表 CLAP 逐檔（batch 1）重算結果

- 契約：`docs/experiments/negprompt_clap_batch1_rescore_20260917_contract.json`；產出：`~/nvme_experiment_artifacts/meanaudio/negprompt_clap_batch1_rescore_20260917/summary.json`
- 協定：MusicCaps 5,521、MF25、seed 42、NoMask、fp32；CFG0 12 arm（novocal_reeval）＋CFG3+fidelity8 14 arm（negprompt_reeval_cfg3.0），音檔重生後以 `phase4_eval.compute_clap_score`（batch 1）評分。
- 忠實度閘：64-clip AES 重現歷史值，26/26 通過，最大絕對差 1.7e-6。
- queue 狀態 `held/` 是契約預期（eval-only job 不產 CFG0 report），不是失敗。

## 表（依 b1 排序；b32 為歷史發表值）

### CFG0

| arm | CLAP b1 | CLAP b32（舊） | b32−b1 |
|---|---:|---:|---:|
| c2p0_fair013_best_full | 0.2288 | 0.2299 | +0.0011 |
| c2p0_slot0_full_seed27182818 | 0.2191 | 0.2234 | +0.0043 |
| c2p0_slot0_q5_full_q9 | 0.2174 | 0.2235 | +0.0062 |
| c2p0_slot0_q5_full_q0 | 0.2162 | 0.2212 | +0.0051 |
| c2p0_slot0_full_noq | 0.2149 | 0.2201 | +0.0052 |
| c2p0_slot0_q3_full_q9 | 0.2145 | 0.2190 | +0.0045 |
| c2p0_slot0_q3_full_q0 | 0.2136 | 0.2172 | +0.0036 |
| c2p0_slot2_full_noq | 0.2135 | 0.2143 | +0.0008 |
| c2p0_fair013_worst_full | 0.2109 | 0.2195 | +0.0086 |
| p7v1_fullq_control_q9 | 0.1929 | 0.1860 | -0.0070 |
| fulltrack_q3_full_q9 | 0.1821 | 0.1870 | +0.0049 |
| fulltrack_noq_full | 0.1791 | 0.1845 | +0.0054 |

### CFG3+neg

| arm | CLAP b1 | CLAP b32（舊） | b32−b1 |
|---|---:|---:|---:|
| c2p0_fair013_best_full | 0.2628 | 0.2762 | +0.0135 |
| c2p0_slot0_full_seed27182818 | 0.2502 | 0.2608 | +0.0106 |
| c2p0_slot0_q3_full_q0 | 0.2496 | 0.2618 | +0.0122 |
| c2p0_slot0_q3_full_q9 | 0.2494 | 0.2619 | +0.0125 |
| c2p0_slot0_q5_full_q9 | 0.2477 | 0.2618 | +0.0141 |
| c2p0_slot0_q5_full_q0 | 0.2476 | 0.2621 | +0.0145 |
| c2p0_slot0_full_noq | 0.2462 | 0.2605 | +0.0143 |
| c2p0_fair013_k3_full_q9 | 0.2391 | 0.2472 | +0.0081 |
| c2p0_fair013_worst_full | 0.2387 | 0.2521 | +0.0134 |
| c2p0_slot2_full_noq | 0.2373 | 0.2459 | +0.0086 |
| p7v1_fullq_control_q9 | 0.2057 | 0.2104 | +0.0047 |
| fulltrack_noq_full | 0.1878 | 0.1890 | +0.0012 |
| fulltrack_q3_full_q9 | 0.1809 | 0.1806 | -0.0004 |
| a3_mfshort100k_direct_noq | 0.1708 | 0.1772 | +0.0064 |

## negprompt 增益（CFG3+neg − CFG0，同 arm）

| arm | b1 | b32（舊） |
|---|---:|---:|
| c2p0_slot0_full_noq | +0.0313 | +0.0404 |
| fulltrack_q3_full_q9 | -0.0011 | -0.0064 |
| c2p0_fair013_worst_full | +0.0277 | +0.0326 |
| c2p0_slot0_q5_full_q9 | +0.0303 | +0.0383 |
| fulltrack_noq_full | +0.0087 | +0.0045 |
| c2p0_fair013_best_full | +0.0340 | +0.0463 |
| c2p0_slot0_q3_full_q9 | +0.0349 | +0.0429 |
| c2p0_slot2_full_noq | +0.0238 | +0.0316 |
| c2p0_slot0_full_seed27182818 | +0.0311 | +0.0374 |
| c2p0_slot0_q5_full_q0 | +0.0314 | +0.0409 |
| c2p0_slot0_q3_full_q0 | +0.0360 | +0.0445 |
| p7v1_fullq_control_q9 | +0.0128 | +0.0245 |

## 判讀

1. b32 系統性偏高且 CFG3+neg 偏更多（中位 ~+0.013 vs CFG0 ~+0.005）→ 舊表**高估 negprompt 的 CLAP 增益約 0.01**（slot0 full +0.040 → +0.031）。增益方向與 c2p0 > fulltrack 的結論不變。
2. 排名相關：CFG0 Spearman 0.944、CFG3+neg 0.912。翻序：CFG0 下 p7v1 fullq q9 從墊底變成 fulltrack 之上；fair013 worst 從第 6 掉到第 9；CFG3+neg 下 fulltrack noq/q3 對調。
3. fair013 best 在兩個協定都第一（CFG0 0.2288、CFG3+neg 0.2628）。
4. 往後 negprompt 核心表的 CLAP 一律引用 b1 欄。ablation/single/044/045 仍是組內同 scorer，未重算。
