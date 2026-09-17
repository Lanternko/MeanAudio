# 061 結果：固定 LUFS 下的 crest 介入（MusicCaps 5521, MF25, CFG3, fidelity8）

預註冊：`docs/experiments/crest_intervention_cfg3_20260916.md`（contract sha256
`b51fd1e5b6decc509a81cb30d9d087f9ec6f7317104c9f9978d1b5d53c915466`）。
產出：`~/nvme_experiment_artifacts/meanaudio/crest_intervention_cfg3_20260916/`
（`summary.json` / `per_clip.csv` / `report.md` / `crest_intervention.png`）。

執行：2026-09-16 13:42 → 2026-09-17 00:12，p2 job `005_crest_intervention_cfg3`（061 的插隊名）。
job 以 rc=0 完成、`terminal.json` 寫了 `status: completed`（progress `[5521, 27605]`），
但 queue 仍以 `missing HARN completed evidence or gate` 把它移進 `p2/held/`
（incident `held-005_crest_intervention_cfg3.20260916T161238Z.txt`）。
**這是收尾 gate 的問題，不是資料問題**：report/summary 齊全且 sha256 對得上。

## Primary endpoint

| 量 | 值 | 95% CI |
|---|---:|---:|
| 051 關聯隱含斜率 | +0.0634 PQ/dB | — |
| **061 合併配對劑量反應斜率**（n=19,886） | **−0.0169 PQ/dB** | [−0.0181, −0.0158] |
| fraction of association | −0.267 | — |
| 低調變穩健性斜率（`gain_mod_rms_db ≤ 3.0`, n=4,287） | **+0.0305 PQ/dB** | [+0.0271, +0.0344] |

腳本判定：`intervention_opposes_association: unregistered direction; treat as exploratory`。
預註冊只列了三種結果（因果 / partial / CI 跨零），**沒有列方向相反**，所以主結果落在
預註冊之外，只能寫成探索性，不能當成「crest 是 lever」或「crest 不是 lever」的已註冊裁決。

## 各 arm 配對 delta（vs `ref`，n=5,502）

| arm | achieved crest Δ | PQ Δ | CE Δ | CU Δ | PC Δ | gain_mod_rms | centroid ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| down | −0.739 | −0.0767 [−0.0810, −0.0724] | −0.0404 | −0.0806 | +0.0746 | 1.24 dB | 1.009 |
| up | +2.175 | −0.0240 [−0.0291, −0.0189] | −0.1513 | +0.0064 | −0.1584 | 4.61 dB | 0.972 |
| upmax | +2.514 | −0.0303 [−0.0356, −0.0250] | −0.1760 | +0.0005 | −0.1859 | 5.01 dB | 0.970 |
| rand | +2.700 | −0.1387 [−0.1439, −0.1336] | −0.1979 | −0.1139 | −0.0179 | 4.61 dB | 1.000 |

響度鎖定有效：四個 arm 的 `lufs_error` 量級都在 1e-15 LU 以下。

### 各 arm 自己的 PQ 劑量斜率

| arm | dPQ / dB achieved | 95% CI | n |
|---|---:|---:|---:|
| down | **+0.0891** | [+0.0832, +0.0951] | 4,115 |
| up | −0.0107 | [−0.0128, −0.0085] | 5,257 |
| upmax | −0.0115 | [−0.0135, −0.0095] | 5,257 |
| rand | −0.0404 | [−0.0420, −0.0389] | 5,257 |

## 解讀

1. **處理本身就扣 PQ，主斜率因此被污染。** 四個 arm 的 PQ delta 全為負，包含把 crest 往
   關聯的「壞方向」推的 `down`。合併斜率的負號主要來自「有沒有被處理」而不是 crest 的量，
   所以 −0.0169 不能當成 crest 的因果效果讀。
2. **劑量與處理強度混在一起。** `down` 只換到 −0.74 dB 卻只用 1.24 dB 調變，
   `up`/`upmax` 換到 +2.2/+2.5 dB 但用了 4.6/5.0 dB 調變並把頻譜質心壓到 0.97。
   預註冊已預告可達範圍是片段性質，這裡確認了：擴張方向的劑量必須付更高的處理代價。
3. **扣掉處理強度後方向與 051 一致。** `down` 的斜率 +0.089、低調變子集 +0.031（約關聯值的 48%），
   兩者 CI 都不跨零且為正。所以 051 的關聯成分沒有被排除。
4. **crest 這個數字本身不是關鍵，跟不跟隨音樂結構才是。** `rand` 的 achieved crest 比 `up` 更高
   （+2.70 vs +2.175）、調變深度相同，PQ 卻多掉 0.115、CE 多掉 0.047。
   與預註冊的判讀規則一致：這指向 transient 結構而非 crest 標量。
5. **決策面**：**不要把 crest 當訓練目標** —— 在固定響度下主動拉高 crest 讓 PQ 變差
   （up/upmax 都是負的），刷這個數字拿不到 PQ。但因為 `down` 與低調變子集的正斜率，
   **arm 間 crest 有差的 PQ 比較仍不能視為乾淨**：關聯的因果成分未被排除，
   只是被處理造成的劣化蓋住。

## 排除

n_total 5,521 → n 5,502（排除 19）。事前篩選（baseline LUFS < −40 或靜音比例 > 0.40）命中 19 筆，
其中 3 筆同時屬於「未能守住響度或破峰值」。合格 5,502 > postflight 門檻 5,400，通過。

## 限制

單一 checkpoint、單一 generation seed（沿用 051 的 baseline FLAC，未重新生成）。
主結果落在預註冊三種結果之外，屬探索性。只做衰減與受限擴張（cap 6 dB RMS 調變），
不可外推到更重的放大或壓縮。劑量分布受片段結構限制且不均，這是劑量反應證據，
不是固定劑量的隨機化對比。低調變子集是 post-hoc 條件化於介入後的共變量，不是隨機化子集。
AES 不是人類品質判斷。
