# 108 結果：短負向措辭組 × 081 arm／control 補 3 seed

2026-10-07 收線。設計與預登錄見 `docs/experiments/shortneg_arm081_seeds_20261007.md`。p2 108 完成，terminal `completed`（02:10Z → 05:04Z，約 3 h）。

- Summary：`~/nvme_experiment_artifacts/meanaudio/shortneg_arm081_seeds_20261007/summary.json`（sha256 `4c4973fd…`）
- A = nmv2pair control，C = 081 arm（`qlabel`），各 3 seed 的 quarter S2 50k；s14159265 的 A 格沿用 086、C 格沿用 106，其餘 28 格本次新跑
- 協定：MusicCaps subset1024、MF25、生成 seed 42、CFG 3；增益 = 格 − 同 ckpt cfg0，逐 clip 配對；差中差（DiD）= C 增益 − A 增益
- CI：10,000 次 clip bootstrap（seed 20261007），3 seed 合併；主讀數 PQ lvl30

## 預登錄判定

| 規則 | 量到（3 seed 平均 [95% CI]） | 各 seed（s14／s16／s27） | 判定 |
|---|---|---|---|
| 1 標籤專一：DiD(lqrec) − DiD(lqnoisy) ≥ 0.19，3 seed 皆 > 0 | **+0.835** [0.799, 0.871] | +0.44／+1.03／+1.03 | **成立** |
| 2 整槽放大：DiD(lqnoisy) 與 DiD(fid8) 皆 ≥ 0.19，3 seed 皆 > 0 | lqnoisy **+0.240** [0.207, 0.273]；fid8 **+0.343** [0.309, 0.377] | lqnoisy +0.54／+0.11／+0.08；fid8 +0.44／+0.25／+0.34 | **成立**（見下方注意） |
| 3 乾淨 E1 = C lqrec 增益 − A lqnoisy 增益 | **+0.388** [0.354, 0.423]，占原 DiD(lqrec) +1.075 的 **36%** | +0.43／+0.34／+0.40 | 報數 |
| 4 CLAP（獨立讀數） | 乾淨 E1 CLAP **+0.0029** [−0.0005, +0.0064] | +0.001／+0.004／+0.004 | ≈0，CI 跨 0 |
| 5a 探索：arm 上 `low quality` 失效（DiD(lq) < 0 三 seed） | DiD(lq) +0.064 | −0.21／+0.10／+0.30 | **不成立**（106 的 −0.21 只在 s14） |
| 5b 探索：stored-null CFG3 懲罰在 arm 消失 | A none −0.076 [−0.109, −0.043]；C none −0.007 [−0.038, +0.024] | A −0.13／−0.10／+0.01；C −0.02／−0.02／+0.02 | 大致成立（s27 的 control 本來就沒有懲罰） |

**規則 2 的注意**：照預登錄（平均 ≥ 0.19 且三 seed 同號）判成立，但 DiD(lqnoisy) 的平均主要由 s14 撐起。s16／s27 只有 +0.11／+0.08，各自都低於 0.19；raw PQ（非 lvl30）更只有 +0.05／+0.05。fid8 的放大在三個 seed 都較穩（+0.25～+0.44）。所以「整槽放大」要寫成：**對 fidelity8 穩定存在；對 `low quality, noisy` 方向一致但大小依 seed 而變**。

## 各格增益（PQ lvl30，對同 ckpt cfg0；3 seed 合併）

| 負向文字 | A control | C arm | DiD [95% CI] | DiD 各 seed |
|---|---|---|---|---|
| lqrec `Low quality recording.` | −0.009 | **+1.066** | **+1.075** [1.035, 1.114] | +0.98／+1.13／+1.11 |
| lqnoisy `low quality, noisy` | +0.678 | +0.918 | +0.240 [0.207, 0.273] | +0.54／+0.11／+0.08 |
| lq `low quality` | −0.083 | −0.019 | +0.064 [0.034, 0.093] | −0.21／+0.10／+0.30 |
| none（stored null） | −0.076 | −0.007 | +0.069 [0.039, 0.099] | +0.11／+0.08／+0.01 |
| irrel（無關文字） | +0.321 | +0.283 | −0.038 [−0.065, −0.011] | −0.04／−0.15／+0.07 |
| fid8（fidelity8） | +0.843 | **+1.186** | +0.343 [0.309, 0.377] | +0.44／+0.25／+0.34 |

- **arm 上 lqrec 是最強的短句**：C lqrec 1.066 比 C lqnoisy 0.918 高 0.15（s16／s27 −0.23／−0.32 的反向在 s14 是 +0.11，即 s14 上 lqnoisy 較強）。在 control 上則相反：lqrec ≈0、lqnoisy +0.68。
- **arm 上 `low quality` 三 seed 都 ≈0**（−0.01／−0.05／+0.00）。DiD(lq) 會翻號是因為 control 的 lq 增益翻號（+0.20／−0.15／−0.30），不是 arm 變了。
- arm cfg0 比 control cfg0 低 −0.148 lvl30（−0.11／−0.17／−0.17），與 081 的「無前綴 CFG0 −0.15」一致。

## 其他指標

**CLAP DiD**（3 seed 合併）：lqrec +0.019 [0.016, 0.023]、lqnoisy +0.006、lq +0.010、none +0.007、irrel −0.008、fid8 +0.002（CI 跨 0）。

- arm 上 lqrec 的 CLAP 增益（+0.036）比 control（+0.017）大。但乾淨 E1 的 CLAP ≈0，因為 control 用 lqnoisy 也有 +0.033。所以 CLAP 上看不到「標籤專一」以外的東西。

**乾淨 E1 的其他 AES 軸**：raw PQ +0.401 [0.365, 0.437]；CE +0.029（跨 0，各 seed −0.12／+0.11／+0.09）；PC **−0.204**（−0.41／−0.00／−0.20）。

**DiD 的其他軸**：
- lqrec：CE +0.975、PC +0.356（arm 用 lqrec 時 PC 掉得比 control 少）
- fid8：CE +0.257、PC +0.147
- irrel：CE −0.183、PC −0.292（arm 用無關文字時 PC 掉更多）

**位準**（s16／s27；s14 見 086／106）：
- C lqrec 靜音 3／0，低於 C cfg0 的 15／6；LUFS −15.0／−15.6，比 cfg0 大聲約 3～4 LU，crest 5.2／5.3。lvl30 已對齊響度，但靜音少本身會抬高平均。
- irrel 在兩個 ckpt 都是靜音最多的格（A 37／30、C 46／23）。

## 判讀

1. **106 的拆解在 3 seed 上複製**：081 arm 的 LQ 負向增益（原 DiD +1.08）裡，
   - 約 36%（乾淨 E1 +0.39，三 seed +0.34～+0.43）是扣掉 control 措辭劣勢後仍留下的；
   - 其餘約 64% 是 control 對 `Low quality recording.` 這個措辭本來就沒反應（086、107）造成的。
   - 106 單 seed 的 44% 落在 3 seed 範圍的上緣，3 seed 平均是 36%。
2. **標籤專一成立**：只有被當成訓練前綴的那句（lqrec）在 arm 上被放大到 +1.07；意思相近的 `low quality` 在 arm 上完全沒有被帶起來（≈0）。訓練學到的是那串字，不是「低品質」這個概念。
3. **整槽放大也存在**：arm 對 fidelity8 的負向反應比 control 大 +0.34，三 seed 穩定。這部分和標籤字面無關，是 arm 整體對負向槽更敏感。
4. **CLAP 沒有讀到乾淨 E1**（+0.003，跨 0）。和 081 一樣，增益只在 AES PQ 這條被標籤定義的軸上；PC 反而降 0.20。
5. **stored-null CFG3 懲罰**：control 有 −0.08，arm 幾乎沒有，三 seed DiD 同號。可能是 arm 的條件分支與 null 分支距離被前綴訓練改變，本實驗沒有直接量。

## 不能這樣寫

- 「081 的 E1 是措辭假象」：乾淨 E1 +0.39 三 seed 都 > 0、CI 不跨 0，扣掉措辭劣勢後仍有效。
- 「081 讓模型學會『低品質』的概念」：`low quality` 在 arm 上 ≈0。
- 「乾淨 E1 代表品質提升」：只有 PQ（標籤與端點同為 AES PQ）；CLAP ≈0、PC −0.20；081 的 FAD 在 105 已未過。本實驗沒算 FAD。
- 「`low quality, noisy` 的整槽放大三 seed 都 ≥ 0.19」：只有 s14 達到。
- 限制：只有一個生成 seed、subset1024、無 FAD；s14 的格沿用 086／106，與本次新跑的 28 格是同協定但不同批次。
