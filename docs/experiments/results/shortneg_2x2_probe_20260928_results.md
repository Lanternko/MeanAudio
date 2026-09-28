# 086 短負向 prompt 2×2 probe 結果（2026-09-28）

設計見 `../shortneg_2x2_probe_20260928.md`。原始數字：`/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_2x2_probe_20260928/summary.json`（sha256 `b4f767ca…`，queue terminal `completed`，2026-09-28 21:20 本地時間；11 格全齊，音檔已刪，cell JSON 保留逐 clip 分數）。

## 一句話

**081 control 的 ≈0 主要是措辭造成的，不是這個 checkpoint 對負向文字比較鈍。**
- `Low quality recording.` 在兩個 checkpoint 上都是最弱的措辭。在 A（081 control）上，它甚至比無關文字低 0.48 PQ。
- 換成 `low quality, noisy`，A 就有 +0.59。
- checkpoint 本身還有一個次要效應：同一措辭 B 一律比 A 多 +0.27～+0.38。

09-03 對照點逐位重現（diff 0.0、逐 clip r = 1.0），因此今天的程式碼與 09-03 的矛盾不是 code 漂移。

## 每格絕對分數（MusicCaps subset1024；MeanFlow 25 步、seed 42、fp32、NoMask、`--no_q`）

| ckpt | 格 | 負向 | PQ raw | PQ lvl30 | CLAP | CE | CU | PC | LUFS | crest | 靜音 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | cfg0 | — | 6.482 | 6.688 | 0.1981 | 6.048 | 6.631 | 5.104 | −19.1 | 6.41 | 12 |
| A | none | stored null | 6.370 | 6.559 | 0.2070 | 5.899 | 6.531 | 4.611 | −18.5 | 6.32 | 23 |
| A | lqrec | `Low quality recording.` | 6.540 | 6.729 | 0.2156 | 6.043 | 6.745 | 4.506 | −18.5 | 6.24 | 27 |
| A | lq | `low quality` | 6.693 | 6.887 | 0.2264 | 6.344 | 6.857 | 4.820 | −17.6 | 5.97 | 15 |
| A | lqnoisy | `low quality, noisy` | 7.062 | 7.279 | 0.2270 | 6.886 | 7.199 | 5.079 | −15.8 | 5.74 | 2 |
| A | irrel | 貓／試算表／印刷字 | 7.063 | 7.207 | 0.2301 | 6.785 | 7.257 | 4.845 | −17.7 | 6.01 | 16 |
| A | fid8 | fidelity8 | 7.307 | 7.514 | 0.2216 | 6.820 | 7.400 | 4.733 | −16.3 | 5.87 | 12 |
| B | cfg0 | — | 6.582 | 6.788 | 0.2142 | 6.311 | 6.705 | 5.144 | −18.5 | 6.25 | 6 |
| B | lqrec | `Low quality recording.` | 6.906 | 7.095 | 0.2471 | 6.671 | 7.049 | 4.940 | −16.6 | 5.71 | 4 |
| B | lq | `low quality` | 7.148 | 7.366 | 0.2484 | 6.972 | 7.265 | 5.321 | −14.5 | 4.70 | 0 |
| B | lqnoisy | `low quality, noisy` | 7.553 | 7.746 | 0.2385 | 7.281 | 7.579 | 5.240 | −16.1 | 5.67 | 0 |

- A = `phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265_stage2_50000`（081 control）。
- B = `phase8_qwen_caption10s_multisent_noq_full_stage2_200000`（09-03 的 c2p0_slot0）。
- 除 `cfg0` 外都是 CFG 3。

## 對自己 cfg0 的增益（逐 clip 配對，bootstrap 10000）

| 負向 | A ΔPQ lvl30 [95% CI] | B ΔPQ lvl30 [95% CI] | B − A | A ΔCLAP | B ΔCLAP |
|---|---|---|---|---|---|
| stored null | **−0.129** [−0.170, −0.088] | ≈0（09-03：6.579 vs 6.582 raw） | — | +0.009 | — |
| `Low quality recording.` | **+0.041** [+0.000, +0.082] | +0.307 [+0.267, +0.347] | +0.266 [+0.221, +0.309] | +0.017 | +0.033 |
| `low quality` | +0.199 [+0.158, +0.241] | +0.577 [+0.534, +0.622] | +0.378 [+0.332, +0.426] | +0.028 | +0.034 |
| `low quality, noisy` | +0.591 [+0.543, +0.640] | +0.958 [+0.910, +1.006] | +0.367 [+0.318, +0.416] | +0.029 | +0.024 |
| 無關文字 | +0.519 [+0.479, +0.559] | +0.357 raw（09-03，未重跑） | — | +0.032 | — |
| fidelity8 | +0.825 [+0.778, +0.874] | +1.067 raw（09-03，未重跑） | — | +0.023 | — |

所有格的 raw 與 lvl30 都同號、幅度接近，所以這些增益不是響度效應。

**交互項**（B − A 的「措辭 − lqrec」差）：
- lqnoisy：+0.102 [+0.054, +0.150]
- lq：+0.113 [+0.077, +0.149]

交互項存在但很小，主效應是兩個加法項。

## 讀法

1. **措辭是主因。** 兩個 checkpoint 的排序完全相同：lqrec < lq < lqnoisy。
   - 在 A 上，`low quality, noisy` 比 `Low quality recording.` 多 +0.55。
   - 這些差異大致拆成兩部分：
     - `Low quality recording.` → `low quality`（拿掉大寫、`recording` 與句號）：A +0.16，B +0.27。
     - 再加上 `noisy`：A +0.39，B +0.38。
2. **A 不是對文字鈍的模型。** A 的無關文字 +0.52，比 B 在 09-03 的 +0.357 還大，所以設計 doc 的「A 整體比較鈍」假說被否定。
   `Low quality recording.` 在 A 上比無關文字低 0.48。這個短句本身是特別差的負向，不是「任何文字」層的下限。
3. **checkpoint 是次要效應。** 同一措辭，B 比 A 多 +0.27～+0.38（CI 都不跨零）。
   A 與 B 同時差了預算（quarter 50k vs full 200k）和語料（slot0clean_nmv2 vs multisent），本 probe 無法再往下拆。
4. **A 的 stored null 負向扣分**（−0.13 PQ，PC −0.49，靜音 12→23），B 則 ≈0。以 stored null 為基準的話，A 的 lqrec 是 +0.17。
5. **「caption 自帶 low quality」假說不成立**（事後拆讀，非預登錄）。subset 有 239/1024 條 caption 含 `low quality`。按這個分組讀 lvl30 增益：
   - A lqrec：+0.047 vs +0.039。
   - B lqrec：+0.29 vs +0.31。
   - 其他措辭兩組也都重疊。

   因此 lqrec 弱的原因不是「負向文字撞到 prompt 內容而互相抵銷」，機制未明。
6. **全量與子集一致。** 081 全量上 control lqrec 是 raw +0.018／lvl30 −0.007；這裡是 +0.058／+0.041。兩者都小，子集沒有改變結論。

## 對 081 的影響

- **E1 差中差（+1.061）本身的定義不變**：arm 與 control 用的都是 `Low quality recording.`，control 的 ≈0 已獲重現。
- **敘述要改**：「未訓練的模型對短標籤沒反應」只對 `Low quality recording.` 這個措辭成立。
  - control 在 `low quality, noisy` 上有 +0.59，在無關文字上有 +0.52。
  - 所以 E1 的幅度有一大部分來自「081 恰好選了 control 上最弱的措辭」。
- **比較粗略的對照（不同子集，只作參考）**：arm 全量 lqrec 增益 +1.05，control 在子集上最好的短措辭是 +0.59，fidelity8 是 +0.83。標籤訓練仍然超出 control 的所有短負向，但沒有差中差那麼誇張。
- **要做乾淨的比較，需要在 arm 上跑同一組措辭。** 本輪沒排（未排隊）。

## 限制

- 單一生成 seed、單一訓練 seed（A = s14159265），只跑 subset1024，沒有 FAD。
- B 的 none／irrel／fid8 沿用 09-03 的 raw 數字。09-03 的 AES 用 batch 32；B 的兩個對照點今天逐位重現，所以沿用風險低，但 lvl30 沒有這三格。
- 措辭的四個差異（大寫、`recording`、句號、`noisy`）只拆出了 `noisy` 這一項。其餘三者合在一起。

## 產物

- 腳本：`scripts/eval/shortneg_2x2_probe_20260928.py`、`scripts/experiment_harness/shortneg_2x2_probe_20260928_guest.py`
- 輸出：`/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_2x2_probe_20260928/`（`summary.json`＋`cells/*.json`，17 MB）
