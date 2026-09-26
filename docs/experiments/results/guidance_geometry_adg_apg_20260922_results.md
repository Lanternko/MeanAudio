# 073 收線：guidance 幾何（ADG／APG）——純 CFG 的浪費換不回 PQ；ADG γ0.5 讓 fidelity8 的 cfg 4.5 不再回落

2026-09-27 補寫（實驗 2026-09-22～23 跑完，summary 當時未寫成文件）。預註冊：
`docs/experiments/guidance_geometry_adg_apg_20260922.md`。延伸格（先 normalize 再減）另見
`guidance_geometry_prenorm_20260923_results.md`。

- driver：`scripts/eval/guidance_geometry_adg_apg_20260922.py`
- 產物：`~/nvme_experiment_artifacts/meanaudio/guidance_geometry_adg_apg_20260922/summary.json`、`cells/*.json`
- checkpoint：`phase8_qwen_caption10s_multisent_noq_full_stage2_200000`（c2p0_slot0 full NoQ）
- 協定：MusicCaps（Stage A 1,024 子集／Stage B 全量 5521）、MF25、seed 42、fp32、NoMask、CLAP batch 1
- 數字都是**同 seed 逐檔配對** Δ（geometry − 同 cfg 的 vanilla），CI 是 bootstrap 95%

## 閘門

| 閘門 | 結果 |
|---|---|
| vanilla 複製（N8 cfg3.0 全量 vs 051 manifest，音訊 sha256） | 5521/5521 相同、0 mismatch ✅ |
| 等價式（γ=0、η=1 對 vanilla 的 latent 相對誤差） | 1.60e-7／1.52e-7（門檻 1e-5）✅ |
| early-kill → Stage B | 已觸發（`A__N0__cfg4.5__adg_g1.0_refB` crest +0.53 > 0.5） |
| 響度閘門 | 只有 N8 cfg4.5 ADG γ0.5（−0.64 LU）超過 0.5 LU → 已補響度對齊讀數 |
| **FAD（第三端點）** | ❌ **沒算出來**：所有格 `fad=-1, pairs=0`。`eval_metrics.py` 的預設參考目錄 `/mnt/HDD/kojiek/music_semantic_fidelity/original_audio` 不存在；MusicCaps 參考音訊在 `/mnt/HDD/kojiek/musiccaps_reference`（5,132 wav），要明確傳 `--ref_dir` |

## Primary：N0（純 CFG，無負向 prompt）cfg 4.5

| geometry | 全量 ΔPQ | ΔCLAP | Δcrest | ΔLUFS |
|---|---|---|---|---|
| ADG γ0.5 | **+0.0225** [+0.015, +0.030] | +0.0025 [+0.002, +0.003] | +0.12 | +0.19 |
| APG η0 | +0.0063 [−0.000, +0.013] | +0.0004 | +0.23 | −0.10 |

N0 cfg 4.5 vanilla 的 PQ 是 6.490，比 CFG0（≈6.579）低 0.089。ADG 收回其中的 0.022，
而 fidelity8 負向 prompt 在同一個 checkpoint 上是 +1.02。

**判讀（依預註冊判定表）**：PQ 有小幅、CI 不跨零的上升，crest 回升，CLAP 持平或微升。
但是規模只有 fidelity8 的 **2%**，而且小於全量推論 seed 底線（PQ 0.142）。
所以**不支持**「範數放大是純 CFG 拿不到 PQ 的主因」。negative 文字的作用**不能**用幾何取代，
這與 080 的「負向槽要的是文字」一致。

## Secondary：N8（CFG＋fidelity8）cfg 3.0 → 4.5 的 PQ 回落

| 格 | PQ | CLAP | LUFS | crest_min |
|---|---:|---:|---:|---:|
| vanilla cfg3.0（= 標準 CFG3+neg） | 7.599 | 0.2462 | — | — |
| vanilla cfg4.5 | 7.560 | 0.2493 | — | — |
| **ADG γ0.5 cfg4.5** | **7.671** | **0.2505** | −19.18 | 1.61 |
| APG η0.5 cfg4.5 | 7.567 | 0.2483 | −18.36 | 1.59 |
| ADG γ0.5 cfg3.0 | 7.642 | 0.2473 | −19.17 | 1.73 |

配對 Δ（全量）：

| 對比 | ΔPQ | ΔCE | ΔCU | ΔPC | ΔCLAP | ΔLUFS |
|---|---|---|---|---|---|---|
| ADG γ0.5 cfg4.5 − vanilla cfg4.5 | +0.110 [+0.099, +0.122] | +0.100 | +0.075 | +0.080 | +0.0012 | **−0.64** |
| 同上，**響度對齊後**（n=3245） | **+0.096** [+0.081, +0.110] | +0.113 | +0.072 | +0.095 | +0.0017 [+0.000, +0.003] | 0 |
| ADG γ0.5 cfg4.5 − vanilla cfg3.0（標準格） | +0.072 [+0.063, +0.080] | +0.059 | +0.058 | +0.045 | +0.0043 [+0.003, +0.005] | −0.43 |
| ADG γ0.5 cfg3.0 − vanilla cfg3.0 | +0.043 [+0.037, +0.048] | +0.038 | +0.027 | +0.036 | +0.0010 | −0.42 |
| APG η0.5 cfg4.5 − vanilla cfg4.5 | +0.006 | +0.015 | +0.011 | −0.018 | −0.0010 | +0.18 |

**判讀**：

- ADG γ0.5 消掉了 cfg 3.0 → 4.5 的 PQ 回落，而且比標準 CFG3+neg 格高 +0.072 PQ／+0.0043 CLAP，
  四個 AES 軸同向。響度對齊後仍然成立（+0.096）。所以這**不是**「變小聲所以 PQ 高」。
- APG 在兩個家族都 ≈ 0，而且讓輸出變大聲、CLAP 微降，只能寫「本設定下無效」。
- γ1.0 與 frame 軸的 pilot 讀數比 γ0.5 更安靜（−1.4～−1.6 LU）、PC 更高，
  但 PQ 對齊後與 γ0.5 同級，沒有進全量。

## 限制（決定這些數字能寫到哪一層）

1. **響度對齊子集有選擇偏誤**。ADG 比較小聲，所以對齊是把 ADG 放大，
   會削波的 2,276 個 clip 被排除，剩 59%。被排除的偏向高能量 prompt，
   而那正是 cfg 4.5 飽和的地方。所以 +0.096 **只能**寫成「在不會削波的 59% 上成立」。
2. **單一 checkpoint、單一推論 seed**。配對 CI 窄，但全量推論 seed 底線是 PQ 0.142。
   +0.07～+0.11 還沒有跨 seed 驗證，**不能**寫成「新的最佳推論設定」，只能寫成
   「同 seed 配對下的正向訊號，待第二個推論 seed 確認」。
3. **FAD 沒算**。negprompt 線唯一的負帳（FAD +0.046）沒有得到回答。
   ADG 讓 crest 上升、響度下降，方向上可能離參考分布更遠也可能更近，沒有數據就不寫。
4. ADG 只是最小實作（範數保持一項），不是完整復現。

## 可寫層級

| 已證明（observation） | 高可信推論 | 不能這樣寫 |
|---|---|---|
| 在 c2p0_slot0 上，N8 cfg4.5 ADG γ0.5 的同 seed 配對 ΔPQ +0.110，對齊響度後在 59% 子集上 +0.096 | 範數保持能讓「負向 prompt＋高 cfg」多拿一點 PQ，而不以響度換 | 「ADG 是新的最佳推論設定」（缺第二個 seed） |
| N0 的 ADG 只收回 0.022 PQ（fidelity8 的 2%） | 純 CFG 拿不到 PQ 的原因主要**不是**範數放大 | 「幾何可以取代負向 prompt」 |
| APG 在兩個家族 ≈ 0 | — | 「APG 在音訊上無效」 |

## 後續

- 在 084（負向 prompt 蒸餾進 S2 target）之後，如果需要推論期的最後一哩，
  再做「ADG γ0.5 cfg4.5 × 第二個推論 seed × FAD」的補格。三件事一次跑，只花推論成本。
  這一格**不**排在 084 前面：它的上限約是 +0.1 PQ，而 084 問的是能不能把 1.0 PQ 的增益收進權重。
