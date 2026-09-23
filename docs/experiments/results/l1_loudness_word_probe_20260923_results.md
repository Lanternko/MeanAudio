# L1 響度詞 probe 結果（2026-09-23，收線）

**問題**：D1 顯示缺陷詞推不出被點名的缺陷。響度是更容易的控制目標——訓練語料（peak-norm 視窗）裡含 "loud" 的 caption 平均 −14.1 LUFS / crest 4.4，含 "quiet|subdued|whisper" 的 −19.5 LUFS / crest 6.2（n=400 各；隨機 −16.8），即 5.5 LU 的監督訊號，而且是由壓縮（crest）而不是峰值承載。模型能不能從一個響度詞重現其中一部分？

**設定**：4 個 stem（rock / piano / edm / acoustic）× 5 種 caption（base、loud/quiet 形容詞＝訓練語料寫法、"played very loudly/quietly at … volume" 明確音量句）× 64 樣本 = 1,280 clip/格。兩個 NoQ checkpoint（c2p0 full、066 control quarter）× {CFG0, CFG3 null-neg}；MeanFlow 25 步、seed 42、fp32、NoMask。響度閘門：`level_match_rescore.py` 全部 scalar-gain 到 −30 LUFS 重評（1280/1280 ok，無 peak cap）。
腳本：`scripts/eval/{l1_loudness_probe_tsv.py,run_l1_loudness_probe.sh}`；輸出 `~/eval_output_nvme/l1_loudness_*`、`~/eval_output_nvme/l1_lvl30/`。

**統計**：每個 stem 內「變體 − base」取均值再對 4 個 stem 平均；95% CI 為 stem 內重抽樣 bootstrap（1000 次）。**變體與 base 的 noise 不成對、caption 也不同** → CLAP 差同時含「caption 換了」與「音訊換了」，只能當輔助。

## 結果（Δ vs base）

| 格 | 變體 | ΔLUFS | Δcrest | ΔPQ raw | ΔPQ @−30 LUFS |
|---|---|---|---|---|---|
| c2p0 CFG0 | loud adj | +1.46 [+1.09,+1.85] | −0.53 | −0.45 | −0.39 [−0.49,−0.31] |
| c2p0 CFG0 | quiet adj | −1.27 [−1.67,−0.92] | +0.41 | +0.12 | +0.16 [+0.08,+0.24] |
| c2p0 CFG3 | loud adj | +2.81 [+2.60,+3.02] | −0.77 | −0.59 | −0.52 [−0.60,−0.44] |
| c2p0 CFG3 | quiet adj | −2.84 [−3.13,−2.60] | +0.68 | +0.11 | +0.11 [+0.04,+0.18] |
| 066 CFG0 | loud adj | +1.41 [+1.00,+1.80] | −0.62 | −0.56 | −0.53 [−0.63,−0.43] |
| 066 CFG0 | quiet adj | −0.93 [−1.39,−0.47] | +0.33 | −0.01 | +0.03 [−0.06,+0.12] |
| 066 CFG3 | loud adj | +2.30 [+2.08,+2.51] | −0.91 | −0.77 | −0.74 [−0.83,−0.66] |
| 066 CFG3 | quiet adj | −2.06 [−2.30,−1.77] | +0.45 | −0.11 | −0.12 [−0.19,−0.05] |

明確音量句（loudvol/quietvol）方向相同但響度位移較小（+0.3～+1.5 / −1.1～−1.5 LU），PQ 代價與形容詞同量級或更大（066 CFG3 loudvol −0.87）。完整表見本 commit 的重算指令（`per_clip.tsv` 以 id 第 2 欄分組）。

## 讀法

1. **響度是文字可達的方向（與 D1 的缺陷相反）**：loud−quiet 形容詞在 CFG0 撐開 2.3～2.7 LU、CFG3 4.4～5.7 LU（訓練訊號 5.5 LU），而且 crest 同向移動 → 模型重現的是語料裡「壓縮承載的響度」，不只是增益。CFG 把它放大約 2×。
2. **"loud" 的 PQ 代價是處理不是音量**：響度對齊到 −30 LUFS 後 ΔPQ 幾乎不變（−0.39～−0.74）。這與 064 limiter 結論同型：用壓縮換來的響度扣 PQ，level 本身只佔一小部分。
3. **"quiet" 不是免費的 PQ 旋鈕**：c2p0 上 +0.11～+0.16 可測，066 control 上 CFG3 反而 −0.12；兩個 checkpoint 不一致 → 不能宣稱 "quiet" 提升品質。
4. **CLAP**：原始 CLAP 對 loud 為 +0.01～+0.03，響度對齊後翻成 −0.01～−0.03 → loud 的 CLAP 增益是 063 的「越大聲 CLAP 越高」效應；quiet 的 CLAP 增益在對齊後仍在（+0.03～+0.06），但 caption 不同、不可解讀為對齊變好。

## 對既有線的意義

- **D1/D2 的對照組**：模型不是沒有任何「屬性方向」——語料裡有、且由 caption 監督的屬性（響度/壓縮）它學得到；缺陷方向不存在是因為語料裡沒有（supports 031／075 的解讀，非 proves）。
- **Attribute 控制的取捨已量到**：「學會大聲」＝用 PQ 換 CLAP，與 meeting 2026-09-22「大聲的音量 / attribute」條目的預測一致。
- 限制：只有 4 個 stem、非成對 noise、單一推論 seed。
