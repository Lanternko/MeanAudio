# 075 補 seed：defectlab / defectunlab × {27182818, 16180339}（2026-09-24 預註冊 addendum）

**Operator 指示（2026-09-24）**：「GO」——回應「lab／unlab 各補兩個 seed，把 CFG0 打平與 negprompt 增益反向兩個結論確認下來」。

**為什麼**：075（`results/d2_defect_negsample_075_results.md`）只有訓練 seed 14159265；E2 判定借 control 的 3-seed 全距，處理臂自身的 seed 變異沒量。control（066/068/070）已有 14159265 / 27182818 / 16180339 三顆，補齊後三臂都是同一組 3 seed，可逐 seed 配對。

**配方**：與 075 完全相同（`scripts/training_pipelines/d2_defect_negsample_action.sh`，同一份 arm inputs / npz_farm / overlay_farm），只改 `D2_SEED`。每個 job 走 p2 queue，收尾時只瘦身自己的 S1 EMA 快照（留 30k/50k/80k/100k × 兩個 sigma），因為 NVMe 只剩 ~130G。

| queue | arm | seed |
|---|---|---|
| 076 | defectlab | 27182818 |
| 077 | defectunlab | 27182818 |
| 078 | defectlab | 16180339 |
| 079 | defectunlab | 16180339 |

每個 ~7.7 h（S1 3.8 h＋S2 2.9 h＋eval 0.9 h）→ 共 ~31 h。

## 判定（啟動前寫定）

每個 seed s 各自算，再對 3 個 seed 報均值與 t 型 95% CI（df=2）；三個值同號才寫「一致」。
所有 AES/CLAP 同時報原始與 −30 LUFS 響度對齊（`level_match_rescore.py`）。

1. **E2 主端點**：`ΔG_PQ(s) = G_lab(s) − G_unlab(s)`，`G = PQ(CFG3+neg) − PQ(CFG0)`（同 checkpoint 逐 clip）。
   「反向成立」＝ 3 個 seed 皆 < 0 且 CI 上界 < 0。同法報 lab−control。
2. **E3 CFG0 非劣性**（對 control 同 seed）：CLAP 界 −0.004（075 原界）、PQ 界 −0.155。
   lab「打平」＝ CI 下界 > 界；unlab「有掉」＝ 3 個 seed 皆 < 0 且 CI 上界 < 0。
3. **不宣稱**：CI 跨零者寫 inconclusive，不寫打平；n=3 的 CI 很寬，這是預期。
