# 107 結果：無關文字負向 × nmv2pair control（3 seed 全量）

2026-10-07 收線。設計與預登錄見 `docs/experiments/irrelneg_nmv2pair_probe_20261007.md`。p2 107 完成，terminal `completed`，實際約 2 h。

- Summary：`~/nvme_experiment_artifacts/meanaudio/irrelneg_nmv2pair_probe_20261007/summary.json`（sha256 `ba4592f2…`）
- 負向文字（irrel）：`a photograph of a cat, a spreadsheet, printed text`
- 增益 G_k = 格 k − cfg0，逐 clip 配對；CI 為 10,000 次 clip bootstrap（seed 20261007），3 seed 合併

## 主要比較（預登錄）

D = G_rev − G_irr，PQ lvl30：**−0.074 [−0.082, −0.066]**。各 seed −0.083／−0.102／−0.037，三個 seed 同號。

- |D| < 0.10 → 照預登錄判 **「兩者相當」**（`reversed_equivalent_to_irrelevant`）。
- CI 上界 < 0，但沒有達到 −0.10 的門檻，所以不判「reversed 較低」。方向上，reversed 小幅但穩定低於無關文字。
- 排除靜音 clip 後 D = −0.079 [−0.087, −0.071]，結論不變。
- 預測是 D ≤ 0，落在「相當」或「reversed 較低」，結果命中。

**文件改法（照預登錄）**：084 Stage C 的 38% 是 **「非空文字層」**，不是 fidelity 領域詞彙。

## 各格增益（PQ lvl30，對 cfg0）

| 負向文字 | s14159265 | s16180339 | s27182818 | 3 seed 合併 [95% CI] | R = G / G_neg |
|---|---|---|---|---|---|
| fidelity8（neg） | 0.828 | 0.854 | 0.828 | **+0.837** [0.819, 0.855] | 1.00 |
| 無關文字（irrel） | 0.542 | 0.248 | 0.210 | **+0.333** [0.318, 0.348] | 0.40 |
| reversed（revneg） | 0.459 | 0.146 | 0.173 | **+0.259** [0.243, 0.275] | 0.31 |
| `Low quality recording.`（lqneg） | 0.057 | −0.092 | 0.015 | **−0.007** [−0.021, +0.008] | −0.01 |

- 預測 G_irr 約 +0.4～+0.6、R_irr 約 0.5～0.7。實際 +0.33、R_irr 0.40，**比預測低**。086 在 s14 子集上的 +0.519 代表的是 s14，不代表其他 seed；s14 本來就是三個 seed 裡增益最大的。
- 無關文字的增益隨 seed 變動很大（0.54 → 0.21），和 reversed 的變動形狀一樣。

## 對比（PQ lvl30，3 seed 合併）

| 對比 | 平均 [95% CI] | 各 seed |
|---|---|---|
| neg − irrel | **+0.504** [0.492, 0.516] | +0.286／+0.606／+0.618 |
| revneg − irrel（主比較 D） | −0.074 [−0.082, −0.066] | −0.083／−0.102／−0.037 |
| lqneg − irrel | **−0.340** [−0.350, −0.330] | −0.484／−0.341／−0.195 |

排除靜音後：neg − irrel +0.499，lqneg − irrel −0.351。

## 其他指標（增益對 cfg0，3 seed 合併）

| 負向 | raw PQ | CE | CU | PC | CLAP |
|---|---|---|---|---|---|
| irrel | +0.417 | +0.296 | +0.452 | −0.628 | +0.0201 |
| revneg − irrel | −0.060 | −0.019 | +0.013 | +0.208 | −0.0041 |
| lqneg − irrel | −0.399 | −0.402 | −0.376 | −0.031 | −0.0065 |
| neg − irrel | +0.465 | +0.435 | +0.367 | +0.239 | +0.0101 |

- 無關文字的 PC 掉最多（−0.63），比 fidelity8（−0.39）、reversed（−0.42）都多。
- CE 上 revneg 和 irrel 各 seed 翻號（−0.19／+0.21／−0.08），只有 PQ lvl30 和 CLAP 的 D 三 seed 同號。

**irrel 格的位準與 FAD**（各 seed s14／s16／s27）：

| 讀數 | irrel | 對照 |
|---|---|---|
| 靜音 clip | 105／176／189 | cfg0 52／39／39；revneg 185／20／137 |
| FAD | 4.566／5.079／5.492 | cfg0 3.81／3.93／3.82；neg 5.23／6.36／6.68；lqneg 3.85／4.15／4.27 |
| CLAP | 0.2273／0.2191／0.2103 | cfg0 約 0.198 |
| LUFS | −17.77／−20.32／−19.64 | — |
| crest | 6.10／7.35／7.67 | — |

- irrel 的 FAD 介於 lqneg 和 fidelity8 之間；PQ 增益也介於兩者之間。這和 084 的「FAD 代價與 PQ 增益約等比例」一致。
- irrel 在每個 seed 都比 cfg0 多 2～5 倍靜音。revneg 的靜音逃逸只出現在 s14／s27，irrel 三個 seed 都有。

## 判讀

1. **084 Stage C 的 38%（非極性部分）是「非空文字層」**：
   - 把任意一段和音樂無關的文字放進負向槽，推論期拿到 G_neg 的 40%。
   - reversed 文字拿到 31%，沒有超過無關文字，反而小幅較低。
   - 所以 reversed 那 38% **不能**歸功於「fidelity 領域詞彙」。
2. **「描述缺陷」的專屬效果** = neg − irrel = +0.50，約占 G_neg 的 60%。這和 Stage C 用 rev 算出的 62% 幾乎一樣（rev 與 irrel 相當）。
3. **`Low quality recording.` 比無關文字低 0.34**，三個 seed 同號。這個短措辭不是「沒有作用」，而是低於任何非空文字的底線。
   - 它只有 5 個 token；irrel 和 rev 有 10～20 個。是長度還是措辭造成的，本 probe 分不開。
   - 這也是 109–111（lq100 訓練期）的重要對照：如果 R_train ≈ R_inf 成立，lq100 的 E1 應該 ≈ 0。
4. **與 negprompt 消融定論的關係**：
   - 2026-08-31 的定論寫的是「增益來自 fidelity 領域詞彙不是缺陷極性（reversed 複製 51%）」。
   - 在 nmv2pair control 的全量 3 seed 上，reversed 的這部分可以用無關文字取代。所以「領域詞彙」的說法要降級：reversed 能拿到的，只是非空文字層。
   - 本 probe 沒有重跑 08-31 消融的設定。在那個設定下，reversed 是否超過無關文字仍未驗證。

## 不能這樣寫

- 「任何非空文字放進 NegMF 的 ∅ 分支都有 38% 效果」：本 probe 只量推論期，訓練期的 irrel arm 沒跑。這只能靠 R_train ≈ R_inf 推論。
- 「無關文字和 reversed 完全等價」：D 的 CI 不跨 0，只是小於預登錄門檻。
- 「無關文字是免費增益」：FAD 變差（3.8 → 4.6～5.5），靜音變多 2～5 倍，PC 降 0.63。
- 只有一種無關文字（086 的同一句）。
