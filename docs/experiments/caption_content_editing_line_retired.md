# Caption 內容編輯線 — 收線（2026-09-22）

**決定（Lanternko 2026-09-22）**：停止在 caption **文字內部**做編輯的實驗線。不再排新 arm，既有 arm 的數字保留作為 null 證據。

## 收線理由：四類干預、同一個答案

| 線 | 干預 | MusicCaps 結果 | 文件 |
|---|---|---|---|
| 050 / 055 slot4、slot4v2 | 剝掉數字（8.6% 列） | quarter/full × CFG0/CFG3+neg 二十個比較全平手 | `caption2p0_slot4_no_digits_line.md` |
| 057 / 058 / 059 slot0nm | 去污染 ＋ 去量測 | CFG0 平手；CFG3+neg 的正差在成對 seed 翻成六格全負 | memory `project_slot0nm_057_result` |
| 066–071 slot0nmv2 | 只去量測、保留風格數字（3 seed 成對） | 兩格 CLAP 貼 0（+0.0017 / +0.0001），逐 seed 翻號；非劣性未判定 | `results/phase8/nmv2pair_three_seed_results.md` |
| 034 / 048 / 049 rotation | 換輪替的 slot、換 K（3 vs 10） | 五指標全落在 seed 雜訊內 | `c2p0_truerandom_q_granularity_line.md` |

對照組（換整個 captioner，不是編輯文字）：

- 038 / 039 / 042 MF 全覆蓋：CLAP 落後 Qwen ~0.012，四項 AES 全在雜訊內
- paired59k captioner-only control：**唯一非零**，Qwen +0.0073 CLAP（24× 同協定 seed 底線），但 AES 仍全在雜訊內

**模式**：caption 文字內部的編輯測不出東西；只有整個換 captioner 才動得了 CLAP，而且只動 CLAP。同一個 checkpoint 加 negative prompt 換到 +0.046 CLAP / +0.95 PQ，4× 訓練預算只換 +0.012 CLAP（`project_step_budget_vs_negprompt_roi`）—— 推論期那側的 ROI 高一個量級。

## 收線不等於證明的部分（重要）

1. **MusicCaps CLAP/AES 量不到 tempo / key / meter 的遵循度**（nmv2 contract D4 已登錄）。「去掉量測詞沒有傷害」這句話這兩個指標沒有資格回答；要回答需另建節奏／調性對齊評估。
2. 以上全是 **quarter 預算**（S1 100k + S2 50k）。若 caption 品質的效果存在但很小，可能被預算與 seed 底線一起蓋住。
3. slot0nmv2 的**非劣性沒有判定**（CFG0 CLAP CI 下界 −0.0097 vs 界線 −0.0084），只是不再為了判它而投 GPU。補 3 個 seed（約 21 h）只會收窄 CI，不會改變「沒改善」。
4. 一個未解釋的 arm-level 差異留著：去量測 arm 的 crest 一致較高、平均略小聲、靜音 clip 略多（6/6、5/6、5/6 格）。不是本線的結論，但比語料 arm 時仍要掃 `level_silent_n`。

## 可重啟的條件

- 有了能量測節奏／調性遵循度的評估 → 去量測那組 arm 值得重看（現成 checkpoint 可直接評）
- 換到 full 預算且有明確機制假說 → 才值得重開，不是再挑一個 caption 欄位來刪
