# 2026-03-27 — 資料品質、評估指標、Phase 7 方向

## 資料品質過濾的效果（尚未定論）

兩種假說：
- **假說 A（大數法則）**：資料夠多，爛的 caption 被平均掉，過濾不重要
- **假說 B（方法問題）**：我們的過濾方式本身不夠好，才沒看到效果

教授直覺傾向假說 B：「如果 caption 是爛的，怎麼可能訓出好的模型？」→ 需要實驗驗證。

## 評估指標：聽感 > CLAP

> 「CLAP 很好啦，通常也可以相信，只是 CLAP score 到底代表了什麼，我個人覺得還是耳聽為憑。你聽起來覺得有變好就是有變好。這些其實都是輔助。」

→ 主觀評估不能省，metrics 是輔助工具。

## 新的資料過濾方向：Caption-Audio CLAP 相似度

- 不用 cross-model consistency（現行做法）
- 改算 **caption ↔ audio CLAP 相似度**，高相似度 = 好 caption
- 預先計算存起來，訓練時直接讀，成本可攤平

## ⚠️ Data Leakage 原則

> 「如果訓練資料過濾用了 CLAP，evaluation 就不能用 CLAP。」

| 過濾方法 | 可用 eval | 不可用 eval |
|----------|-----------|------------|
| Caption-audio CLAP 過濾 | FAD、Meta Aesthetics | ❌ CLAP |
| 不用 CLAP 過濾 | CLAP + FAD | — |

→ Phase 7 若採用 CLAP 過濾資料，evaluation 需改用 FAD 或 Meta Audiobox Aesthetics。
