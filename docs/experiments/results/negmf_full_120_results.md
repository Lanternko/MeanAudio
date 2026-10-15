# NegMF N100 在論文 recipe 上（p2 120–122）結果

設計與預先登錄門檻：`../negmf_full_120_20261012.md`（端點沿用 084）。
分析：`scripts/analysis/negmf_full_120_analysis.py`；數字：`negmf_full_120_summary.json`（2026-10-15 三 seed 全部到齊後重跑）。

- 對照組：114／116／117 full control（slot0clean nmv2matched，NoQ）。
- arm：從同一個 S1 it 400k 分支，只重訓 S2 200k，∅ 分支改讀 fidelity8 特徵。
- 三個 seed 的 12 格都齊全，`cells_missing` 為空。

## 主端點：全過

差值都是 arm − control，逐 clip 配對（n = 5521），CI 是 clip bootstrap。

| 端點 | 合併 [95% CI] | 各 seed（14159265／16180339／27182818） | 門檻 | 判定 |
|---|---|---|---|---|
| E1：CFG0 lvl30 PQ | **+0.956** [+0.940, +0.972] | +0.958／+0.920／+0.991 | ≥ +0.31、CI 下界 > 0、三 seed 同號 | 過 |
| E2：CFG0 CLAP raw | +0.0118 [+0.0103, +0.0133] | +0.006／+0.014／+0.016 | CI 下界 > −0.008 | 過 |
| 靜音逃逸 | arm CFG0 5／12／7 段，control 14／22／38 段 | — | arm > 2× control 且 > 1% | 未觸發 |

E1 比 084（quarter、S2-only，+0.948）大小幾乎一樣，R = 1.04。

## 描述性讀數

**其他 CFG0 讀數**（arm − control）
- CE +1.108
- CLAP lvl30 +0.017

**arm CFG0 vs control CFG3＋neg**
- PQ +0.038，三 seed 裡有一個是負的
- CLAP raw −0.023

→ arm 不用推論期負向，PQ 就追平 control 的推論期負向；但 CLAP 輸 0.023。

**疊加：兩邊都跑 CFG3＋neg**
- arm − control PQ +0.252
- CLAP −0.005

**1-NFE CFG0**：arm − control PQ +0.843。

**對 MusicGen-medium（lvl30 格平均）**

| arm 格 | PQ | CLAP |
|---|---|---|
| CFG0 | +0.190 | −0.014 |
| CFG3＋neg | +0.404 | +0.008 |

**FAD**（越低越好；arm 三 seed 都更差，與 084 相同）

| 格 | control | arm | arm − control（平均） |
|---|---|---|---|
| CFG0 | 3.19–3.34 | 5.78–6.07 | +2.67 |
| CFG3＋neg | 5.00–5.44 | 6.64–6.93 | +1.60 |

## 判讀

- E1 過，照預先登錄的措辭，只能寫「full recipe 上複製 084 的 PQ 位移」。
- 這是 PQ 導向的訓練，所以 AES 欄不進系統排名。
- FAD 代價與 084 同方向、同量級，不可寫成「不需推論期負向的品質提升」。
- 無人類評分。

## 後續

- 120–122 跑完，三個 full control 的 S1 `ckpt_last` 與 mf25 音檔的「122 完成前不可刪」限制解除。
- 2026-10-15：s16180339 的 S1 400k 目錄搬到 `/mnt/seagate/nvme_archive/exps_nvme/`（留 symlink）；三個 S1 現在都在 seagate，S2 留 NVMe。
