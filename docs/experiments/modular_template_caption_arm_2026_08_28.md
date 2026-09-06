# 模塊化模板 caption arm — 語料與校準 gate

日期：2026-08-28　狀態：**語料已建，校準 gate 未過，等 operator 決定是否排入 queue**

動機來自 [`results/phase8/fulltrack_pq_gap_mechanism_2026_08_28.md`](results/phase8/fulltrack_pq_gap_mechanism_2026_08_28.md)：
在所有控制之後，fulltrack（AES 強）與 Caption 2.0（AES 弱）之間唯一存活的差異是
**訓練期文本分布的寬度**。這個 arm 要單獨操縱那個變因。

---

## 設計

`scripts/preprocess/build_modular_template_captions.py` 把每列 Caption 2.0 caption
拆成受控槽位（genre / tempo / instruments / mood / texture / rhythm / dynamics /
space / timbre / production），再用 **8 個固定 frame** 之一（依 `clip_id` sha256 決定）
重新組裝。

關鍵：**只改形式，不改內容**。每列仍然描述它自己那 10 秒，所以這是
「窄化文本分布」的乾淨控制，而不是「換掉 caption 來源」。

槽位填充率（251,599 列）：

| slot | 填充率 | | slot | 填充率 |
|---|---:|---|---|---:|
| instruments | 92.0% | | rhythm | 39.1% |
| mood | 89.5% | | timbre | 36.4% |
| tempo | 88.2% | | dynamics | 34.2% |
| genre | 86.5% | | texture | 28.2% |
| production | 81.8% | | space | 15.2% |

空槽不丟句子，改用 4 選 1 的固定 fallback（`hard to characterise` 等），
否則長度會腰斬到 22 字。

**產物**：`/home/kojiek/eval_tsvs_p100/phase8_caption2p0_modular_template_train.tsv`
（251,599 rows，sha256 `d465b93fc9122a94ea0d27a25905da80f27c75be73d9b3bc34161c31082ec920`）

範例：

> The music is an amalgamation of acoustic and folk, taken at a slow tempo. Acoustic
> guitar, piano, bass guitar, drums and snare dominate the mix. The rhythm is steady,
> the dynamics soft, the texture neither sparse nor dense. The timbre is clean and the
> space is unremarkable. The overall mood is soothing and relaxed and the production is
> clear and balanced.

---

## 校準 gate：**未過**

目標是落在 fulltrack 附近（相對位置 1.0 = 剛好等於 fulltrack，容許帶 0.7–1.6），
避免重演 Phase 8 V4 `[consistency=X.XX]` prefix 那次 CLAP 崩 67% 的形態。

| 指標 | C2P0 | FULLTRACK | MODULAR | 相對位置 | |
|---|---:|---:|---:|---:|---|
| 平均字數 | 47.6 | 44.3 | 50.5 | — | ✅ 長度對了 |
| 唯一 caption | 100% | 14.56% | 99.99% | — | 內容保住了 |
| trigram 重複率 | 76.22% | 88.98% | 98.31% | 1.73× | ❌ 過頭 |
| trigram 熵 | 14.842 | 13.268 | 10.089 | 3.02× | ❌ 過頭 |
| 開頭 4-gram 變異度 | 0.1552 | 0.0301 | 0.1333 | 0.18× | ❌ 沒收窄 |
| 兩兩 Jaccard | 0.1852 | 0.2030 | 0.3265 | 7.97× | ❌ 大幅過頭 |

四輪校準（單一 frame → 8 frames → 加槽位 → 保留空句）都無法同時命中四個指標。
規則式改寫器打不到 fulltrack 的文本幾何：它靠的是 MIR 論述本身的長固定片語
（`with a tempo of X BPM`、`It employs a 4/4 time signature`），不是句型模板。

**過頭的方向**：模板連接詞在各列之間共用，Jaccard 因此衝到 0.33。
但與 P8 V4 不同的是，**語義內容有保留** —— 每列都指名實際的 genre / 樂器 / mood，
不是一個語義空白的常數前綴。所以崩潰風險比原始統計看起來低，但不是零。

---

## 要排的話需要什麼

| 項目 | 成本 |
|---|---|
| Text overlay 編碼（T5+CLAP，251,599 列） | **~76 GB** NVMe（比照 `text_overlays/slot2`）+ 數小時 GPU |
| Quarter 訓練 S1 100k + S2 50k | 共用 GPU 下約 12 小時以上 |
| Eval（MusicCaps cfg0 + novocal） | ~1 小時 |

NVMe 目前剩 315 GB，放得下但會吃掉四分之一。

**與現有 queue 的衝突**：`p2/pending/` 已排 `025_true_random_full` 與
`026_fake_random_full`（各 S1 400k + S2 200k，全尺度）。模板 arm 若排 `027`，
要等好幾天才會輪到。要更早跑就得插隊（`001`–`009`），那會延後 025/026。

**建議的預註冊門檻**（若決定跑）：
- 主要：CE 與 PQ 相對 `c2p0 slot0 quarter`（CE 6.1185 / PQ 6.5364）要進步
- 護欄：CLAP 不得低於 0.18（低於此視為進入 P8 V4 崩潰區，該路線退場）
- 早期停損：S1 100k 完成後先 eval，CLAP < 0.15 就中止，不要進 S2

---

## 誠實結論

語料備好了，但**我無法事先保證它會提升分數** —— 校準 gate 沒過，它比 fulltrack 更極端。
它測的是「大幅窄化文本分布」而不是「複製 fulltrack 的文本幾何」。
不論結果如何都有資訊量（升 → 方向對；CLAP 崩 → 找到上限），但它是一次探索，不是穩賺。
排不排、插不插隊，是 operator 的資源決定。
