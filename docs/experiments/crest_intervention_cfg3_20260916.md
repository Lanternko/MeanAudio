# 061 預註冊：固定 LUFS 下的 crest 介入（MusicCaps 5521, MF25, CFG3, fidelity8）

2026-09-16 設計。結果寫在 `docs/experiments/results/crest_intervention_cfg3_20260916_results.md`，
**本檔被 061 contract 以 sha256 釘住，定案後不要改**。

## 問題

051（`loudness_aes_cfg3_20260911`）發現 crest 是 AES PQ 最強的關聯量：crest Q5−Q1 的 PQ 差
**+0.4396** [+0.3692, +0.5070]，且在五個 LUFS 組內分層後仍全正、CI 全不跨零，所以不只是響度代理。
但依 051 預註冊那是**描述性關聯，不宣稱因果**。

本實驗問：**把 crest 移動、同時把整合響度鎖住，PQ 會不會跟著動？**

決策相關性：如果會動，crest 就是可刷的旋鈕，拿它當訓練目標等於直接優化 eval metric（Goodhart），
而且所有 arm 間 crest 有差的 PQ 比較都要回頭重讀；如果不會動，+0.4396 就是混淆，
crest 只是某個真實因素的代理，該去找那個因素。

## 關聯隱含的預測值（預註冊錨點）

從 051 的 `per_clip.csv`（gain 0 列）重算：

| 量 | 值 |
|---|---|
| crest Q5 − Q1 gap | 6.929 dB |
| 對應 PQ 差 | +0.4396 |
| **關聯隱含斜率** | **+0.0634 PQ / dB crest** |
| 同批 clip 的 OLS 斜率 | +0.0525 PQ / dB |

介入若為因果，配對斜率應接近 +0.0634/dB。

## 材料

**不生成任何音訊。** 直接重用 051 保留的 5,521 個 baseline FLAC
（`.../loudness_aes_cfg3_20260911/_audio/baseline/`），以其 `audio_manifest.json` 的 sha256 逐檔驗證。
Checkpoint、solver、steps、cfg、negative prompt、generation seed 全部繼承 051，因此與 051 的數字直接可比。

## 變換

**只用平滑時變增益**：attack/release 包絡追蹤器 → 單極平滑的 dB 曲線 → 逐樣本相乘。
沒有逐樣本非線性，所以不引入諧波失真；每首逐檔記錄頻譜質心比值當稽核共變量。

- clamp ±12 dB、**增益調變上限 `gain_mod_max_db = 6.0`**（RMS）
- ratio 用 24 點 **grid search**，不是 bisection
- 響度正規化**迭代**至殘差 ≤ 0.01 LU（gated loudness 對純量不是嚴格等變）
- 墊底 `pad_db = -6`；只有最大峰值會破 `-0.1 dBFS` 天花板的片段，**整首所有 arm 一起**額外衰減
  （`pad_db_effective`）。pad 在片段內是常數，所以在配對對比中相消。

### 為什麼不是「打中 +3 dB」

試跑推翻了原始設計，兩點都寫進 contract：

1. **crest 對 ratio 非單調。** 兩個方向的重度處理都會把 peak/RMS 推回去，bisection 因此會靜默地
   回傳一個把 crest 移往**反方向**的邊界值——原始 `down` arm 有 4/6 首實際 achieved 是 **+3.8～+7.6 dB**。
   改成 grid search 並以「達成值最小 / 最接近目標」為目標函數。
2. **可達範圍是片段自身的性質。** 強迫命中 +3 dB 會把增益調變推到 17–21 dB RMS、質心壓到 0.82，
   而那些片段的 crest 仍然不動（baseline crest 已高者，峰值是孤立樣本，擴張時 RMS 同步上升）。
   那量到的會是「處理強度」不是 crest。所以改為**限制處理強度、達成多少算多少**，
   分析一律用 **achieved** crest delta。

## Arms（5 × 5,521 = 27,605 筆 AES）

| arm | 內容 |
|---|---|
| `ref` | 純純量增益到 base LUFS + pad。所有配對對比的參考 |
| `up` | 跟隨結構的擴張，目標 +3 dB crest（目標不是宣稱，achieved 才是劑量） |
| `upmax` | 上限內可達的最大 crest 增量，供劑量散佈 |
| `down` | 上限內可達的最大 crest 減量（以達成值選，因為映射非單調） |
| `rand` | `up` 的增益包絡循環位移。調變深度相同但與音樂結構去相關 |

`rand` **不是零效果安慰劑**——慢速隨機調幅本身就會提高 crest，這正是重點：
它把「crest 這個數字」和「跟隨結構的擴張」分開。若 PQ 不管調變跟不跟隨結構都跟著 crest 走，
crest 就是 lever；若只有跟隨結構的那支會漲，那是 transient 結構不是 crest。

## Primary endpoint 與判讀規則

**Primary**：所有介入 arm 對 `ref` 的**配對劑量反應斜率 dPQ / dB achieved crest delta**（過原點），
bootstrap seed 20260916、10,000 次、pointwise 95% CI，over clips。

| 結果 | 判讀 |
|---|---|
| CI 不跨零、為正、且 ≥ 0.5 × 0.0634 | **crest 是 PQ 的因果 lever** → 不可當訓練目標；所有 arm 間 crest 有差的 PQ 比較須重讀 |
| CI 不跨零但遠低於關聯斜率 | partial，關聯多半混淆 |
| CI 跨零 | **關聯未被介入重現** → crest 是代理，不訓練它，去找背後因素 |

**Secondary**：各 arm 的配對 AES delta 與各自斜率（CE/CU/PC）；`up` vs `rand` 在相同 achieved delta 下的比較；
只取 `gain_mod_rms_db ≤ 3.0 dB` 片段的穩健性斜率。

## 排除規則

片段層級排除（整首所有 arm 一起），兩個來源都要報數：

- **前處理篩選**：baseline LUFS < −40 或靜音比例 > 0.40
- **未能守住**：任一 arm 無法鎖在目標響度，或破峰值天花板

近乎靜音的片段無法被鎖在固定的 gated loudness——擴張會把它們佔多數的安靜部分推到 gate 以下，
目標因此不可達。合格列數低於 5,400 則 postflight 失敗。

## 限制（預先聲明）

單一 checkpoint、單一 generation seed。只做衰減與受限擴張，不可外推到放大或壓縮。
可達 crest 變化受片段結構限制且分布不均，因此這是**劑量反應**證據，不是固定劑量的隨機化對比。
CI 跨零是證據不足，不是等效（未設等效界值）。AES 不是人類品質判斷。
