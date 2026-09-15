# ATTM 合成 tag 三元組 prompt 重跑（2026-09-07）

延續 `attm_protocol_benchmark_2026_09_04.md`。該輪用器樂 MusicCaps 自然 caption 當
prompt；ATTM 實際上用的是**合成的 (genre, instrument, mood) 三元組**。這輪把 prompt
換成同一種構造方式重跑，補上當時列為未解的 domain-match confound。

## 協定

- **Prompt**：1,000 條合成三元組，`scripts/attm/build_tag_triplet_prompts.py`，seed 42。
  Tag 池取自器樂 MusicCaps aspect_list 中出現 ≥30 次者：genre 9 / instrument 17 /
  mood 27（共 4,131 種組合，抽 1,000 條相異）。均勻抽樣，模板固定為
  `{art} {mood} {genre} track featuring {instrument}.`（a/an 依母音修正）。
  產物：`/home/kojiek/eval_tsvs_p100/attm_tag_triplet_1000.tsv` + `attm/tag_triplets.json`
- **抽 1,000 不是 100**：ATTM 只有 100 條，n=100 的抽樣 sd 約 0.010–0.012，會蓋掉
  arm 間的真實差異。要看他們那個尺度的散布，從 per-clip 值重抽即可。
- **FAD 參考不變**：器樂 MusicCaps 2,382 條。這輪 prompt 分布（合成 tag）與參考分布
  （真實錄音）不再同源 —— 這其實**更接近**他們的設定（合成 prompt + Jamendo 參考）。
- **CCS**：`scripts/attm/ccs_triplet.py`。這是 Eq. 2 的原生形式 —— 1/3N 本來就假設每個
  clip 恰好 3 個目標概念，這次是 prompt 自己要求的那 3 個，不再從人工 aspect 湊。

## 主表

| arm | prompt | CLAP-ATTM | CLAP-ours | FAD ↓ | CE | PQ | CCS raw | CCS 校正 |
|---|---|---|---|---|---|---|---|---|
| ours cfg3+neg | caption | 0.3112 | 0.2806 | 0.2451 | 6.924 | 7.436 | 0.8833 | 0.836 |
| ours cfg3+neg | **triplet** | **0.3315** | 0.2946 | 0.3693 | 7.627 | 7.823 | 0.6947 | **0.699** |
| ours cfg0 | caption | 0.2801 | 0.2389 | 0.1993 | 5.943 | 6.487 | 0.8841 | 0.830 |
| ours cfg0 | **triplet** | 0.2890 | 0.2554 | 0.2589 | 6.818 | 6.848 | 0.7137 | 0.685 |
| MeanAudio-S-Full | caption | 0.1304 | 0.0951 | 0.4504 | 2.998 | 4.990 | 0.5630 | 0.224 |
| MeanAudio-S-Full | **triplet** | **0.0756** | 0.0678 | 0.4990 | 3.107 | 4.988 | 0.3813 | 0.284 |
| MeanAudio-L-Full | caption | 0.1149 | 0.0860 | 0.4887 | 2.882 | 4.999 | 0.4927 | 0.220 |
| MeanAudio-L-Full | **triplet** | **0.0448** | 0.0394 | 0.5676 | 2.833 | 5.058 | 0.3193 | 0.210 |

CCS 三種讀法（triplet）：raw = 全部 3,000 次判定；verifiable = 只算通過 ATTM criterion 2
（recall ≥ 0.85）的 tag，1,373 次；校正 = verifiable 再做 chance correction。

| arm | raw | verifiable | 校正 | genre | instrument | mood |
|---|---|---|---|---|---|---|
| ours cfg3+neg | 0.6947 | 0.8004 | **0.6990** | 0.665 | 0.650 | 0.769 |
| ours cfg0 | 0.7137 | 0.7859 | 0.6848 | 0.644 | 0.668 | 0.829 |
| MeanAudio-S-Full | 0.3813 | 0.4873 | 0.2838 | 0.284 | 0.320 | 0.540 |
| MeanAudio-L-Full | 0.3193 | 0.4057 | 0.2100 | 0.284 | 0.179 | 0.495 |

## 發現

### 1. 短 tag prompt 讓 topline 崩掉，卻讓我們變好
我們 CLAP +0.020，topline S **−0.055**、L **−0.070**（掉一半以上）。差距從 2.4× 拉到 4.4×。
已排除生成故障：抽 40 檔，10 s / 16 kHz、RMS 0.21、crest 6.2、零靜音。

### 2. ⚠️ 與 ATTM 論文對不上，topline 這欄不可外用
他們表上 MeanAudio-S-Full 在**他們的** tag prompt 上是 CLAP 0.210，高於我們用自然
caption 量到的 0.130；我們用自製三元組量到 0.076。同樣是「合成 tag prompt」方向相反。
→ 模板或 tag 詞彙與他們差異顯著。**這輪 topline 數字不得對他們的表**；我們自己 arm
之間的比較仍成立。

### 3. domain-match confound 有了方向性證據
所有 arm 的 FAD 都升（我們 +0.125 / +0.059，topline +0.049 / +0.079）。prompt 換成合成
tag、參考仍是真實錄音，分布就拉開 → **我們異常低的 FAD 確實有一部分來自 caption 與
參考同源**。但 cfg3+neg 的 0.369 仍低於他們表上最佳 0.417，沒有被完全解釋掉。

### 4. CCS 的絕對值降了，但 yes-bias 的結論反轉方向
raw CCS 從 ~0.88 降到 ~0.70：合成三元組要求的概念是**指定**的，不像人工 aspect 那樣
本來就描述該 clip，所以難度高得多，這是預期內的。
值得注意的是**校正後 topline 反而略升**（S 0.224→0.284、L 0.220→0.210 持平）：
09-04 那輪 topline 的高 raw CCS 幾乎全是 yes-bias，這輪 raw 本來就低，校正拿掉的比例
較小。差距從校正前的 1.8× 變 2.5×，仍然同向。

### 5. negprompt 在 ATTM 計分法下是明確負值（比先前更差）
| 指標 | caption 集 | triplet 集 |
|---|---|---|
| CLAP | +0.0311 好 | **+0.0425** 好 |
| FAD | +0.0458 差 | **+0.1104** 差 |
| CCS 校正 | ≈0 | +0.014（在雜訊內） |

Borda 等權下 FAD 的懲罰比先前更重 → **「negprompt 打 ATTM 榜」的淨效果由「約為零」
下修為明確負值**。negprompt 在我們自己 CLAP+AES 體系裡的價值不變。

### 6. n=100 尺度的散布（per-clip 重抽 20,000 次）
| arm | 全量 | sd | 95% 區間 |
|---|---|---|---|
| ours cfg3+neg | 0.3315 | 0.0099 | 0.3122 – 0.3510 |
| ours cfg0 | 0.2890 | 0.0106 | 0.2680 – 0.3097 |
| MeanAudio-S-Full | 0.0756 | 0.0120 | 0.0520 – 0.0994 |
| MeanAudio-L-Full | 0.0448 | 0.0106 | 0.0243 – 0.0659 |

即使只抽 100 條，我們的區間下緣仍遠高於 topline 上緣。**ATTM 只用 100 條 prompt 本身
就是噪聲相當大的榜**，對 topline 量級尤其如此。

## 未解

- **模板 / tag 詞彙與他們的差距未知**，且發現 2 顯示這個差距足以翻轉 topline 的方向。
  沒有他們的 prompt 就無法收斂。
- **FAD 參考換 held-out 器樂 Jamendo** 仍未做 —— 這輪只證明 domain match 有貢獻，
  沒有量出它的大小。
- **從未做過 MOS**。

## 產物

`/home/kojiek/nvme_experiment_artifacts/meanaudio/attm/`：`*_tri1k.json`（CLAP/FAD/AES +
per-clip）、`ccs_tri_*.json`、`tag_triplets.json`、`_audio/*_tri1k/`（4,000 clips）。
log：`~/logs/attm_triplet_2026_09_07.log`、`~/logs/attm_triplet_ccs_2026_09_07.log`。
