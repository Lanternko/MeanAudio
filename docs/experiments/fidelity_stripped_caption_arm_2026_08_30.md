# 保真度用語剝除 arm — 語料與校準 gate

日期：2026-08-30　狀態：**語料已建、gate 通過，等 GPU 與磁碟空間**

這是針對 canonical CFG 0 協定下「非 fulltrack arm 達到 PQ 6.9」目標，
**機制對準度最高**的一個 arm。

---

## 為什麼是這個介入

`results/phase8/fulltrack_pq_gap_mechanism_2026_08_28.md` 的逐檔拆解顯示：

- c2p0 slot0 最差 10% 的 clip 有 **47.1%** 是低保真提示詞（最好 10% 只有 27.2%）
- MusicCaps 有 **37%** 的提示詞含低保真語言
- c2p0 訓練語料有 **82.8%** 的 caption 提到 quality，且幾乎都是正面的
- fulltrack（canonical 下 AES 較高）只有 **7.3%**

推論：c2p0 學到了一條銳利的保真度軸，於是忠實地跟著低保真提示詞產生劣化音訊；
fulltrack 幾乎沒看過保真度語言，所以大致忽略那些詞，輸出停在它的預設水準。

**若把保真度語言從訓練 caption 拿掉，模型就不會學到那條軸。**
這比模板化 arm 更直接：它只動一個已被指認的變因，不改變其他表面形式。

---

## 做法

`scripts/preprocess/build_fidelity_stripped_captions.py`：

- **整句刪除**：只有當一個句子談的**只有**保真度時才刪（同時提到樂器／情緒／編制的句子保留並就地編輯），
  所以樂曲內容不會被連帶波及。
- **就地移除**：保真度形容詞與承載它們的片語（`high-quality`、`well-mixed`、
  `professionally recorded`、`clear mix`、`amateur`…）。
- **保底**：若剝除後 caption 少於 5 個字，退回原句（13,712 列，5.4%）。

**產物**：`/home/kojiek/eval_tsvs_p100/phase8_caption2p0_fidstrip_train.tsv`
（251,599 rows，sha256 `1f4d920126f75701022ddfb0924fb28613bde9d9fb0145fdd0872cfbdeb8e436`）
83.3% 的列被修改，平均字數 52.1 → 41.3。

範例：

> **前**：The audio features a soft, ambient track with guitar strumming as the primary
> instrument. The music is slow-paced and has a dreamy, melancholic mood.
> *The production quality is high, with clear mixes and good recording fidelity.*
>
> **後**：The audio features a soft, ambient track with guitar strumming as the primary
> instrument. The music is slow-paced and has a dreamy, melancholic mood.

---

## 校準 gate：**通過**

| 語料 | 字數 | "quality" | 高保真用語 | 低保真用語 | 樂器 | 情緒 |
|---|---:|---:|---:|---:|---:|---:|
| FULLTRACK（目標） | 47.9 | **7.3%** | 5.8% | 6.6% | 93.0% | 84.4% |
| **FIDSTRIP（新）** | 41.3 | **10.2%** | 15.6% | 0.1% | **92.5%** | **83.7%** |
| C2P0 slot0（來源） | 52.1 | 82.8% | 76.8% | 1.6% | 92.5% | 83.7% |
| MusicCaps（eval） | 49.0 | 33.7% | 1.0% | 34.3% | 76.3% | 19.5% |

**要移動的變因移動到位**：quality 提及率 82.8% → 10.2%，落在 fulltrack 的 7.3% 附近。
**不該動的沒動**：樂器 92.5%、情緒 83.7% —— 與來源語料**完全相同**。

對照之下，模板化 arm（`modular_template_caption_arm_2026_08_28.md`）的 gate 沒過：
它在 trigram 重複率、熵、Jaccard 三項上都比 fulltrack 更極端。
**這個 arm 的設計乾淨得多。**

殘留：高保真用語仍有 15.6%（fulltrack 5.8%），因為 `clear` / `balanced` 也會用在
非保真度語境（`clear melody`、`balanced arrangement`），刻意不動。

---

## 尚未執行的前置作業

| 項目 | 成本 | 狀態 |
|---|---|---|
| Text overlay 編碼（T5+CLAP，251,599 列） | **~78 GB** NVMe + ~50 min GPU | ❌ 未做 |
| Quarter 訓練 S1 100k + S2 50k | 共用 GPU 下約 5 小時 | ❌ 未做 |
| Canonical cfg0 eval | ~40 min | ❌ 未做 |

**阻擋原因：磁碟。** NVMe 目前剩 **112 GB（97%）**，`/mnt/HDD` 剩 74 GB（100%）。
再寫一份 78 GB overlay 只會剩 34 GB，而模板 arm 正在訓練中，不該冒這個險。

**建議的排序**：等模板 arm（`001_modular_template_quarter`）跑完。
若它未通過自身門檻（CE/PQ 要贏 c2p0 slot0 quarter 的 6.1185 / 6.5364，CLAP 護欄 ≥ 0.18），
它的 78 GB overlay `/home/kojiek/text_overlays/modular_template` 即可刪除，
正好騰出這個 arm 需要的空間。

**建議的預註冊門檻**（與模板 arm 對齊）：
- 主要：CE 與 PQ 要贏 `c2p0 slot0 quarter`（CE 6.1185 / PQ 6.5364）
- 護欄：CLAP ≥ 0.18
- 早期停損：S1 100k 完成後先 eval，CLAP < 0.15 即中止

---

## 這個 arm 測什麼、不測什麼

**測**：訓練期保真度語言的存在與否，是否改變模型在低保真提示詞上的劣化行為，
進而改變 canonical CFG 0 下的 aggregate AES。

**不測**：它不能單獨證明 fulltrack 的優勢就是這個機制造成的 —— fulltrack 同時還差在
caption 粒度、文本分布寬度與訓練軌跡。這是一個**充分性**測試
（「拿掉保真度語言是否足以取得增益」），不是對 fulltrack 的因果歸因。

**預期**：若假說成立，CE/PQ 應上升而 CLAP 可能小幅下降（模型不再回應提示詞裡的保真度
資訊，那部分語意對齊會流失）。若 CLAP 掉超過 0.02，代表代價過高。
