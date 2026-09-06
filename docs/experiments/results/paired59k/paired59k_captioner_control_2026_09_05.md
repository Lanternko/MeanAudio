# paired59k captioner-only control (quarter)

**狀態**：✅ 兩臂完成（2026-09-05 18:37）

## 為什麼要跑

036/037 拿 Music Flamingo 語料對比 c2p0 Qwen 語料時，**同時動了三個變因**：
captioner、語料筆數（100,000 vs 251,599）、以及 clip 集合（只重疊 59.6%）。
036 contract 自己的 `not_controlled` 欄位就承認了前兩項。

本實驗跑兩份語料的**交集**：相同 audio latent、相同 59,614 筆、相同順序、
相同 recipe 與預算，**唯一不同的是 caption 文字**。

## 兩臂

| Arm | Caption 來源 | Text overlay |
|---|---|---|
| `paired59k_mf_noq_quarter` | Music Flamingo `short_direct_v2` 重新 caption（2026-09-05 驗收通過） | `text_overlays/paired59k_mf_recaption`（1-cap，本次新編） |
| `paired59k_qwen_noq_quarter` | c2p0 slot0 | `arm_inputs/qwen_text_overlay` → symlink 到 `text_overlays/true_random` **slot 0**，`cap_index_fixed=0` |

訓練 log 已確認 Qwen 臂實際載入 `cap_index_fixed: 0` ＋ symlink farm overlay ＋ paired Qwen TSV。

預算：quarter（S1 100k / S2 50k），batch 8、lr 1e-4、seed 14159265、NoQ、NoMask。
Eval：MusicCaps 5521 / MeanFlow 25 / **CFG 3.0 + fidelity negative** / seed 42 / `--no_q`
（與 slot0-vs-fulltrack 表同協定，**不可**與 canonical CFG 0 表比較）。

## 為什麼幾乎不用新資源

- **Audio 不重抽**：沿用 `mfshort100k_direct_noq_c2p0recipe_npz`（35 G），
  以 `arm_inputs/cache_train.txt` 明確綁定；`paired59k_mf_npz_row_index.txt` → MF 100k TSV id
  **59,614/59,614 零錯配**、索引唯一。
- **Qwen 臂零編碼、零新增磁碟**：抽 200 筆驗證，paired Qwen caption 的 sha256
  **200/200 命中 true_random 的 slot 0**。因為 `ExtractedAudio.__getitem__` 對 audio 與
  text overlay **共用同一份檔名列表**，所以用 symlink farm 把 overlay 改名到 audio cache
  的命名空間，就能讀到與 c2p0 arms **位元相同**的特徵。
- MF 臂只新編 18 G 的 1-cap overlay（3 分鐘）。
  **兩臂 `text_encoder_fingerprint` 同為 `27e88fac…`**，captioner-only 的宣稱才站得住。

## ⚠️ 已知限制：`require_text_overlay` 開不起來

MF 的 audio NPZ 沒有 `clip_id` 欄位（早於該欄位），Qwen overlay 的 `clip_id` 又保留
Qwen 的 `_<slot>` 後綴，兩邊都過不了 loader 自檢。改以離線 audit 取代，記在
`arm_inputs/bindings.json`，並在 action script Step 1 每次啟動前重跑 300 筆
（兩臂皆 300/300 通過）。**任何輸入變動都必須重跑這個 audit。**

## 上游：MF recaption 驗收（2026-09-05 08:39）

| 項目 | 結果 |
|---|---|
| Rows | 59,614 / 59,614 |
| **R2：77-token 超窗** | **0 筆**（原語料 79% 被截斷）|
| 完整句結尾 | 100.00% |
| **R1：唯一率** | **92.22%**（原語料 73%）|
| 重複 | 4,638 筆（7.78%），最大群 105 clips 共用一條 |
| attempts | mean 3.056（probe n=82 是 2.76）|
| error / restart | 0 / 0，總耗時 8h50m |

報告：`~/logs/mf_recaption_paired59k_v2_acceptance.json`

> ⚠️ 訓練時那 4,638 個 clip 會有完全相同的 text embedding。這是 greedy decoding 在同質
> EDM/trance/rock clip 上的必然結果，不是 bug。

## ⚠️ 兩臂的 caption 性質**沒有**對齊（2026-09-05 08:50 實測，同一把尺）

| 語料（同 59,614 clips）| 唯一率 | 最大重複群 | T5 token 平均 | p95 | **超 77 token** |
|---|---|---|---|---|---|
| MF recaption（本次新編）| 92.22% | 105 | 62.7 | 76 | **0.00%** |
| Qwen c2p0 slot0 | **100.00%** | 1 | 75.4 | 112 | **44.92%** |
| MF short_direct（舊，已被取代）| 77.44% | 172 | 103.9 | 142 | 78.92% |

**這代表 "captioner-only" 的宣稱要打折**：recaption 把 MF 的截斷修掉了，但 Qwen 臂沒有
同等處理，仍有 **44.92% 的 caption 在 77-token 窗口被截**。兩個殘餘差異方向相反 ——
Qwen 唯一率較高（利），截斷率高很多（不利）——所以任一方向的結果都不能單純歸因給 captioner。

緩解證據：memory `reference_caption_corpus_t5_truncation.md` 量過 c2p0 的截斷是
**局部損失**（p50 CLAP cos 1.0000），不像 MF/LP-MC 是全域偏移；所以 44.92% 的
語義代價可能遠小於比例本身的暗示。但這是**既有量測，不是本實驗的控制**。

**允許寫的**：本對照已鎖住 audio、筆數、順序、recipe、預算（036/037 三者皆未控制）。
**不允許寫的**：把差異單一歸因為 captioner。要拿掉這個 confound，需再跑一臂
「Qwen caption 也 enforce 到 77 token」的版本。

（原本標記的「Qwen 唯一率未測」缺口已於此關閉。）

## 檔案

| 用途 | 路徑 |
|---|---|
| 驗收閘 | `scripts/preprocess/accept_mf_recaption_paired59k.py` |
| 輸入組裝 + audit | `scripts/preprocess/build_paired59k_arm_inputs.py` |
| 1-cap overlay 編碼 | `scripts/preprocess/build_single_cap_text_overlay.py` |
| 兩臂 action | `scripts/training_pipelines/paired59k_captioner_control_quarter_action.sh` |
| 啟動 + 續跑 | `~/logs/paired59k_control.launch.sh`（log `~/logs/paired59k_control.log`）|


---

## 結果

### MF 臂（2026-09-05 13:42 完成）

協定：MusicCaps 5521 / MeanFlow 25 / CFG 3.0 + fidelity negative / NoMask / seed 42 / `--no_q`

| Arm | rows | Caption | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|---|---|
| `mfshort100k_direct_noq_c2p0recipe_quarter` | 100,000 | short_direct（79% 截斷、77.4% 唯一）| 0.2079 | 6.0727 | 6.9292 | 4.6104 | 6.7557 |
| **`paired59k_mf_noq_quarter`** | **59,614** | **short_direct_v2 enforced（0% 截斷、92.2% 唯一）** | **0.2221** | **6.8845** | **7.3303** | **5.1565** | **7.1394** |
| Δ | −40% 資料 | | **+0.0142** | **+0.812** | **+0.401** | **+0.546** | **+0.384** |

**五個指標全部上升，而且是在少 40% 訓練資料的情況下。** 筆數減少應該要傷害結果卻沒有，
所以 caption 改寫的效果**至少**有觀察到的這麼大。

⚠️ **兩個變因同時動了**（caption 品質 ↑、rows ↓），方向相反，所以這是「效果下界」的讀法，
不是乾淨的單變因量測。要精確歸因需要 100k 全量的 enforced 版本。

⚠️ **雜訊底線未在同協定量過**：CE +0.812 / PQ +0.384 約為推論 seed 底線（0.296 / 0.142）的
2.7×，但**訓練 seed 底線比推論底線大**且隨協定變動（見 `reference_training_seed_pq_noise_floor.md`，
門檻應設 2×）。CLAP +0.0142 遠大於推論底線 0.0003，但同樣缺訓練 seed 對照。
**定案前需補 seed 對照臂。**

音檔 sanity：5,521/5,521、0 零位元組、id 全唯一、16 kHz / 9.98 s、crest 4.98–10.57、0% clipping。
S1+S2 全程 0 次 `loss:nan` / 0 次 `grad_norm:nan`。

### 兩臂對照（2026-09-05 18:37，本實驗的主結果）

同 audio、同 59,614 筆、同順序、同 recipe、同 quarter 預算、同 eval 協定，只有 caption 不同。
顯著性用**同協定實測**的訓練 seed 底線判定（CFG 3.0 + fidelity negative，
見 `reference_training_seed_pq_noise_floor.md`；門檻依該 memory 設 **2×**）。

| 指標 | MF recaption | Qwen c2p0 slot0 | Δ (Qwen−MF) | seed 底線 | ×底線 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| **CLAP** | 0.2221 | **0.2294** | **+0.0073** | 0.0003 | **24.3×** | ✅ **顯著** |
| CE | **6.8845** | 6.6814 | −0.2031 | 0.2960 | 0.69× | ❌ null |
| CU | 7.3303 | **7.3440** | +0.0137 | 0.1053 | 0.13× | ❌ null |
| PC | **5.1565** | 4.8685 | −0.2880 | 0.1884 | 1.53× | ⚠️ 未達 2×，當 null |
| PQ | 7.1394 | **7.2381** | +0.0987 | 0.1416 | 0.70× | ❌ null |

**可以寫的**：在鎖住 audio／筆數／順序／recipe／預算之後，兩個 captioner 的差異
**只出現在 CLAP（text-audio 對齊），Qwen 語料高 +0.0073**；四項 AES 美學指標
（CE／CU／PC／PQ）**全部落在 seed 雜訊內，無法區分**。

**不可以寫的**：
- ❌「Qwen 音質較好」——  PQ +0.0987 只有底線的 0.70×，是雜訊。
- ❌「MF 的 CE／PC 較好」—— 0.69× 與 1.53×，都沒到 2× 門檻。
- ❌ 把 CLAP 差異單一歸因給 captioner —— **caption 性質仍未對齊**（見上節）：Qwen 唯一率 100%
  但 44.92% 截斷，MF 92.22% 唯一但 0% 截斷。Qwen 是**在截斷劣勢下**仍拿到較高 CLAP。
- ⚠️ 只有兩顆 seed 的底線給的是一個差值不是分布；主張小效果需多 seed（同 memory 規則 3）。

### 對照組：recaption 本身的效果

MF 臂 vs `mfshort100k_direct_noq_c2p0recipe_quarter`（同 recipe 同預算，舊 caption、100k rows）：
CLAP +0.0142 = **47× 底線**，顯著；但 caption 品質與 rows 兩變因同動方向相反，只能讀成效果下界。

### 執行品質

兩臂 S1+S2 全程 **0 次 `loss:nan` / 0 次 `grad_norm:nan`**；兩臂各 5,521/5,521 音檔、
0 零位元組、id 全唯一、16 kHz / 9.98 s、0% clipping、crest 5.0–10.6（遠離飽和警戒線）。

### 下一步（未執行，等指示）

1. **多 seed 底線**：目前底線來自另一個 arm 的兩顆 seed；要把 CLAP +0.0073 寫成結論，
   最好在本 arm 上跑第二顆訓練 seed。
2. **拆掉截斷 confound**：跑一臂「Qwen caption 也 enforce 到 77 token」。
3. **拆掉 rows confound**：100k 全量的 enforced MF 版本。
