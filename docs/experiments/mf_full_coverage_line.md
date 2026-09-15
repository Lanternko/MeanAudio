# Music Flamingo 全覆蓋線（mf_fullcov）

> 2026-09-06 建立。對應 queue job `038_mf_fullcov_quarter.sh` / `039_mf_fullcov_full.sh`，
> contract 在 `docs/experiments/mf_fullcov_{quarter,full}_contract.json`。

## 這條線要回答什麼

「Music Flamingo 在 c2p0 的完整 251,599-clip 規模上撐不撐得住？」

前面兩次都答不了：

| 先前實驗 | 為什麼答不了 |
|---|---|
| 036 / 037（`mfshort100k_direct_noq_c2p0recipe`） | 訓練語料是 100k 切片，caption 本身沒過 corpus audit（唯一率 73.17%、35.48% 的 row 與別的 clip 共用 caption、79.05% 在 T5 77-token 窗口被截斷）。它的落後無法歸因到 captioner。037 在 it 27,062 依 036 的 early-kill 規則手動停掉。 |
| paired59k（`paired59k_{mf,qwen}_noq_quarter`） | 乾淨的對照：audio latents、row、順序、recipe、budget 全固定，只動 caption 文字。但只涵蓋 59,614 clip 的交集，即語料的 23.7%。**覆蓋率**是這個對照吸收不掉的唯一不對稱。 |

paired59k 的結果（MusicCaps 5521 / MF25 / CFG 3.0 + fidelity negative）：

| arm | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| Qwen | **0.2294** | 6.6814 | 7.3440 | 4.8685 | **7.2381** |
| MF | 0.2221 | **6.8845** | 7.3303 | **5.1565** | 7.1394 |

差距只在 CLAP（+0.0073 ≈ 24× seed 底線），四項 AES 全落在 seed 雜訊內。

全覆蓋 recaption（`short_direct_v2` + `--enforce`，77-token 窗口、最多 5 次嘗試）把
covera­ge 這個不對稱移掉：語料 row 從 59,614 → 251,599，captioner、prompt preset、
enforcement、audio latents 全部不變。

**唯一沒控制住的**：對 036/037 而言，prompt preset 和覆蓋率是一起動的（v1 無 enforcement
→ v2 有）。對 paired59k 而言只有覆蓋率在動。

## 設計

- **不重新抽取任何 audio**。兩個 arm 都透過 c2p0 自己的 cache list 讀
  `/mnt/HDD/kojiek/phase8_qwen_official_matched_npz`，所以 audio 側與每一個 c2p0 arm
  逐位元相同。caption 從 `text_npz_dir` overlay 進來（`~/text_overlays/mf_fullcov`，約 76G）。
- **recipe 照抄 c2p0 launcher**：seed 14159265、batch 8、lr 1e-4、
  `lr_schedule_steps=[999999,999999]`、NoQ、no text attention mask。
  quarter = S1 100k / S2 50k；full = S1 400k / S2 200k。
- **`require_text_overlay=true`**（訓練 split）。這是 paired59k 做不到的：MF 100k 的
  audio NPZ 早於 `clip_id` 欄位。c2p0 的 audio NPZ 有 `clip_id`，而
  `build_mf_fullcov_arm_inputs.py` 送出的 id 是帶 slot 後綴的 c2p0 id，兩邊對得上，
  所以 loader 會在每個 batch 的每一 row 驗證 audio clip_id / overlay clip_id / TSV
  caption sha。這補上 `project_c2p0_corpus_provenance_2026_08_26.md` 記的守門缺口。
  - 為此改了 `meanaudio/data/data_setup.py`：`require_text_overlay` 改成先讀 per-dataset
    再讀 global。全域旗標會連 val split 一起打到，而 val 沒有 overlay，啟動就會 raise
    —— 這正是這個守門一直全關的原因。預設值沒變。

## id 正規化（會靜默出錯的地方）

不對稱：**只有 c2p0 側**要剝一層 `_<digits>`。

```
c2p0     00_1014400_segment_2_0   → 剝一次 → 00_1014400_segment_2
MF/recap 00_1014400_segment_2     → 不要動
```

MF id 結尾就是 segment 編號，一起正規化會把同一 track 的所有 segment 併成一個 key。
實測：誤正規化後 205,006 個 id 塌成 36,512 個，覆蓋率變 0 而且不會 raise。
見 memory `reference_c2p0_id_slot_suffix.md`。

## 判讀規則（launch 前登記）

CFG0 / MusicCaps 5521 / MF25 / NoMask / seed 42 / full precision / `--no_q`：

**quarter（038）**，與 0.80M samples 對齊的比較對象是 MF 100k quarter 0.1774
與 c2p0 slot0 quarter 0.2029：

| quarter CFG0 CLAP | 判讀 |
|---|---|
| ≥ 0.2029 | 全覆蓋 + enforced caption 直接把 gap 補平，MF 在規模上可用 |
| 0.1900 – 0.2029 | 覆蓋率買回大部分落後，full arm 值得那 ~19h |
| ≤ 0.1900 | 覆蓋率不是缺的那塊；039 自己中止 |

**full（039）**，比較對象是 c2p0 slot0 full 0.2149（CFG0）／0.2605（CFG 3.0 + neg）：

| full CFG0 CLAP | 判讀 |
|---|---|
| ≥ 0.2149 | 同覆蓋同預算下 MF 追平 Qwen，captioner 之爭倒向 MF |
| 0.2029 – 0.2149 | MF 落後幅度小於 quarter→full 的增益；寫成 captioner 的細微差異，不是語料失敗 |
| ≤ 0.2029 | 同覆蓋下仍落後；缺陷在 captioner 不在語料 |

### early-kill 是寫進 action 的，不是靠人

queue **沒有** dependency 機制（`lib_scheduler.py` 全檔沒有相關實作，036 contract 裡的
`ordering_dependencies` 只是文件），排序純粹是字典序。所以 `SCALE=full` 時
`mf_fullcov_action.sh` 的 Step 0b 會自己去讀 quarter 的 CFG0 report，`clap_score < 0.1900`
就 exit 5。要強制跑就 `touch ~/exps_nvme/mf_full_coverage/PROCEED_TO_FULL_ANYWAY`。

這正是 037 的失效模式：036 回報的數字依 036 自己的 contract 應該取消 037，但 037 還是被
自動 seat，最後在 it 27,062 被手動殺掉。

## 交接時機

Job 進 `p2/pending` 但不會馬上開跑：`probe_foreign()`（`lib_scheduler.py:99`）會把任何
非 queue 名下、佔用 > 3072 MiB 的 GPU process 當成 foreign 而 hold 住座位。目前擋著的是
手動的 recaption job（tmux `mf_recap_full`，pid 1188932，18.9 GiB）。它一退出，p2 host
下一輪就會 seat 038。

預估 GPU 空出時間：2026-09-07 06:00 前後（實測 6,720 clips/h，剩 ~48k clip）。

## 檔案

| 角色 | 路徑 |
|---|---|
| queue 進入點 | `gpu_queue/p2/pending/03{8,9}_mf_fullcov_{quarter,full}.sh` |
| contract | `docs/experiments/mf_fullcov_{quarter,full}_contract.json` |
| per-scale wrapper | `scripts/training_pipelines/mf_fullcov_{quarter,full}.sh` |
| 共用 action | `scripts/training_pipelines/mf_fullcov_action.sh` |
| arm inputs builder | `scripts/preprocess/build_mf_fullcov_arm_inputs.py` |
| recaption 產出 | `~/eval_output/mf_recaption_full_coverage/caption.jsonl` |
| 訓練 TSV（runtime 產生） | `~/exps_nvme/mf_full_coverage/arm_inputs/mf_fullcov_train.tsv` |
| text overlay | `~/text_overlays/mf_fullcov` |

---

## 結果（2026-09-07）

### 038 quarter — 完成，rc=0

| 指標 | CFG0 canonical | CFG3 + fidelity neg |
|---|---|---|
| CLAP | **0.1892** | 0.2172 |
| CE | 6.2116 | 6.9567 |
| CU | 6.7810 | 7.5810 |
| PC | 5.0496 | 4.7247 |
| PQ | 6.5981 | 7.4771 |

5,521/5,521 生成、16 kHz mono、report `status: passed`。
S1 100k = 4h13m、S2 50k = 2h12m，全程約 7h20m。

**語料 deviation**：本 arm 的 caption unique rate 是 0.8916，低於 CLAUDE.md checklist
item 2 的 0.90。2026-09-07 由 Lanternko 授權把 gate 放寬到 0.89 才得以啟動，理由與範圍
記在兩份 contract 的 `deviations` 欄。**任何引用這些數字的地方都必須帶上這個 caveat。**

### 039 full — 依 pre-registered gate 自我中止

```
[FAIL] quarter CFG0 0.1892 < 0.1900; the coverage hypothesis failed at quarter budget
```

Gate 差距 **0.0008**。對照 `reference_inference_seed_noise_floor.md`，全量 n=5521 的推論
seed CLAP 雜訊底線是 0.0003，所以這個差距約 2.7× 推論雜訊 —— 但訓練 seed 的底線更大，
單就這一點無法宣稱 0.1892 與 0.1900 有實質差別。**判讀上這是「打在門檻上」，不是「明確
失敗」。**

### coverage 假說的實際收益

| arm（皆 quarter，0.80M samples，CFG0/MF25/n=5521） | CLAP |
|---|---|
| MF 100k-slice quarter | 0.1774 |
| **MF full-coverage quarter** | **0.1892** |
| c2p0 slot0 quarter | 0.2029 |

全覆蓋 recaption 把 MF 對 c2p0 的 0.0255 缺口補回 **0.0118（46%）**，方向明確為正，但
沒有補完。可以寫成 observation：coverage 是 MF 劣勢的一部分成因，不是全部。剩下的
0.0137 仍未歸因。

### CFG 3.0 + fidelity negative 的 budget-matched 對照（2026-09-07 補跑）

038 的 CFG3 cell 原本沒有 c2p0 對照。補跑 `phase8_qwen_caption10s_multisent_noq_quarter`
（= slot0 quarter，CFG0 0.2029 那一個）於同一格：MusicCaps 5521 / MF25 / CFG 3.0 /
`low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi` /
NoMask / seed 42 / full precision / `--no_q`，flags 逐字取自 `mf_fullcov_action.sh` Step 6。
5,521/5,521 生成，產出在 `~/eval_output_nvme/c2p0_slot0_quarter_mc_mf25_cfg3_neg/`。

| arm（CFG 3.0 + neg，n=5521，MF25） | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| c2p0 slot0 quarter | **0.2248** | 6.6952 | 7.3871 | 4.6661 | 7.3101 |
| MF full-coverage quarter (038) | 0.2172 | **6.9567** | **7.5810** | **4.7247** | **7.4771** |
| paired59k qwen（非 budget-matched 到 slot0） | 0.2294 | 6.6814 | 7.3440 | 4.8685 | 7.2381 |
| paired59k mf | 0.2221 | 6.8845 | 7.3303 | 5.1565 | 7.1394 |

**CLAP gap 在這一格縮到 0.0076**（CFG0 是 0.0137），而 038 的四項 AES 全部高於 slot0
——CE +0.2615、CU +0.1939、PQ +0.1670，PC +0.0586。對照
`reference_training_seed_pq_noise_floor.md`，CFG3+neg 協定下的訓練 seed 底線比 CFG0 大
2–3 倍，因此 PC 這 0.0586 不該解讀；CE/CU/PQ 的量級則超過已知底線。

判讀（observation 層）：**MF 全覆蓋語料在 CFG3+neg 下換到的是 AES 全面小幅領先，代價是
CLAP 略低**。這與 paired59k 的形狀一致（Qwen 贏 CLAP、MF 贏 CE/PC），只是這次 MF 連 CU/PQ
也贏。不能據此宣稱 captioner 優劣反轉——CFG0 canonical 那格仍是 slot0 領先 0.0137，而
039 的 gate 讀的是 CFG0，判定不變。0.8916 unique rate 的 deviation caveat 同樣適用於本表
的 038 那一列。

---

## 重複 caption 修復線（mf_dedup，2026-09-07 開）

quarter 的 0.1892 帶著兩件糾纏的事實：打在 gate 上（差 0.0008），以及語料
unique rate 只有 0.8916。13.9% 的 row 與別的 clip 共用完全相同的 caption，
本身就是被稀釋的 conditioning 訊號。這條線把兩者拆開。

### 修的是什麼

7,589 個重複字串涵蓋 34,853 row；每組留一個代表，其餘 **27,264 個 clip 重新 caption**。

關鍵是不能原封不動重跑：那 34,853 row 是 `enforced_caption()` **已經用完** 5 次
per-clip 重採樣之後的殘留，byte-identical 的重跑會逐字重現同一份語料。唯一會改變
結果的槓桿是更長更熱的採樣梯：`--max-attempts` 5 → 8，溫度上限 1.5 → 2.1
（ladder 是 `0.7 + 0.2*attempt`，attempt 0 仍是 greedy）。

**3,287 筆重生後吐回一模一樣的字串**（幾乎正好等於 `enforced_ok=false` 的那批）——
即使溫度上到 2.1，captioner 在那些 generic EDM/rock clip 上仍只有一種說法。
Lanternko 2026-09-07 決定就留著，不再加碼。最終語料因此是「重複從 13.9% 壓到約 2%」，
不是零重複；引用 dedup arm 數字時要這樣寫。

out_dir 用 224,335 筆已經唯一的 caption 種好，所以 `--resume` 會（a）不碰那些 row、
（b）把 uniqueness 集合種到全語料，新 caption 必須對整份語料唯一，不只對彼此唯一。
captioner 呼叫的其他每一項都逐字相同（short_direct_v2、window 77、
max-new-tokens 80、batch 16、seed 4242），語料在 prompt 與截斷政策上維持同質。

**沒控制住的**：那 27,264 row 來自比周圍 224,335 row 更熱的 sampler。這個
inhomogeneity 只落在原本是完全重複的 row 上，但它是真的，contract 有記。

### 訓練 arm

`040_mf_dedup_quarter.sh` → `mf_dedup_noq_quarter`，contract
`docs/experiments/mf_dedup_quarter_contract.json`。recipe、audio latents、row 順序、
budget 與 `mf_fullcov` 逐項相同，只有 caption 文字動。**audit gate 回到 CLAUDE.md
原本的 0.90，不帶例外** —— 這個語料存在的目的就是把 0.89 deviation 移掉。

判讀（launch 前登記）：

| dedup quarter CFG0 CLAP | 判讀 |
|---|---|
| > 0.1892 | 重複確實在吃 conditioning 訊號；這份語料成為 MF 的 corpus of record |
| ≤ 0.1892 | 重複不是綁住效能的那條；殘餘缺口屬於 captioner |

margin 小於約 0.003 要當平手寫，不能當贏（推論 seed 底線 0.0003，訓練 seed 底線更大
且此協定下未量）。

### full 的歸屬（2026-09-07 更新：絕對 gate 已取消）

Lanternko 2026-09-07 明確取消 039 那個 pre-registered 的絕對 gate（quarter CFG0
CLAP ≥ 0.1900），並授權不必再問：**誰 CLAP + AES 好就跑誰**。

理由站得住：0.1900 那道 gate 問的是「MF 到底可不可用」，quarter 已經回答了
（0.1892，差 0.0008，落在訓練 seed 合理擺幅內）。現在的問題是兩個 MF 語料哪一個值得
那 ~19h —— 那是比較，不是門檻。

規則寫在 `scripts/eval/decide_mf_full_arm.py`，在任何一邊的數字進來之前就定稿：

- **兩個 eval cell 都算**（CFG0 canonical 與 CFG3+neg）。MF vs Qwen 時這兩格結論相反，
  沒有理由假設這裡會一致。5 指標 × 2 cell = 10 個比較。
- 每個比較**只有在 margin 超過該 cell 實測訓練 seed 底線的 2 倍時才計分**，否則記平手。
  底線是協定專屬的（CFG3+neg 把 AES seed 雜訊放大 2–3 倍，CLAP 反而縮小），不可互換。
- 勝者 = 計分勝場多的一方；平手時比總效果量（以底線為單位，單一指標上限 10 倍，
  免得 CFG3+neg CLAP 那個 0.0003 的底線一項獨大）；完全平手歸 `mf_dedup`，
  因為它不帶 audit deviation —— 這是聲明的偏好，不是量測，報告時照此寫。

兩個 full arm 都已排進 queue（`041_mf_fullcov_full.sh`、`042_mf_dedup_full.sh`），
各自 assert「我是贏家」。先被 seat 的那個若不是贏家就 exit 5 讓位，另一個接手。
不需要人介入。

**報告義務**：任何 full arm 的結果都必須註明原本的 0.1900 gate 是在 2026-09-07 經授權
取消的，且 arm 是比較選出來的，不是通過門檻進來的。兩份 full contract 的 `deviations`
都記了這一條。

### 交接

recaption 在 tmux `mf_dedup_recap`（`~/logs/mf_dedup_recaption.launch.sh`），
27,264 clip、實測約 1,900 clips/h、ETA ~14h。040 已在 `p2/pending`，
`probe_foreign()` 會因 captioner 佔 20 GiB 而 hold 住座位，captioner 一退出就接手——
與 038 同一個機制。若 captioner 中途死掉而 jsonl 不完整，
`build_mf_fullcov_arm_inputs.py` 會在 Step 1 直接 raise（缺 caption 的 clip 會被點名），
不會拿半份語料開訓。

---

## 結果：mf_dedup full 完成（2026-09-09 08:18）

`042_mf_dedup_full` rc=0 → `done`（harn 的 CFG0 completion evidence 通過，5,521/5,521
audio、unique_ids 5,521、16 kHz mono）。

**語料組成（先講清楚，`dedup` 這名字會誤導）**：mf_dedup **不是**丟掉重複 row，而是把
27,264 個「5 次抽樣都只吐出看過的 caption」的 clip **重新 caption**。結果仍是
**251,599 rows、unique_ids 251,599**，與 c2p0 逐 row 對齊；唯一率從 89.16% 拉到
**98.45%**（247,687 unique，尚有 5,948 rows 共用、3,913 rows 仍 best-effort）。
所以 coverage confound 在 full budget 上是**關掉的**，兩邊 row-matched。

### CFG0 canonical（MusicCaps 5521 / MF25 / NoMask / seed 42 / full precision / --no_q）

預算全部對齊（S1 400k / S2 200k），語料全部 251,599 rows：

| arm | CLAP | CE | CU | PC | PQ |
|---|---|---:|---:|---:|---:|
| **MF `mf_dedup_noq_full`** | **0.2078** | 6.2509 | 6.7922 | 4.9649 | 6.5752 |
| Qwen `c2p0_slot3_noq_full` | 0.2194 | 6.1976 | 6.7172 | 5.0576 | 6.5189 |
| Qwen `c2p0_k3_true_random_noq_full` | 0.2221 | 6.3893 | 6.8719 | 5.1883 | 6.6513 |
| Qwen `caption10s_multisent_noq_full` (seed 27182818) | 0.2191 | 6.1527 | 6.6700 | 5.0839 | 6.5270 |

CFG3+neg：MF `mf_dedup_noq_full` CLAP **0.2420** / CE 6.7534 / CU 7.3490 / PC 4.8045
/ PQ 7.2140。

### 判讀

CFG0 CLAP 的訓練 seed 底線是 0.0042，2× = **0.0084**。MF 對三個 Qwen full arm 的 CLAP
差距是 **0.0113–0.0143**，全部越過門檻 → **實質落後**。四項 AES 的差距則幾乎全部落在
各自 2× 底線內（唯一例外是 CU 對 multisent 的 +0.122，勉強越線，方向對 MF 有利）。

**這與 paired59k 在 23.7% 覆蓋率下的簽名完全相同**：captioner 差異只出現在 CLAP，
AES 進不了雜訊之上。全覆蓋沒有改變結論，只是把 CLAP 差距從 +0.0073 擴到 ~0.012
（arm 不完全可比，別當成精確的放大倍數）。

對照本線 launch 前登記的 band（comparator = c2p0 slot0 full 0.2149）：0.2078 落在
**0.2029–0.2149 中段** → 「MF 落後幅度小於 quarter→full 的增益（mf_dedup 自己
0.1865→0.2078 = +0.0213），寫成 captioner 的細微差異，不是語料失敗」。換成上表三個
可驗證的 Qwen arm（差距 0.0113–0.0143）這個判讀依然成立。

### 兩個 cell 不一致

CFG0 差距 0.0071（對 slot0 引用值）≈ 1.7× 底線，**低於** 2× 門檻；CFG3+neg 差距
0.0185 對上 0.0003 的底線 = 62× 底線。同一組 checkpoint，兩個 cell 對「差距有多確定」
給出很不一樣的答案 —— 與 `decide_mf_full_arm.py` docstring 記的「兩個 cell 對 MF vs
Qwen 曾經不一致」一致。報告時兩個 cell 都要列，不能只挑一個。

### 報告義務（沿用上節）

原本 0.1900 的絕對 gate 於 2026-09-07 經授權取消；mf_dedup 是**比較選出來**的 arm，
不是通過門檻進來的。另外 c2p0 slot0 full 的 0.2149 / 0.2605 是引用既有文件，
`cfg0_eval_runtime/reports/` 裡沒有 slot0 full 的報告可即時複核；上表三個 Qwen arm
的數字才是本次直接讀檔驗證過的。
