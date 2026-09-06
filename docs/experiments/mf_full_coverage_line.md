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
