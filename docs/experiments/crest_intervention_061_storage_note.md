# 061 storage：需要空間時怎麼刪、怎麼重建

操作備忘，**不是**預註冊內容（設計檔 `crest_intervention_cfg3_20260916.md` 被 contract 釘住，不要改那份）。

## 現況

061 的 transform 產物在 `/home/kojiek/nvme_experiment_artifacts/meanaudio/crest_intervention_cfg3_20260916/`：

| 內容 | 大小 |
|---|---|
| `arms/{ref,up,upmax,rand,down}/*.wav` | 5 × 3.4 GB ≈ **17 GB** |
| `transform/*.json`（逐檔訊號統計與 hash） | ~30 MB |

061 排在 059、060 之後，所以這 17 GB 會停在 NVMe 上一段時間。NVMe 長期在 96–97%。

## 需要空間時

**兩個目錄一起刪，不能只刪 `arms/`：**

```bash
rm -rf /home/kojiek/nvme_experiment_artifacts/meanaudio/crest_intervention_cfg3_20260916/{arms,transform,transform_manifest.json}
```

只刪 `arms/` 會讓 061 靜默壞掉：transform 階段是看 `transform/<id>.json` 在不在決定要不要跳過，
不檢查音檔。留著 record、刪掉 WAV → transform 整個跳過 → score 階段在缺檔上 raise → job 標成 `held`，
白佔一次 GPU seat。

刪掉之後不需要做任何事：061 排到時 `commands.run` 的第一個 phase 就會重建，代價約 7 分鐘 CPU
（跟 058/059/060 搶 CPU 時會慢到 ~15 分鐘），這段時間 GPU 是閒著的。

## 重建是數值精確可重現的

變換完全決定性：ratio 走固定 24 點 grid，`rand` 的位移種子來自
`sha256(shift_seed_namespace + clip_id)`，響度正規化迭代到固定容差。同一個 clip 跑兩次，
baseline LUFS / crest / target / shift / fitted_ratio / 各 arm 的 lufs 與 crest 全部到小數 12 位一致。

**注意 manifest hash 的定義**：`transform_manifest.json` 存的是**解碼後 float32 樣本 + 取樣率**的 sha256，
不是檔案位元組的 sha256。因為 libsndfile 會在 float WAV 表頭寫一個含 Unix timestamp 的 `PEAK` chunk，
同一份音訊每次寫出的檔案位元組都不同（2026-09-16 實測：只有 offset 60 那個位元組隨時間變，資料區完全相同）。
改成內容雜湊之後，重建出來的 manifest 可以直接和舊的逐項比對。

## 重建後的驗證

```bash
cd /home/kojiek/MeanAudio && /home/kojiek/venvs/dac/bin/python scripts/eval/validate_crest_intervention_cfg3_20260916.py
```

要確認重建與前一次一致，重建前先留一份 `transform_manifest.json`，重建後比對 `arm_sha256`：
兩份應該完全相同（27,605 個 entry）。不同就代表 contract 或程式碼在中間被動過，
此時 record 會因 `contract_sha256` 不符而被拒絕，不會靜默混用。

---

## 排序：059 之後、060 之前（2026-09-16 操作者要求）

`p2_host.sh` 的 `next_pending` 就是對 `pending/` 做 `LC_ALL=C sort` 取第一個，純字典序。
`QUEUE.md` 規定插隊要**丟一支 `001_`–`009_` 的新腳本**，不要把後面的 job 重新編號。

現在就改名會連 059 一起插過去，跟要求相反。所以掛了一支 watcher
（`scripts/runs/crest061_cutin_watcher.sh`，tmux session `crest061_cutin`，log 在
`~/logs/crest061_cutin_watcher.log`）：**等 059 離開 `pending/`（＝已入座）之後**，
才把 `061_crest_intervention_cfg3.sh` 改名為 `005_crest_intervention_cfg3.sh`。
那時 pending 只剩 060 和 005，於是 059 跑完換 061，再換 060。

改名是安全的，檔名沒有任何東西綁定：
- `accept_guest` 只雜湊**腳本內容**（內容不含自己的檔名），不比對路徑
- contract 是從腳本裡的 `# GPU_QUEUE_CONTRACT=` 註解找的，與路徑無關
- 已實測：同一支腳本在 `005_` 與 `061_` 兩個名字下 `accept_guest` 都回 `(True, 'ok')`

contract 裡的 `queue_name` 與 `bindings.launcher` 已經預先寫成 `005_...`，
所以改名後這兩個欄位才是正確的；檔案在 watcher 觸發前仍叫 `061_`。

**要取消插隊**：`tmux kill-session -t crest061_cutin`。若已經改名，改回去即可：
`mv /home/kojiek/gpu_queue/p2/pending/005_crest_intervention_cfg3.sh /home/kojiek/gpu_queue/p2/pending/061_crest_intervention_cfg3.sh`

## 順帶修掉的 acceptance bug

061 原本宣告 `resume.kind = "per_item_idempotent"`，但 `_accept_resume_checkpoint` 的
cold-start 白名單只認 `from_scratch_with_autoresume` + `iteration == 0`，其餘一律
`resume checkpoint binding required` → 排到就被丟進 `held/`。**兩個檔名都會中**，與插隊無關。

同一個閘還有第二個陷阱：cold-start 分支若發現 `autoresume` 指的檔案已存在就拒絕。
原本 guest 在 pause 時會寫進那個路徑，等於被 P1 搶佔一次之後 061 就永遠不可能再被接受。
現在 `autoresume` 留空（這個 job 本來就沒有 checkpoint，續跑靠逐檔 record），
pause 進度改寫到 `pause_progress`，沒有任何 acceptance 閘會讀它。
