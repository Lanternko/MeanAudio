# 128 nvme_r5：修正新資料誤放 Seagate，恢復 NVMe 並承接已完成分數

使用者指出「seagate 不是放舊資料而已嗎？新資料都在 nvme」。先前 agent 因 root 剩餘空間接近 50 GiB floor，把新評分暫存與新分數也設到 Seagate；這偏離熱資料配置，造成主要耗時在磁碟。**本次已修正為新分數、日誌、批次音訊全部在 NVMe；Seagate 用於舊資料封存。**

## 實際速度與完工估算

| 批次 | 音訊準備 | AES／CLAP | 總計 |
|---|---:|---:|---:|
| 舊 Seagate，首個 C1 1,000 段 | 200.2 s | 6.3 s | 206.5 s |
| 新 NVMe，新 C1 1,000 段 | 3.6 s | 6.9 s | 10.5 s |
| 新 NVMe，C2 1,000 段 × 4 views | 4.3 s | 72.7 s | 77.1 s |

這是同一格、同一 mode 的不同已完成 chunk 的操作耗時對比，不是比較不同 chunk 的科學分數。evaluator/action SHA256 相同。

暫估新 129 約 **5–6 小時**，130 約 **3–4 小時**，合計約 **8–10 小時**，以這次啟動推算約 **2026-10-18 上午 10–12 點（台北）**。來源為十格各 32 段的 CPU-only NVMe 準備測速，配上已完成 A 各系統 clean unit 的實際 GPU 耗時、chunk setup、identity／CPU 分析 allowance 和 25% reserve；已用新的完整 C1／C2 unit 交叉核對。r5 完整 C3 尚未完成首批，預測仍有誤差，不能保證時刻。

## 執行與資料位置

新 run 於 **2026-10-18 01:39:54 +08:00** 入座啟動。PID/start time、3 份啟動／preflight／handoff 送達回執、所有 raw hash、schema 與容量 gate 均驗證通過；新 C1/C2 score hash、ID 與有限值有效。129 持續執行，130 已排隊等待成功依賴。

- 新輸出與 supervisor：`/home/kojiek/eval_output_nvme/metric_invariance_128_20261017_nvme_r5/<part>`。
- 新音訊暫存：`/home/kojiek/eval_output_nvme/metric_invariance_128_20261017_nvme_r5_tmp/<part>`。
- 已完成 A 及舊 r4 原紀錄保持在 Seagate，作為凍結證據；本次原始輸入不移動。
- 原 `claims_r4` 正常中斷，3 個 C1 × 1,000 段與 224 段 identity unit 驗證後逐位複製承接，**未重算這 3,000 段分數**。未完成 unit 不承接；舊 130 因成功依賴缺失 held，不把它當完成 NegMF。

## 舊資料封存與安全線

只處理已完成 115 的六格 `lvl23/audio`，全部排除於本次 input manifest；沒有移動目前實驗 raw 音訊。逐檔 copy、fsync 與 SHA256 驗證，以原目錄／symlink 的 atomic exchange 保留舊路徑；再驗證後才移除 NVMe 重複副本。原音訊完整留在 Seagate，沒有丟棄資料。

目前已驗證 6/6 格，13,462,978,858 bytes；封存狀態：**全部完成**。第一次單檔 copy 後立即 fsync 的排程較慢，另有保留原始來源的 CPU copy 排程修正記錄；改為批次 copy 後 fsync 所有目標，完成同樣的耐久性與雜湊檢查才交換目錄。兩次 CPU attempt 和 partial-copy 處理均保留。

啟動容量閘在完成足夠封存後通過。同一 NVMe filesystem 合併預算：50 GiB hard floor、7 GiB 新增峰值（包含暫存及報告）、100 MiB atomic duplication、2 GiB recovery reserve。runtime 持續每 5 秒監控，沒有降低 hard floor，也沒有動到共享服務或其他程序。

## 科學協定與驗證

- 整份 `protocol` 與 r4 完全一致；m、80%、K1–K4、主要 identity 門檻、fallback 排除與 C3 sensitivity-only 都保留。沒有改分數或丟棄不利結果。
- 沿用已驗證的完整 A，沒有重跑 A。17 h 數值與原 r4 B start clock 保留，包含此次搬移時間，沒有以 restart 延長預算；130 若超標仍依原登錄規則 deferred。
- evaluator source 未變，267,368 筆、0 invalid 的三份全量掃描 source／manifest hash 仍有效，逐份比對後原樣承接。封存 lvl23 不改動任何掃描輸入。
- 33 個科學／supervisor CPU 測試通過；新增 exact-cache-resume、不承接未完成 unit、action/hash/ID 錯誤拒絕。queue／schema／receipt／scheduler fixture 與 runtime 未變，承接 r4 的已通過證據；新 contract 的 bundle 及 live preflight 另行通過。
- 仍沒有完整表 B／K1–K4／節奏分層結果；不能把啟動或首批成功當成研究結論。

證據位於 `harn/metric_invariance_128_nvme_r5/`：`storage_correction_plan.json`、`archive_*.json`、`archive_gate.json`、`score_resume.json`、`registration_verification.json`、`scan_reuse.json`、`nvme_prepare_timing.json`、`live_storage_timing.json`、`live_eta.json`、`launch_verification.json`、`live_nvme_C1_1000.tsv`。新機器契約與 typed HARN bundle 同時提交。
