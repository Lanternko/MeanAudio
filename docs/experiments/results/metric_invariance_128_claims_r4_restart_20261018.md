# 128 claims_r4：依結論指標檢查身分，沿用 A 重啟 129 → 130

查核時間：2026-10-18T01:20:21.429678+08:00。**129 正在執行；130 已在 P2 pending，等待新 129 的成功證據。** 這是啟動與診斷報告；K1–K4、完整表 B 和節奏分層尚未完成。

使用者在看到 budget_r3 的 CU 尾批次差異後，對「只由結論使用的指標阻斷，CU 等其他軸改為診斷警告」回覆 **GO**。先前整條 run 停下是身分閘範圍過廣，不能把它當成科學結果為負；正面與負面分數都保留。新 source／契約於 GPU 啟動前推送至 `3977f766a42c0f9f4f4293616dc21001ff1abeab`。

## 新規則與不變的限制

- 128b：PQ、CLAP 為阻斷指標；128c：PQ 為阻斷指標。其他軸仍按原門檻判讀、存數值與警告；超標不停止整個研究。主要指標門檻原值保留：AES max 0.01、|signed mean| 0.001；CLAP max 1e-4。無效分數與 provenance 錯誤仍拒絕繼續。
- 身分檢查仍比較每格前 200 段；後續 24 段真實音訊只補足 AES 最後一批至 32，不列入比較。正式評分與 Scorer 不變。這項工程調整與決策範圍在 §13 和 amendment 明列。
- 使用同一 input manifest、原始音訊、seed、m、80%、C* 選擇、fallback 排除與 K1–K4 判定。不因結果不好刪除資料。
- 沿用 budget_r3 已完成的 A，報告 SHA256 `2e8dacee5d46775473d7f5443590a39c034454744c6c030a0d6bb1deb69e4a23`。成功 terminal／通知回執、來源 git blob、校正與 Scorer AST、評分依賴均驗證一致，**沒有重跑 A**。
- 原 A 的負面結果保留：四個指標沒有方案通過，皆回退 C3 `sensitivity_only=true`；blastnote 的 C0/C1 PQ 變好率同為 80.35%，原先「降到 50% 以下」的預測未成立。後續 K 結果仍受 sensitivity-only 限制，不能提升成正式結論。
- 17 h 的 C 順延門檻保留，累計已完成 A 的 26,762.696 秒與本次 B 的 wall time，排除舊 run 停止後等待。暫存改到 Seagate 專用目錄，每個檔案系統仍維持 50 GiB hard floor；原始音訊、舊 run 和共享服務保留。

## 實際查核

30 個科學／supervisor CPU 測試、33 個 queue 測試，以及 schema、通知回執、scheduler 檢查均通過。實 GPU pause/resume fixture 未重跑。對實際舊 200 段重驗，新規則通過 PQ/CLAP、保留 CU 警告；舊規則仍重現拒絕。完成 A 可合法沿用，換錯 report hash、Scorer AST、manifest 或 carried time 的負向檢查均拒絕。

新來源綁定 CPU 全量掃描：A 179,422、B 54,820、C 33,126，合計 **267,368** 筆、**0 invalid**；fallback 事件分別 100／11／1，ID 與原掃描完全相同。2,763 個歷史檔案雜湊全部未變。

129 於 **2026-10-18 01:15:16 +08:00** 入座；主機 preflight、所有資料 SHA256、來源、儲存與通知 gate 通過，supervisor PID 2349966、child PID 2350036 與 start time 相符。4 份啟動／交接／preflight／磁碟 advisory 通知回執已驗證送達。root 在 warning 線以下、hard floor 以上；暫存放 Seagate 避免吃穿 root 安全線。

首格 `meanaudio_s14159265_cfg0` 實際評 224 段、比較原 200 段：

| 指標 | max abs | signed mean |
|---|---:|---:|
| PQ | 0 | 0 |
| CE | 0 | 0 |
| CU | 0 | 0 |
| PC | 0 | 0 |
| clap | 0 | 0 |

五個指標與歷史分數完全一致；這一格的尾批次差異已消失。第一個 **1,000 段 C1 正式 unit 已完成**（計時 206.5 s），ID／矩陣維度／AES 有限值／scores.npz 與 done.json 雜湊均通過。其中音訊準備 200.2 s、AES 評分 6.3 s；本次 Seagate 暫存較原 NVMe 慢，不能沿用原先估計的完工時間。這證明續跑已越過舊 CU 停點，並非只有 queue 啟動訊息。不能將首格通過推論為十格已全部通過。

## 可重現證據

- `harn/metric_invariance_128_claims_r4/launch_verification.json`：程序身分、首格診斷、首個正式 unit hash、通知回執驗證、budget carry。
- 同目錄的 `verify_launch.py`：唯讀重新查核活躍程序與 score unit，再寫啟動快照；只適用於此 run 活躍時。
- `live_identity_224.tsv`、`live_first_C1_1000.tsv`、`live_score_exports.json`：實際逐段分數、上下文標記與來源／匯出雜湊。
- `offline_identity_recheck.json`、`calibration_reuse_validation.json`、`registration_verification.json`、`scan_comparison.json`、`preservation_verification.json`：離線與主機驗證。
- `metric_invariance_128b_20261017_claims_r4_contract.json`、`metric_invariance_128c_20261017_claims_r4_contract.json`：新機器契約；typed HARN bundle 及原 source-bound CPU 掃描全部上傳。
- 先前結果與事件：`metric_invariance_128_budget_r3_outcome_20261018.md`。舊失敗及其逐段證據原樣保留。

主機持續由既有 P2 host 和新 run supervisor 監看，健康輪詢不呼叫模型；完成後依成功證據接續 130，超過 17 h 則照契約順延，不能把 deferred 寫成完成 NegMF 評分。
