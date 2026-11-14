# 128 budget_r3：已批准的 9 小時吞吐預估上限

2026-10-18 查核更新：128a 已於 10-17 17:16:20 完成，四種方案未有任何完整通過；129 因 CU 尾批次身分閘超標 failed，130 依賴 held，GPU 已釋放。完整結果、200 段 CPU 音訊重建與可離線重算證據見 [後續結果報告](metric_invariance_128_budget_r3_outcome_20261018.md)。以下保留開跑時紀錄。

開跑時驗證：2026-10-17 09:50:07（台北）P2 已啟動 budget_r3。五系統 C0 對 126a／115 身分檢查與全部 20 個 smoke units 完成；新的 9 h feasibility gate 通過，選定 N=400，含原 25% reserve 的預估為 8.715 h。09:53:26 第一個 400 段校準單元完成，已正式進入 128a，129／130 按成功依賴在 pending。尚無校準、表 B 或 NegMF 結論。

使用者對「只把 128a 預估時間上限由 7 h 改為 9 h，以 N=400 另登錄新 run」回覆「同意」。amendment 原檔為 `harn/metric_invariance_128/budget_amendment_20261017.json`，保留原 quiet_r2 提案及失敗證據。預估仍包含原本的 25% reserve；候選 1000／600／400、m、80%、seed、資料、變體、identity／quiet-LUFS、排除敏感度及 17 h deferral 均不變。依前次 8.694 h 實測預期選 400，新 run 仍須通過新的 smoke。

Evaluator 不再固定比較 7 h，改讀 `protocol.budget_seconds_a`；登錄命令明確指定 `--budget-hours-a 9`，寫入 32400 秒。smoke 報告保存實際使用的上限。缺少參數或非正／非 finite 上限會失敗。

25 個科學／supervisor CPU 測試、33 個 queue 情境，以及 schema、通知收據及 scheduler 檢查通過；新增測試直接套用 quiet_r2 的實測預估，確認 7 h 不通過、9 h 選 N=400。真實 GPU pause/resume fixture 未跑。三份 source-bound CPU 全量掃描重新產生，不沿用舊 source 的報告。掃描及新登錄證據保存在 `harn/metric_invariance_128_budget_r3/`。

新 suffix 為 budget_r3；新輸出預定 `/mnt/seagate/kojiek_data/metric_invariance_128_20261017_budget_r3/{a,b,c}`，NVMe 暫存使用獨立 budget_r3_tmp。使用既有 P2 controller，128→129→130。舊 quiet_r2 的 139 份 contract／bundle／分數／事件證據雜湊已保存並核對不變。

三份 CPU 掃描已全通過：128a 179,422 個（fallback 格 100）、128b 54,820 段（fallback 11）、128c 33,126 段（fallback 1），合計 267,368 個輸入、0 invalid。與 quiet_r2 的 scope、ID、分類及數量逐欄比對完全相同，只有 evaluator SHA256 更新。

三份新 contract 已用 `--run-suffix budget_r3 --budget-hours-a 9` 登錄並通過 HARN schema；與 quiet_r2 的 protocol 逐欄比對，唯一差異為 budget_seconds_a 25200→32400，原 manifest 完全相同。每份 contract 都綁定三份新掃描、approved amendment、source hashes 與操作人核准紀錄。quiet_r2 的 139 份保護檔案再次核對不變。

開跑驗證已另存 `launch_verification.json`、`live_smoke.json` 與 supervisor snapshot：20 個 smoke unit 分數檔 SHA256 正確；五系統對兩份歷史參照獨立重驗通過；第一個校準單元的 400 個 ID 與凍結 manifest 的前 400 個一致、全部分數 finite；P2 PID／start_time／job_id 及六份開始／preflight／smoke 等通知收據驗證有效。quiet_r2 的 139 份保護檔案雜湊仍不變。容量由既有 supervisor 持續監測。
