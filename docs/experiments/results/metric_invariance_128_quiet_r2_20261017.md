# 128 quiet_r2 主機驗證與重新登錄

最新結果：quiet_r2 於 2026-10-17 09:32:57（台北）啟動，完成五個系統 × 四種方案的 20 個 50 段 GPU smoke units；五系統 C0 對 126a／115 身分檢查均通過，C1／C3 未再發生 LUFS 問題。但吞吐預估 gate 不通過：N=400 最小候選仍需 8.694 h，超過已登錄 7 h。程式未開始校正，128 failed、129／130 因科學依賴 held，09:36:27 queue idle，GPU 已釋放。

使用者提供 PR #2 `eff41ef`，批准非零 finite 音訊的 quiet-LUFS 替代規則，並要求在任何新 GPU 結果產生前登錄 fallback 排除敏感度分析、完成三份 CPU 全量掃描。主機已拉取該版；新增主機執行與證據綁定修補，source 在 `d5e3a16`。

## CPU 掃描結果（GPU 開跑前）

| 部分 | 檢查輸入數 | fallback 格數 | 無法處理 |
|---|---:|---:|---:|
| 128a | 179,422 | 100 | 0 |
| 128b | 54,820 | 11 | 0 |
| 128c | 33,126 | 1 | 0 |
| 合計 | 267,368 | 112 | 0 |

128a 的 100 個 fallback 格涉及 8 段 MusicGen 音訊，涵蓋最大 1,000 段候選 × 5 系統 × 35 變體，以及 402 × 11 個 MIDI 輸入。128b 的 11 段分別為 MusicGen 9、MusicLDM 1、MeanAudio s16180339 CFG0 1；128c 的 1 段是同一個 MeanAudio CFG0 控制組音訊。格數和各部分的數量不可當作跨部分不重複的音檔數。完整 ID／變體／來源雜湊綁定見 `harn/metric_invariance_128_quiet_r2/lufs_scans/{a,b,c}.json`。

三份掃描依序執行，最多 8 個 CPU workers，沒有 GPU 模型呼叫。每個 job 專用暫存目錄，避免不同系統同 ID 的 raw→lvl30 檔案互相覆寫；MIDI 報告保留 case 名稱。掃描前後核對 evaluator 與 input manifest 的 SHA256。進度與雙檔案系統容量由獨立 CPU driver 每 5 秒記錄，已發送開始／通過通知；CPU driver、contract、通知返回狀態均保存。

## 主機測試與規則落實

24 個科學／supervisor 測試、33 個 queue 情境、schema、通知收據及 scheduler 檢查通過。用實體 Python 檔案執行 spawn CLI fixture：正常段通過、近靜音段記 fallback、全零段令掃描失敗且報告列出 invalid。真實 GPU pause/resume fixture 未跑；新的 GPU smoke 仍是必要 gate。

fallback ID 會從四種方案的全部 calibration 格一起排除，重選 C* 並重判 Q3；K1–K4 會去除任一側 fallback 的配對重判。新增配對回歸測試確認 verdict 改變時 `robust_to_lufs_fallback=false` 且 `eligible_as_conclusion=false`。另補排除後 Q3 欄位、table fallback 數量；若 calibration C* 改變或原本只是敏感度分析，也不把該 K 判定當作結論。

## 新登錄與保留證據

新 run suffix 為 `quiet_r2`，三份 contract／HARN bundle、輸出及通知 ID 均獨立。三份成功掃描報告加入每份 contract 的 immutable files；登錄與 GPU preflight 都檢查三份 part、完整筆數、零 invalid，以及 source／manifest SHA256，拒絕缺漏或過期證明。

與 identity_r1 逐欄比對：輸入 manifest、m、80%、identity gate、seed、變體、C* 及 K 判定規則保持一致；protocol 唯一新增為已批准的 quiet_lufs_rule。identity_r1 的 55 份 contract／bundle／分數／事件證據檔案雜湊保持不變。不套用 optional AES padding。

新輸出：`/mnt/seagate/kojiek_data/metric_invariance_128_20261017_quiet_r2/{a,b,c}`。NVMe 暫存採獨立 `quiet_r2_tmp`，使用既有 P2 controller 與 GPU lock，順序 128→129→130。128c 維持 17 小時順延規則；約 21 小時仍須由新 smoke 吞吐決定。登錄與掃描通過不代表科學結果完成。

## 吞吐 gate：目前實際阻擋原因

| N_a | 預估時間（含既有 25% reserve） | 7 h gate |
|---:|---:|---|
| 1,000 | 19.805 h | 不通過 |
| 600 | 12.398 h | 不通過 |
| 400 | 8.694 h | 不通過 |

`smoke.json` 的 selected_n 為 null。這次沒有改動門檻、偷偷縮小 N 或進入正式校準；沒有表 B、NegMF、C* 或敏感度結論。20 個 unit 的 1,000 行分數（同一批每系統 50 段、四方案各評一次）已匯出 `budget_incident/smoke_1000_rows.tsv`；未評分的 CLAP／MEva 格為空白。獨立重算確認全部五系統 C0 對兩份歷史參照通過。MusicGen 的 C1／C3 各有一段用到 fallback，超過 0 dBFS 的峰值照實保留。

`budget_incident/` 保存完整 smoke timing／CPU transform benchmark、逐段分數、unit SHA256、三份 terminal、supervisor snapshot、13 份有效通知收據的驗證及重新核對過的舊 run 雜湊。`scripts/analysis/check_metric_invariance_128_budget.py` 只用標準 Python 從已提交的 timing 重算三個預估，無模型呼叫。

待核准提案只把 128a 可接受的吞吐預估上限從 7 h 改為 9 h，依同一組候選會選 N=400；既有 25% reserve 已包含在 8.694 h 內。m、80%、資料、seed、identity／LUFS 規則、敏感度分析與 17 h deferral 均不變。提案在 `budget_incident/budget_amendment_proposal.json`，尚未套用、尚未批准。若批准，需要新 contract／run，並讓 evaluator 讀取 contract 的 budget_seconds_a；目前 evaluator 的 feasibility 比較仍是固定 7 h。若 evaluator source hash 改動，三份 source-bound CPU 掃描也必須重建，不沿用舊證明。
