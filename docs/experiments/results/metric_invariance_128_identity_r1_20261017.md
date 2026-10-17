# 128 身分閘 amendment 後的主機重新登錄

使用者提供 PR #2 的 `4830bd1`，並明確指示「開始實驗」。此次重新登錄為 `identity_r1`，不續跑已失敗的 run，也不套用可選的 AES 尾批次 padding `3b1ead4`。

AES 四軸每段最大絕對差 ≤ 0.01，各軸 signed mean 差的絕對值 ≤ 0.001；CLAP／MEva 保持每段最大差 ≤ 1e-4。m、80% 平手率、劣化階梯及 K1–K4 判定規則不變。

新 contract 使用原本凍結的 `inputs.json`，保留音檔 SHA256、抽樣順序、seed 與樣本。新輸出位於 `/mnt/seagate/kojiek_data/metric_invariance_128_20261017_identity_r1/{a,b,c}`；暫存位於 NVMe 的獨立 `identity_r1_tmp` 目錄。登錄腳本新增 `--run-suffix`，拒絕覆寫既有 contract；三份 contract 都綁定 amendment 原檔 SHA256。

主機驗證：18 個科學／supervisor 測試通過，33 個 queue 情境通過；schema、通知收據、scheduler 測試通過。真實 GPU pause/resume fixture 未跑。舊 smoke 的 50 段分數由新 `check_c0()` 離線重驗，對 126a、115 都通過。舊 run 的 25 份 contract／bundle／輸出檔案 SHA256 均保持不變。新 contract 除身分閘外的 protocol 欄位，已逐欄與原 contract 比對一致。

所有驗證與新 HARN 文件位於 `docs/experiments/harn/metric_invariance_128_identity_r1/`。離線驗證不代表新的 GPU smoke 完成；新 run 仍必須通過五個系統的身分及吞吐 gate，才會進入 128a 校正。128c 維持 17 小時後順延規則。
