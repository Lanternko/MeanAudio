# 128 identity_r1：新身分閘通過，C1 因未定義 LUFS 停止

2026-10-17（台北）已按使用者指示，以 PR #2 的 `4830bd1` amendment 重新登錄三份 contract，使用新的 `identity_r1` run。08:48:09 由既有 P2 host 接手；08:48:57 完成 MusicGen C0 的 50 段 GPU 評分，新身分閘對 126a、115 都通過。隨後 C1 響度正規化遇到未定義 LUFS，128 於 08:49:07 被歸類 failed；129／130 因缺少成功的 128a 報告而 held。08:50:10 queue idle，GPU 已釋放。正式校正、N_a 選擇、表 B 與 NegMF 重判均未執行。

本次 AES 每段最大差 ≤ 0.01，各軸 signed mean 差絕對值 ≤ 0.001；CLAP／MEva 保持 1e-4。m、80%、劣化階梯、C* 與 K1–K4 不變，未套用 optional padding `3b1ead4`。新 C0 的 AES 最大差為 0.00053，signed mean 最大絕對值 3.31e-5，CLAP／MEva 最大差約 4.9e-7。新 run 使用原始凍結 inputs.json，音檔 SHA256、抽樣順序與 seed 不變。舊 run 的 25 份 contract／bundle／輸出檔案雜湊核對不變；新 run 的 contracts、分數、通知及 terminal 證據也均保留。

## 真正的停止原因

`RDCXvJPzT5k_520` 是第一組 50 段 MusicGen 音檔之一。峰值 0.0032959、RMS 0.00010688，並非全零，但 integrated LUFS 的絕對閘會將它視為無有效區塊，回傳負無限。只為診斷而加 40／80 dB 再扣回增益，均得到約 −81.274 的估計。五個系統各 50 段共 250 段 CPU 量測，只有這一段回傳未定義 LUFS。

設計 §9 已明定「無法定義 LUFS 的音訊使 gate invalid，不默默排除或替代」。因此這次停止符合已登錄協定，不能用換樣本、保留 raw 或臨時改響度規則繼續。這也不是 AES 身分閘再次失敗。

例外處理還發現一個程式問題：ProcessPool 的 map 在一個 worker 出錯後，其餘 worker 仍可能寫檔，清理暫存資料夾因此發生 `Directory not empty`，遮住原始錯誤。三個 guests 都已 terminal 後，已修正為先 shutdown／等待 worker，再清理；新增回歸測試確認保留原始例外並清完晚寫入檔案。19 個科學／supervisor 測試通過。此修正不改 LUFS、分數或任何科學門檻，尚未用於新的 GPU run。

## 可審查的下一步提案（尚未批准／未套用）

對 finite LUFS 完全沿用現有處理；只有非全零且 finite 的訊號被 LUFS 回傳負無限時，先除以峰值取得量測 probe，再用相同 pyloudnorm 量測、縮放到 −30 LUFS，不 limiting。全零、空訊號、非 finite 或重測仍失敗者繼續 invalid。記錄所有 fallback 的 id 與數量，不排除樣本。

獨立 CPU 候選對這段音訊量得輸出 −30 LUFS，峰值 1.2069（保留超過 0 dBFS）。正常音訊輸出不變、跨絕對閘的合成音增益不變性、零／NaN／空訊號拒絕等檢查通過。這只是候選，沒有改動正式 `normalize()`，也未 GPU 評分。它會放大近靜音的雜音，改變這些音訊的絕對閘遮罩，可能影響不變性和劣化結論；提案是在看過失敗後制定，必須另經負責操作人批准並建立新 contract／run。

提案：`docs/experiments/harn/metric_invariance_128_identity_r1/incident/quiet_lufs_amendment_proposal.json`。候選程式：`scripts/analysis/propose_metric_invariance_128_quiet_lufs.py`。

## 證據與驗證

主機啟動前：18 個科學／supervisor 測試、33 個 queue 情境，以及 schema、通知收據、scheduler 測試通過；真實 GPU pause/resume fixture 未跑。完整輸入雜湊 preflight 通過，三份新 HARN bundle 結構一致，重複登錄會拒絕覆寫。啟動與終止等 12 份通知收據已獨立驗證。

`docs/experiments/harn/metric_invariance_128_identity_r1/incident/` 保存 LUFS audit、fresh C0 done、terminal 三份、supervisor snapshot、host log、收據驗證、CPU 候選證據及清理修正測試輸出。`scripts/analysis/audit_metric_invariance_128_lufs.py` 可在主機重現 250 段 CPU 量測及 fresh C0 檢查，不呼叫 GPU 模型。

NVMe 約 59 GiB 可用，高於 50 GiB hard floor、低於 60 GiB warning floor；容量警告已送達，停止原因為 LUFS，不是磁碟。約 21 小時仍是未由完整 smoke 驗證的原估計。

## 後續：使用者同意（2026-10-17）

使用者同意採用上述替代規則，並加兩個條件：所有結論在去掉替代規則段落後重算一次，翻轉就報不穩健；開跑前用 CPU 全量掃描三份 contract 的 C1／C3 輸入。已實作於 `scripts/eval/metric_invariance_128.py`，規則與條件見設計 doc §11 與 `harn/metric_invariance_128/quiet_lufs_amendment_20261017.json`。正式重跑需新 contract 與新 run。

