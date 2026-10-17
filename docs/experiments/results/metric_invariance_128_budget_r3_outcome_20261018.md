# 128a 已完成：四種方案皆未通過；129 在 CU 尾批次身分檢查停止

截至 2026-10-18 本次查核，budget_r3 的 128a 已於 **10-17 17:16:20（台北）完成**；129 於 17:16:57 failed，130 因前項未成功而於 17:17:07 held，17:17:08 queue idle，GPU 已釋放。129 只完成第一格的 200 段 C0 身分檢查，尚未評分 C1／C2／C3 全量表 B；NegMF 與 128d 節奏分層也未執行。

## 128a 的已完成結果

本次是五系統各 400 段、35 個變體、四種方案，加上 402 段 × 11 個 MIDI case。700 個 calibration units、44 個 MIDI units 及 20 個 smoke units 均已完成；重新執行原 contract 的 postflight，數量、唯一 ID、評分矩陣、必要有限值與全部分數檔 SHA256 都通過。

**PQ、CE、CU、CLAP 均沒有任何方案通過完整門檻。** 按預登錄規則，四項都回退到 C3，且 `sensitivity_only=true`。這是本次樣本與指定方案下的負結果，不能把 C3 稱為已通過的不變性修補。所有方案的失敗格及劣化階梯資料完整保存。

C3 的部分平手率如下；每格 N=400，判定要求五系統的指定平手變換都達 80%。m 仍使用原 126a 值（PQ 0.091892、CLAP 0.0103826），沒有重算或放寬。

| 系統 | PQ lead 10 ms | PQ lead 50 ms | CLAP PCM_16 | CLAP lead 200 ms |
|---|---:|---:|---:|---:|
| MusicGen | 82.00% | 81.00% | 89.50% | 45.25% |
| MusicLDM | 69.00% | 67.00% | 70.50% | 49.00% |
| ours CFG0 s14159265 | 66.25% | 66.50% | 72.50% | 48.00% |
| ours CFG3+neg s14159265 | 80.25% | 78.75% | 73.00% | 51.00% |
| 真實錄音 | 77.00% | 75.25% | 75.25% | 59.00% |

真實錄音的 resample44k 在 C3 下，PQ 平手率另只有 **52.75%**。完整 2,625 格 calibration 統計與 44 格 MIDI 統計另附 TSV。

**Q3 預測不成立。** MIDI 單音 blastnote_x1 的 PQ 變好率，在 C0 與 C1 響度正規化後都是 **323／402＝80.35%**，沒有降到預測的 50% 以下；C2／C3 是 329／402＝81.84%。這不支持「響度正規化足以消除此現象」，不直接斷言其餘機制。

quiet-LUFS fallback 在本次實際 N=400 校準的 union 涉及 3 段 MusicGen（MHgPpImV7b4_20、RDCXvJPzT5k_520、SHYGoXwgKtk_50），其他系統與 MIDI 為 0。按已登錄規則，從四方案的全部格一起移除這三段後重判，C* 選擇和 Q3 判定均不變，`robust_to_lufs_fallback=true`。這不表示每個單格比例或失敗格列表都相同。

## 129 的停止原因與 CPU 檢測

第一格 `meanaudio_s14159265_cfg0` 的前 200 個排序 ID，與 115 歷史 5,521 段的排序前綴一致。預設 AES batch_size=32，因此此次有六批 32 段與一批 8 段。

| 指標 | 前 192 段最大絕對差 | 最後 8 段最大絕對差 |
|---|---:|---:|
| PQ | 0 | 0.00904083 |
| CE | 0 | 0.00288630 |
| CU | 0 | **0.01289082** |
| PC | 0 | 0.00153661 |
| CLAP | 0 | 0 |

只有 CU 超過核准的每段 0.01 身分門檻。出問題的是 0-based index 193（第 194 段）`0ewQiU3q5jM_30`，signed diff = −0.0128908157；CU 全 200 段 signed mean = −0.0000595093，仍在平均差絕對值 0.001 之內。

已在 CPU 重新執行這 200 段的原 raw→lvl30→clean→C0 輸入準備，核對原 lvl30 SHA256，並與歷史 lvl30 檔案比較；**200 段採樣率與 float32 波形逐位元一致**。這次沒有呼叫 GPU 模型、沒有重判評分或套用 padding。

前 192 段逐分完全相同、所有 AES 差異都在 8 段尾批次、CLAP 全部完全相同，加上輸入波形一致，強烈支持 AES forward batch 大小是差異來源；固定 32 padding 是否能解決這一例，仍需另做 GPU 驗證，不能只憑此診斷視為已修好。

## 已確認與仍未回答

128a 能確認：本次四個候選方案均不足以通過完整不變性門檻，且 LUFS 正規化沒有讓單音爆音的 PQ 誤判比例降到 50% 以下。排除 fallback 不改變這兩項登錄判定。

**K1–K4、表 B 優勢是否縮成平手、NegMF E1 是否仍成立、節奏分層都尚無 128 的結果。** 130 是因缺少成功的 129 報告而 held，並非觸發 17 h 順延規則。原門檻與 run contracts 均保留，沒有自動放寬、重啟或改寫舊評分。

## 證據與重現

資料位於 `docs/experiments/harn/metric_invariance_128_budget_r3/outcome_20261018/`：

- `a_summary.json`：原始 128a summary 的逐位元副本，SHA256 與完成 terminal 的 registered report 證據相符。
- `calibration_rates.tsv`／`midi_rates.tsv`：全部 2,625／44 格統計。
- `b_identity_200.tsv`：200 段實測值及歷史參照，可離線重算完整身分閘、尾批次位置與失敗 ID。
- `b_audio_replay.json`：200 段 CPU 波形重建雜湊與比對結果。
- `score_unit_manifest.json`：765 個已存在 unit 的原始分數路徑及 SHA256（a 764、b 1）。原始 NPZ 留在主機，此檔不是 raw scores 的替代品。
- 三份 terminal、supervisor snapshots、host log excerpt、17 份有效通知收據的驗證、原 contract postflight 重驗及完整 snapshot provenance。

`python3 -S scripts/analysis/audit_metric_invariance_128_outcome.py` 可僅以標準 Python，在 clone 的 repo 中核對 snapshot 雜湊，重新套用 C* 門檻、fallback 排除後判定、Q3 與 129 身分閘，不依賴主機音檔、NumPy 或 GPU。host 與 portable audits 結果一致，另以修改 snapshot 一個 byte 的反例確認完整性檢查會拒絕。需要重新核對原始音檔時，才使用 `--export-host --verify-audio --scratch <本機暫存目錄>`。

此前通過的 25 個程式／supervisor CPU 測試與 queue 33 項是實作驗證；本次重新核對的是實際評分、完成證據及停止原因，沒有新增 GPU 工作。
