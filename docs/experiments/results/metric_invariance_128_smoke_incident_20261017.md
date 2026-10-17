# 128 啟動、smoke 失敗與檢測紀錄

128 已實際啟動，但第一個 **MusicGen、C0、50 段**的重現性檢查沒有通過。AES 最大絕對差為 **0.000527508**，超過預登錄 `1e-4`，因此停止正式校準。129、130 因缺少有效的 128a 完成證據而 held；終止／held 通知收據均已驗證。

本紀錄的結論是前置數值重現檢查失敗。128a 的方案校準、N_a 選擇、表 B／NegMF 重判與節奏分層尚未完成，因此不能從此次 smoke 判定論文結論是否成立。

## 發生順序

以下時間皆為台北時間，2026-10-17。

| 時間 | 事件 |
|---|---|
| 約 02:09 | 126b 刺激的 89,329 個檔案逐檔 SHA256 比對完成；25,872,793,877 bytes 移到 Seagate，原路徑保留 symlink。NVMe 可用空間回到約 64 GiB，超過 50 GiB 底線。 |
| 02:12 前 | 補齊腳本、P2 supervisor、三份 contract；schema bundle、完整輸入雜湊與儲存空間 preflight 通過。 |
| 02:12:53 | P2 啟動 `128_metric_invariance_calibration`，程式／contract revision 為 `45bf011`。 |
| 02:13:29 | 第一個 MusicGen C0 unit 完成 50 段評分，隨即比較歷史分數。 |
| 02:13:36 | 身分閘失敗，128 以 rc=1 結束；failure 通知收據已交付。 |
| 02:14:18 | 129 preflight 因缺少成功的 128a 報告而 held。 |
| 02:14:28 | 130 同樣因科學依賴未滿足而 held。其後 P2 idle，沒有繼續正式評分。 |

原有預登錄檔案與失敗證據保留。隔離修補候選沒有套用到這次已登錄執行。

## 檢測一：逐段重現比較

將此次 50 段 C0 分數，按 clip ID 對齊 126a `clean/musicgen/scores.tsv`。門檻固定為每段絕對差 ≤ `0.0001`；沒有放寬門檻。

| 指標 | 50 段最大絕對差 | 超標段數 | 前 32 段最大差 | 最後 18 段最大差 |
|---|---:|---:|---:|---:|
| AES PQ | 0.000360847870 | 5 | 0.000001025543 | 0.000360847870 |
| AES CE | 0.000288858917 | 7 | 0.000002666168 | 0.000288858917 |
| AES CU | 0.000493323669 | 3 | 0.000000932190 | 0.000493323669 |
| AES PC | 0.000527508057 | 6 | 0.000000857025 | 0.000527508057 |
| CLAP | 0.000000486207 | 0 | 0.000000486207 | 0.000000388978 |
| MEva | 0.000000489487 | 0 | 0.000000489487 | 0.000000483597 |

AES 以每批最多 32 段評分，50 段分為 **32＋18**。所有超標都落在索引 32–49 的尾批次（索引從 0 起算）。同一組 50 個 ID 的歷史 126a 與 115 分數最大差低於 `3e-6`，因此歷史參照彼此一致，這次新分數才是失配的一側。

逐段原始分數與三組歷史參照已匯出成 [smoke_50.tsv](../harn/metric_invariance_128/smoke_evidence/smoke_50.tsv)，可以直接重算上述數字。

## 檢測二：音訊與 CPU 重採樣

使用相同的 50 個來源檔案，檢查輸入 SHA256，再分別重建：

1. 126a 原有 `make_one(..., clean)` 產生的 FLAC。
2. 128 的 `prepare_one(..., C0)` 產生的浮點 WAV 容器。

兩者解碼成 float32 後，**50／50 段逐位相同，最大樣本差為 0**。另外，以 CPU 1 thread 與 2 threads 執行相同的 32 kHz→16 kHz 重採樣，最大差也為 **0**。此次重查沒有執行模型或 GPU 推論。

這排除了本次受測音訊樣本改變、以及上述 CPU thread 重採樣差異的解釋；沒有排除所有可能的推論後端因素。每段取樣數、取樣率與兩份解碼資料的 SHA256 見 [audit_results.json](../harn/metric_invariance_128/smoke_evidence/audit_results.json)。

## 檢測三：歷史 clean／clean_shuf

再比較 126a MusicGen 全部 5,131 段的原有 `clean` 與 `clean_shuf` AES 分數：

| 指標 | 歷史兩組最大差 | 差異超過 1e-4 的段數 |
|---|---:|---:|
| PQ | 0.002257 | 16 |
| CE | 0.004949 | 20 |
| CU | 0.002693 | 17 |
| PC | 0.005162 | 17 |

這是既有結果的再分析，沒有重新評分。完整逐段資料見 [historical_clean_vs_shuffled_5131.tsv](../harn/metric_invariance_128/smoke_evidence/historical_clean_vs_shuffled_5131.tsv)。歷史整體差異的成因仍需另外控制變因；不能把每個歷史差異都直接歸因於尾批次。

目前證據強烈指向此次失配與不足 32 段的尾批次有關。**尚未以固定實際 batch 大小的 GPU 對照試驗確認因果。**

### 補充（2026-10-17）：歷史差異的位置

126a 的 `clean` 以排序後的 ID 順序送進 AES，`clean_shuf` 則先 `random.Random(126).shuffle()`（`metric_win_tie_126a.py`），兩者都是每批 32 段；5,131 段在兩種順序下各有一個 11 段的尾批次。重建兩種順序後：

| | 段數／值 |
|---|---|
| 四軸任一軸差 > 1e-4 | 22 |
| 其中位於任一種順序的尾批次 | **22**（11＋11） |
| 不在尾批次的段，四軸最大差 | **3e-6** |

不在尾批次的 5,109 段，同批的其他段在打亂後全部不同，分數仍吻合到 3e-6，所以 AES 分數受 **批次大小** 影響、不受同批內容影響（全部段長相同，318,080 samples，不是 padding）。上面「歷史整體差異的成因仍需另外控制變因」一句據此可回答：歷史 clean／clean_shuf 的差異全部可由尾批次解釋。重現：`python3 scripts/analysis/tail_batch_position_check_128.py`（只需標準函式庫）。

### 後續：身分閘 amendment

使用者同意把 AES 的身分閘從每段 1e-4 改成「每段 ≤ 0.01 且平均差絕對值 ≤ 0.001」，CLAP／MEva 維持 1e-4（設計 doc §10、`harn/metric_invariance_128/identity_gate_amendment_20261017.json`）。以 `smoke_50.tsv` 離線重驗，此次 smoke 在新門檻下對 126a 與 115 都通過。重跑另登錄新 run。

## 修補候選的狀態

隔離候選 [`3b1ead4`](https://github.com/Lanternko/MeanAudio/compare/45bf011...codex/metric-invariance-128-tail-batch) 將 AES 最後不足 32 段的批次，用最後一段的重複路徑補齊至 32；50 段例子會補 14 次模型輸入。沿用的 per-path score mapping 保留 50 筆觀測，重複路徑只作運算上下文。候選保留原始 cohort、其他評分器及 `1e-4` 身分門檻。

候選版合計 **20 個 CPU fixture 通過**（原有 17 個，加 3 個補齊／觀測計數檢查）。[測試輸出](../harn/metric_invariance_128/smoke_evidence/candidate_tests_20.txt) 驗證程式邏輯，**不代表 AES GPU 分數已重現或修補已成功**。候選尚未套用，尚未執行 GPU smoke；重啟須依原政策批准並另登錄新 run，保留這次失敗記錄。

## 重現檢測

只需 Python 標準函式庫，即可用已提交的快照重算分數差異與批次分布：

```bash
python3 scripts/analysis/audit_metric_invariance_128_smoke.py
```

在保留原輸入與相同環境的主機，可另外重查音訊與 CPU 重採樣：

```bash
/home/kojiek/venvs/dac/bin/python scripts/analysis/audit_metric_invariance_128_smoke.py \
  --recheck-audio --scratch /path/to/scratch --output /path/to/audit.json
```

預期輸出為 `identity_gate_passed: false`、`all_exceedances_in_last_18: true`；音訊重查三項差異為 0。診斷程式正常結束，表示成功重現這個失敗狀態，並非 smoke 通過。

快照的來源路徑與雜湊見 [provenance.json](../harn/metric_invariance_128/smoke_evidence/provenance.json)；三份 terminal JSON 與通知收據驗證狀態見 [queue_status_snapshot.json](../harn/metric_invariance_128/smoke_evidence/queue_status_snapshot.json)。
