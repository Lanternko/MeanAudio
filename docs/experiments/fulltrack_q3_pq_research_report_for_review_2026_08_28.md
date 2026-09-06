# Fulltrack Q3 高 PQ 現象：研究稽核與驗證計畫審查稿

報告日期：2026-08-28  
報告狀態：提交操作者審查；尚未取得 Gate 2，未註冊或啟動實驗  
研究主題：解釋歷史 fulltrack Q3/q9 的 PQ 6.9337，區分評估錯誤、推論 Q token 效應與 checkpoint-family／訓練歷程效應，並測試 non-fulltrack PQ 是否可達 6.9。

## 一、執行摘要

歷史 fulltrack Q3/q9 結果為 CLAP 0.1821、CE 6.8458、CU 7.1468、PC 5.3016、PQ 6.9337。現有證據不支持「該次評估混入舊音檔」：凍結 log 只有一組 eval arguments，且有 5,521 筆 `Audio saved`、5,521 個唯一輸出路徑。不過生成音檔已刪除，當時也沒有逐音檔雜湊，因此無法宣稱 byte-level reproduction。

已確認的缺陷是 provenance 標示錯誤與 caption granularity mismatch。該 checkpoint 實際使用 upstream fulltrack Q3 track-level caption corpus，而不是 Caption 2.0 slot012；同一首 track 的 caption 被廣播至該 track 的約 10 秒 segments。沒有證據顯示 caption 跨到不同 track，而歷史 TSV-to-NPZ row binding 因原 NPZ 已覆寫且當時 `require_text_overlay=false`，目前無法判定。

Q 路徑確實被使用；q0–q9 在 Stage 2 前是 trained q10 的 bit-exact copies，因此「結果來自未初始化的隨機 Q embedding」被排除。但這並不能證明 Q 是高 PQ 的主因。16 個去重歷史 arms 中，PQ 與 CE、CU 的 Pearson 相關分別為 0.969184、0.987095，與 CLAP 為 -0.126996。這表示 PQ 很可能主要反映 no-reference aesthetic axes，而不是 prompt adherence；因此只追 PQ 可能犧牲 CLAP。

一個未註冊、非 canonical 的 511-row prefix probe 顯示，fulltrack NoQ 相對 segment slot0 NoQ 的 paired PQ 差為 +0.221375，95% CI [0.139169, 0.301193]；在相同 neutral text 下差為 +0.250005，95% CI [0.169971, 0.329147]。fulltrack 內 q9 相對 NoQ 的差較小，為 +0.057474，95% CI [0.017206, 0.095714]。這組探索性結果把主要嫌疑指向 checkpoint family／audio prior，而不是單純 prompt alignment 錯誤，但它不能作為正式結論。

目前最合理的工作假說是：歷史高 PQ 主要來自 checkpoint-family、caption corpus 與訓練 trajectory 的合成效應，Q inference token 可能提供較小的額外位移。評估 bug 尚未被證實；fulltrack 的因果優勢也尚未被證實。

## 二、研究問題與判定邊界

本研究回答四個問題：

1. 歷史 PQ 6.9337 是否可在同一凍結 stack 下重現？
2. q9 相對 q0 的推論 token 是否在 fulltrack 與 segment-slot0 Q-trained families 中產生一致方向的 PQ 位移？
3. fulltrack 相對 segment-slot0 的 checkpoint-family 差異，是否在 q9、q0、NoQ 三種條件下穩定存在？
4. canonical non-fulltrack arm 是否達到 aggregate PQ ≥ 6.9？

本計畫是 evaluation-only。即使三組 checkpoint-family contrasts 都顯著，也只能描述 checkpoint-family association，不能把效果歸因於 Q training、caption granularity 或某一訓練資料因素。若要做因果主張，仍需另行批准 matched fresh-training replication。

## 三、歷史證據稽核

### 3.1 歷史指標

| Arm | CLAP | CE | CU | PC | PQ |
|---|---:|---:|---:|---:|---:|
| fulltrack Q3/q9 | 0.1821 | 6.8458 | 7.1468 | 5.3016 | 6.9337 |
| segment slot0 Q3/q9 | 0.2145 | 6.2474 | 6.7019 | 5.1752 | 6.5437 |

兩者呈現明顯 tradeoff：fulltrack 的 PQ 高 0.3900，但 CLAP 低 0.0324。這不等於 fulltrack 的整體品質較好；它只表示 aesthetic predictor 與 prompt-alignment metric 對兩個 checkpoints 的排序不同。

### 3.2 已確認、已排除與未決事項

| 狀態 | 事項 | 研究含義 |
|---|---|---|
| 已排除 | stale-audio contamination 可解釋該次歷史 eval | log 的輸出筆數與唯一路徑完整；但仍非 byte-level reproduction |
| 已確認 | 歷史 contract 的 Caption 2.0/slot012 身分是 mislabel | checkpoint 實際綁定 fulltrack Q3 upstream corpus |
| 已確認 | track caption 對約 10 秒 segments 的 granularity mismatch | 是 same-track broadcast，不是已證實的 cross-track mismatch |
| 未決 | 歷史 TSV-to-NPZ row binding 是否逐列正確 | 原 NPZ 已覆寫，無法追溯 |
| 已排除 | Q pathway 未使用或 q0–q9 是隨機初始化 | runner 會消費 q level，且 q0–q9 起始為 trained q10 copies |
| 已排除 | same-track sharing 單獨足以造成 PQ 增益 | R-Shared control 的 PQ 未上升；但其他交互作用仍未決 |
| 未決 | fulltrack 是可重現的因果優勢 | 缺少 matched fresh training 與完整歷史 training provenance |

### 3.3 指標結構

去除重複 `exp_label` 後共有 16 個 arms。PQ 的 Pearson correlation 為：

| 對照指標 | 與 PQ 的相關 |
|---|---:|
| CE | 0.969184 |
| CU | 0.987095 |
| PC | 0.459229 |
| CLAP | -0.126996 |

Audiobox Aesthetics 將 CE、CU、PC、PQ 定義為獨立的 no-reference per-item axes，不是 prompt alignment 的替代品。因此，PQ ≥ 6.9 必須與 CLAP 一起解讀，且不能直接等同於人類偏好或「更符合 prompt」。

## 四、探索性 511-row probe

此 probe 未 preregister、沒有 HARN／queue ownership、採用 MusicCaps 前 511 rows 而非隨機樣本、CFG metadata 寫作 0.0、沒有 CLAP，也沒有逐音檔 hashes。它只用來產生後續假說，不列入 canonical evidence。

### 4.1 Aggregate PQ

| Arm | n | PQ |
|---|---:|---:|
| fulltrack NoQ | 511 | 6.806277 |
| segment slot0 NoQ | 511 | 6.584902 |
| fulltrack Q3/q9 | 511 | 6.863751 |
| segment slot0 + neutral text | 511 | 6.619611 |
| fulltrack + neutral text | 511 | 6.869616 |

### 4.2 Paired PQ contrasts

| Contrast | Mean delta | 95% paired bootstrap CI |
|---|---:|---:|
| fulltrack Q3/q9 − fulltrack NoQ | +0.057474 | [0.017206, 0.095714] |
| fulltrack NoQ − segment slot0 NoQ | +0.221375 | [0.139169, 0.301193] |
| neutral-text fulltrack − neutral-text segment slot0 | +0.250005 | [0.169971, 0.329147] |

在 neutral text 下仍出現約 +0.25 的 gap，使「只是原 prompt 比較方式造成」變得較不可能。另一方面，這些 checkpoint-family contrasts 同時混合了 caption corpus、training trajectory 與 Q-training history，不能識別單一原因。

## 五、預註冊 B-matrix 驗證設計

所有新 primary／fair-comparison evaluation 都使用 MusicCaps 5,521、MeanFlow 25、literal CFG 0、seed 42、NoMask、full precision，並計算 CLAP、CE、CU、PC、PQ。每個 arm 使用全新 no-skip output directory、5,521 個逐音檔 SHA-256、逐項 metrics 與完整 provenance。

| Arm | Checkpoint family | 推論條件 | 分類 | 目的 |
|---|---|---|---|---|
| B1 | fulltrack Q3 | q9 | canonical repeat | 重現歷史 fulltrack 結果 |
| B2 | segment slot0 Q3 | q9 | canonical repeat, non-fulltrack | 重現歷史 segment 結果 |
| B3 | fulltrack NoQ | NoQ | canonical | NoQ family contrast |
| B4 | fulltrack Q3 | q0 | secondary diagnostic | fulltrack q9−q0 |
| B5 | segment slot0 Q3 | q0 | secondary diagnostic, non-fulltrack | segment q9−q0；secondary 6.9 target |
| B6 | segment slot0 NoQ | NoQ | canonical, non-fulltrack | NoQ family contrast；canonical 6.9 target |

### 5.1 Reproduction gate

B1 與 B2 的五個 aggregate 指標，必須各自以 `Decimal(str(value))`、`ROUND_HALF_UP` 量化至四位小數後，逐欄完全等於歷史向量。任何 source/hash/count/finite-value 錯誤都判為 `reproduction_invalid`；有效但不相等則判為 `historical_repeat_failed`。兩者任一未通過，B3–B6 全部 hold。

這個 gate 很嚴格，故失敗只能說「在目前凍結 stack 下沒有重現」，不能倒推出歷史評估必然混入舊音檔。依賴版本、硬體、程式或非決定性 drift 都仍可能造成失敗。

### 5.2 Paired analysis

分析單位為 MusicCaps ID，使用 10,000 次 paired percentile bootstrap、seed 20260828、95% CI。主要 contrasts：

- q inference：B1−B4、B2−B5；實用門檻 |ΔPQ| ≥ 0.05。
- checkpoint family：B1−B2、B4−B5、B3−B6；實用門檻 |ΔPQ| ≥ 0.15。
- interaction：`(B1−B4)−(B2−B5)`。

`positive_supported` 要求 mean delta 達正門檻且 CI lower bound > 0；`negative_supported` 對稱定義。CI 接觸或跨過 0、或 mean 未達門檻，均歸為 `small_or_uncertain`。

### 5.3 Non-fulltrack PQ ≥ 6.9 判定

優先看 canonical B2、B6；任一有效且 aggregate PQ ≥ 6.9，即判 canonical target achieved。只有 secondary B5 達標時，必須標成 secondary q0 target，canonical target 仍未達成。若三者都有效且低於 6.9，target 保持開放，等待既有 024／025／026 結果，再提出下一個 training proposal；不得因此重排現有 queue。

## 六、Gate 1 實作與測試狀態

操作者已批准 Gate 1／AUTO。Gate 1 僅授權 no-GPU 實作、測試與封存，不授權 queue mutation、final launch contract 或 GPU launch。

| 項目 | 結果 |
|---|---|
| Science selftests | 8/8 passed |
| Security selftests | 36/36 passed |
| B1–B6 runner dry-run | 6/6 exit 0；未使用 GPU |
| HARN dry-run | passed；明確回報 launch/queue mutation 均為 false |
| Candidate direct launch | exit 75，依預期 fail closed |
| Sealed inputs | 142 registered entries，12,459,504,053 bytes |
| Seal integrity | actual path set 等於 receipt set；無 symlink/non-regular；無 owner/mode/link-count 錯誤 |
| Final contract | 未建立 |
| Live HARN state | 未建立 |
| Queue mutation／GPU | 均未發生 |

Gate 1 實作狀態是 `implementation_complete_awaiting_exact_review`，不是 launch-ready。

## 七、目前 blocker 與所需增補

1. Offline CLAP load 發現未納入原 Gate 1 copy sources 的 `bert-base-uncased` tokenizer dependency；`roberta-base` 也是已知 transitive dependency。現有 seal 正確地 fail closed，未下載也未擅自加入。
2. 現有 Plan 同時要求 final launch-enabled contract 位於 sealed source tree，又禁止在 Gate 1 exact review 前建立 final contract，形成 materialization／seal sequence 的循環。
3. Plan-specific launch-control fields 與 strict HARN schema 的接合方式仍需明確化。
4. Full four-document HARN validation、live lease proof、notification delivery 與完整 Gate 2 hash binding 尚未演練完成。

建議的下一步不是直接批准 Gate 2，而是先審查一份範圍受限的 Gate 1b additive amendment：只授權加入精確雜湊的 BERT／RoBERTa 本地 assets、解決 schema 與 final-contract materialization 順序、重新 seal，並完成 no-GPU exact review。Gate 1b 仍不得註冊 queue 或啟動 GPU。

## 八、研究結論

目前最佳結論如下：

- 沒有證據證明歷史 fulltrack Q3 高 PQ 是 stale-audio eval bug。
- 已確認 provenance mislabel 與 same-track caption granularity mismatch；未確認 cross-track mismatch。
- Q inference path 有效，但 Q 是否造成主要增益仍未知。
- 歷史 arm correlation 與探索性 paired probe 都更支持「checkpoint-family／audio prior 是主要來源、Q token 可能是較小修飾」這個工作假說。
- fulltrack Q3 的因果優勢、歷史 byte-level reproduction、以及 non-fulltrack PQ ≥ 6.9，全部尚未成立。
- B1–B6 設計可以把 reproduction、Q inference association、checkpoint-family association 與 non-fulltrack target 分開判定；在完成 Gate 1b、獨立安全審查與 fresh-context verification 前，不應進入 Gate 2。

## 九、請審查者特別判定

1. 是否同意「現有證據未證實 eval bug，但也不足以證明 fulltrack 因果優勢」？
2. 是否接受 B1/B2 的逐欄四位小數 exact reproduction gate，或認為應另設 tolerance-based secondary analysis？
3. 是否接受 Q-effect 0.05、family-effect 0.15 的 practical thresholds？
4. 是否同意 canonical non-fulltrack target 只由 B2/B6 判定，B5 僅能標為 secondary？
5. 是否同意先進行 no-GPU Gate 1b amendment review，而不是直接批准 Gate 2？

## 十、證據索引與 SHA-256

| 產物 | SHA-256 |
|---|---|
| `fulltrack_q3_pq_audit_2026_08_28.json` | `4dc0c9b2a07b788862fbe89d24f486a039d369879ad4a34fcf25fffb54e04e26` |
| `fulltrack_q3_pq_audit_2026_08_28.md` | `b862eaba2a733ad4d3b6eb4cbff8aec5316f32397273fbf029bafb0f6ab68e90` |
| `fulltrack_q3_pq_probe_supplement_2026_08_28.json` | `d532f90754680a684322dccb3203797293cd654ad29485118b990c6bdfb1f323` |
| `designs/fulltrack_q3_pq_bmatrix_v1.json` | `e5d160d1e708fbd06318029b5525fa923a7f0a2219e5818acf133266d0495379` |
| `fulltrack_q3_pq_bmatrix_contract.candidate.json` | `97a5419d430fa6bbe370b8a37272024b765975c6b76c8fbd9771db7e11187dda` |
| Gate 1 approval record | `73167c97725ab90ff2812da3c9724aa94a75a81027cb155843899e5adcaed078` |
| Gate 1 test report | `eea73f89e323baeee6bf72d50c41eceaeca7bd2d137039802ba084f04e0b1ec9` |
| Sealed copy receipt | `6da0a766dce8a1a15908d0facbf0ef68f7732e875aeed6ec0eda3f8375b522b9` |
| Runner | `6ee39d22adce67223e81d32d1617373ef386aa86d13544e21cae867144d33cb4` |
| Per-item scorer | `7725979e90087f404ed34518221f9e5b8ef7a6ff6896f3a951cc21303237cf14` |
| Paired analyzer | `5affb0df2ca056f59077ec7e4c54bbcb71c76a1d4b502396264ca575398b8630` |
| HARN | `c1895c79fe9317c9f895028a1078f7df098da7d7be90906bb0b8828b712f2c5f` |
| Science selftest | `503ea9110af9cbfb196f94015be23df61001d38329279c5126378c9852360e6a` |
| Security selftest | `86a93de5180e7cf5b3b265392e68793f477b9c9274909acb0546563d992d22c1` |
| Queue candidate（未註冊） | `9d68015a655f91ca3ab536332eaf7c6b46797dc2f182e6ef0a1375fb943313c2` |
| Security receipt v1 | `ef640486ecb9bba64095e2ce4cf46cdd0d939c8ac2ef0b30d489ab809640f8f1` |
| Security receipt v2 | `a09f33abeeaeb8b3e0a44fce465c63eaa0958a7362ac2ac21fdd346d5b58c3ec` |
| Security receipt v3 delta | `e6af56290b66dbc93625d1aff2cd7eb41b85f61367eb8d9f2320609561d8d7b4` |

完整 machine-readable audit、probe supplement、Plan 與 Gate 1 test report 是本報告的權威證據；本報告若與其不一致，以凍結且雜湊綁定的原始產物為準。
