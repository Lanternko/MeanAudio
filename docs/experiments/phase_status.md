# Phase 狀態總表

> Phase 編號作內部追蹤用；對外報告和論文使用描述性名稱（`資料集-Caption策略-Q信號`）。
>
> Caption 2.0 (phase8_c2p0) 系列不走 4-token，見下方「Caption 2.0 (phase8_c2p0) arm 命名」。

---

## 4-token paper-facing 命名（2026-05-08 統一）

```
{Caption}-{Sel}-{Q}              ← 預設 eval = MC（MusicCaps human captions）
{Caption}-{Sel}-{Q} (eval_token) ← 非預設 eval 加括號
```

| Token | 取值 |
|---|---|
| Caption | `LP` (LP-MusicCaps captioning model) / `Qwen` (Qwen2.5-Omni-3B 5-task) |
| Sel | `Rnd` (seed=42 static random) / `BC` (BestConsensus) / `Multi` (multi_cap=True per-iter random) |
| Q | `Q` (use_q_conditioning=true, mean_sim bin) / `NoQ` |
| Eval (括號內) | `JM` (Jamendo seed42 + LP captions) / `JMQ` (JM + Qwen captions) / `MCQ` (MC + Qwen captions, 未生成) |

### Phase 編號 → 4-token 對照速查

| Phase 內部 | 4-token 名 | MC CLAP |
|---|---|---|
| Phase 4 V2 | LP-BC-NoQ | 0.191 |
| Phase 7 V1 | LP-Rnd-Q | **0.198** (歷史最佳) |
| Phase 8 | LP-Rnd-NoQ | 0.185 |
| Phase 9 V1 bugfix | LP-Multi-NoQ | ~~0.065~~ **INVALID: cache misaligned** |
| Phase 9 V2 bugfix | LP-Multi-Q | ~~0.040~~ **INVALID: cache misaligned** |
| Phase 9.5 V1 | Qwen-Multi-NoQ | ~~0.061~~ **INVALID: same cache-writer bug** |
| Phase 9.5 V2 | Qwen-Multi-Q | (SKIPPED) |
| P8-Qwen | Qwen-Rnd-NoQ | 0.061 |
| P7V1-Qwen | Qwen-Rnd-Q | 0.069 |
| P4V2-Qwen | Qwen-BC-NoQ | 0.061 |

| Phase（內部） | 對外名稱 | 核心改動 | 狀態 |
|--------------|---------|---------|------|
| Phase 4 V2 | `JamendoFull-BestConsensus-NoQ` | 基礎 MeanFlow 兩階段訓練 | ✅ Baseline（歷史參考） |
| Phase 5 V1 | `JamendoHalf-BestConsensus-NoQ-HardFilter` | 117K 硬過濾 | ✅ 完成，退步（資料量 -53%） |
| Phase 5 V2 | `JamendoHalf-BestConsensus-NoQ-Random` | 117K 隨機抽樣 | ✅ 完成，≈ V1（量是問題） |
| Phase 6 V1 | `JamendoFull-BestConsensus-MeanSim-Q-S2Only` | q 只在 Stage 2 | ✅ 完成，效果受限 |
| Phase 6 V2 | `JamendoFull-BestConsensus-MeanSim-Q` | q Stage 1+2 | ✅ 完成 |
| Phase 7 V1 | `JamendoFull-Random-MeanSim-Q` | Caption 隨機選一（seed=42） | ✅ 完成，**目前最佳** |
| Phase 7 V2 | `JamendoFull-CLAPBest-MeanSim-Q` | Caption 取 CLAP 最高 | ✅ 完成，劣於 V1 |
| Phase 7 V3 | `JamendoFull-WorstConsensus-MeanSim-Q` | Caption 取最低共識 | ✅ 完成，≈ V1 |
| Phase 8 | `JamendoFull-Random-NoQ` | 無 q conditioning（消融） | ✅ 完成，q embedding 有獨立貢獻 |
| Phase 8 V2 | `JamendoFull-Random-AudioboxPQ-Q` | q 信號改用 Audiobox PQ | ✅ 完成，劣於 Phase 7 V1 |
| Phase 8 V3 | `JamendoFull-Random-CLAP-Q` | q 信號改用 audio-text CLAP sim | ✅ 完成，全面退步（信號語義錯誤） |
| ~~Phase 8 V4 (舊)~~ | `JamendoFull-Qwen2Audio-MeanSim-Q` | Caption 換用 Qwen2-Audio-7B | ❌ 廢棄（僅1 cap/clip，不支援 true random）→ V4 名額重用給 PromptConsistency 實驗 |
| **Phase 8 V4** | `JamendoFull-Random-PromptConsistency-NoQ` | mean_similarity raw float 寫進 caption prefix `[consistency=X.XX]`，訊號全走 text encoder（QA-MDT 風格），不依賴 q_embed。eval inference prefix 固定 `0.90`（in-support，median–p90 區間） | ✅ **完成 2026-04-27**。**Natural-ref CLAP**：MusicCaps 0.0571 / Jamendo seed42 0.0591（vs P8 NoQ baseline 0.1851 / 0.1986），AES 僅小跌（PC 甚至略升）。⚠️ 這是 **natural-ref**（metric tsv 用原始未 prefix caption）— 只能說 cross-format alignment 弱，不等於 prompt-following 失敗。Prompt-following（prefixed-ref CLAP）+ dual-ref backfill 排在 priority queue（Codex Round 2 P1 驅動）。Working hypotheses（行為類似 P9 multi-cap 但 mechanism 未證）見下方分析段 |
| **Phase 8 V4 Q** | `JamendoFull-Random-PromptConsistency-S2OnlyQ` | 同 P8 V4（保留 `[consistency=X.XX]` prefix），但 S2 開 `use_q_conditioning=true`（複用 P8 V4 S1 NoQ ckpt）。測試問題：text prefix + q_embed 雙 pathway 並存能否救回 semantic CLAP？ | ✅ **完成 2026-04-27**（dual-ref）。**MusicCaps**：q=6 prefixed_ref **0.0626** / natural_ref 0.0562；q=9 prefixed_ref 0.0598 / natural_ref 0.0539。**Jamendo seed42**：q=6 prefixed_ref 0.0450 / natural_ref 0.0447；q=9 prefixed_ref 0.0417 / natural_ref 0.0411。**讀數**：(1) prefixed_ref 一致高於 natural_ref（MC ~+11%、JM ~+1%）→ 模型有在 follow prompt format；(2) 加 Q pathway 沒救回 CLAP（vs P7 V1 ~0.20 仍差 3-4×）；(3) MC q=6 > q=9（P7 V1 是 flat，這裡不同）。⚠️ caveat：S1 沒看過 Q → S2-only Q regime（half-Q 等級劣化已知）|
| Phase 9 V1 (buggy) | `JamendoFull-TrueRandom-NoQ` | LP-MusicCaps 5 caps 動態採樣，無 Q | ❌ 廢棄（帶 q=9→10 bug、undrop 別名 bug，Jamendo CLAP 0.0260 崩盤）|
| **Phase 9 V1 bugfix** | 同上（修 bug 後）| 修 networks.py q=10 + runner_meanflow.py undrop clone | ❌ **結果失效 2026-07-16**：0.0650 checkpoint 使用全量 audio–caption 錯配 cache；只保留為 corrupted-data artifact，不得比較 multi-cap 效果 |
| Phase 9 V2 (half Q) | `JamendoFull-TrueRandom-MeanSim-Q` | 同 V1 + Q=pairwise MeanSim of 5 caps | ❌ 廢棄於 iter 31k（發現 runner_flowmatching.py 沒讀 q；artifact 保留為 `phase9_v2_s1noq_s2q_partial_*`）|
| **Phase 9 V2 bugfix** | 同上（真 Q end-to-end） | 額外修 runner_flowmatching.py 6 處傳 q | ❌ **結果失效 2026-07-16**：0.0403 同樣使用全量 audio–caption 錯配 cache；q sweep 只描述 corrupted-data checkpoint，不支撐 aggregate-q 或 multi-cap claim |
| **Phase 9.5 V1** | `JamendoFull-QwenOmni-TrueRandom-NoQ` | Qwen2.5-Omni-3B 5 task caps，從零 S1+S2 | ❌ **結果失效 2026-07-16**：由同一舊 writer 生成 sequential multi-cap cache，Qwen captions 也與 mapped audio 全量錯配；0.0609/steering 只保留為 artifact |
| Phase 9.5 V2 | `JamendoFull-QwenOmni-TrueRandom-MeanSim-Q` | 同上 + Q=pairwise MeanSim of 5 task caps | ❌ **SKIP 2026-05-04**；原 launch gate 引用了失效的 P9 V1/P9.5 V1 數字，不能再解讀成 Q variant 必然失敗 |
| **P8-Qwen** | `JamendoFull-QwenOmni-Random-NoQ` (single-cap) | Qwen 5 caps random pick (seed=42, static), single-cap, NoQ | ✅ **完成 2026-05-06**。MC CLAP **0.0611**, JM s42 0.0582, PE-AV peav **−0.038**, steering max 0.120。有效 observation：Qwen single-cap 本身也呈低 prompt conditioning；因 P9.5 multi-cap 對照失效，不再解讀為「拿掉 multi-cap 沒救回」 |
| **P7V1-Qwen** | `JamendoFull-QwenOmni-Random-MeanSim-Q` (single-cap) | Qwen single-cap random + Qwen-local mean_sim Q | ✅ **完成 2026-05-07**。MC CLAP q=6 0.0687 / q=9 0.0686, JM s42 q=9 0.0599, PE-AV −0.038, steering max 0.057。**加 Q 也救不回**；Qwen-local q sweep flat (q=6 ≈ q=9) |
| **P4V2-Qwen** | `JamendoFull-QwenOmni-BestConsensus-NoQ` (single-cap) | Qwen single-cap BestConsensus (argmax of pairwise mean_sim row), NoQ | ✅ **完成 2026-05-08**。MC CLAP **0.0611**, JM s42 0.0596。BestConsensus 選法對 collapse 無影響 — 加入 +0.020 Qwen-prompt boost cluster（第 7 個 collapsed 模型）|

> 完整 Qwen rerun 三組對照與翻盤的 paper-narrative 修正見 `docs/experiments/history/phase8/qwen_rerun_summary.md`。

## Caption 2.0 (phase8_c2p0) arm 命名（2026-08-26 統一）

> 這組 arm 不走上面的 4-token 規則（那套是 LP/Qwen collapse 時代的）。
> Caption 2.0 = Qwen2.5-Omni-3B first-10s `multisent_max160_stop_clean_v1`，
> **per-segment** caption（不是整軌一條）。

```
{Slot 來源} {選法} {scale}          ← NoQ 是預設，不寫；有 Q 才標
```

| Token | 意義 |
|---|---|
| `slot0` / `slot1` / `slot2` / `slot3` | 第幾條 captioner 取樣。`slot0` = base `phase8_qwen_caption10s_multisent_train.tsv`（磁碟上沒有 `slot0_train.tsv` 這個檔） |
| `013` | slot **0/1/3** 三條組成的 stacked overlay pool（`/home/kojiek/text_overlays/`，每個 npz `(3, 77, 1024)`） |
| `fulltrack` | **不是 Caption 2.0**。整軌一條 Qwen caption 複製給該軌所有 segment 的舊語料 |
| `true random` / `fake random` | 同一個 013 pool 的取槽方式：true = `multi_cap=True`（每 epoch 依 clip_id hash 重抽）、fake = `multi_cap=False`（每列固定槽） |
| `bestof3` / `worstof3` | 013 pool 內依 CLAP 選最高 / 最低 |
| `q3` / `q5` | q bucket 數（`q3` → q∈{0,5,9}，`q5` → q∈{0,2,5,7,9}）。**出現即代表 `use_q_conditioning=true`** |
| `quarter` / `full` | quarter = S1 100k + S2 50k；full = S1 400k + S2 200k |

### ⚠️ 舊寫法 `k3` / `k5` 已停用（歧義）

`K` 在不同 arm 是兩件事，混用會推出錯誤結論：

- `k3_balanced` / `k5_balanced` → K = **q bucket 數**，`use_q_conditioning=true` → 改寫 `q3` / `q5`
- `k3_true_random` / `k3_fake_random` → K = **caption 槽數**，`use_q_conditioning=False` → 改寫 `013 true random` / `013 fake random`

### 正名對照（本週 CFG0 / MF25 / MusicCaps n=5521）

| 正名 | 實際 exp id | MC CLAP |
|---|---|---|
| `slot0 q5 full` | `phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced` | **0.2174** |
| `slot0 full` | `phase8_qwen_caption10s_multisent_noq_full_stage2_200000` | 0.2149 |
| `slot0 q3 full` | `phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced` | 0.2145 |
| `013 bestof3 quarter` | `phase8_qwen_caption2p0_bestof3_noq_quarter` | 0.2129 |
| `fair013 bestof3 quarter` | `phase8_qwen_caption2p0_fair013_bestof3_noq_quarter` | 0.2114 |
| `fair013 worstof3 full` | `phase8_qwen_caption2p0_fair013_worstof3_noq_full` | 0.2109 |
| `slot1 quarter` | `phase8_qwen_caption2p0_slot1_noq_quarter` | 0.2047 |
| `slot0 quarter` | `phase8_qwen_caption10s_multisent_noq_quarter` | 0.2029 |
| `slot2 quarter` | `phase8_qwen_caption2p0_slot2_noq_quarter` | 0.2017 |
| `013 true random quarter` | `phase8_qwen_caption2p0_k3_true_random_noq_quarter` | 0.2013 |
| `013 fake random quarter` | `phase8_qwen_caption2p0_k3_fake_random_noq_quarter` | 0.2005 |
| `fair013 worstof3 quarter` | `phase8_qwen_caption2p0_fair013_worstof3_noq_quarter` | 0.1985 |
| `fair013 q3 quarter` | `phase8_qwen_caption2p0_fair013_k3_quarter` | 0.1966 |
| `013 worstof3 quarter` | `phase8_qwen_caption2p0_worstof3_noq_quarter` | 0.1957 |
| `qwen3cap q3 quarter` | `phase8_qwen_caption2p0_qwen3cap_k3_quarter` | 0.1894 |
| **`fulltrack q3 full`** | `phase8_qwen_s2q_from_noq_full_k3_balanced` | 0.1821 |

**anchor**：`rmatched s2 mf25 cfg0.5` = 0.2157（n=5521）。各 REPORT.json 的
`fair_compare_anchor` 欄把它寫成 `caption2p0_s2_mf25_cfg0`，**那是錯的** —
磁碟上產出 0.2157 的是 `rmatched_s1_s2_steps_cfg_matrix_seed14159265_s2_mf25_cfg0p5`，
不同 corpus。真正的 caption2p0 S2 是 `caption_granularity_..._caption2p0_s2_mf25_cfg4p5`
= 0.2419 @ cfg4.5（換協定不可比）。

### `fulltrack q3 full` 不是 slot（2026-08-26 更正）

`phase8_qwen_s2q_from_noq_full_k3_balanced`（0.1821）與
`phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced`（0.2145）只差 exp id 裡的
`caption2p0` 一個 token，但語料完全不同：

| | `fulltrack q3 full` | `slot0 q3 full` |
|---|---|---|
| 訓練 TSV | `phase8_qwen_meansim_k3_balanced.tsv` | `phase8_caption2p0_k3_balanced_train.tsv` |
| caption 粒度 | 整軌一條，複製給該軌所有 segment | per-segment |
| 訓練日 | 2026-08-02 | 2026-08-24 |
| S1 來源 | official_matched NoQ 400k | caption10s_multisent NoQ 400k |

500 列 sha256 比對：`k3_balanced` / `k5_balanced` 的 caption **500/500 等同 slot0**，
只有 `q_level` 欄不同；`fulltrack` 0/500。

- ⚠️ 它掛 slot 編號會誤導 —— slot0/1/2/3 都是 Caption 2.0 的不同 captioner 取樣，
  fulltrack 不在那個維度上。
- ⚠️ 綁它的 `docs/experiments/caption2p0_k3_full_cfg0_eval_contract.json`
  `experiment_id` 寫成 `phase8-caption2p0-k3-slot012-full-cfg0-eval`，
  與該 checkpoint 訓練 log 的 TSV 矛盾 —— **contract 的 provenance 標錯，尚未修**。

### 語料保存狀態（2026-08-26 查核）

- `/mnt/HDD/kojiek/phase8_qwen_official_matched_npz` 於 **2026-08-22 被整批原地覆寫**
  成 Caption 2.0 text features（抽驗 `33.npz` 的 `caption_sha256` = c2p0 TSV row0，
  ≠ fulltrack TSV row0）。`fulltrack q3 full` 訓於 08-02，在覆寫**之前**，
  數字有效 —— 但**該語料已不在磁碟上，那個 run 無法從現有檔案重現**。
- 該目錄 owner 是 `admin123:admin123`、權限 `drwxrwxrwx`（共用機器上他人可寫）。
- 這批 run 的 `require_text_overlay` 皆為 `False`，等於
  `extracted_audio.py::_check_caption_binding()` 那道 TSV↔NPZ caption 對齊守門是關的
  —— 就是當初為防 Phase 9 錯配寫的。時序上這次躲過，但 08-22 之後任何拿舊 TSV 的重跑
  都會靜默訓在錯配 caption 上且無警告。**建議後續 full-scale run 一律開 `True`**（未執行）。
- `/home/kojiek/text_overlays/` 現只剩 `true_random/` 與 `_index/`。`fake_random/` 與
  `worst013/` 是 **2026-08-26 刻意刪除的**（查出兩者是 true_random slot 的重新編碼複本，
  251,599/251,599 全可重建，152 GB 純重複），保留 `_index/*.slot_index.tsv` 索引即可還原。
  ⚠️ 重建是**數值等價非逐位元一致**（相對誤差 median 2e-6 / max 8e-5，float32 re-encode 噪音），
  不能宣稱與舊 run bit-identical。詳見 `docs/experiments/text_overlay_dedup_2026_08_26.json`。

## Phase 9 NPZ 前處理狀態（歷史；2026-07-16 判定失效）

- `gen_multicap_npz.py` 已跑完，iter 6243 崩潰原因為 `~/phase9_multicap_npz/990.npz` 和 `1218.npz` 缺 `text_features_c`
- 已透過 `gen_multicap_npz.py --resume` 重新生成，251,599/251,599 齊全
- `train_pipeline_phase9_v1.sh` 當時加入的 pre-flight 只驗 schema，**無法驗證 pairing**。

### ⚠️ 2026-07-16 audit：Phase 9 true-random 結果因 audio--caption mapping 錯配而失效

- 歷史 `gen_multicap_npz.py` 以 TSV row `i` 直接讀取
  `src_npz/i.npz`；正確 audio latent 必須經 `npz_cache_train.txt` 映射
  （例如 row 0 → `33.npz`，不是 `0.npz`）。
- 全量 audit：sequential filename 與 canonical mapping 相同者為
  **0 / 251,599**。因此 Phase 9 cache 的五個 captions 與 audio
  `mean/std` 系統性錯配。
- 舊 validator 只檢查 count / index / file size / keys / shapes，沒有檢查
  TSV ID ↔ captions ↔ audio latent alignment，所以 preflight 通過不代表
  pairing 正確。
- Phase 9 V1 bugfix 的 0.0650 仍重用受影響的 S1 checkpoint 和同一份錯配
  cache；**不得再用此數字主張 true random / all-five-caption training 有害**。
- Phase 9.5 V1 的 Qwen multi-cap cache 由同一 writer 生成，且 pipeline 未傳
  `gt_cache`、依 row index 載入；0.0609 也失效。有效的 Qwen single-cap
  controls 不受這項特定 multi-cap cache bug 影響。
- P0 TODO：按 `npz_cache_train.txt` 重建含 attention mask 的 multi-caption
  cache，加入 mapped `mean/std` exact-equality audit，使用修正後 NoQ/q-null
  與 CFG code，S1+S2 從零重訓，並以同 code/steps 的 clean static-random
  baseline 比較。詳細 checklist 見
  `docs/reviews/ismir2026-487-promptcc/CORRECTNESS_VALIDATION_PLAN.md`。

**2026-07-16 tooling fix**：`gen_multicap_npz.py` 現在強制要求
`--gt-cache`，source 與 output 均使用 canonical filename，並寫入綁定
TSV id / filename / ordered-caption SHA-256 的 `MANIFEST.tsv`；
每個 NPZ 也內嵌 `clip_id / row_index / caption_sha256`；
`validate_multicap_npz.py` 預設全量驗證 provenance、manifest、caption membership，以及
output 對 mapped source 的 `mean/std` exact equality。舊 cache 沒有 v2 manifest，
不能通過新 pre-flight。

## Phase 9 V1/V2 bugfix 核心發現（2026-04-20/21）

**三個結構性 bug（Codex 抓到兩個關鍵）**：
1. `networks.py:526/558` MeanAudio q=None 填 9（應為 10 null token）→ `use_q_conditioning=False` 實驗 train/eval mismatch。Codex 2026-04-19 發現。
2. `runner_meanflow.py:238-239/268-269` `text_f_undrop = text_f` 是別名不是 clone → in-place null mask 污染 CFG target。Claude 2026-04-19 獨立發現。
3. `runner_flowmatching.py` 完全沒讀 q_level、沒傳 q 到 FluxAudio → 所有 Phase 6+「+Q」實驗 S1 都沒訓 q_embed[0-9]。Codex 2026-04-20 發現。已修 6 處（L224/252/262/285/307-309/414-416）。

**歷史 artifact（不得用於方法比較）**：
- P9 V1 0.0650 與 P9 V2 0.0403 是在系統性錯配 audio–caption pairing 上量到的行為。
- 這些數字可用來診斷 corrupted-data checkpoint，不能用來估計 multi-cap、true-random 或 Q 的效果。

**Codex 2026-04-21 警告的 confound**：
- V2 比歷史 Phase 7 V1 (0.1975) 差，但那是 half-Q；V2 是真 full Q → 混了 (a) multi_cap 效應、(b) full Q vs half Q、(c) q=9 vs 最適 q 三個變量
- 不能直接下「multi_cap 本質性不適合」的定論

**已完成（2026-04-21）**：
1. **P9 V2 q sweep**：q=6/7/8/9 皆 flat（CLAP 0.0402–0.0417）→ P9 V2 failure 不能歸因於 q=9 選錯
2. **P7 V1 q-sweep**（既有 checkpoint）：q=0/3/6/9 顯示 support-set gating — q=0/3 OOD 區 CLAP ≤ 0.045，q=6/9 in-support 區 CLAP ~0.197 等價。Q 表現為 coarse regime marker，非 ordinal quality controller

**已完成（2026-04-22）**：
3. **Phase 7 V1 full-Q control rerun**（~36 hr wall clock）：乾淨 implementation (S1 q-passing fix + S2 text_f_undrop clone fix) 下的 full-Q E2E。全 5 eval 一致低於歷史 P7 V1 best ~8-12% CLAP：
   - Jamendo q=6: 0.1816 vs 0.1980（−8.3%）
   - Jamendo q=9: 0.1799 vs 0.1984（−9.3%）
   - Jamendo native_q: 0.1801 vs 0.1977（−8.9%）
   - MusicCaps q=9: **0.1748 vs 0.1975（−11.5%）**
   - MusicCaps q=6: 0.1759（歷史無直接對照）
   - support-set gating 行為 replicate（q=6/9/native_q 內部差 ≤ 0.002）

**這個 control 的活躍 implementation 差異只有 2 個**（非 3 個）：
- S1 runner_flowmatching q-passing fix（S1 現在真訓 q_embed[0-9]）
- S2 runner_meanflow `text_f_undrop.clone()` fix（CFG target 不再被污染）
- `networks.py q=None→10` fix 不活躍（train+eval 都用顯式 q_level）

### Puzzle — apparent tension with historical P6 V1 vs V2

Historical P6 V2 outperformed P6 V1, but this should not be interpreted as evidence that Stage 1 successfully trained q embeddings. At that time, the runner_flowmatching q-passing bug was still present, so P6 V2 tested the presence of a q_embed layer in the Stage 1 architecture, not effective Stage 1 q learning. The current P7 full-Q rerun shares the former but differs in two active respects: Stage 1 q embeddings are now actually trained, and the Stage 2 text_f_undrop alias bug is fixed. Therefore, the current drop should not be summarized as "full-Q is harmful"; it remains compatible with at least two unresolved contributors: effective Stage 1 q training, the Stage 2 clone fix, or their interaction.

### 2026-07-16 更正：P9 gap decomposition 撤回

先前把 P9 V2 gap 拆成 clean-implementation penalty 與 ~0.13 P9-specific
residual；由於 P9 training pairing 全量錯配，後者不是合法的 multi-cap
effect estimate。只有獨立的 clean P7 full-Q / S2-only ablation 結論仍有效。

### ✅ 已解決（2026-04-24 ablation chain 完整後）

- ~~「clone fix 造成 drop」~~ → **已 falsify**：s2only 5/5 eval ≈ historical，clone fix 非主因
- ~~「pseudo-EMA bootstrap 膨脹 ema_final」~~ → **已 falsify**：兩實驗 EMA gap 一致（~13-14%），結構性現象
- **現在可以寫**：`The primary remaining contributor is Stage 1 effective q training.`

### 仍不能寫的 strong claims

- 「full-Q 本身有代價」/ `S1 q training 本質上有害` — 有代價是觀察，mechanism 未證
- 「S1 q-training 造成 drop」（mechanism claim）— 只能說 primary remaining contributor，不能說 causation
- 「multi-cap 本質不適合 MeanAudio」— clean all-five-caption rerun 尚未完成，現階段沒有有效 LP-MC 對照可支持

### Confound 記錄

- **A. gt_cache / TSV alignment**：❌ 2026-07-16 推翻。training loader 的 mapping 設定不等於 cache writer pairing 正確；舊 writer 直接讀 `i.npz`，造成 0/251,599 canonical matches。
- **B. Pseudo EMA bootstrap**：僅適用 Clean S2 only ablation，不適用 finished full-Q control rerun（後者是從零訓 S1）。
- **C. Eval pipeline 版本**：歷史 (Mar 2026) 與 rerun (Apr 2026) 都用當前 eval 流程 + num_samples=2048 metric。顯式驗證 TODO（若有疑問可跑歷史 ckpt eval 驗證是否還得 0.1984）。

### ✅ 完成（2026-04-23）

4. **Clean S2 only ablation** (`phase7_v1_s2only_ablation`, tmux `p7v1_s2only`)：用歷史 P7 V1 S1 weights (wrapped into load-compatible pseudo training-state ckpt) + 只重訓 S2 with clone fix + 5 eval（Jamendo q=6/q=9/native_q + MusicCaps q=6/q=9）。

   **EMA final 結果（5/5 eval 完成）**：
   | Eval | Historical | fullq_control | s2only | s2only Δ vs hist |
   |---|---|---|---|---|
   | Jamendo q=6 CLAP | 0.1980 | 0.1816 | **0.2008** | +0.0028 (+1.4%) |
   | Jamendo q=9 CLAP | 0.1984 | 0.1799 | **0.1993** | +0.0009 (+0.5%) |
   | Jamendo native_q CLAP | 0.1977 | 0.1801 | **0.1995** | +0.0018 (+0.9%) |
   | MusicCaps q=6 CLAP | — | 0.1759 | **0.1981** | — |
   | MusicCaps q=9 CLAP | 0.1975 | 0.1748 | **0.1951** | −0.0024 (−1.2%) |

   **Attribution（5/5 一致訊號）**：s2only ≈ historical across all evals；fullq_control 持續低 ~8-12%。
   **結論**：The Stage 2 text_f_undrop clone fix is not the main driver of the ~8-12% CLAP drop in fullq_control. The primary remaining contributor is Stage 1 effective q training (enabled by the runner_flowmatching q-passing fix).

   - **Pseudo-EMA bootstrap confound**：兩條 ema_models (sigma 0.05 / 0.1) 都從同一份 `_ema_final.pth` 起跑，**非歷史真實雙軌跡**，是 load-compatible approximation，**不是 semantic equivalent**。
   - **Last.pth insurance ✅ 完成（2026-04-24）**：
     | | ema_final q=9 | last.pth q=9 | EMA gap |
     |---|---|---|---|
     | s2only ablation | 0.1993 | 0.1757 | +13.4% |
     | fullq_control | 0.1799 | 0.1575 | +14.2% |
     兩者 EMA-vs-online gap 一致（~13-14%）→ **pseudo-EMA bootstrap confound 排除**，gap 為 S2 訓練結構性現象。

5. **P7 clean control attribution（P9 residual 部分已於 2026-07-16 撤回）**：
   - **General penalty ~0.02 CLAP**：主要來自 S1 effective q training（runner_flowmatching q-passing fix 啟用後）
   - ~~**P9-specific residual ~0.13 CLAP**~~：**無效**；混入全量 audio–caption mapping corruption
   - **S2 clone fix**：不是 fullq_control drop 的主因（已 falsify）
   - **Pseudo-EMA bootstrap**：不影響 ema_final 比較結論（已 falsify）

### Stage 1 PromptCC：結論層級與後續 TODO（2026-07-16 記錄）

**已確認的 observation / attribution**：
- 在目前實作與單次訓練設定下，S1+S2 full-Q 相較於 historical half-Q / clean S2-only，CLAP 一致下降約 8–12%。代表性比較為 MusicCaps q=9：0.1748 vs 0.1951；Jamendo q=9：0.1799 vs 0.1993。
- clean S2-only ablation 回到 historical baseline 附近；S2 `text_f_undrop.clone()` fix 與 pseudo-EMA bootstrap 已排除為主要解釋。因此目前可寫：**Stage 1 effective q training is the primary remaining contributor to the observed drop.**
- q sweep 顯示 q=6–9 幾乎 flat，而低支援 q 值大幅退化；現有證據較支持 q 是 coarse support/data-regime marker，而不是細粒度 ordinal consistency controller。

**目前的 mechanism hypothesis（尚未證明）**：
- S1 從零建立 text–audio alignment；離散、低維且經 global AdaLN 注入所有 blocks 的 q，可能成為比完整 caption 更容易使用的資料分區捷徑，使模型較少依賴文字內容。S2-only 加 q 時，既有 text grounding 已形成，因此 q 較可能只扮演 residual uncertainty cue。
- q 是由同一 audio 的五個 captions 聚合而得，未必描述當次餵入單一 caption 的正確性；這個 aggregate-q / selected-caption mismatch 可能增加 shortcut 或 label-noise 效應。
- 現行 `runner_flowmatching.py` 的 CFG dropout 會將文字換成 empty features，但不會同步將 q 換成 null token 10；因此 text-null 樣本仍保留 target-derived q。這可能鼓勵 q-only prediction，屬最優先驗證的 implementation-level hypothesis。

**TODO（未完成）**：
1. **P0 — tied CFG dropout ablation**：S1 文字被 drop 時同步設 `q=10`，與目前保留真實 q 的版本做 matched comparison。
2. **P1 — clean single-variable rerun**：在相同 code/data/eval 下，從零比較 `S1 NoQ` vs `S1 +Q`；資源允許時使用多個 training seeds，避免把單次 run 差異當作一般性結論。
3. **P2 — shortcut diagnostics**：測試 shuffled-q、delayed/zero-init gated q，並比較 S1 checkpoints 的 prompt steering、conditional–unconditional gap、text-conditioning activation magnitude，以及 q 與 genre/音色/captioner bias 的關聯。
4. **P2 — conditioning granularity**：比較 aggregate-q 與 per-caption confidence/consistency signal，確認問題是否來自 clip-level q 與 selected caption 不匹配。

**論文措辭邊界**：
- ✅ 可寫：「In our current setup, enabling effective prompt-consistency conditioning in Stage 1 is the primary remaining contributor associated with the observed 8–12% CLAP drop, while applying it only in Stage 2 restores the historical performance range.」
- ✅ 可將 shortcut learning / CFG q leakage 寫為與結果一致、待驗證的解釋。
- ❌ 不可寫：「Stage 1 PromptCC 本質上有害」或「shortcut learning 已被證明」。目前缺少 tied-dropout、完全 matched multi-seed 與直接 probe 證據。

## Phase 9 caption responsiveness — corrupted-cache artifact（2026-07-16 更正）

> 下列 probe 數字本身可重現，但 P9 V1/V2 checkpoint 是由全量錯配的
> audio–caption cache 訓練。它們只能顯示「錯配訓練會削弱 steering」，不能再
> 描述成 single-cap vs multi-cap association，也不能當成 multi-cap failure 證據。

**方法**：
- 固定 cfg=0.5, num_steps=1（benchmark-matching）
- 4 個 A/B prompt pair（樂器、人聲、鼓、編制密度），每 pair 同 seed 對打
- 3 seeds × 2 prompts × 4 pairs = 24 檔/model
- 量 `(A-B L2) / (noise L2)` ratio — **noise L2 是同 prompt 不同 seed 的 L2 baseline**
- Probe battery 375-state grid（6 ckpts S1+S2、5 seeds、5 timesteps、10 prompt pairs、4 metrics）determinism check d=0 通過
- P9 V2 q=9 sanity 與 q=8 差距 |Δ| ≤ 0.006，q sweep 結論穩定

**結果 — 4 模型 2x2 分組（A/B same-seed L2 / noise L2）**：

| Model | 01 instr | 02 vocals | 03 drums | 04 density |
|---|---:|---:|---:|---:|
| **P7 V1** (Q, single-cap) | 1.457 | 1.071 | 1.702 | 0.884 |
| **P8** (NoQ, single-cap) | 1.121 | 0.950 | 1.723 | 0.913 |
| **P9 V1** (NoQ, multi-cap) | 0.075 | 0.025 | 0.068 | 0.147 |
| **P9 V2** (Q=9, multi-cap) | 0.015 | 0.012 | 0.021 | 0.056 |

- ratio > 1 ⇒ prompt 效應 > noise 效應（single-cap 組）
- ratio < 0.2 ⇒ P9 corrupted-data checkpoints 中 noise 主導、prompt 微弱
- ~~single-cap vs multi-cap 是行為分界線~~ → **撤回**；資料 pairing corruption 是未控制的主導變量

**可說**：
- Same-seed prompt steering 在歷史 P9 corrupted-cache runs 中很弱
- 這些 P9 checkpoints 不是完全不看 caption；prompt effect 小於 noise effect
- P9 V1 殘留最弱反應維度：density（0.147）與 instrument/drums（~0.07），vocals 最弱（0.025）
- P9 V2 在所有維度比 V1 更弱（0.01-0.06）
- Probe battery 一致：P9 a/c ratio 0.001-0.015 vs P7 0.10-0.21（差 20-200x）；P9 S1→S2 ratio 再跌 4-6x，P7 沒跌

**不能說**：
- ❌ multi-cap 與 conditioning failure 有效相關或有因果關係（training pairs 已知全量錯配）
- ❌ text_cond_proj 梯度被毒、weight 崩壞等 mechanism
- ❌ P9 "unconditional generation"（殘留 ratio 非 0）
- ❌ P9 "完全不看 caption"

**Artifacts**（`eval_output/probe_subjective_v2/`）：
- `p7v1/`、`p8/`、`p9v1/`、`p9v2/`（q=8）、`p9v2_q9/`（sanity）各 24 wav
- `probe_battery_results.json` 3450 條 records

## Phase 9.5 Qwen captioning corpus 狀態（caption 完成；multi-cap training 失效）

**Caption corpus 狀態**：5 slots × 251,599 全部完成，自動 merge →
`phase9_omni_captions.jsonl`（251,599 行 / 182 MB）。JSONL 本身仍可用；
失效的是舊 multi-cap NPZ 中 captions 與 audio statistics 的 pairing。

| Slot | Task framing | 行數 | 狀態 |
|------|-------------|------|------|
| 0 | Writing（詳細自然描述句） | 251,599 | ✅ 4/26 完成 |
| 1 | Summary（壓縮為短句） | 251,599 | ✅ |
| 2 | Paraphrase（豐富詞彙改寫） | 251,599 | ✅ |
| 3 | Attribute（屬性為主） | 251,599 | ✅ |
| 4 | NaturalProse（中性敘述） | 251,599 | ✅ |

> **修正**：先前 docs 記錄的 slot 順序誤標（slot 0 寫成 NaturalProse、slot 4 寫成 Attribute）。
> 實際 PROMPTS 順序見 `gen_qwen25omni_captions.py:55-65`。

- 模型：`Qwen/Qwen2.5-Omni-3B`（Thinker-only，SDPA attention）
- 輸入：`phase7_v1_train.tsv` (251,599 clips) + audio root `/home/hsiehyian/dataset/segments_no_vocals`
- 輸出：
  - 個別：`/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions_slot{0..4}.jsonl`
  - Merge：`/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl`（每行 `{"id":..., "captions":[c0..c4]}`）

### 階段二 resume run 實測（2026-04-28 20:13 → 2026-05-02 12:53，~88h wallclock）

完成的工作量：slot 1 殘餘 183,471 + slots 2/3/4 各 251,599 = **938,268 captions**（約佔總 1,257,995 的 75%）

| 階段 | 速度 (s/batch, batch=20) | 備註 |
|------|------------------------|------|
| GPU 共用 hsiehyian (7.7 GB) | ~10.5 s/it（約 2 caps/s） | slots 1, 2 早期 |
| GPU 獨佔 | ~5 s/it（約 4 caps/s） | hsiehyian 結束後 |

- BATCH_SIZE 從 32 降至 20（避免與 hsiehyian training 共用時 OOM；獨佔時可回 32）
- 一次 silent SIGKILL 崩潰（啟動後 ~20 min），`--resume` 完整恢復無資料遺失
- 全程自動 merge（`--slot all` 跑完 5 slots 後自動觸發 `merge_slots()`）

### n=11 早期 diversity sample（slot 0 vs slot 1，2026-04-25 21:34）

| 評估 | 數量 | 解讀 |
|------|------|------|
| ✅ Consistent (multi-task 真產生 valid 多角度) | 6/11 = 55% | 同 genre / 同 instruments / 同 mood，差別只是 verbosity 或 task focus |
| ⚠️ Mild contradiction / ambiguous | 3/11 = 27% | 同 genre 但 energy 描述偏移（e.g. soft vs vibrant reggae） |
| ❌ Clear contradiction (hallucination) | 2/11 = 18% | 互斥屬性（acoustic vs electric guitar、somber slow vs feel-good upbeat） |

**對 P9.5 訓練的意涵**（Day 4 全 100 筆完整檢查再定論）：
- 82% at-least-同義 → multi-cap 有真正的 diversity 信號（hypothesis 仍成立）
- 18% hallucination 給訓練信號加噪聲，但**不是垃圾資料**（語意仍在，只是樂器/情緒 misjudge）
- **觀察點**：mean_sim 信號可能被「captioner stability」而非「audio difficulty」污染 → 與 `project_mean_sim_interpretation_hypothesis.md` 反向假設可能對應，P9.5 訓完後值得分析 q 分布 vs audio 特徵

詳細設計見 `docs/experiments/history/phase9/phase9_design.md`，Lane A/B/C 排程見 `../meetings/2026-04-18_lane_abc_and_lpmc.md`。

## Phase 8 V4 結果與分析（2026-04-27 完成）

### 數字

| Benchmark | CLAP ↑ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ |
|-----------|--------|------|------|------|------|
| MusicCaps n=5521 | **0.0571** | 5.7708 | 6.5038 | 5.2716 | 6.4163 |
| Jamendo seed42 n=2048 | **0.0591** | 5.7342 | 6.4665 | 5.2631 | 6.3741 |

### 與 P8 NoQ baseline 比較

| 指標 | P8 V4（PromptConsistency-NoQ） | P8（Random-NoQ） | Δ |
|------|-------------------------------|-----------------|---|
| MusicCaps CLAP | 0.0571 | 0.1851 | **−67%** |
| Jamendo CLAP | 0.0591 | 0.1986 | **−70%** |
| MusicCaps CE | 5.771 | 5.913 | −2.4% |
| MusicCaps PQ | 6.416 | 6.544 | −2.0% |
| MusicCaps PC | **5.272** | 4.983 | **+5.8%** ↑ |

### 診斷：低 CLAP / 高 AES 外觀（不再連結 P9 multi-cap）

**Pattern (behavior-level observation)**：CLAP（natural-ref）大跌（~70%）但 AES 僅小跌（CE/PQ −2-3%），PC 反而上升。先前與 P9 V1 連結的解讀已撤回，因 P9 V1 0.0650 來自錯配資料；數值外觀相似不再提供 training-mechanism 證據。

**⚠️ Codex P1 2026-04-27 caveat**：上述 CLAP 0.0571 / 0.0591 是 **natural-ref**（metric tsv = 原始未 prefix caption；generation tsv = 帶 `[consistency=0.90]` prefix）。**這只是 cross-format alignment，不是 prompt-following metric**。要真正測「模型有沒有跟著 prompt 走」必須補 prefixed-ref pass（generation 與 metric 同 TSV）。dual-ref backfill 排在 2026-04-27 priority queue #1.5。

**Working hypotheses（需 embedding/probe evidence 才能升級為 mechanism claim）**：
- H1: `[consistency=X.XX]` 前綴占據 T5 token sequence 前幾個位置，可能影響 text embedding 主方向（**未測 embedding norm/方向變化**）
- H2: 模型可能學到「consistency 數值 → 音質/風格」捷徑而 underweight 語義 caption（**未做 attention/probe 驗證**）
- ~~H3: 與 multi-cap collapse 模式相似~~：**撤回**；參照 checkpoint 的 audio–caption pairing 已知失效

**對 QA-MDT 類方法的觀察**（design suggestion，非 finding）：
- QA-MDT 原始設計用離散 quality token（`[high]`/`[medium]`/`[low]`），semantic caption 完整保留
- P8 V4 用 raw float prefix `[consistency=0.83]`，T5 encode 成連續向量
- 一個可能的設計改進：quality token 接 special-vocab embedding 或走 separate q_embed pathway（與 P7 V1 同層）

### 暫定結論（pending dual-ref result + ablation chain）

**已證（behavior-level）**：在 natural-ref CLAP 上，P8 V4 NoQ 的 prompt prefix 設計顯著低於 P8 NoQ baseline。AES 維持/PC 微升。

**尚不能證**：
- ❌ 「prefix 主導 T5」（mechanism claim — 需 embedding 變化證據）
- ❌ 「shortcut learning」（需 probe / attention evidence）
- ❌ 「複製 multi-cap text conditioning collapse」（行為相似 ≠ 同機制）

**待 dual-ref + 後續 ablation 補完**：
- ~~prefixed-ref CLAP（P8 V4 Q）~~ ✅ **完成 2026-04-27 14:25**：MC q=6 prefixed=0.0626 / natural=0.0562；q=9 prefixed=0.0598 / natural=0.0539。**Δ ~+11% on MC**, **~+1% on JM**。Prefixed_ref 高於 natural_ref → 模型有 follow prompt format，只是與 natural caption 對齊度低
- prefixed-ref backfill for P8 V4 NoQ p=0.90 baseline — 跑中（priority queue #1.5）
- prefixed-ref + natural-ref for P8 V4 NoQ p=1.00 — 跑中（priority queue #2）
- P7 V1 q=0..9 q-sweep on MusicCaps — 跑中（priority queue #3）

### Dual-ref 結果讀解（P8 V4 Q，2026-04-27）

|  | MC q=6 | MC q=9 | JM q=6 | JM q=9 |
|---|---|---|---|---|
| prefixed_ref CLAP | **0.0626** | 0.0598 | 0.0450 | 0.0417 |
| natural_ref CLAP  | 0.0562 | 0.0539 | 0.0447 | 0.0411 |
| Δ (P>N) | +0.0064 (+11%) | +0.0059 (+11%) | +0.0003 (+1%) | +0.0006 (+1%) |
| vs P7 V1 baseline | × 3.2 較低 | × 3.3 較低 | × 4.4 較低 | × 4.7 較低 |

**可說（observation）**：
- Prefixed_ref 一致高於 natural_ref，**MC 上幅度顯著（~11%）**、JM 上接近 noise（~1%）
- 即使用 prefixed_ref（對 P8 V4 Q 較有利的 metric），CLAP 仍遠低於 P7 V1（~0.20）→ **加 q_embed pathway 無法補回 prefix 設計造成的 CLAP 缺口**
- q=6 略高於 q=9（與 P7 V1 在 in-support 區的 flat 行為不同；可能 S2-only Q regime 的副作用）

**不能說**：
- ❌ 「prefix 沒影響語意」— prefixed_ref 仍遠低於 P7 V1 baseline，只是 prefixed 比 natural 多 ~11%
- ❌ 「q_embed pathway 無效」— 這次 S2-only Q + 跟 prefix 共用 text encoder 是混在一起的（confound：S2-only regime + prefix 同存）
- ❌ MC/JM 差異反映「prefix vs no-prefix」— JM 也用 prefix，差異更可能是 cross-domain vs in-domain

### 後續 ablation：prefix value sweep（2026-04-27 排隊中）

**P8 V4 NoQ + `[consistency=1.00]` eval**（用既有 ckpt，不重訓）：
- 動機：0.90 是 in-support p90，是否 push 到 1.00 邊界值會更好/更差？
- caveat：1.00 是 OOD-edge（PM 2026-04-26 警告原則），這是 paper-completeness ablation
- TSV：`~/eval_tsvs_p100/phase8_v4_{musiccaps_test,jamendo_seed42_2048}_p100.tsv`
- Pipeline：`~/MeanAudio/eval_p8v4_noq_p100.sh`
- 排隊：等 P8 V4 Q (S2 + 4-eval) 跑完才插隊（避免 GPU 衝突）
- ETA：P8 V4 Q ~14:15 完 → p100 eval ~14:35 完

### 平行進行：P8 V4 + Q 變體（2026-04-27）

`phase8_v4_q_stage2_200000`（tmux `p8v4q`）：複用 P8 V4 S1（NoQ ckpt），S2 開 `use_q_conditioning=true`。測試假說：text prefix + q_embed 雙 pathway 並存能否救回 semantic CLAP？S2 from scratch q_embed → S2-only Q regime，已知 half-Q 等級劣化。Eval q=6 + q=9 × MusicCaps + Jamendo seed42 (4 跑)。S2 速度 0.110 s/iter, ETA 13:15 訓練完、~14:15 全 eval 完。

## P7 V1 完整 q-sweep on MusicCaps（2026-04-27 完成）

對 P7 V1 既有 EMA（Mar 28 訓的 historical baseline）跑 q=0..9 完整 sweep，補齊歷史只測過 q=0/3/6/9 中間值（q=1/2/4/5/7/8）。

### 完整曲線

| q | CLAP | CE | CU | PC | PQ | 區段 |
|---|------|-----|-----|-----|-----|------|
| 0 | 0.0446 | 3.32 | 4.49 | 4.19 | 4.90 | OOD |
| 1 | 0.0481 | 3.66 | 4.85 | 4.31 | 5.25 | OOD |
| 2 | 0.0247 | 3.33 | 4.41 | 4.11 | 4.86 | OOD |
| 3 | **−0.0113** | 2.74 | 3.84 | 3.97 | 4.34 | OOD（負值！）|
| 4 | 0.0591 | 3.52 | 4.71 | 4.30 | 5.18 | **OOD-edge** |
| **5** | **0.1871** | 5.93 | 6.71 | 4.69 | 6.64 | **支援集邊界** |
| 6 | 0.1960 | 6.00 | 6.79 | 4.69 | 6.65 | in-support |
| 7 | 0.1973 | 5.99 | 6.79 | 4.71 | 6.62 | in-support |
| 8 | 0.1968 | 5.95 | 6.77 | 4.66 | 6.60 | in-support |
| **9** | **0.1975** | 6.02 | 6.81 | 4.67 | 6.68 | in-support |

### 重要驗證：q=9 = 0.1975 與歷史完全一致（Δ = 0.00%）

之前 docs 記錄的 P7 V1 MusicCaps q=9 baseline = **0.1975**。本次 rerun **逐位數匹配** → eval pipeline / model ckpt / code path 全部 reproducible，無 silent regression。

### Support-set gating 再確認 + 精確邊界

歷史 memory `reference_p7v1_q_support_gating_2026_04_21.md` 的觀察：「q=0/3 OOD ≤ 0.045，q=6/9 in-support ~0.197」全部還原。**新資訊：邊界在 q=4↔q=5**：
- q=4 = 0.0591（OOD edge）
- q=5 = 0.1871（已 in-support，但低於 q=6-9 平台 ~0.0089）
- q=6/7/8/9 = 0.1960/0.1973/0.1968/0.1975（極窄區間 0.0015）

**訓練分布解釋**（從 P8 V4 train.tsv 看，繼承自 P7 V1）：q=3 僅 1 筆、q=4 僅 312 筆、q=5+ 才開始有實質量。對應曲線：q=0..3 是 random（甚至 q=3 負相關）、q=4 過渡、q=5+ 支援集 plateau。

**論文意義**：Q 不是 ordinal quality controller（值高低不影響輸出細粒度）；它是 **coarse regime marker**，把模型推到「訓過的分布內」或「訓過的分布外」。

## P8 V4 NoQ s=1.00 — PE-AV eval（2026-04-27 完成）

用既有 P8 V4 NoQ s=1.00 audio（priority queue #2 跑出來的）跑 PE-AV，dual-ref 與 CLAP 同方法論。

### 結果（dual-ref）

| Benchmark | Ref | peav_score | t2a R@1/5/10 | a2t R@10 | median rank |
|-----------|-----|-----------|--------------|----------|-------------|
| MusicCaps n=5521 | natural | −0.0378 | 0.018/0.091/0.199% | 0.217% | 2741 |
| MusicCaps n=5521 | prefixed | −0.0416 | 0.018/0.072/0.217% | 0.254% | 2744 |
| Jamendo n=2048 | natural | +0.0073 | 0.098/0.342/0.488% | 0.537% | 1003 |
| Jamendo n=2048 | prefixed | −0.0037 | 0.098/0.293/0.488% | 0.439% | 1009 |

### 對照歷史 P7 V1（n=30 random）

| 指標 | P7 V1 (30 random) | P8 V4 NoQ s=1.00 (full) |
|------|-------------------|---------------------------|
| peav_score | **+0.052** | **−0.04 ~ +0.007** |
| t2a R@10 | 5.4%（30× random）| ~ random baseline |

### 關鍵讀數

1. **PE-AV retrieval 降至 random baseline**：MC 隨機 R@10 = 10/5521 = 0.181%、JM 隨機 = 0.488%，本次測得值與隨機**完全一致**
2. **peav_score MC 為負值（−0.038）**：與 prompt 反相關（vs P7 V1 的 +0.052）
3. **median rank ≈ n/2**：retrieval 完全沒有對齊 signal
4. **prefix 0.90 vs 1.00**：對 PE-AV 也幾乎無影響（與 CLAP 同結論）

### Prompt-following 三 metric 一致 verdict（P7 V1 baseline → P8 V4 NoQ s=1.0）

| Metric | P7 V1 | P8 V4 NoQ s=1.0 | 劣化倍率 |
|--------|-------|------------------|----------|
| CLAP (MC, prefixed_ref) | 0.1975 | 0.0676 | **2.9×** |
| PE-AV peav_score (MC) | +0.052 | −0.038 | **sign flip** |
| PE-AV t2a R@10 (MC) | 5.4% | 0.20% | **27×** |

PE-AV 上劣化幅度遠大於 CLAP（27× vs 2.9×）—— PE-AV 是 fine-grained retrieval，對語意對齊更敏感。**三 metric 一致指向：prefix 設計實質傷害 prompt-following**，且傷害程度比 CLAP 數字所暗示的更嚴重。

## P8 NoQ q-sweep control（2026-04-27/28 完成）

### 背景

Wei-Jaw 建議：Fig.2 加入 P8 NoQ baseline 橫線，並跑 q=5..9 control 確認 random init q_embed 的效應（對照 P7 V1 trained Q curve）。同時發現 P8 NoQ 本身存在 `networks.py q=None→9` bug（null token 訓在 q[9] 而非 q[10]）。

### P8 NoQ q=9 結果 — 揭露 bug

| q | CLAP | 解釋 |
|---|------|------|
| q=5 | 0.0920 | random init (untrained) |
| q=6 | 0.0681 | random init |
| q=7 | 0.0418 | random init |
| q=8 | 0.0709 | random init |
| **q=9** | **0.1907** | **bug 暴露 trained null（本應在 q[10]）** |

q=9 比其他 q 高 2-5×，與 P8 bug 確認（null 訓在 q[9]）完全吻合。q=9 可視為「最接近 bug-free P8 baseline 的 proxy」。

### Bug-free P8 baseline 估計 vs P7 V1

- P8 q=9（exposed trained null）= **0.1907**
- P7 V1 MusicCaps q=9（true trained）= **0.1975**
- 估計缺口：~0.007（3-4%），非歷史所說的 +6.7%
- 若 retrain P8 NoQ bug-free，預期結果 ~0.190 — 仍略低於 P7 V1

### P8 NoQ --no_q baseline rerun（2026-04-28，同 pipeline）

等同「訓練 null 在 q[9]，但 eval 強制走 q[10]（untrained random）」→ train/eval mismatch 的 pipeline-consistent baseline。

### P8 q=10 sanity check

目的：驗證 eval.py 內部 `--no_q ≡ --quality_level 10`（兩者均走 q[10]）。若完全相同，pipeline 一致性確認。

### P8 V4 NoQ q-sweep control（bug-free ckpt，q=5..10，dual-ref）✅ 2026-04-28 完成

針對 P8 V4（2026-04-26 訓練、bug 已修）:

**Preflight**（q_embed row diff, S1→S2）：
- q[0..9] delta_l2 = 0.000000（untouched）✅
- q[10] delta_l2 = 2.268874（TRAINED null）✅

**完整結果（MusicCaps n=5521）**：

| q | prefixed_ref CLAP | natural_ref CLAP | 解釋 |
|---|---|---|---|
| `--no_q` baseline | **+0.0665** | **+0.0571** | 訓練 null token（q[10]） |
| 5 | −0.0306 | −0.0480 | random init，主動干擾 |
| 6 | −0.0042 | −0.0173 | random init，近乎 random |
| 7 | −0.0318 | −0.0397 | random init，主動干擾 |
| 8 | −0.0009 | −0.0182 | random init，接近 zero |
| 9 | +0.0008 | −0.0154 | random init，prefixed 勉強正 |
| **10** | **+0.0665** | **+0.0571** | ✅ 訓練 null = baseline（sanity 通過） |

**關鍵讀數**：
1. **q=10 ≡ --no_q baseline**（兩 ref 完全一致）— `--quality_level 10` 與 `--no_q` pipeline 行為等效，P8 V4 code path 一致
2. **q=5..9 全部 ≤ +0.001**（natural_ref 全負）— random init embedding 主動干擾生成，不只是「沒有幫助」
3. **P8 NoQ qsweep 的 q=9 異常（0.1907）在 P8 V4 不存在**：P8 V4 用 bug-free code，null 正確在 q[10]，q=5..9 都是 random
4. **⚠️ 注意：絕對值不能與 P8 NoQ / P7 V1 直接比較**（P8 V4 有 prefix dominance，CLAP magnitude 不同量級）

**Fig.2 用途**：P8 V4 NoQ control 只作 supplementary inset（不同訓練機制，Y 軸量級不同），不放進 P8 NoQ vs P7 V1 主圖。

### Fig.2 設計（2026-04-28 data 齊全）

**Main figure**（同 Y 軸，同模型系列）：
- P7 V1 trained-Q curve：q=5: 0.1871、q=6: 0.1960、q=7: 0.1973、q=8: 0.1968、q=9: 0.1975
- P8 NoQ horizontal baseline：`--no_q` = 0.1851（pipeline-consistent，注意 train/eval mismatch caveat）
- P8 NoQ random-q curve：q=5: 0.0920、q=6: 0.0681、q=7: 0.0418、q=8: 0.0709、q=9: 0.1907（trained null via bug）

**Supplementary inset**（P8 V4 NoQ，不同 Y 軸，量級 ~0.06 vs ~0.18）：
- Baseline --no_q: prefixed=0.0665, natural=0.0571
- q=5..9 all negative (random init hurts)
- q=10 = baseline (trained null verified)

## TODO：Retrain P8 NoQ bug-free — **DONE 2026-07-16（結果偏離）**

**EXPERIMENT_COMPLETE** · exp：`phase8_bugfix_rerun_stage2_200000` · tmux `p8_clean_baseline`（已結束）

### Timeline
- 01:20 launch · clean NPZ 251,599 驗證通過 · S1 01:32–13:14 · S2 ~13:14–19:03 · MusicCaps gen+metrics ~19:03–19:15
- S1/S2 訓練健康：~0.10–0.105 s/it、loss≈0.986、grad≈1.9–2.1、GPU≈93%、VRAM≈14.8 GB
- ckpt：`exps/phase8_bugfix_rerun_stage1_400000/*_ema_final.pth`、`exps/phase8_bugfix_rerun_stage2_200000/*_ema_final.pth`
- log：`/home/kojiek/logs/phase8_bugfix_rerun_pipeline.log`
- metrics：`eval_output/metrics/phase8_bugfix_rerun_stage2_200000_musiccaps/metrics.txt`
- audio：`eval_output/phase8_bugfix_rerun_stage2_200000_musiccaps/audio/`（5521 flac，RMS 正常、非靜音）

### MusicCaps `--no_q` 結果 vs 預想

| metric | 本 run | 預想 | 歷史髒 baseline | 判定 |
|---|---:|---:|---:|---|
| **CLAP** | **0.0615** | 0.185–0.195（proxy ~0.190） | 0.1851 | 🔴 **嚴重偏離**（<0.17） |
| aes_CE | 5.977 | ~5.91 | 5.91 | ✅ |
| aes_CU | 6.575 | ~6.75 | 6.75 | ✅ 略低 |
| aes_PC | 5.328 | ~4.98 | 4.98 | ✅ 略高 |
| aes_PQ | 6.542 | ~6.54 | 6.54 | ✅ |

### Eval 核對（非路徑/指令錯）
- `eval.py`：`meanaudio_s` + S2 `ema_final` + MusicCaps TSV + `--use_meanflow --num_steps 1 --encoder_name t5_clap --text_c_dim 512 --cfg_strength 0.5 --no_q --full_precision`
- `phase4_eval.py`：5521 clips，CLAP 0 skip；AES 173 batches 完成
- 音訊抽樣：10s @16kHz、mean RMS≈0.12、frac near-zero = 0 → **非 silent / 非空檔**
- AES 正常 + CLAP 崩 → 美學/生成能量 OK，**語意對齊失敗**（類似 P8 V4 natural-ref ~0.057 / EXP-A ~0.061 量級）

### 解讀（暫定，勿當最終論文數字）
- 訓練迴圈表面健康，但 bug-free NoQ clean retrain **未** 重現歷史 0.185 或 q=9 proxy 0.190。
- **不要輕率重訓**；先查：(1) clean NPZ text 與 TSV caption 對齊、(2) `--no_q` / q-embed null 路徑是否仍與 train 一致、(3) 歷史髒 P8 EMA 用**同** eval 指令能否仍得 ~0.185（sanity）、(4) 抽聽 gen audio 是否 prompt-follow。
- 論文 Table 1：此 run **不可** 當 clean NoQ baseline 數字；標記為 failed/outlier 直至根因釐清。

### 設定（實際）
- `USE_Q_CONDITIONING=false` · S1 400k + S2 200k · single-cap · clean NPZ `npz_phase7_clean`
- `networks.py` q=None→10（bug-fix）· batch 8 · lr 1e-4

### 根因與後續（2026-07-17 forensics → 2026-07-18 排程）

見 `docs/experiments/history/phase8/phase8_baseline_forensics_2026_07_17.md`：七月 clean retrain 用了與 Phase-7 TSV ID 幾乎全量錯配的 audio latent；歷史 0.1851 實際吃的是 **extraction catalog 配對 + 訓練時 Q（runner 忽略 `use_q_conditioning=false`）**。

| 實驗 | 狀態 | 單變量 |
|---|---|---|
| `phase8_legacy_repro` | **✅ 完成（2026-07-19）** MusicCaps CLAP **0.1684**（CE 5.36 / PQ 6.49；q=9 + NoMask eval）。vs 歷史 q=9 條件 0.1907 delta −0.022（audit ±0.03 內；量級=S1-effective-q penalty ~0.02）。⚠️ 首次 eval 誤用 `--no_q` 得 0.0134（Q-trained 模型的 q=10=uncond 記號）— pipeline/audit 已修，見 forensics addendum | catalog-matched + **Q=true** + NoMask |
| `phase8_catalog_matched_noq` medium gate | **✅ PASSED（2026-07-19）** `p8_catalog_noq_gate`：100+100+64；S1=`fluxaudio_s` / S2=`meanaudio_s`；`use_q=false` + NoMask；eval `'no_q': True`；short-run CLAP −0.0352（**wiring only**，非收斂指標） | 同 n4096 smoke + **Q=false**；只驗 wiring |
| `phase8_catalog_matched_noq` full | **✅ 完成（2026-07-20）** S1 400k + S2 累計 600k；MusicCaps 5,521/5,521，CLAP **0.1888** / CE 5.7252 / CU 6.4241 / PC 4.8893 / PQ 6.4174；final audit PASSED。單次 AMP grad NaN 為已恢復 overflow，未造成 checkpoint corruption | 同 full cache + **Q=false** + NoMask |
| `phase8_catalog_matched_s2_realq` | **🟢 RUNNING（2026-07-20 12:19 起）** 共用 clean-NoQ S1 400k，只訓 S2 200k with real per-row Q；MusicCaps q9 primary + q6 secondary；tmux `p8_s2_q_ablation` | 相較 0.1888，只新增 S2 real-Q |
| `phase8_catalog_matched_s2_shuffledq` | **⏳ QUEUED** Real-Q final contract 通過後自動接續；seed 424242 只打亂 Q，Q histogram 與所有 audio/text pairing 不變 | Real-Q 的 information control；區分真 Q signal 與額外 embedding/regime token |

**Gate 假陽性修復（2026-07-19）**：首跑 audit 誤查 hydra `training_stage`（此 key 不存在；stage 由 `set_training_stage.py` 切 runner，config 上的 durable signal 是 `model`）。已改為驗 `model ∈ {fluxaudio_s, meanaudio_s}` + `no_q=True` eval log；既有 gate 產物 re-audit → PASSED → scheduler 開 full。

- Gate：`scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq_medium_gate.sh`
- Full：`scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq.sh`
- Scheduler：`scripts/training_pipelines/schedule_catalog_matched_noq_after_legacy.sh`（audit → gate → full）
- 監控 loop：`meanaudio_chain_watch`（60s）+ `meanaudio_repair`（120s 安全自癒）
- 啟動紀錄：`~/logs/phase8_legacy_repro_guard/next_experiment_catalog_matched_noq.json`
- Gate sentinel：`~/logs/phase8_legacy_repro_guard/noq_medium_gate_PASSED.json`
- Audit / gate 失敗時 **不會** 開 full train
- multi-cap clean rebuild 仍 P0 但 HDD 空間不足（~413G need / ~314G free）

---

## Qwen collapse root-cause EXP 系列（2026-05-08 起）

> 完整設計 / 結果 / 機制分析見 `docs/experiments/history/phase8/qwen_collapse_root_cause_2026_05_08.md`

| 實驗 | 介入 | MC CLAP | 假說影響 |
|---|---|---|---|
| EXP-A (LP-MC-destructured) | LP-MC boilerplate prefix 全部移除 | **0.0608** | ✅ H10 confirmed：boilerplate template 是 inductive anchor |
| EXP-B (Qwen-Slot0-Fixed) | Qwen 全部固定 slot 0（移除 5-task variance） | **0.0615** | ❌ H11 falsified：variance 不是 collapse 原因 |
| EXP-C (Qwen+Boilerplate) | Qwen slot-0 + LP-MC prefix string prepend | **0.0580** | ❌ H12 falsified：prefix string alone 不夠 |
| EXP-D2/D3 (activation probe) | eval-only：text projection MLP activation 解剖 | — | ✅ Collapse 在 S1 已發生；MLP ÷7–14 attenuation；weight shrinkage 只 −25% |
| EXP-D4 (projection transplant) | 把 P8 healthy text projection weights 移植至 EXP-A/B/C | −5%~−27% vs original | ✅ Projection collapse = 症狀；joint_blocks 已 co-adapt，transplant 反而更差 |
| EXP-F (50% LP-MC + 50% Qwen) | 50-50 per-audio 混合訓練 | **0.0610** | ❌ G4：50% LP-MC anchor 不足以阻止 collapse |
| **EXP-G (LP-MC S1 → Qwen S2)** | reuse P8 LP-MC S1 ckpt + Qwen slot-0 S2 200K (NoQ) | **0.0679** | ❌ G5：S1 anchor 不能保護 S2 Qwen co-adaptation。PE-AV peav MC **−0.034**（負）、steering 0.068-0.098 全 collapse cluster。完成 2026-05-15 |

**在 EXP / Qwen collapse audit 測試的所有 variants（EXP-A~G + 所有 Qwen-trained runs）中，P8 healthy control 以外全部 collapsed**（MC CLAP 0.058–0.069；P7 V1 / LP-Rnd-Q 不在此 universe 內，仍為 healthy）。  
**已測的唯一健康共同因素**：完整 LP-MC writing-task style（含 ~45% boilerplate prefix density）+ S1+S2 全程 LP-MC supervision。注：此為 tested configuration 範圍內的觀察，非必要條件的完整證明。

**~~尚未測試的關鍵路徑：全 LP-MC S1+Qwen S2~~** ✅ 已測 EXP-G，NULL 結果。所有 currently designed stage/data intervention 都 collapse；剩下 untested 高層級 hypothesis = caption-audio granularity mismatch。更精確地說，LP-MC 是約 30s local-segment caption → first-10s NPZ；歷史 upstream track-level Qwen 則是一筆 track caption 廣播到該 track 的多個 local segments → first-10s NPZ。來源、共享規則與乾淨控制組見 [`caption_provenance_granularity_and_aes_controls.md`](caption_provenance_granularity_and_aes_controls.md)。

---

## Caption 2.0 queue：025/026 full-scale true-vs-fake random（2026-08-26 排入 p2/pending）

**問題**：K=3 stack 的 per-epoch caption rotation 有沒有用？

**quarter 的 null 不算數**。`true_random` 0.2013 vs `fake_random` 0.2005（delta 0.0008）是在 S1 100k 測的，而
251,599 rows ÷ batch 8 = 31,450 it/epoch，100k iter 只有 **3.18 epoch**：

- quarter：每個 clip 期望看到 3·(1−(2/3)^3.18) = **2.19 / 3** 條 caption — 廣度只實現 73%
- full（S1 400k = 12.72 epoch）：3·(1−(2/3)^12.72) = **2.98 / 3** — 實質完整覆蓋

而且 augmentation 是 regularizer，3.18 epoch 仍在 undertrained 區（matched pair 顯示多跑 4 倍步數還能再拿
+0.012 CLAP），在這種 regime 測 regularizer 預期就是 null。**原結論必須降級為「quarter / 3.18-epoch 條件下沒有可測訊號」**。

**設計**：兩 arm 都 **cold-start 0→400k S1 + 200k S2**，NoQ，MusicCaps MF25 cfg0 `--no_q`。

| Queue | Arm | Caption 供給 | multi_cap | cap spec |
|---|---|---|---|---|
| `025_true_random_full` | `phase8_qwen_caption2p0_k3_true_random_noq_full` | 每 epoch 重抽（`extracted_audio.py:_true_random_cap_index`，epoch 進 hash） | true | — |
| `026_fake_random_full` | `phase8_qwen_caption2p0_k3_fake_random_noq_full` | 固定一條/clip，uniform SHA-256 分派（slot0/1/3 = 83,716/83,494/84,389） | false | `column:cap_index` |

**為什麼不沿用 022/023/024 的 `same_arm_100k_restart_boundary`**：`fake_random` 的 quarter S1 `ckpt_last.pth`
已不存在（只剩 480MB weights-only `_last.pth` / `_ema_final.pth`），無法重現該 boundary。若讓 `true_random`
從 100k resume 而 `fake_random` 連續跑，兩 arm 就同時差在 (a) rotation 與 (b) restart boundary — 而 restart
會重跑 `linear_warmup_steps=1000` 並重置 optimizer/RNG，其擾動量級與我們要測的效果（~0.01）同級。
兩邊都 cold-start 才讓差異只剩 rotation。preflight 新增 `from_scratch_with_autoresume` resume kind
（不綁前置 ckpt，靠 arm 自己的 `ckpt_last` 在 crash/pause 後自動接續，符合 GPU idle backlog guardrail #2）。

**Overlay 零成本**：兩 arm 共用既有 `~/text_overlays/true_random`（3-cap stack，225 GB，`DONE.json` status=passed）；
`fake_random` 專屬 overlay 已於 2026-08-26 刪除回收 152 GB，改用 per-row `cap_index` 欄位取同一條 caption。

**判讀規則（先寫死）**：`true_random` 必須贏 `fake_random` 超過 quarter 的 0.0008 才支持 caption breadth；
**若 full coverage 下再度 null，K-stack rotation 這條線就收掉**。

**同 budget 下已知**：`best-of-3` 0.2122 > `true_random` 0.2013（差 0.011，兩者同為 quarter）— 挑最好的那條
贏過輪替全部。廣度輸給選擇。

**⚠️ 資料狀態**：`021_true_random_quarter` 目前在 `p2/held/`（terminal reason `cfg0 report not passed`），
它的 0.2013 只有 `eval_output/metrics/.../metrics.txt`，**沒有 contract 要求的 `cfg0_eval_runtime/reports/*_REPORT.json`**。
數字本身（n=5521、正確 TSV、cfg0 noq）可用，但不是 contract-verified。

### ✅ 結果（2026-09-02，兩 arm 皆 contract-verified）

| arm | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| `true_random` full | **0.2221** | **6.3893** | **6.8719** | **5.1883** | **6.6513** |
| `fake_random` full | 0.2186 | 6.2217 | 6.7331 | 5.1341 | 6.5543 |
| **Δ (true − fake)** | **+0.0035** | +0.1676 | +0.1388 | +0.0542 | +0.0970 |

`true_random` 五項數字都比 `fake_random` 高（quarter 時 PC 還輸）。

> ⚠️ **2026-09-03 更正：這張表曾被判為「支持 caption breadth」，該判定已被推翻。**
> 當時的依據是「CLAP +0.0035 > 預先寫死的 0.0008 門檻」，但 **0.0008 是從 quarter 觀測差
> 推出來的，不是量測過的 noise floor**。實測 seed 位移出來後（031，見下），CFG 0 的
> CLAP 雜訊是 **0.0042 > +0.0035**、PC 雜訊 **0.0554 > +0.0542** —— 兩項當場出局；
> 其餘三項在同協定檢驗下也沒過關。**完整結論見下方「seed noise floor」節。**

兩 arm 走同一個 wrapper、同 batch，所以 CLAP 的 batch-size 敏感度（~0.0045）**不適用**於這個
within-protocol 對比。這點仍然成立。

**scale 效應**：true 0.2013 → 0.2221（+0.0208）、fake 0.2005 → 0.2186（+0.0181）。
**兩 arm 一起漲，漲幅差只有 0.0027**（< CLAP seed 雜訊 0.0042）—— 從 quarter 到 full 的
增益是**訓練量帶來的，不是 rotation 帶來的**。原本寫的「full coverage 下訊號才出現」
已隨 2026-09-03 更正作廢。

### ⚠️ CFG 3 + negative prompt 下 CLAP 優勢消失（2026-09-02，secondary protocol）

協定：MusicCaps 5521 / MF25 / **CFG 3.0** / NoMask / seed 42 / full /
`negative_prompt="low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi"`。
**同時變了 cfg 與 negative 兩個變數，不可與 CFG 0 表逐格對比**；非 contract-verified（canonical
wrapper 只收 `_mf25_cfg0_`）。產出：`nvme_experiment_artifacts/meanaudio/negprompt_random_full_cfg3/`。

| metric | true | fake | Δ | paired t | true 勝率 |
|---|---|---|---|---|---|
| CLAP | 0.2651 | 0.2647 | **+0.0004** | **+0.60** | **50.1%** |
| CE | 7.1205 | 7.0657 | +0.0548 | +6.15 | 60.0% |
| CU | 7.6737 | 7.5459 | +0.1279 | +17.92 | 65.4% |
| PC | 4.8476 | 5.1398 | **−0.2923** | **−38.37** | 20.5% |
| PQ | 7.6111 | 7.4610 | +0.1502 | +20.81 | 65.2% |

**飽和檢查通過**（true crest_min 2.204 / fake 3.340、clipping 皆 0.0）→ 數字不是 cfg≥2 波形飽和假象。

**跨協定一致性**：

| metric | Δ @ CFG 0 | Δ @ CFG 3+neg | 讀法 |
|---|---|---|---|
| CU | +0.1388 | +0.1279 | 同向、同量級 |
| PQ | +0.0970 | +0.1502 | 同向、同量級 |
| CE | +0.1676 | +0.0548 | 縮小但同向 |
| **CLAP** | **+0.0035** | **+0.0004** | **崩到 0，勝率 50.1% 等同擲硬幣** |
| **PC** | **+0.0542** | **−0.2923** | **翻轉** |

> ⚠️ **2026-09-03 更正**：這裡原本寫「CU/PQ 跨協定穩定 → 是真效果」。**該推論無效** ——
> 它拿 CFG 3 量到的效果去對照 CFG 0 量到的 seed 雜訊。同協定 seed 對照跑出來後
> （見下節），CFG 3+neg 的 seed 雜訊比 CFG 0 大 2–3 倍，CU/PQ 兩項都掉進雜訊裡。

**paired t 的界線（重要，別再誤讀）**：paired t 移除的是 prompt/eval 噪音（同 5,521 個 clip
配對），**不含 training-seed 變異**。下節的 seed 對照示範了這件事的嚴重性：**同配置只換訓練
seed，一樣能產生 t=26.7 的「穩定」差異**。所以 t 大只能說「這兩個 checkpoint 之間差得很
穩定」，**不能**說「這個差來自 rotation」。

### 🔴 Seed noise floor：同協定對照推翻上面兩節的結論（2026-09-03）

arm = `phase8_qwen_caption10s_multisent_noq_full`（= `c2p0_slot0` full），兩顆訓練 seed，
**其餘完全相同**，所以每一格 Δ 依定義都是雜訊。true/fake 兩 arm 的訓練 seed 都是 **14159265**
（contract `training.seed`），跟這裡的 baseline 同一顆，是直接可比的 proxy。

**CFG 0**（031，contract-verified，`cfg0_eval_runtime/reports/…_seed27182818_…_REPORT.json`）：

| 訓練 seed | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| 14159265 | 0.2149 | 6.2870 | 6.7220 | 5.1393 | 6.5793 |
| 27182818 | 0.2191 | 6.1527 | 6.6700 | 5.0839 | 6.5270 |
| **\|Δ\|** | **0.0042** | 0.1343 | 0.0520 | 0.0554 | 0.0523 |

**CFG 3.0 + neg**（`negprompt_reeval_cfg3.0/`，13-arm sweep，非 contract-verified）：

| 訓練 seed | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| 14159265 | 0.2605 | 7.2114 | 7.6251 | 5.1059 | 7.5992 |
| 27182818 | 0.2608 | 6.9153 | 7.5198 | 4.9175 | 7.4576 |
| **\|Δ\|** | **0.0003** | **0.2960** | **0.1053** | **0.1884** | **0.1416** |

**CFG 3+neg 的 seed 雜訊比 CFG 0 大 2–3 倍**（CE 0.134→0.296、PC 0.055→0.188、
PQ 0.052→0.142、CU 0.052→0.105），而效果沒有跟著放大。

**同協定對照（效果 vs 雜訊，皆 CFG 3+neg、n=5521 paired）**：

| metric | 效果 Δ (true−fake) | 效果 t | seed \|Δ\| | seed t | 效果/雜訊 | 判定 |
|---|---|---|---|---|---|---|
| CLAP | +0.0004 | 0.60 | 0.0003 | −0.35 | 1.62 | ❌ 兩者都不顯著 |
| CE | +0.0548 | 6.15 | 0.2960 | 26.72 | **0.19** | ❌ 雜訊是效果的 5 倍 |
| CU | +0.1279 | 17.91 | 0.1053 | 13.42 | 1.21 | ❌ |
| PC | −0.2923 | −38.36 | 0.1884 | 22.77 | 1.55 | ❌ |
| PQ | +0.1502 | 20.81 | 0.1416 | 18.29 | 1.06 | ❌ |

**🔴 定論：五個指標沒有一個達到 2× 雜訊門檻**（最高 PC 1.55，且方向與 CFG 0 相反；
PQ 只有 1.06）。加上 CFG 0 那邊 CLAP 與 PC 本來就已被雜訊吃掉，**沒有任何一個指標、
在任何一個協定下，撐得住「per-epoch caption rotation 有幫助」**。

依 contract `caption2p0_true_random_full_cfg0_contract.json` 事前寫死的判讀規則 ——
*"a second null at full coverage retires the K-stack rotation line"* —— **K-stack rotation
這條線收掉**。

**caveat**：(1) seed replicate 做在 `slot0` arm，不是 true/fake 本身；(2) 兩顆 seed 只給
一個差值，不是分布；(3) `seed27182818` 在 CFG 3 下 **crest_min 1.768 < 2.0**（另一顆 3.43），
落在波形飽和警戒線內（clipped_fraction 仍為 0.0），這顆 checkpoint 的 CFG 3 數字帶飽和疑慮。
要完全關掉 caveat (1)(2) 需要對 true 或 fake 其中一 arm 再訓 seed replicate。

**子集**：novocal / vocal / lofi-prompt / clean-prompt 四切下 true−fake 的 Δ 幾乎不變
（CLAP −0.0013~+0.0025、PQ +0.14~+0.17）→ 兩 arm 的差異**不是** lofi-specific，
negative prompt 的缺陷語意沒有和 rotation 產生交互作用。

---

## Caption 2.0 queue：034 true012 rotation quarter（2026-09-02 排入 p2/pending 尾端）

**問題**：等 budget 下，per-epoch caption rotation 能不能贏過它輪替的**每一條** caption？

013 那組（021/025/026）答不了這題 —— **slot3 從來沒單獨訓練過**，所以 `true random` 只能跟
`fake random`、`best/worst-of-3` 比，沒有「rotation vs 它自己的組成槽」的對照。**012 是唯一
三個組成槽都有 budget-matched quarter 數字的 pool**：

| 對照組（皆 quarter） | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| slot0 | 0.2029 | 6.1185 | 6.7031 | 5.0350 | 6.5364 |
| slot1 | **0.2047** | 6.3008 | 6.7593 | 5.1632 | 6.5668 |
| slot2 | 0.2017 | 6.2071 | 6.7487 | 5.0814 | 6.5623 |
| 012 best-of-3 | 0.2129 | | | | |
| 012 worst-of-3 | 0.1957 | | | | |

**判讀規則（先寫死）**：`true012` 要**超過最好的單槽 slot1 0.2047** 才支持 rotation。落在
0.2017–0.2047 的單槽帶內就是 null —— 那代表 rotation 只是重現了組成槽的平均值。贏過
best-of-3 0.2129 不是門檻，也不預期。

**零磁碟做法（重點）**：012 stack **沒有**被編碼過，而重編一份要 225 GB、NVMe 只剩 142 GB。
但 slot0/slot1 就是既有 013 stack 的 index 0/1，slot2 有自己的單槽 overlay，**兩者
text encoder fingerprint 相同**（`27e88fac…`）→ 改成 loader 於載入時組裝 caption pool
（`text_npz_sources`），新增磁碟 0 bytes、不動 GPU 編碼。之後任何 slot 組合都免費。

**pairing audit**（Phase 9 錯配事故後的必要守門，全部 passed）：
- 1,500 列 × 3 source：clip_id 全數對上 TSV id
- pool position 1/2 分別對照 `phase8_caption2p0_slot1_train.tsv` / `slot2_train.tsv` 驗 caption sha
- rotation 分布 0.3397 / 0.3327 / 0.3277（4 epoch）
- loader smoke test：15 組 (idx, epoch) 回傳的 embedding 與宣稱來源**逐位元相同**；跨 epoch
  換槽 1364/2000（期望 2/3）；負控制（餵錯 pool）被正確拒絕
- 訓練期 `require_text_overlay=true` 逐列再驗一次

**繼承的 caveat**：S1 100k = 3.18 epoch，rotation 覆蓋率只有 2.19/3，仍在 undertrained 區測
regularizer。但所有對照組同樣是 quarter，**內部比較成立**。

Queue：030(running) → 031 → 032 → 033 → **034**。contract
`caption2p0_true012_random_quarter_cfg0_contract.json`、pool spec
`caption2p0_true012_caption_pool.json`。

### ✅ 034 結果（2026-09-02 完成，contract-verified）

`p2/done/034_true012_random_quarter.terminal.json` status=completed。當初 rc=1 是
`validate_caption2p0_cfg0_report.py` 的 checkpoint-binding sidecar 檔名以
`reports/` + `cell_id` 命名，導致所有用 `cell_id=canonical_noq` 的 CFG 0 contract
撞同一個檔；binding_path 改成 per-cell 命名後於 2026-09-02 12:22 revalidate
為 `STRICT_REPORT_OK`。**不是實驗失敗。**

| arm | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| slot0 | 0.2029 | 6.1185 | 6.7031 | 5.0350 | 6.5364 |
| slot1（最好單槽） | 0.2047 | 6.3008 | 6.7593 | **5.1632** | 6.5668 |
| slot2 | 0.2017 | 6.2071 | 6.7487 | 5.0814 | 6.5623 |
| **034 true012 rotation** | **0.2053** | **6.3422** | **6.8654** | 5.0760 | **6.6997** |
| Δ vs 最好單槽 | +0.0006 | +0.0414 | +0.1061 | −0.0872 | +0.1329 |
| CFG 0 seed 底線 | 0.0042 | 0.1343 | 0.0520 | 0.0554 | 0.0523 |
| 效果/底線 | **0.14×** | 0.31× | **2.04×** | 輸 | **2.54×** |

**依事前寫死的規則 → null。** 034 的 0.2053 技術上超過 slot1 0.2047，但只贏 +0.0006 =
CLAP seed 雜訊的 **0.14 倍**。判讀帶（0.2017–0.2047，寬 0.0030）**比雜訊底線 0.0042 還窄**，
所以「過線」不帶資訊 —— rotation 只是重現了組成槽的平均值，與原文寫的 null 判準一致。
**這條 CLAP 規則本身是不可用的**（設計時沒有實測底線可參照）。

**AES 側未定**：PQ +0.1329（2.54× 底線）、CU +0.1061（2.04×）跨過 2× 門檻，且是贏過
**最好的**單槽、同 budget —— 這是 013 那組給不出的對照（slot3 從未單獨訓練）。
**但還不能當結論**：(1) PQ/CU 不在事前規則裡，事後換指標宣告勝利是 moving the goalposts；
(2) 013 的 CU/PQ 在 CFG 0 也曾看似過關（2.67×/1.85×），換到 CFG 3+neg 量同協定底線後
掉到 1.21×/1.06× —— **CFG 0 底線會低估**，而 034 目前只有 CFG 0 數字；
(3) 底線量在 full-scale arm，034 是 quarter。

**要定案需補**：034 + 三個單槽跑 `negprompt_reeval_full_arms.py --cfg=3.0`
（目前該 sweep 只收 full arms），在同協定下驗證 PQ/CU。4 arm × ~42 min ≈ 3 hr GPU。
