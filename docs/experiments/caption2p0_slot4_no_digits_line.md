# c2p0 slot4 — 把 slot0 的數字寫掉

> 2026-09-09。queue job `050_c2p0_slot4_no_digits_quarter.sh`，
> contract `docs/experiments/caption2p0_slot4_no_digits_quarter_cfg0_contract.json`。

## 要回答什麼

MusicCaps 5521 句裡 BPM 只有 1 句（0.02%），有阿拉伯數字的也只有 4.1%，而且幾乎是 `4 on the floor` / `808` / `80s`。Eval 考的是樂器、mood、genre、人聲、情境。

c2p0 slot0 卻有 8.6% 的 caption 帶數字，其中 6.6% 是明確 BPM（含 `136.49 BPM` 這種假精度）。有實驗指出 caption 寫詳細數字會拖累發揮。slot4 只動這一個變因：把 slot0 裡有數字的列改寫到 `\d` 為 0，其餘 byte-identical，看 quarter CFG0 相對於 slot0 的 0.2029 有沒有起來。

## 做法

- 來源：`phase8_qwen_caption10s_multisent_train.tsv`（slot0，251,599 列）
- 只改有數字的列（約 21,644）。Qwen2.5-Omni-3B 純文字改寫，最多 5 次；失敗走詞彙表（`808`→drum machine、`4/4`→common time、`80s`→eighties）再剝殘留數字。
- **不**改成 LP-MusicCaps 句型，不發明樂器。
- Overlay：hardlink `text_overlays/true_random`，只重編改過的 slot-0 特徵（約 19G，不是 76G）。訓練 `cap_index_fixed=0`。
- Quarter：S1 100k / S2 50k，seed 14159265，batch 8，lr 1e-4，NoQ，NoMask。CFG0 canonical + CFG3+neg。

## 判讀

對 slot0 quarter CFG0 CLAP **0.2029**。2× seed floor CLAP = 0.0084。

| CFG0 CLAP | 判讀 |
|---|---|
| ≥ 0.2113 | 剝數字有幫助 |
| 0.1945 – 0.2113 | 平手；MusicCaps 不考 BPM，寫了也沒被罰到可測的程度 |
| < 0.1945 | 剝數字有害（可能連 808/80s 這種詞彙也被誤傷） |

---

## 050 quarter — 完成（2026-09-12）

S1 100k + S2 50k 全跑完，CFG0 report `status: passed`（5,521/5,521、16 kHz mono）。

| quarter | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| **slot4（無數字）** CFG0 | **0.2050** | 6.1661 | 6.7525 | 5.0268 | 6.5832 |
| slot0 CFG0（comparator） | 0.2029 | — | — | — | — |
| **slot4** CFG3+neg | **0.2285** | 6.8031 | 7.4767 | 4.7489 | 7.3663 |
| slot0 CFG3+neg | 0.2248 | 6.6952 | 7.3871 | 4.6661 | 7.3101 |

**判讀：平手。** CFG0 差 +0.0021 = **0.5× seed floor**（0.0042），落在 contract 預先寫好的
0.1945–0.2113 平手帶正中央。CFG3+neg 差 +0.0037，同樣在底線內。

意思是：MusicCaps 幾乎不寫 BPM，caption 裡寫了也**沒有被罰到可測的程度**——把 8.6% 的
caption 剝成零數字，在 quarter 尺度上既沒幫助也沒害處。這是原本三個 bucket 裡最中庸的那個。

## 052 full — 已排入 p2 queue（2026-09-12）

- contract `docs/experiments/caption2p0_slot4_no_digits_full_cfg0_contract.json`
- queue seat `052_c2p0_slot4_no_digits_full.sh`，wrapper `scripts/training_pipelines/caption2p0_slot4_full.sh`
- S1 400k / S2 200k（其他 c2p0 單槽 full arm 的標準預算），語料與 overlay 直接沿用 050 的產物，
  Step 1–3 會偵測到既有檔案而跳過。
- comparator：slot0 full CFG0 **0.2149**（win ≥ 0.2233 / tie 0.2065–0.2233 / loss < 0.2065）；
  次要 cell 對 slot0 full CFG3+neg 0.2605。
- **沒有 early-kill gate**：quarter 是平手，依照原本的 gate 邏輯這條線會就地收掉；full arm 是
  依 operator 指示照排，目的是在板上 headline 數字所在的尺度再量一次。

> shared action 於 2026-09-12 加上 `full)` 預算 case 與 scale-aware 的 NVMe 門檻
> （quarter 30G / full 80G），sha256 由 `085fca48…` 變成 `4e15b289…`。quarter wrapper 的
> pin 已同步更新；quarter 的預算/語料/eval cell 都沒被這次編輯動到。

### 052 第一次執行：訓練完成、評估未跑（2026-09-13）

- S1 400k、S2 200k 全部跑完；loss 0 次 NaN（S1 有 63 次 `grad_norm:nan`，屬 AMP 正常率）。
- S2 `ema_final.pth` 於 09-13 12:42 寫出：40 個 EMA 快照大小一致，203 個 tensor 全部 finite。
- 之後 `train.py` 印完 `Evaluation:` config dump 就卡在程序結束階段 11.5 小時（quarter 與 full S1 在同一行後都是秒退），
  09-14 00:14 被外部 SIGTERM → torchrun rc≠0 → action 在 pipefail 下以 rc=1 結束，兩個 eval cell 都沒跑到，
  queue 記 `failed`，並接著排上 053。
- 46 秒後佇列寫入 runtime resume binding（`manual-requeue-eval-only-after-s2-teardown-hang`，綁 S2 `ckpt_last` it=600000，
  sha `f6be924a…`），052 移回 `pending/`，`accept_guest` 驗證通過。重跑時 action 看到 S2 EMA 已存在，會跳過訓練，只做
  CFG3+neg → CFG0。
- `set_training_stage.py` 的 5 個 patch 全部在 `MeanFlow.loss()` 裡，取樣路徑不受影響，所以在 Stage 1 tree 上評估
  與 quarter（當時 tree 還停在 Stage 2）可以直接比較。

## 052 full — 完成（2026-09-14 02:12 CST，eval-only 重跑 rc=0）

EMA 為 09-13 12:42 那顆（mtime 未變；sha `02868ccb…`）。CFG0 report `status: passed`，5,521/5,521。

| full | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| **slot4** CFG0 | **0.2210** | 6.2184 | 6.7502 | 5.1030 | 6.5896 |
| slot0 CFG0（seed 14159265） | 0.2149 | 6.2870 | 6.7220 | 5.1393 | 6.5793 |
| slot0 CFG0（seed 27182818） | 0.2191 | 6.1527 | 6.6700 | 5.0839 | 6.5270 |
| **slot4** CFG3+neg | **0.2455** | 6.9443 | 7.5477 | 4.8675 | 7.4655 |
| slot0 CFG3+neg（seed 14159265） | 0.2605 | 7.2114 | 7.6251 | 5.1059 | 7.5992 |
| slot0 CFG3+neg（seed 27182818） | 0.2608 | 6.9153 | 7.5198 | 4.9175 | 7.4576 |

**CFG0（preregistered primary）：平手。** CLAP +0.0061 = 1.45× floor，落在 0.2065–0.2233；而且在 slot0 兩個訓練 seed
（0.2149 / 0.2191）的範圍附近。AES 四項差距都 < 1× floor。

**CFG3+neg：CLAP 明確較差，AES 平手。** CLAP −0.0150，低於 slot0 兩個 seed（0.2605 / 0.2608）。同協定 CLAP floor 只有
0.0003（僅 2 seed，可能低估），就算改用 CFG0 floor 0.0042 也是 3.6×。AES 各項以同協定 floor 計：CE 0.90×、CU 0.74×、
PC 1.27×、PQ 0.94× —— 全部在雜訊內。

**判讀（observation 層）：**
- 剝數字在 canonical 協定下沒有可測效果，quarter（0.5×）與 full（1.45×）一致。
- 在加 negative prompt 的協定下，剝數字讓 CLAP 掉 0.015，這是這條線唯一超出雜訊的訊號，而且方向是「有害」。
  兩個協定不同號，依 decision rule 不下「有幫助」的結論。
- 不寫成機制。可能的解釋（未驗證）：改寫讓 8.6% caption 的措辭偏離 slot0 voice，或 `808`/`80s` 這類詞彙被一併寫掉，
  在 CFG3 放大條件信號時才顯現。

**收線建議**：數字不是 MusicCaps 上的瓶頸；不建議再開「只剝 BPM」的後續 arm，除非要專門查 CFG3+neg 那 −0.015 的來源。

### 052 第二次執行：eval-only，完成（2026-09-14）

09-14 00:56 CST 入座，Step 1–5 全部 `already complete`，只跑 Step 6 CFG3+neg → Step 7 CFG0，
02:12 CST `status: completed`，CFG0 report `passed`（5,521/5,521）。

**CFG0，vs slot0 full（primary）：五項全平**

| metric | slot4 full | slot0 full | delta | ×floor | 判定 |
|---|---|---|---|---|---|
| CLAP | 0.2210 | 0.2149 | +0.0061 | 1.45× | tie |
| CE | 6.2184 | 6.2870 | −0.0686 | 1.02× | tie |
| CU | 6.7502 | 6.7220 | +0.0282 | 1.08× | tie |
| PC | 5.1030 | 5.1393 | −0.0363 | 1.31× | tie |
| PQ | 6.5896 | 6.5793 | +0.0103 | 0.39× | tie |

CLAP 0.2210 落在登記的平手帶 0.2065–0.2233 內。

**CFG3+neg，vs slot0 full（secondary）：五項全平**

⚠️ **CLAP 必須用 batch 32 的數字比。** slot0 full 的 0.2605 來自 `negprompt_reeval_full_arms.py`（CLAP batch 32），
本 arm 的 Step 6 走 `phase4_eval.py`（逐檔）。同一份音檔兩者差 +0.004～+0.025 且因 arm 而異（AES 逐位相同，
見 memory `reference_clap_batch_size_sensitivity.md`）。本 arm 已用 `scripts/eval/rescore_clap_batch32.py`
重算（該腳本在 013 full 上逐位重現 negprompt_random_full_cfg3 的 0.2651）。

| metric | slot4 full | slot0 full | delta | ×floor | 判定 |
|---|---|---|---|---|---|
| CLAP（batch 32） | 0.2607 | 0.2605 | +0.0002 | 0.67× | tie |
| CLAP（逐檔，不可與 0.2605 比） | 0.2455 | — | — | — | — |
| CE | 6.9443 | 7.2114 | −0.2671 | 0.90× | tie |
| CU | 7.5477 | 7.6251 | −0.0774 | 0.74× | tie |
| PC | 4.8675 | 5.1059 | −0.2384 | 1.27× | tie |
| PQ | 7.4655 | 7.5992 | −0.1337 | 0.94× | tie |

若誤用逐檔 0.2455 會得到 −0.0150（50× floor）的假 LOSS。

quarter 用 batch 32 重看也一樣：slot4 0.2374 vs slot0 0.2372（+0.0002），與逐檔的 +0.0037 同為平手。

### 結論

**兩個尺度、兩個協定、共二十個比較，全部平手。** 把 slot0 裡 8.6% 帶數字的 caption 改寫成零數字，
在 MusicCaps 上既沒幫助也沒害處 —— 與 quarter 結論一致，full 尺度沒有翻案。這條線收掉。
