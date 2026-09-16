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

> ⚠️ **本段與下方「收線建議」已作廢**（2026-09-14）：−0.0150 是拿逐檔 CLAP 0.2455 去比 batch 32 的 0.2605，
> 混用 scorer。batch 32 重算為 +0.0002，見下方「052 第二次執行」。

~~**CFG3+neg：CLAP 明確較差，AES 平手。**~~ CLAP −0.0150，低於 slot0 兩個 seed（0.2605 / 0.2608）。同協定 CLAP floor 只有
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

### ⚠️ 語料品質 caveat（2026-09-14 事後審計）

對 `~/exps_nvme/slot4/arm_inputs/rewrites.jsonl` 做 slot0→slot4 逐列 diff：

- 改寫列 21,644（8.6%）：`llm` 10,700、`lexicon` 10,934、`lexicon_from_original` 10。全語料被刪或改的字只佔 **2.40%**，
  但在被改的列裡佔 **24.4%**（word-level 相似度中位數 0.81）。
- **Qwen no-EOS trap 重演**：`rewrite_slot4_no_digits.py` 的 `model.generate` 沒傳 `eos_token_id`，輸出後面會接上
  「下一輪對話」的垃圾文字（例如 "Given the current trend of incorporating artificial intelligence..."）。
- **`fallback()` 拿最後一次 LLM 輸出做 lexicon strip，不是拿原句**（L196）→ 垃圾文字被保留下來，而數字直接刪掉會留下
  "with a tempo of." 這種斷句。
- 啟發式計數：注入殘留 llm 5.3% / lexicon 21.1%；斷句 lexicon 68.6%；合計有問題的列約 **8,848 = 改寫列的 40.9%，
  全語料的 3.5%**。另外 LLM 也順手刪了不含數字的句子（36–44% 的無數字句消失，約 1/4 的列丟掉 production/mix 描述）。

所以 slot4 **不是乾淨的「只拿掉數字」**，而是「拿掉數字＋3.5% 列被污染＋部分描述被刪」。平手的結論仍然成立（污染沒有
造成可測的傷害），但不能據此宣稱「數字本身無影響」已被乾淨地測過。

### 結論

**兩個尺度、兩個協定、共二十個比較，全部平手。** 把 slot0 裡 8.6% 帶數字的 caption 改寫成零數字，
在 MusicCaps 上既沒幫助也沒害處 —— 與 quarter 結論一致，full 尺度沒有翻案。這條線收掉。

---

## slot4v2 — 修好改寫 bug 重產語料（2026-09-14 排入 055 quarter）

> operator：「修好 rewrite bug 重產語料，排 quarter」。contract
> `docs/experiments/caption2p0_slot4v2_no_digits_quarter_cfg0_contract.json`，queue seat
> `055_c2p0_slot4v2_no_digits_quarter.sh`，wrapper `scripts/training_pipelines/caption2p0_slot4v2_quarter.sh`。

`scripts/preprocess/rewrite_slot4v2_no_digits.py`（舊的 `rewrite_slot4_no_digits.py` 保留不動，050/052 的 contract 還 pin 著它）：

| slot4 的缺陷 | v2 的修法 |
|---|---|
| `generate` 沒傳 `eos_token_id` → 後面接上下一輪對話 | 傳 `<\|im_end\|>` 當 eos／pad，首輪 greedy |
| 整句 caption 丟給 LLM → 36–44% 無數字句被刪 | **句子層級**：只有含數字的句子送進 LLM，無數字句逐位元組複製（audit 會 assert） |
| fallback 拿有垃圾的 LLM 輸出去剝數字 → 斷句 | fallback 只拿**原句**：先用 token 詞彙表，再整段刪掉含數字的子句；主子句本身就是數字才整句刪 |
| 沒有品質閘門 | LLM 輸出要通過：無數字、單句、無對話／指令標記、不能新增斷句、無 `bpm` 字、4/4 不能變 four-on-the-floor、替換詞只能在原句有對應數字時出現（擋「in waltz time」這種幻覺）、最多 1 個新字、非數字相關的內容字**一個都不能少** |

改寫完成後，Step 2 會再做一次 audit（無數字列逐位元組相同、無數字句都還在、無注入、無新斷句），不過就不訓練。
訓練／評估協定與 050 完全相同；CFG3+neg 會多跑 `rescore_clap_batch32.py`，直接得到可以跟 slot0／slot4 的 batch 32 數字比較的 CLAP。

判讀比照 050：slot0 quarter CFG0 CLAP 0.2029，≥ 0.2113 算贏、0.1945–0.2113 平手、< 0.1945 算輸；
次要對照是 slot4 quarter（CFG0 0.2050、CFG3+neg b32 0.2374）。

### slot4v2 語料完成＋QA（2026-09-14 21:13 CST 放進 p2 pending）

`~/exps_nvme/slot4v2/arm_inputs/phase8_caption2p0_slot4v2_train.tsv`（251,599 列，數字列 0）。22,277 句含數字句的處理方式：

| 方法 | 句數 | 說明 |
|---|---|---|
| `llm`（Omni-3B 首輪） | 12,905 | 只刪數字 |
| `llm_restructure`（Omni-3B 重組） | 3,301 | 動詞只能由原句分詞變形（creating→creates） |
| `llm_pass3`（Qwen2.5-7B-Instruct） | 2,218 | 3B 做不了的「數字嵌在文法裡」句型 |
| `lexicon` / `lexicon_span` / `lexicon_clause` / `lexicon_prefix` | 961 / 100 / 2,187 / 25 | 對**原句**做決定性編輯 |
| `llm_none` / `lexicon_drop` | 213 / 367 | 整句只有數字、或無法安全保留 → 刪句 |

QA（`scripts/preprocess/qa_slot4v2_corpus.py --strict`，已加進 action Step 2）：數字、單位字、拼寫數字、對話殘留、斷句、重複字、
小寫句首、`X-on-the-floor`、BPM 的 80s 被寫成年代、原句沒有的速度形容詞 → **全部 0**。改寫期間 QA 逐輪抓到並補成閘門的錯誤：
`2/4`→"two-on-the-floor"、"seven-eighths"、"tempo in the eighties"、"117.9 bpm"→"moderate"（幻覺）、刪掉 80s／4/4／8-bit 帶的資訊、
"a eighties"、在形容詞列表中間切逗號、`16-bit`→"vintage digital." 殘句、"mid-tempo tempo"。另人工分層抽樣 >200 句。
被改的列裡 9.7% 的字被刪或改、逐字相似度中位數 0.941；全語料 0.95%（slot4 分別是 24.4%、0.81、2.40%）；仍有 93 列遺失 meter／decade 資訊（隨含數字子句一起刪掉），367 句整句刪除，列為已知限制。

⚠️ 入座阻塞：arale Irodori-TTS（pid 1868671）目前 3,810 MiB > probe 門檻 3,072 MiB，p2 host 會靜默等待。

## 055 slot4v2 quarter — 完成（2026-09-15 04:36 CST，rc=0，`status: completed`）

S1 100k（loss NaN 0、grad_norm NaN 0）→ migrate → S2 50k（NaN 0，程序正常退出，無 052 式 teardown hang）→ eval。
CFG0 report `passed`（5,521/5,521、16 kHz mono）；CFG3+neg 5,521 檔，batch-32 CLAP 已由 action 自動重算。

**CFG0（preregistered primary）**

| quarter CFG0 | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| **slot4v2（乾淨剝數字）** | **0.2004** | 6.0955 | 6.6823 | 5.0968 | 6.5178 |
| slot4（污染語料） | 0.2050 | 6.1661 | 6.7525 | 5.0268 | 6.5832 |
| slot0（comparator） | 0.2029 | — | — | — | — |

vs slot0：CLAP −0.0025 = 0.6× floor，落在登記平手帶 0.1945–0.2113 → **平手**。
vs slot4：CLAP −0.0046（1.1×）；AES CE 0.53× / CU 1.35× / PC 1.26× / PQ 1.25× —— 全部 < 2× floor。

**CFG3+neg（secondary，CLAP 用 batch 32）**

| quarter CFG3+neg | CLAP b32 | CLAP 逐檔 | CE | CU | PC | PQ |
|---|---|---|---|---|---|---|
| **slot4v2** | **0.2309** | 0.2206 | 6.7146 | 7.4070 | 4.8494 | 7.2998 |
| slot4 | 0.2374 | 0.2285 | 6.8031 | 7.4767 | 4.7489 | 7.3663 |
| slot0 | 0.2372 | 0.2248 | 6.6952 | 7.3871 | 4.6661 | 7.3101 |

vs slot0：CLAP b32 −0.0063；AES CE 0.07× / CU 0.19× / PC 0.97× / PQ 0.07×（同協定 floor）。
CLAP −0.0063 若除以 CFG3+neg CLAP floor 0.0003 是 21×，但那個 floor 是 **full 尺度、只有 2 個 seed** 量的（本檔先前已註記可能低估）；
除以 CFG0 floor 0.0042 只有 1.5×。逐檔 CLAP 同方向（−0.0042）。

**判讀（observation 層）：**
- 預先登記的 primary（CFG0 CLAP）平手，AES 全在雜訊內 —— 乾淨剝數字在 canonical 協定下沒有可測效果，與 050／052 一致。
- CFG3+neg 的 CLAP 比 slot0、slot4 都低約 0.006，是這條線唯一值得留意的訊號，方向是「略差」。但沒有 quarter 尺度的
  CFG3+neg seed floor，不能判定超出雜訊；不寫成「有害」。
- 污染語料（slot4）與乾淨語料（slot4v2）在 CFG0 差 0.0046（1.1×），不支持「slot4 的污染傷害了結果」這個說法。
- 本實驗只能說：**移除 caption 裡的數字，在 MusicCaps 上沒有可測的幫助**。

**建議**：不排 full。若要確認 CFG3+neg 那 −0.006，成本最低的是補一個 slot0 或 slot4v2 的 quarter 第二訓練 seed。

## 後續：057 slot0nm（去污染＋去量測）

slot4v2 的語料再往前一步——除了數字，還移除調性/調式/和弦性質與拍號，並套用 slot0 污染清洗的 1,499 列重生版本。
quarter CFG0 CLAP **0.2060**（vs slot0 0.2029，0.74× floor，平手）；CFG3+neg b32 **0.2417**（vs slot4v2 0.2309）。
完整數字、判讀與命名規則見 `docs/experiments/slot0_semantic_audit_20260915/README.md`「057 slot0nm quarter」。
