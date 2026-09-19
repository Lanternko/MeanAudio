# Slot0 semantic contamination audit

## Current execution — full_v1

The full API-only audit is running in tmux session `slot0-semantic-audit`, with
an independent supervisor and no GPU use. `status.json` is a dated summary;
the live state is
`/home/kojiek/exps_nvme/slot0_semantic_audit_20260915/full_v1/state.json`.
The original 251,599-row TSV remains unchanged.

The operator resolved the semantic definition: bare no-music assertions without
concrete sound descriptions are INVALID; concrete speech/environmental sounds
and explicit silence are retained. Numbers are not automatically contamination.

Calibration v4 passed 39/39. Fresh heldout_v2 found a meta-preface miss (47/48);
its failed report remains unchanged. After registering explicit meta-preface
handling, calibration_v5 passed 40/40 and fresh heldout_v3 passed 28/28. These
small enriched tests do not establish a population miss rate. The runtime pins
a 3,000-row uniform all-source sample for independent review, without filtering
by regex or the model's verdicts.

### Runtime behavior

- `slot0_semantic_audit.py` remains the bounded calibration/shared classifier.
  `slot0_audit_controller.py` is the separate full runner; no 64-row cap.
- Four concurrent workers, eight captions per request, persistent SQLite state,
  exact-prefix edits only, and an independent API check of every trimmed result.
- 429 responses have bounded backoff and persistent attempt counts. Timeouts
  and 5xx are ambiguous: recover a local receipt or query stored completion
  metadata; otherwise isolate that batch and continue other batches. Empty
  lookup results never prove zero billing and do not authorize blind reposts.
- A real stored-response recovery probe retrieved the same response ID. Its
  first lookup was empty due to indexing delay; this limitation is recorded.
- Authentication/model-access errors, notification failure, storage floors and
  the conservative $75 cost ceiling block further submissions.
- Credential file is outside Git, owned by the user with mode 0600. No key is
  included in argv or the audit child's environment.
- Notifications use durable receipts; the supervisor polls without model calls.

### Evidence and outputs

`full_v1_contract.json` pins the source, scripts, prompt, limits and gate reports.
`harn_v1/` contains the four schema-v1 registration documents. Runtime copies are
under `full_v1/harn/`; both registration and running snapshots were validated.
The missing CPU-ownership event in the first live snapshot was backfilled from
the actual owned process/file descriptor and lock, without changing runtime,
prompt, source or budgets. Evidence: `full_v1/cpu_ownership_evidence.json`.
The acceptance suite has 22 local tests, including receipt recovery, duplicate
lock, failed-batch isolation, retry bound, source drift, notification failure,
storage/cost holds, incomplete responses and prefix preservation.

The generated-source admission report allows the untrusted corpus into a
read-only auditor. It explicitly does **not** certify training cleanliness.
This contract generates audit artifacts, not audio captions or text features.

At audit completion, the controller exports `row_decisions.jsonl`,
`quarantine.jsonl`, `regeneration_requests.tsv`, and `full_gate_report.json`.
No training TSV is released. REVIEW adjudication, original-prompt audio
regeneration for INVALID rows, independent real-data review and downstream
feature provenance remain required. GPU regeneration is not automatically
launched by this API-only contract, and no GPU queue entry was inserted.

Resume the same immutable audit after resolving an operational hold with:

```sh
python3 /home/kojiek/MeanAudio/scripts/preprocess/slot0_audit_supervisor.py \
  --contract /home/kojiek/MeanAudio/docs/experiments/slot0_semantic_audit_20260915/full_v1_contract.json \
  --key-file /home/kojiek/.config/meanaudio/luna_api_key
```

## Historical preparation record (superseded by the execution above)

Operator requested use of their Luna API access to clean slot0 thoroughly,
following the preceding discussion of preserving wording and removing polluted
suffixes. The access credential is not persisted here. This directory contains
development calibration, not a released corpus or a launch-ready HARN bundle.

## Preserved scope

- Source: 251,599 slot0 rows, all unique IDs and unique captions. See
  `source_manifest.json` for the pinned source SHA-256.
- Inspect every row semantically, including digit-free captions; regex is not
  a candidate-selection mechanism.
- Preserve musical numbers, phrasing, and metadata. An API reviewer labels
  spans; deterministic code only keeps an exact original prefix.
- Embedded contamination, invalid boundaries, and unresolved cases have no
  automatic output. No source row is silently dropped.
- No regeneration, feature extraction, training, or queue mutation has run.
- A text-only audit cannot establish audio-caption fidelity or zero residual
  semantic errors. The original caption-generation prompt has not changed.

## Calibration evidence

`calibration_v1.json` contains 36 hand-labeled development fixtures. API results
are under `/home/kojiek/exps_nvme/slot0_semantic_audit_20260915/`:

- `calibration_v1`: two mismatches (speech-only description and Markdown).
- `calibration_v2`: stopped on an edit-boundary failure after 24 completed rows.
- `calibration_v3`: 36/36 matched after development fixes. This is a development
  set, not independent evidence of generalization.
- `heldout_v1`: 24 randomly selected slot0 captions (seed 20260915), eight new
  adversarial suffixes, and three known real slot0 defects. 34/35 label matches;
  all 24 random captions preserved and all eight injected suffixes removed at
  the registered exact boundary. One semantic-label conflict remains below.

The legacy structural classifier is reused, except that benign short captions
are not rejected. It is an additional gate, not the semantic detector. It still
needs a complete production taxonomy/fixture acceptance review before release.

## Unresolved definition

Real source ID `57_17957_segment_2_0`:

> The provided audio is not a music clip. Please provide the correct file or
> describe the audio in detail if you have any other questions.

Luna retained exactly `The provided audio is not a music clip.` and removed the
request. The held-out label says INVALID, but the audit prompt allows factual
descriptions of absence of music. This is a policy/label conflict, not evidence
that Luna missed the request. Preserve the failed report rather than silently
changing its expected label after inference. The original row remains untouched.

The production definition must decide whether such a bare no-music assertion
may remain, or requires same-prompt recaptioning using the actual audio. It
must not infer that the audio truly contains no music from this sentence alone.

## Execution boundary

The current CLI intentionally accepts at most 64 fixtures and cannot export a
training corpus. A complete immutable production contract, resumable full
controller, HARN bundle, notification/storage acceptance tests, all-row output
gate, independent review and provenance manifests remain prerequisites for a
long run and corpus release. `status.json` records this preparation hold.

Run deterministic edit-safety checks with:

```sh
python3 scripts/tests/test_slot0_semantic_audit.py
```

API credentials are entered with terminal echo disabled and stay in process
memory. API messages are not exposed on HTTP errors. Successful response
receipts are persisted before decision validation in the current version.

## 2026-09-15 12:25 — full_v1 paused for cost; switching to local LLM

Operator judged the projected Luna API cost (~US$21, cap US$75) too high and
asked to run on the local RTX 5090 instead. `full_v1` was stopped with SIGTERM
to the supervisor (graceful drain): 769 audit batches done (6,152 rows),
2 quarantined, 0 inflight/ambiguous, 30,679 pending, known cost US$0.50.
State in `full_v1/state.sqlite` is resumable; nothing was deleted.

Plan: local model (start with cached Qwen2.5-7B-Instruct via vLLM in
`~/venvs/vllm`) must pass the same calibration_v5 / heldout_v3 gates, plus
agreement against the 6,152 Luna-audited rows, before any full local run.
The full local run goes through the p2 GPU queue after 053; it is a new
contract, not a continuation of `full_v1`.

### Local run wiring (2026-09-15 12:56)

- Env: `~/venvs/vllm` (vLLM 0.29.0, torch 2.13+cu130, sm_120 present).
- Script: `scripts/preprocess/slot0_audit_local.py` (imports PROMPT/SCHEMA/
  `parse_response` from the hash-pinned audit module unchanged). Failed batch →
  per-row retry → per-row quarantine; TRIM accepted only if retained prefix
  re-audits KEEP. CPU smoke test of this logic passed; no GPU test yet.
- Chain: tmux `slot0_local_audit` → `scripts/runs/run_slot0_local_audit.sh`.
  Blocks on `gpu0.lock` (after 053), runs calibration_v5 + heldout_v3, agreement
  vs the 6,152 Luna rows, then full audit only if gate passes: fixtures exact,
  Luna non-KEEP recall ≥ 0.80, KEEP flag rate ≤ 0.02. Output
  `~/exps_nvme/slot0_semantic_audit_20260915/local_qwen7b_v1/`. Holds
  gpu0.lock for the whole chain, so later p2 jobs wait behind it.

### v2 redesign after review (2026-09-15 13:35) — supersedes the gate above

Review points accepted: same-model re-audit is not independent; Luna is not
ground truth; a 68-row fixture pass says nothing about full-corpus misses; text
review cannot check audio fidelity. Changes:

- **No quality gate in the GPU chain.** Only an operational stop (quarantine
  > 5% or exact agreement with Luna < 0.90 → local model unusable as a screen).
  Fixture and Luna-agreement results are reports.
- `trim_verified` renamed `trim_selfcheck_keep` (same model; consistency only).
- **Miss-rate measurement** (`scripts/preprocess/slot0_audit_crosscheck.py`):
  F = every local non-KEEP/quarantined row, K = 5,000 uniform local-KEEP rows
  (seed 2026091505); Luna reviews both (rows already in full_v1 reused), cap
  US$8, `--execute` required to spend. Report = Luna-flag rate in K with exact
  Clopper-Pearson 95% CI (checked against scipy) and projected residual rows;
  plus Luna agreement on F. These are Luna-flag rates, not true contamination.
- **Blind human pack**: `crosscheck/human_label_blind.tsv` (100 K + 100 F,
  shuffled); key in `human_label_key_DO_NOT_OPEN_BEFORE_LABELING.json`.
- Acceptance still needs three separate measures: residual unrelated text
  (K sample + human), wrongful removal (F vs Luna + human), audio mismatch
  (not measured by any text step; needs a listening protocol).
- Chain restarted 13:35 (tmux `slot0_local_audit`, v2 script), still waiting
  on gpu0.lock behind 053.

Accidental spend: a dry run of the cross-check on synthetic captions called
Luna because its cap (US$0.50) exceeded the projection; stopped after 166
requests, **~US$0.08**, receipts under the session scratchpad. The `--execute`
guard was added as a result.

Captioner truncation sizing on the current slot0 TSV (Qwen2.5 tokenizer):
max 159 tokens vs `max_new_tokens=160`; 75 rows ≥ 140 tokens, all end with
terminal punctuation; only 2 rows lack terminal punctuation and both are
metatext ("The caption of the audio is: '…'"), which the semantic audit covers.
Truncation impact on this corpus is therefore negligible. The generator now
reports `hit_max_new_tokens` rows as errors (opt-in `return_truncation=True`,
also used by `regen_multisent_defect_ids.py` so regeneration never accepts a
cut caption), and pins `MODEL_REVISION=f75b40e3…` (the only cached snapshot).

### v3 operator flow (2026-09-15 13:42) — supersedes v2 cross-check design

Operator set the flow to save API cost:
1. Local LLM audits every row (`slot0_audit_local.py full`).
2. Every locally flagged row (TRIM/REVIEW/INVALID/quarantined; **regenerated,
   not trimmed**) is recaptioned with the original Qwen2.5-Omni-3B + PROMPT,
   pinned revision, seed + attempt×1000 (`gen_slot0_regen_candidates.py`);
   truncated or structurally invalid candidates (multiline included, never cut
   to the first line) fail; valid candidates are re-audited locally and only
   KEEP is accepted (`slot0_regen_loop.py`, max 6 attempts, resumable by replay).
   Rows unresolved after 6 attempts go to `regen/unresolved.tsv` and are left
   out of `regen/slot0_regen_candidate_corpus.tsv` (reported, not silently kept).
3. Luna spot check (`slot0_audit_crosscheck.py`): 500 random regenerated + 500
   random untouched rows, cap US$2 (~US$0.10 expected), full_v1 decisions
   reused only when the caption hash matches. Both strata KEEP rate ≥ 0.98 →
   PASS; otherwise REPORT_TO_OPERATOR with failing captions. Clopper-Pearson
   CI reported. PASS ≠ zero contamination; audio fidelity still unchecked.

The 5,000-row K sample and human pack from v2 were dropped per operator.
Chain: tmux `slot0_local_audit` restarted 13:42 with v3 script; offline tests of
loop replay/assembly/unresolved exclusion and spot-check dry run passed.

### v4 (2026-09-15 17:00) — definition A + Qwen2.5-32B-Instruct-AWQ; supersedes v3

Why v3 was stopped (16:24, before any full chunk): Qwen2.5-7B passed the
operational check (98.4% agreement) but caught **4/28** Luna-flagged rows in
full_v1 and 45% of heldout_v3 problems; misses were mostly metatext. The v3
spot check (KEEP rate ≥ 0.98 on 500 rows) could not have detected this: at a
~0.2–0.5% base rate a screen with zero recall still scores ~99.6%.

Operator decisions: **definition A** (flag contamination only: metatext,
instruction/prompt echo, requests/refusals, model commentary, unrelated
content, bare no-music, format wrappers, non-English; grammar slips and
contradictions are KEEP) and **32B local model**.

- `slot0_contamination_a.py`: binary KEEP/FLAG + category, shared by local screen
  and Luna. Of Luna's 28 full_v1 non-KEEP rows, 11 are A-contamination
  (hand-classified, listed in `build_slot0_defA_probe.py`).
- Probe v1 (dev, 442 rows: relabeled fixtures, Luna real 11 flag / 17 hard keep,
  300 random Luna-KEEP, 55 pattern-anchored "The caption … is:" / "The caption
  should" rows). First 32B prompt: recall 1.0 but **29% false flags** (read "The
  audio features …" as metatext). Prompt fixed (metatext = text referring to
  itself), then held-out **probe v2** (555 rows, seed 20260916, no v1 overlap):
  recall 1.0 (55/55), false flags 0/500; v1 also recall 1.0 / 0 false flags.
  Limitation: v2 positives are pattern-anchored (easy); subtle metatext recall
  is only evidenced on the dev set.
- vLLM needs `VLLM_USE_FLASHINFER_SAMPLER=0` on this host (no nvcc; FlashInfer
  sampler JIT crashes warmup). ~12 rows/s → full screen ≈ 5.8 h.
- Spot check redesigned as paired: 6,000 random ids, Luna-A reviews original and
  cleaned captions; PASS iff base ≥ 5, residual ≤ max(1, ⌊0.25·base⌋) and
  regenerated KEEP ≥ 0.98; base < 5 → INCONCLUSIVE. Also reports the local
  screen's recall on Luna flags. Projected ~US$0.65, cap US$2.
- Chain: tmux `slot0_local_audit`, output `local_qwen32b_defA_v4/`, started 16:59.

## PASS 後自動排 056 quarter（2026-09-15 17:19 設定）

操作者指示：「完成並通過後 自動接上 quarter 訓練」。

- gate：tmux `slot0clean_gate` → `scripts/runs/run_slot0clean_queue_after_pass.sh 771553`（PID gate，等 v4 chain 結束）。
  log：`local_qwen32b_defA_v4/queue_after_pass.log`。
- 只有 `spotcheck_report.json` verdict == `PASS` 才動作；INCONCLUSIVE / REPORT_TO_OPERATOR / INCOMPLETE / chain 失敗 → 什麼都不排，回報操作者。
- PASS 後：`build_slot0clean_arm_inputs.py`（驗 corpus sha = 抽樣時的 sha、id 順序 = source 扣掉 unresolved、非重生列與 source 逐位元相同、pandas/csv 解析一致）→ 寫 contract `docs/experiments/caption2p0_slot0clean_defA_quarter_cfg0_contract.json` → `accept_guest` 乾跑 → 才把 launcher 放進 `p2/pending/056_c2p0_slot0clean_defA_quarter.sh`。
- 訓練：`caption2p0_slot0clean_action.sh quarter`，配方與 055 slot4v2 完全相同（S1 100k + S2 50k、seed 14159265、NoQ、cap_index_fixed=0、require_text_overlay），只換語料；unresolved 列從 TSV 與 cache list 同步排除（deviation D1）。
- 判讀：CFG0 CLAP 對 c2p0 slot0 quarter 0.2029，±0.0084（2× seed floor）內算平手。只動約 0.2% 的列，**平手是預期結果**，不能據此說污染無害。

## v4 結果（2026-09-15 23:22）：REPORT_TO_OPERATOR，056 未排

- 本機 32B 全量：FLAG 1,492 + quarantine 7 = 1,499 列（0.60%）；抽看估計約一半是誤標（metatext 誤判 "The music features ..." 開頭）。
- 重生：1,499/1,499 被本機接受（attempt 1: 1,490、attempt 2: 9），unresolved 0，語料仍 251,599 列。
- Luna paired 抽查（US$0.546，882 batch 無失敗）：base 17/6000（0.28%，CI 0.17–0.45%）→ residual 5/6000（0.08%，CI 0.03–0.19%）；規則上限 floor(0.25×17)=4，差 1 列不過。
  - 本機召回（以 Luna 為準）12/17 = 0.71；被抓到的 12 列清洗後 0 殘留；重生 500 列 Luna KEEP 100%；probe 一致率 1.00。
  - 殘留 5 列全是本機漏抓的原句：1 列明確（"This audio does not contain any audible music."），4 列是邊界尾句（"limited information on the mix"、"genre cannot be definitively identified without more information"、"scene is set in a spacious studio"、"reminiscent of a retro album cover"）。
  - 本機在 S 中標 35 列、Luna 認為有問題的 12 列 → 以 Luna 為準的精確度約 34%。
- gate 依規則停止，未建 arm inputs、未放 launcher。

## 操作者放行，056 已排（2026-09-16 00:43）

使用者選「1，排 056」：接受 REPORT_TO_OPERATOR（residual 5 vs 上限 4）。build/action 加上 `--operator-override`，manifest 保留真實 verdict 並記錄放行文字；contract deviation `D2-spotcheck-not-pass-operator-accepted`。語料 251,599 列、1,499 列換成重生版本、unresolved 0（cache list 與原檔相同）。launcher `p2/pending/056_c2p0_slot0clean_defA_quarter.sh`，accept_guest 乾跑 ok。

## 056 停止 → 057 slot0nm（污染＋數字＋調性＋拍號）（2026-09-16 01:13 排入）

使用者：「BPM 一定不準…MusicLLM 沒有能力偵測 BPM…詳細數字也要去除，會污染訓練」；範圍選「數字＋調性＋拍號」，056 停掉直接換新 arm。

- 056 於 Stage 1 it ~2,050 停止（p2/failed，操作者指示，非 bug）。
- 語料：`rewrite_slot0nm_no_measurements.py`（Qwen2.5-32B-AWQ，逐句、hard gate、無量測的句子逐位元保留）。
  - 基底：未重生列用 slot4v2（數字已去、QA 過），重生的 1,499 列用重生版本。
  - 移除：數字/BPM/Hz/dB、調性/調式/和弦性質（key of X、in A minor、minor key、major chords）、拍號（time signature、common/waltz time）。保留：fast/slow 等定性速度詞、four-on-the-floor、年代詞。
  - 14,265 個不重複量測句：LLM 12,890、LLM 判整句無內容 591、正則裁切 323、逗號子句刪除 93、整句刪除 391。1 列所有句子都是量測 → 排除。
  - 修過的誤判：`piano keys` / `keys of the piano`、`a major role`（音名改大小寫敏感）、「The chord progression is primarily.」類副詞殘句。
- 結果：251,598 列，與原 slot0 不同 29,956 列（11.9%），全語料 MEASURE 殘留 0。
- 057：`caption2p0_slot0nm_action.sh quarter`，配方同 055；contract 記 D1（污染抽查未過、操作者放行）、D2（排除 1 列）、D3（056 停止，沒有「只清污染」的對照）。
- 判讀限制：兩個改動綁在一起；MusicCaps CLAP 不量 tempo/key/meter 遵循度，不能用來證明數字有害或無害。

## 057 slot0nm quarter — 完成（2026-09-16 00:34 CST，`status: completed`）

S1 100k（loss 0.988）→ migrate → S2 50k → eval。CFG0 report `passed`（5,521/5,521、16 kHz mono、
checkpoint sha `0f33f4d3…`）；CFG3+neg 5,521 檔，batch-32 CLAP 由 action 自動重算。

**CFG0（preregistered primary）**

| quarter CFG0 | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| **057 slot0nm（去污染＋去量測）** | **0.2060** | 6.1891 | 6.7038 | 5.1011 | 6.5672 |
| 055 slot4v2（只去數字） | 0.2004 | 6.0955 | 6.6823 | 5.0968 | 6.5178 |
| 052 slot4（污染語料、去數字） | 0.2050 | 6.1661 | 6.7525 | 5.0268 | 6.5832 |
| slot0（comparator） | 0.2029 | — | — | — | — |

vs slot0：CLAP +0.0031 = **0.74× seed floor**（floor 0.0042），落在登記平手帶 0.1945–0.2113 → **平手**。

**CFG3+neg（secondary，CLAP 用 batch 32）**

| quarter CFG3+neg | CLAP b32 | CLAP 逐檔 | CE | CU | PC | PQ |
|---|---|---|---|---|---|---|
| **057 slot0nm** | **0.2417** | 0.2314 | 6.8336 | 7.5070 | 4.7384 | 7.4360 |
| 055 slot4v2 | 0.2309 | 0.2206 | 6.7146 | 7.4070 | 4.8494 | 7.2998 |
| 052 slot4 | 0.2374 | 0.2285 | 6.8031 | 7.4767 | 4.7489 | 7.3663 |
| slot0 | 0.2372 | 0.2248 | 6.6952 | 7.3871 | 4.6661 | 7.3101 |

vs slot0：CLAP b32 +0.0045、逐檔 +0.0066；CE +0.1384、CU +0.1199、PQ +0.1259、PC +0.0723。
057 在六格裡有五格是四個 arm 中最高（PC 輸 slot4v2），且是唯一同時勝過 slot0 與 slot4v2 的 arm。

**判讀（observation 層）：**
- Primary（CFG0 CLAP）平手。CFG3+neg 全面偏正，但 CFG3+neg 的訓練 seed floor 是 **full 尺度、只有 2 個 seed**
  量的（先前已註記可能低估）；以 CFG0 floor 的 2 倍粗估（CLAP ≈0.008、CE ≈0.27、CU/PQ ≈0.10–0.16），上面的差距
  全部落在底線附近或以下。**不能宣稱贏。**
- slot4v2（只去數字）在 CFG3+neg 是四個 arm 最低，057 高出 +0.0108。兩者語料差別＝污染修復＋去調性/拍號。
  方向有趣，但同樣受限於底線不明，只能當觀察。
- 兩個改動（去污染、去量測）綁在一起，056 在 it ~2,050 被停，**沒有「只清污染」的對照**。
- MusicCaps CLAP 不量 tempo/key/meter 遵循度：本實驗不能證明移除量測有害或有益，只能說在 MusicCaps 上沒有可測損失。

**命名**：語料/實驗 ID 一律 `slot0nm`，表格顯示名「slot0nm（去污染＋去量測）」。不要叫 `slot5`（會被誤認為另一個
原始 caption slot），也不要叫「完全乾淨版」（污染抽查仍有 5/6000 殘留，且排除了 1 列）。

**若要把 CFG3+neg 那個正差變成可宣稱的結果**：需要 quarter 尺度、同協定的成對多 seed
（slot0 與 slot0nm 各 3 個新訓練 seed，共 6 次 quarter，每次約 7 h），逐對算 Δ；且兩 arm 應使用**相同的
251,598 個 id**（057 少一列會改變 sampler 順序），必須另立 contract，不可回頭改 057。

## 058/059 成對 seed（quarter，training seed 27182818，相同 251,598 id）— 完成（2026-09-16）

contract：`caption2p0_slot0nm_pairseed_quarter_cfg0_contract.json`（D4：單一 seed pair，只是重現性檢查，
不得報 p 值/CI/「贏」）。兩者 CFG0 report 皆 `passed`（5,521/5,521）；CFG3+neg 5,521 檔、CLAP b32 自動重算。

**CFG0（primary）**

| quarter CFG0，seed 27182818 | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| 058 slot0 | 0.1991 | 6.1455 | 6.6861 | 5.0551 | 6.5052 |
| **059 slot0nm** | 0.1977 | 6.0569 | 6.6639 | 4.9159 | 6.5167 |
| Δ（059−058） | −0.0014 | −0.0886 | −0.0222 | −0.1392 | +0.0115 |

**CFG3+neg（secondary）**

| quarter CFG3+neg，seed 27182818 | CLAP b32 | CLAP 逐檔 | CE | CU | PC | PQ |
|---|---|---|---|---|---|---|
| 058 slot0 | 0.2449 | 0.2303 | 7.0246 | 7.5871 | 5.0664 | 7.4575 |
| **059 slot0nm** | 0.2402 | 0.2272 | 6.7931 | 7.5008 | 4.7354 | 7.4256 |
| Δ（059−058） | −0.0047 | −0.0031 | −0.2315 | −0.0863 | −0.3310 | −0.0319 |

**兩個 seed 的 Δ 並排（slot0nm − slot0）**

| | CFG0 CLAP | CFG3 CLAP b32 | CFG3 CE | CFG3 CU | CFG3 PC | CFG3 PQ |
|---|---|---|---|---|---|---|
| seed 14159265（057 vs 歷史 slot0；id 集合差 1 列） | +0.0031 | +0.0045 | +0.1384 | +0.1199 | +0.0723 | +0.1259 |
| seed 27182818（059 vs 058；id 相同） | −0.0014 | −0.0047 | −0.2315 | −0.0863 | −0.3310 | −0.0319 |

**同 arm 換 seed 的差距（quarter CFG3+neg 底線的粗略量級）**

| | CFG0 CLAP | CFG3 CLAP b32 | CFG3 CE | CFG3 PQ | CFG3 PC |
|---|---|---|---|---|---|
| slot0：27182818 − 14159265 | −0.0038 | +0.0077 | +0.3294 | +0.1474 | +0.4003 |
| slot0nm：27182818 − 14159265 | −0.0083 | −0.0015 | −0.0405 | −0.0104 | −0.0030 |

**判讀（observation 層）：**
- CFG0 primary 仍是平手（−0.0014 = 0.33× floor 0.0042）。
- 057 的 CFG3+neg 全面偏正**沒有重現**：這個 seed 六格全部轉負。兩個 seed 的 Δ 正負相反、量級相近，平均約為零。
- 反轉幾乎全來自 slot0 在 seed 27182818 的 CFG3+neg 特別高（CE +0.33、PC +0.40 vs 自己的另一個 seed），
  slot0nm 兩個 seed 反而很接近。單一 arm 換 seed 在 quarter CFG3+neg 就能移動 CE ~0.3、PC ~0.4，
  比兩次 Δ 都大 → 057 的正差應視為 seed 雜訊。
- 兩個 seed 都沒看到 slot0nm 在 MusicCaps 上有可測損失；若要登記，比較適合的目標是「非劣性」而不是「較好」。
- CFG0 PC −0.139 與 CFG3 PC −0.331 同號，但 slot0 自己換 seed PC 就差 0.40，不可單獨解讀。

## 逐 clip 追查：slot0nm 的「崩掉」是生成近乎靜音（2026-09-17）

起因：試聽時發現差距常常不是好壞之分，而是其中一邊完全崩掉。腳本與原始輸出在 `silence_probe_20260917/`。

**1. 大部分逐 clip 大差距是雜訊**：arm Δ 的標準差（0.058/0.052）＝同 arm 換 seed（0.058/0.055）；Δ 跨 seed 相關只有 0.06。
但兩個 seed 都 |Δ|>0.15 的 clip 有 15 個（隨機期望 1.9），其中 9 個兩次都是 nm 較低、1 個兩次都是 nm 較高（事後挑選，只當觀察）。

**2. 那 9 個 clip 的音檔**：6/9 是 slot0nm 輸出近乎靜音（RMS −44～−67 dBFS，100 ms 框 RMS<0.01 的比例 0.86–1.00），slot0 同 prompt 是 −15～−36 dB。
全是稀疏 prompt：純鼓組、數拍子、didgeridoo。其餘 3 個（管風琴＋說話、說話蓋 pop、電子鼓＋bass）音量正常，但內容偏了。

**3. 全量 5,521 靜音掃描（RMS < −45 dBFS）**

| 協定 | seed | slot0 | slot0nm | 只有 nm 靜音 | 只有 slot0 靜音 | 兩邊都靜音 |
|---|---|---|---|---|---|---|
| CFG3+neg | 14159265 | 42 | 140 | 107 | 9 | 33 |
| CFG3+neg | 27182818 | 53 | 149 | 101 | 5 | 48 |
| CFG0 | 27182818 | 86 | 111 | 46 | 21 | 65 |
| CFG0 | 14159265 | (無音檔) | 48 | — | — | — |

- 跨 seed 同一批 clip：nm 兩個 seed 都靜音 88 個（獨立期望 3.8）；slot0 為 24 個（期望 0.4）。→ 是 prompt 綁定、跨 seed 重現的**語料效應**，不是單次訓練的偶發。
- 依 prompt 類型（CFG3+neg，seed 14159265 / 27182818）：純打擊 n=155：slot0 7.1/8.4% → nm 11.0/17.4%；
  只有人聲、無樂器 n=215：slot0 1.9/2.8% → nm **13.0/11.6%**；其他 n=5,151：0.5/0.7% → 1.8/1.9%。
- CFG3+neg 放大效應（CFG0 只差 +25，CFG3+neg 差 +98/+96）；nm seed 27182818 的 CFG3 靜音有 81/149 在 CFG0 也靜音。
- 靜音 clip 的 CLAP ≈0.08–0.10，其餘 ≈0.24。**把同 seed 任一邊靜音的 clip 剔除後**，CFG3+neg CLAP b32：
  seed 14159265 slot0 0.2402 / nm 0.2454（+0.005），seed 27182818 0.2477 / 0.2445（−0.003）。平均差很小，而靜音模式是最一致的差異。

**4. 語料端找不到明顯成因**：
- 純鼓 caption 的改寫率 9.9%，低於全體 11.9%；rhythm/groove/fast/slow 字眼保留率 ≈100%，只刪了 BPM/拍號/調性。
- 9 個 prompt 的 TF-IDF 前 200 個近鄰，改寫率 10.9%，與隨機 MusicCaps prompt 的 10.4% 相當。
- silence/quiet/faint/no sound 等字眼的列數幾乎不變。
- 尚未查：1,499 列污染重生列、以及 718 列刪掉 >20 字的列，對應的音訊是否偏安靜或稀疏（要讀 audio latent 或原始音檔）。

**判讀（observation 層）**：slot0nm 的差異主要不是平均品質，而是在稀疏 prompt（純打擊、只有人聲）上把輸出推向靜音，
CFG3+neg 會放大這個效應。這個效應在兩個 seed 上都重現，但目前還不知道語料裡的哪個改動造成。
在「平均 CLAP 平手」的表面下，這可以算是一個 **slot0nm 的劣化模式**，非劣性結論要加上這個 caveat。
另外 AES（PQ 等）對近乎靜音的輸出怎麼評分尚未檢查，靜音比例不同的 arm，AES 比較也可能被影響。

### 補查：改寫列對應的訓練音訊（2026-09-17，`silence_probe_20260917/train_audio_probe.py`）

讀 `segments_no_vocals` 原始 mp3 的前 10 秒，也就是 caption 描述的視窗。「安靜」＝ 超過 50% 的 100 ms 框 RMS < 0.01。

| 組 | n | 中位 dB | <−40 dB | 安靜 | vs 對照 |
|---|---|---|---|---|---|
| 污染重生列（slot0clean 的 1,499 列） | 1,499 | −17.3 | 3.07% | 68 (4.5%) | OR 2.59，Fisher p=3e-7 |
| nm 改寫後字數變多 | 840 | −17.5 | 3.45% | 40 (4.8%) | OR 2.73，p=6e-6 |
| nm 刪掉 >20 字 | 718 | −16.8 | 1.67% | 22 (3.1%) | OR 1.72，p=0.04 |
| nm 改寫列隨機 2,000 | 2,000 | −16.4 | 0.55% | 1.5% | — |
| 未改寫列隨機 3,000（對照） | 3,000 | −16.2 | 0.93% | 54 (1.8%) | — |

- 重生列確實集中在安靜或非音樂的片段（約 2.6 倍），但絕對量很小：兩組合併只有 **69 列安靜片段**，佔語料 0.03%。
- 這些列的 caption 改寫方向：抽樣可見原本描述水流、引擎、機械聲、「music is absent」的 caption，被重生成
  「soft ambient electronic with synth arpeggio / subtle percussion, production quality high」。
  但量化後是雙向的：安靜重生列裡，帶非音樂字眼的比例 26.5% → 17.6%（13 列失去、7 列新增）。
- **判讀**：語料裡「安靜音訊 ↔ 音樂描述＋高保真描述」的配對多了幾列，是一個候選機制，但只有數十列，
  不足以單獨解釋 MusicCaps 上 +100 個靜音輸出。也無法和「去量測」拆開，因為 056 slot0clean 在 it ~2,050 就停了。
  要確認必須補 slot0clean（只清污染）arm，或做一個把這 69 列還原成 slot0 caption 的 arm。

### 全語料靜音比例（2026-09-17，隨機抽 20,000 / 251,598 列，`silence_probe_20260917/window_silence_scan.py`）

照 latent 抽取的順序處理：整個 30 秒檔先 peak-normalize 到 0.95，再切 10 秒窗（`clips.tsv` 全是 `_0`、start=0 → 訓練只看前 10 秒）。「靜音」= 100 ms 幀 RMS<0.01 佔超過一半。

- 前 10 秒靜音：**0.76% ± 0.12**（全語料約 1,900 列）；>80% 幀靜音 0.22%（約 570 列）；整檔全靜音 0。
- 集中在 `segment_0`（曲子開頭）：4.5%；其餘 segment 0.08–0.24%。
- 注意：上表的 1.8% 對照是用**未正規化**的原始 RMS 算的，會高估模型實際看到的安靜比例；組間比較方向不變。
- 三窗：前 10 秒靜音的 151 列中，137 列後兩窗都有聲音。任一窗靜音 1.52%，中段窗只有 0.19%。

## 量測真的是亂猜的嗎？用音訊驗證（2026-09-18，`measurement_accuracy_20260918/`）

**變動列拆解**：29,956 列 = 重生 1,499 ＋ 只在去數字階段（slot4v2）改 14,190 ＋ 只在 nm 階段改 6,958 ＋ 兩階段都改 7,309。全語料的 token 變動率是 2.1%。

**對照音訊**：從 slot0 隨機抽出 600 列有 BPM 的、600 列有「X major/minor」的，對訓練用的 10 s 窗跑 librosa beat_track 和 Krumhansl 調性估計，再拿「在同一批內打亂 caption」當隨機基準（保留宣稱值的分布，2,000 次）：

| 宣稱 | caption 原值 | 打亂基準 95% |
|---|---|---|
| BPM ±8%（允許倍頻/半頻） | **0.748**（CI 0.71–0.78） | 0.264（0.23–0.30） |
| BPM ±8%（不允許倍頻） | 0.633 | 0.220 |
| 調性完全相同 | **0.370** | 0.065（0.05–0.09） |
| 同調號（含關係大小調） | 0.455 | 0.126 |
| 只看大小調 | 0.630 | 0.500 |

排除 120 BPM 後結果不變（0.742 vs 0.259），各速度區段也都成立。**結論：Qwen2.5-Omni 寫的 BPM 和調性明顯帶有和音訊相關的資訊，不是隨機亂填**。但這不代表它們準確：只看大小調的一致率只高出基準 0.13，而且 librosa 本身也有誤差（這兩件事不確定各佔多少）。「量測都是編的」這個前提被推翻了。

**嚴格條件重算（2026-09-18 追加，回應「±8% 太寬、大家都在 120 附近容易猜中」）**：`bpm_strict.py`、`madmom_tempo.py`、`bpm_madmom_cmp.py`。宣稱值的中位數正好是 120，37.7% 落在 110–130，11.3% 就是 120。所以另外加了三種基準：同曲風內打亂、在 60–200 之間挑最好的單一常數猜測，以及換用第二個估計器 madmom（RNN beat tracker；librosa 的 beat_track 預設以 120 BPM 為先驗，可能跟 caption 的 120 偏好一起造成假一致）。

| 條件（不允許倍頻） | caption | 全體打亂 | 同曲風打亂 | 最佳常數 |
|---|---|---|---|---|
| vs librosa ±2% | 0.232 | 0.040 | 0.055 | 0.125 |
| vs librosa ±4% | 0.508 | 0.112 | 0.139 | 0.237 |
| vs madmom ±2% | **0.428** | 0.057 | — | — |
| vs madmom ±4% | 0.595 | 0.097 | — | — |
| 兩估計器一致（±4%，n=390）的子集 ±2% | **0.518** | 0.070 | — | — |

- 容許誤差收緊後，和基準的比值反而變大（±2% 約 6–7 倍，±8% 約 3 倍）。排除宣稱值 110–130、或兩邊都排除 110–130 之後，結論都不變（±2%：0.230 / 0.271 vs 打亂約 0.035）。
- 對 madmom 的誤差分布（倍頻取最佳）：≤2% 51.5%、2–8% 33.7%、8–20% 10.0%、>20% 4.8%。不允許倍頻時 >20% 佔 22.5%，大部分是倍頻/半頻錯誤。
- 33% 的宣稱值帶小數（例如 117.45），人類標註不會這樣寫，比較像是模仿 beat tracker 的輸出；帶小數與整數兩組的命中率相同（0.42 vs 0.43）。
- **限制**：這裡比的是「和兩種 tempo 估計器一致」，不是和人工標註的真值一致。Omni 可能是在訓練時學過工具標出來的 BPM，所以會重現工具的行為，包括倍頻錯誤。Spearman 只有 0.23（兩估計器一致子集 0.54），原因是約半數很準、剩下的明顯偏掉，不是全體都差一點。

**nm 階段的附帶誤刪**：有 333 列的描述性字詞（情緒、曲風、旋律，例如 melancholic mood、folk genre）從整句 caption 消失，原因是 LLM 整段刪掉同一子句。另有 11 列因為正則殘留而變成 `primarilys`/`mainlys`。

**繼承自 slot4v2 的無效換詞**：slot0nm 的基底是 slot4v2，slot4v2 的規則是「`\d` 歸零」，所以非量測類的數字也被改掉了：年代 `80s` → `eighties`（slot0nm 裡的文字年代 944 次、數字年代 0 次）、`8-bit` → `chiptune`、`808` → `drum machine`、`12-bar` → `twelve-bar`。這些都是風格描述，不是聽不出來的量測，換掉沒有清洗效果。而且 MusicCaps 的寫法是數字形式（`80s` 類 44 次 vs 文字年代 5 次、`808` 21 次），換詞反而和 eval 的用詞分布拉開。若要重做，基底應改回 slot0，只刪量測類別。

## slot0nmv2：用更強的本地 LLM 重做去量測（2026-09-19）

使用者要求「用更強的本地 LLM 再做清洗，不要把之前 80s 年代的錯誤復現」。語料在 `~/exps_nvme/slot0nmv2/full/`，腳本 `scripts/preprocess/rewrite_slot0nmv2_measurements.py`（寫手）、`review_slot0nmv2.py`（獨立審查）。（2026-09-20 已建 arm_inputs 並排 3 seed 成對 quarter，見下節。）

**和 slot0nm 的差別**

- 基底改為 slot0clean 全列，不再疊在 slot4v2（`\d` 全歸零）上。
- 只刪量測：BPM／速度數字（含「tempo is in the high 120s」「tempo in the 80s」這類 BPM 區間）、拍號、調性／調式／和弦性質、音名＋八度、Hz／dB、時長與時間戳。
- 含數字的風格詞列為 PROTECT，逐句前後多重集合必須相等：年代（80s、1980s、'80s、80's、mid-1980s）、年份、N-bit、808/909/303、12-bar、12-string、20th century、16th note、4 on the floor、2-step。另有 18 處無法歸類的數字（2-2-2-2 structure、Yamaha YDP143、blink-182、21-year-old…）保留不動。
- 模型：Qwen3.6-27B（`cyankiwi/Qwen3.6-27B-AWQ-INT4`，vLLM 0.29，`max_num_seqs=64`，否則 Gated DeltaNet 的狀態在 cudagraph profiling 時 OOM）。舊版是 Qwen2.5-32B-AWQ。
- 閘門：新增字逐字比對（詞形變化只准換成語料中出現 ≥20 次的真字，`primarilys` 過不了）；量測區段外的內容字零遺失（只有緊貼數字的一個字、以及承載量測的動詞可以一起刪）；定性速度詞（moderate/fast/slow）必須保留；量測名詞（key/scale/signature/time）不得多於原句非量測部分；殘句規則（is with、a steady.、has a tempo.）。
- 裁判：同模型（非思考）對每個已接受的改寫判斷通順／新增描述（重組模式再加遺失描述）。這是操作性過濾，不是驗收。
- 流程演變：thinking 模式在「只准刪字」下只會留下殘句（The tempo, showcasing a moderate pace.），已放棄；改為難句走「重組」模式（可補功能詞與簡單動詞，不可加描述），最後 272 句才用 thinking＋重組，救回 98 句。

**結果**

| | slot0clean | slot0nm（舊） | slot0nmv2 |
|---|---|---|---|
| 列數 | 251,599 | 251,598 | 251,596 |
| 量測 regex 命中列 | 27,373 | 16 | **0** |
| 保留 token（年代/8-bit/808…） | 1,620 | **0** | **1,620** |
| 數字年代 | 996 | 0 | 985（少的 11 個是 BPM 區間） |
| 拼字年代（eighties…） | 6 | **944** | 6 |
| 808 ／ N-bit | 93 ／ 425 | 0 ／ 0 | 93 ／ 425 |
| `primarilys` 類殘字 | 0 | 11 | 0 |
| 與 slot0clean 不同的列 | — | 28,634 | 27,371 |

量測句 28,131 句（去重）：只刪字 23,607、重組 3,029、整句判 NONE 1,321、**無法在閘門內改寫而刪句 174**（`unresolved.json`）。其中 **3 列整段 caption 只有這一句，整列被排除**（`excluded_rows.json`：21_165121_segment_2_0、27_337827_segment_10_0、69_1295569_segment_8_0），所以 id 集合和 slot0 不完全相同，做成對比較前要處理。

**遺失內容（確定性逐字比對）**：只刪字的 23,607 句中 23,604 句量測區段外內容字零遺失；重組的 3,029 句中 2,236 句零遺失、738 句丟 1 字、55 句丟 2 字，丟的大多是承載動詞（alternates、giving、making、all），真正的描述字很少（moderate ×7、high ×7、complex ×6，另有像「somber key of D minor」這種修飾調性的形容詞會跟著消失）。

**獨立審查（Qwen2.5-32B-AWQ，分層抽樣：只刪字 600、重組 300；未觸及列 600）**

- 新增資訊：0/600、0/300（95% 上界 0.6%、1.2%）。
- 量測殘留：改寫句 0/900；未觸及列 1/600，而且這筆是誤報（four-on-the-floor），regex 漏抓率 95% 上界約 0.9%。
- 文法問題：只刪字 7/600（1.2%，CI 0.5–2.4%）、重組 18/300（6.0%，CI 3.6–9.3%），加權約 1.7%。抽看有真問題（「The tempo is creating an upbeat tempo.」「an instrumental rock piece and a chord progression」），也有誤報。
- **lost_info 指標作廢**：審查器把「刪掉調性與 BPM」這個本來就該做的事當成資訊遺失（NOTE 裡一再寫「B major 不是量測」），和自己的指示矛盾，60%／90% 不可用；改用上面的逐字比對。
- 限制：兩個 LLM 都不是真值；文字審查量不到 caption 是否和音訊相符；重組模式有少量屬性位移（「steady 4/4 time signature」→「a steady melody」）沒被抓到。

人工對照樣本：`~/exps_nvme/slot0nmv2/full/sample_100_changed_rows.md`。

## slot0nmv2 vs slot0clean：3 seed 成對 quarter（queue 066–071，2026-09-20 排入）

使用者：「製作 3 個 seed 的 quarter ablation」。

- **對照**：slot0clean（slot0nmv2 的底），兩臂只差去量測。不用原始 slot0，因為那會把污染清洗也綁進來（057–059 的問題）。
- **同 id**：兩臂都是 slot0clean 減掉 slot0nmv2 排除的 3 列，共 251,596 列，cache list 相同。`scripts/preprocess/build_slot0nmv2_pair_arm_inputs.py` 建兩臂 inputs，會擋：量測殘留、保護詞逐列不一致、非 caption 欄位被改、pandas/csv 讀法不一致。
- **overlay**：`~/text_overlays/slot0nmv2`（hardlink true_random，重編 28,703 列）；對照直接用 `text_overlays/slot0clean`。
- **配方**：與 057/058/059 相同（quarter：S1 100k + S2 50k、batch 8、lr 1e-4、NoQ、NoMask、`cap_index_fixed=0`、`require_text_overlay=true`），只換語料、seed，eval 改走 `scripts/eval/mc_mf25_eval.sh`（CFG0 + CFG3+neg，CLAP batch 1）。
- **Seeds / 排程**：14159265（066 clean / 067 nmv2）、27182818（068 / 069）、16180339（070 / 071）。每 job 約 7 h。
- **預登記判讀**（contract `docs/experiments/caption2p0_{slot0clean,slot0nmv2}_nmv2pair_quarter_s*_contract.json`）：報 3 個逐 seed Δ（nmv2 − clean）、平均與 t 95% CI（df=2）。CFG0 CLAP 的 CI 下界 > −0.0084（2× CFG0 floor）→ 非劣性；CI 不含 0 且 3 個同號才能說增益。n=3 的 CI 很寬，沒有結論就照實寫沒有結論，不寫成平手。跨 arm 看 AES/CLAP 時要附 `level_lufs_mean` 與 `level_silent_n`（slot0nm 在 CFG3+neg 有靜音模式）。
