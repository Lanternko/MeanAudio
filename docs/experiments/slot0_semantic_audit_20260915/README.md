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
