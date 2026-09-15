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
