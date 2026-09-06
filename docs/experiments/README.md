# Experiment documentation registry

This directory contains policy, executable experiment contracts, scientific
results, and historical operational material. This registry is the navigation
entry point for those documents. It is not itself an experiment contract and
does not authorize a launch, resume, stop, repair, or cleanup action.

## Authority and precedence

Repository-wide experiment governance starts in `../../AGENTS.md`. The
canonical second-layer policies are:

| Policy | Sole responsibility |
| --- | --- |
| `experiment_notification_policy.md` | Experiment lifecycle, durable queue, launch eligibility, storage gates, event taxonomy, notification delivery, and harness acceptance tests. |
| `watcher_policy.md` | Deterministic monitoring, incident fingerprints, bounded recovery transactions, repair authorization, independent review, and recovery proof. |
| `generated_corpus_policy.md` | Generated-corpus structure, full-corpus gates, validator behavior, and corpus-to-cache provenance. |

An experiment-specific contract may specialize these policies but cannot weaken
them. Historical reports, prompts, handoffs, proposals, status summaries, and
this registry never grant runtime authority. When documents disagree, use this
order:

1. `AGENTS.md` safety and scientific-contract rules;
2. the three canonical policies above;
3. the approved machine-readable contract and its durable ledger;
4. experiment-specific operator notes and implementation scripts;
5. historical analysis, handoffs, prompts, and summaries.

## Document model

Documentation and runtime state are separate systems:

- this registry answers what a document is, where it belongs, what it replaces,
  and whether it may still be relied upon;
- the validated contract, preflight report, queue state, and event ledger answer
  what is running, held, complete, failed, or eligible to run next.

Neither a folder name nor Markdown metadata can authorize a run or determine its
live state. Missing or contradictory runtime evidence is reported as `unknown`;
it is never reconstructed from prose, process names, tmux sessions, logs,
checkpoints, or `.done` files.

### Writing a new document

Write one document for one primary responsibility. Separate observations,
evidence, decisions, and proposed next actions, and link evidence instead of
copying mutable status between files. New Markdown documents begin with YAML
front matter followed by one H1 heading:

```yaml
---
document_id: ma-expdoc:operator-note:example-runbook:v1
document_family: ma-expdoc:operator-note:example-runbook
document_type: operator-note
lifecycle: preregistered
authority: contract-bound
owner: responsible-operator
created_at: 2026-08-11
updated_at: 2026-08-11
review_after: 2026-08-18
expires_at: 2026-08-25T00:00:00+08:00
experiment_ids: [example-experiment]
run_ids: [example-run]
supersedes: []
superseded_by: []
canonical_refs:
  - document_id: ma-expdoc:policy:watcher:v1
    path: docs/experiments/watcher_policy.md
relationships:
  - type: depends_on
    document_id: ma-expdoc:design:example:v1
    path: "docs/experiments/designs/<experiment_id>/<design_id>.md"
evidence_refs: []
contract_ref: "docs/experiments/contracts/<experiment_id>/<contract_id>.json"
ledger_ref: null
---
# Example runbook
```

`document_id` identifies one immutable edition and must be repository-unique.
`document_family` remains stable across editions. IDs use lowercase ASCII,
colon-separated components, and a positive `vN` suffix. A move does not change
an ID; a material replacement gets a new edition ID and reciprocal
`supersedes`/`superseded_by` links. All paths are repository-relative POSIX
paths. A relationship records both the target ID and path when the target is a
governed Markdown document; the ID detects accidental retargeting and the path
makes the link usable.

Relationships are directional and typed:

| Field or type | Meaning |
| --- | --- |
| `canonical_refs` / `governed_by` | The target constrains this document. |
| `depends_on` | This document is not complete or usable without the target. |
| `explains` | This document adds human interpretation without replacing the target. |
| `derived_from` | This document's claims or data were produced from the target. |
| `supersedes` | This edition intentionally replaces the target edition. |
| `superseded_by` | Reciprocal pointer from the replaced edition to its replacement. |
| `evidence_refs` | Immutable artifacts supporting a claim; include path/URI, hash, and producing run where applicable. |
| `contract_ref` / `ledger_ref` | Binds a contract-scoped document to the immutable design and runtime evidence. Neither grants authority by itself. |

Use the narrowest relationship. `supersedes` must have a reciprocal
`superseded_by`, may not form a cycle, and never means that earlier evidence is
deleted. General background links belong in the prose and do not need a
governance relationship.

The front matter above is the target grammar for newly governed Markdown. The
current legacy collection is not retroactively conformant. Missing or duplicate
IDs therefore classify a new document as `review-needed`, while an existing
legacy file retains the classification explicitly recorded in this registry.
Automated uniqueness and relationship checking belongs to the future document
catalog described below; until then, reviewers check both this registry and
repository references.

### Required metadata by class

| Document class | Required behavior |
| --- | --- |
| Repository policy | `authority: repository-policy`, an owner and `review_after`; `expires_at` is normally null. Policy review lateness blocks new launches that depend on the stale rule. |
| Experiment design or preregistration | Experiment IDs, owner, canonical policy references, and a review date before launch. It proposes a contract but does not authorize one. |
| Contract-bound operator note or prompt | Contract/run references, owner, `review_after`, and a finite `expires_at`. It cannot outlive or broaden its contract. |
| Result or completed scientific record | Experiment/run IDs and immutable evidence references. Use `expires_at: null`; scientific evidence is retained even when superseded. |
| Historical or archived operational record | Provenance and replacement links when known. Use `authority: none` and `expires_at: null`; archival classification already removes operational authority. |
| Planning reference or estimate | Source and measurement date plus `review_after`. After that date it must be revalidated before operational use. |

Machine-readable HARN documents use their JSON schemas rather than this
Markdown header. Their IDs, hashes, bindings, and authority rules remain those
defined by `schemas/` and the canonical policies.

## Placement and migration

The directory now uses a transitional mixed layout. Static designs, results,
history, and safe archived operations use responsibility-based subdirectories.
Policies, contracts, queue/backlog inputs, script- or skill-bound prompts, and
canonical indexes remain at their stable root paths until their runtime
consumers receive a separately verified migration. Do not move a file merely
to make the tree look tidy. Migration incrementally adopts this target layout:

```text
docs/experiments/
  README.md
  policies/
  schemas/
  contracts/<experiment_id>/<contract_id>.json
  designs/<experiment_id>/
  results/<experiment_id>/
  history/
  archive/ops/
  registry/documents.json
```

Contracts live at stable paths based on identity, never in lifecycle folders
such as `active/`, `held/`, or `completed/`. Runtime state changes in the queue
and ledger without moving, copying, or rewriting the authoritative contract.
For example, an `active` run that becomes `held` and later `completed` keeps the
same contract path and hash; only authenticated queue state and ordered ledger
events change.

Choose a destination from responsibility, not age: policies govern all runs;
contracts define one immutable run design; designs explain hypotheses; results
preserve evidence and conclusions; history preserves non-operational scientific
context; `archive/ops` preserves expired prompts and handoffs. Every move is a
separate reference-migration transaction: enumerate inbound references, update
them atomically, validate them, and only then remove the old path.

## Document lifecycle

| Label | Meaning |
| --- | --- |
| `canonical` | Current repository-wide policy or scientific index. |
| `preregistered` | Designed before launch, but not launch-authorized by this label alone. |
| `completed` | Retained scientific result or completed contract record; never resume without a new approved contract. |
| `historical` | Evidence or analysis retained for provenance, not current operational instruction. |
| `superseded` | Replaced by a named newer edition; retained for provenance and no longer authoritative. |
| `stale` | Its `review_after` has passed; revalidate before relying on it for a new launch or decision. |
| `expired` | Its `expires_at` has passed; it has no operational authority and must not be executed. |
| `archived-ops` | Old prompt, handoff, watcher, or reviewer instruction; must not control a current run. |
| `reference` | Supporting data or estimates, not a contract or gate. |
| `review-needed` | Lifecycle or conformance is ambiguous; fail closed and obtain operator classification. |

`active`, `held`, `failed`, and `interrupted` are runtime states, not document
labels. Only a machine-readable contract whose policy/runtime hashes, approvals,
preflight report, queue state, and durable ledger all agree may back an active
run. A filename or Markdown lifecycle label is insufficient.

Age alone never makes a document obsolete. `review_after` means the content must
be checked before new operational reliance; it does not erase history.
`expires_at` ends operational authority at that instant; it does not delete the
file. `superseded_by` identifies an intentional replacement. Completed results
and historical evidence do not expire, although newer work may supersede their
interpretation. Any inconsistent timestamps or broken reciprocal replacement
links yield `review-needed`.

Classification examples:

| Evidence | Classification and action |
| --- | --- |
| An operator note is past its finite `expires_at`. | `expired`; retain it, but do not execute it or use it to resume a run. |
| A completed result has `expires_at: null` and intact evidence links. | `completed`; retain it permanently, even if later interpretation supersedes it. |
| A new document has a missing or duplicate `document_id`. | `review-needed`; do not register or rely on it until identity is repaired. |
| A legacy run lacks any member of the validated HARN bundle. | Its document stays `review-needed` and its live state is `unknown`; do not infer completion or activity. |

## Current contract registry

The common `harn-schema-v1` bundle now exists under `schemas/`, but no legacy
contract has been migrated to it. Treat the following as `review-needed` until
migrated and reconciled with their preflight, queue, and ledgers:

| Contract | Existing runtime relationship | Registry note |
| --- | --- | --- |
| `rich_shared_then_matched_full_contract.json` | Read by `scripts/training_pipelines/sequence_rich_shared_then_matched_full.sh`. | Contains `preregistered_before_launch`; reconcile with current terminal report/ledger before classifying completed. |
| `phase8_qwen_official_matched_contract.json` | Read by the official-matched probe queue and monitor. | Legacy schema; lifecycle status is absent. |
| `phase8_qwen_dose_contract.json` | Read by the dose queue and monitor. | Legacy schema; lifecycle status is absent. |
| `phase8_fixedq_attm_chain_proposal.json` | Proposal for the completed fixed-Q/No-Q chain. | Historical contract candidate; status is absent and final comparison lives outside the repository. |

Queue controllers must not infer launch eligibility from this table. They must
validate the contract and ledger under `experiment_notification_policy.md`.

## Common schemas and validator

The structural bundle and its security boundary are documented in
`schemas/README.md`. New harnesses use all four documents together:

| Document | Responsibility |
| --- | --- |
| `schemas/experiment-contract-v1.schema.json` | Immutable experiment design, phases, resources, commands, corpus mode, repair mode, and required events/checks. |
| `schemas/preflight-report-v1.schema.json` | Fresh approval bindings, required checks, storage measurements, and derived preflight verdict. |
| `schemas/event-ledger-v1.schema.json` | Ordered lifecycle and notification events with stable identity and hash links. |
| `schemas/queue-state-v1.schema.json` | Ordered queue entries, dependencies, resource assignment, terminal delivery state, and document bindings. |

`scripts/validate_experiment_harness_documents.py` performs offline structural
and semantic checks. Passing it never authenticates approval or authorizes an
action; runtime enforcement remains mandatory.

## Scientific indexes and references

These documents summarize results or define scientific interpretation. They do
not authorize runtime actions.

| File | Label | Purpose |
| --- | --- | --- |
| `phase_status.md` | `canonical` | Current scientific status and invalidation ledger across phases. |
| `best_results.md` | `canonical` | Current benchmark/result index with historical caveats. |
| `caption_provenance_granularity_and_aes_controls.md` | `canonical` | Current caption-provenance and AES-control design guidance. |
| `training_time_estimates.md` | `reference` | Runtime estimates only; remeasure before resource planning. |
| `results/benchmarks/ten_exp_metrics.tsv` | `reference` | Data table supporting the ten-experiment benchmark record. |

## Completed and historical scientific records

The following files preserve preregistrations, results, audits, or design
history. Embedded commands and thresholds are historical context unless copied
into a newly approved contract that conforms to current policy.

| File | Label |
| --- | --- |
| `history/phase4-phase8/Phase4_to_Phase8_Complete_Summary.md` | `historical` |
| `results/benchmarks/ten_exp_full_benchmark.md` | `completed` |
| `designs/phase8/exp_h_rewrite_spec.md` | `completed` |
| `results/phase8/music_flamingo_ablation_todo.md` | `completed` |
| `results/phase8/p8_qwen_completion_2026_05_21.md` | `completed` |
| `history/phase9/phase9_design.md` | `historical` |
| `history/phase9/phase9_5_summary.md` | `historical` |
| `history/phase8/qwen_collapse_audit_10model.md` | `historical` |
| `history/phase8/qwen_collapse_root_cause_2026_05_08.md` | `historical` |
| `history/phase8/qwen_rerun_summary.md` | `historical` |
| `history/phase8/phase8_baseline_forensics_2026_07_17.md` | `historical` |
| `history/phase8/phase8_post_legacy_comparison_2026_07_22.md` | `historical` |
| `history/phase8/s2q_bug_audit_2026_07_23.md` | `historical` |
| `history/phase8/phase8_halfq_quarter_2026_07_23.md` | `historical` |
| `results/phase8/phase8_qwen_fullq_halfq_quarter_2026_07_23.md` | `completed` |
| `phase8_qwen_bucket_quarter_backlog_2026_07_26.md` | `completed` |
| `results/phase8/caption10s_multisent_quarter_2026_08_09.md` | `completed` |
| `phase8_fixedq_attm_chain_proposal.md` | `completed` |

## Archived operational material

These files describe superseded Grok/Luna/Sol workflows or one-run handoffs.
They remain for incident provenance, but current agents, watchers, and
controllers must not execute their instructions as policy.

| File | Label |
| --- | --- |
| `archive/ops/SOL_QWEN_LIVE_REVIEW.md` | `archived-ops` |
| `phase8_clean_noq_grok_handoff_2026_07_19.md` | `archived-ops` |
| `archive/ops/phase8_s2_q_ablation_grok_handoff_2026_07_20.md` | `archived-ops` |
| `archive/ops/phase8_qsafe_ft_grok_handoff_2026_07_21.md` | `archived-ops` |
| `phase8_fixedq_attm_grok_watcher_prompt.md` | `archived-ops` |
| `phase8_fixedq_attm_luna_loop_prompt.md` | `archived-ops` |
| `phase8_fixedq_attm_sol_review_prompt.md` | `archived-ops` |
| `phase8_qwen_luna_3h_prompt.md` | `archived-ops` |
| `phase8_qwen_sol_adjudication_prompt.md` | `archived-ops` |
| `phase8_qwen_dose_luna_prompt.md` | `archived-ops` |
| `phase8_qwen_dose_sol_approval_prompt.md` | `archived-ops` |
| `phase8_qwen_dose_sol_incident_prompt.md` | `archived-ops` |
| `phase8_qwen_bucket_repair_luna_prompt.md` | `archived-ops` |
| `phase8_qwen_bucket_repair_sol_prompt.md` | `archived-ops` |
| `archive/ops/phase8_qwen_official_matched_operator.md` | `archived-ops` |

## Known second-layer gaps

The registry exposes, but does not solve, these migration tasks:

1. Migrate eligible legacy contracts to the common schema bundle without
   changing their scientific contracts or inferring lifecycle state.
2. Define one event taxonomy and producer for each lifecycle, disk, gate, stall,
   recovery, and GPU-idle event so harness and watcher cannot double-send.
3. Define how the queue controller and incident controller share process
   identity, locks, ledgers, and authority without competing for ownership.
4. Migrate storage contracts from fixed floors to measured peak-write,
   transient-duplication, retention, and recovery-reserve fields.
5. Make generated-corpus gate reports explicit queue eligibility dependencies.
6. Migrate legacy runtime stacks to the common schemas and prove conformance
   with the required no-GPU fixtures before assigning a new long run.

## Fleet-wide experiment view

A reliable overview must be generated, not maintained by editing Markdown. The
future read-only fleet view has two distinct inputs:

1. `registry/documents.json`, generated from governed metadata, answers which
   documents exist, their class, owner, lifecycle, review/expiry condition, and
   relationships.
2. Validated HARN contract, preflight, queue, and ledger bundles answer each
   run's runtime state, assigned GPU, current phase, last durable event,
   notification delivery, storage status, blocker, and next eligible action.

The generated view must include `generated_at` and evidence freshness, show
`unknown` for incomplete or conflicting bundles, and remain read-only. It may
highlight unexpected GPU idle time, stalls, terminal events without delivered
Discord notifications, disk pressure, and an empty approved queue, but it may
not launch, repair, promote, or reclassify a run.

This README is only the human-readable registry until those two generators are
implemented. Building the machine document catalog and the fleet-view command
are separate future slices; neither is silently approximated from the legacy
files listed here.

## Maintenance rules

- Add every new experiment document to this registry in the same change.
- For new Markdown, assign metadata according to its class and verify that its
  ID is unique and all relationship targets exist.
- New operational prompts are temporary artifacts and require an owning
  contract, expiration condition, and final archive classification.
- Change document lifecycle labels only from governed metadata, valid
  replacement links, or explicit operator disposition. Declared
  `review_after`/`expires_at` may yield `stale`/`expired`; do not infer either
  from filename or apparent age. Derive runtime state only from HARN evidence.
- Archive or move files only after updating all inbound references and verifying
  that no active contract, script, ledger, or report depends on the old path.
