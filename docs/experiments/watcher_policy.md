# Experiment watcher and recovery policy

This document is the canonical policy for watchers and recovery controllers used
by long-running MeanAudio corpus builds, training runs, evaluations, and
conditional experiment chains. It complements
`docs/experiments/experiment_notification_policy.md`; that document defines the
experiment harness, storage gates, and notification events, while this document
defines how operational incidents are detected, deduplicated, investigated,
repaired, approved, and proven recovered.

This policy is deliberately implementation-neutral. A shell or Python watcher,
Pilotfish-Codex session, tmux session, or model prompt is not the source of truth.

## Responsibilities and sources of truth

The system has three distinct layers:

1. **Policy (this document):** repository-wide safety invariants and required
   behavior for monitoring and recovery.
2. **Per-experiment machine-readable contract:** experiment/run identity,
   phases, immutable scientific inputs, progress signals, thresholds,
   filesystems, notification events, repair authorization, retry budgets,
   resume commands, and recovery evidence.
3. **Runtime watcher/controller CLI:** deterministic polling, durable state,
   locking, notification delivery, bounded model dispatch, approved command
   execution, and recovery validation.

The contract specializes policy but cannot weaken it. The runtime CLI enforces
both policy and contract; it must not infer authorization from prose, logs, a
`.done` marker, or an LLM response. Experiment-specific scripts are adapters,
not alternate controllers.

The watcher observes and classifies. The controller owns incident transactions
and authorized actions. They may be deployed in one process only if their state
and responsibilities remain explicit and independently testable.

## Deterministic monitoring

Healthy and unchanged polls must make zero LLM calls. They use deterministic,
process-local code or monitoring tools to inspect only the assigned experiment:

- controller and process identity, including the experiment lock;
- current phase and phase-specific progress evidence;
- expected artifact existence, freshness, validity, and provenance;
- log age, bounded fatal signatures, and registered throughput or stall signals;
- GPU/process health without signaling unrelated processes;
- free bytes on every filesystem registered for the active or next phase;
- pending gate, notification, cleanup, and recovery states.

Polling must be quiet for healthy state unless the contract registers a progress
milestone. Status reports are written atomically. A watcher restart reconstructs
state from the contract, verified artifacts, and durable ledger rather than from
in-memory alert sets.

Stall rules are phase-specific. The contract must define the progress signal,
expected quiet interval, alert threshold, and hard-incident threshold for each
phase. Log freshness alone is insufficient when a process can emit output while
making unusably slow progress.

## Incident fingerprints and deduplication

Every new anomaly is normalized into a stable incident fingerprint. The
fingerprint includes at least the experiment/run ID, phase, incident category,
relevant invariant or signal, and material evidence identity. It excludes
volatile timestamps, PIDs, and repeated log noise unless they change the
diagnosis.

One fingerprint maps to exactly one durable incident transaction. Repeated
observations update that transaction without opening another transaction,
resending identical model context, or duplicating Discord alerts. A materially
different phase, invariant, artifact hash, failure signature, or failed recovery
may produce a new fingerprint with an explicit link to the prior transaction.
Only a material local change or explicit operator action may re-arm a suppressed
fingerprint.

The controller acquires an experiment-scoped lock before changing transaction
state or dispatching a model. It persists stage, attempts, model calls, evidence
hashes, candidate revision, review verdict, notifications, commands, and
validation results atomically so restart cannot reset a budget or repeat an
action.

## Bounded evidence

An incident evidence bundle contains only what is needed to reproduce and judge
the incident:

- the exact experiment contract and current controller/ledger state;
- assigned process and phase state;
- relevant machine-readable status and artifact reports with hashes;
- at most 100 relevant log lines;
- the failed invariant, observed value, threshold, and incident fingerprint;
- applicable repository revision and dirty-worktree disclosure.

Secrets, webhook URLs, unrelated user processes, and unrelated experiment data
must not enter evidence or model context. Evidence is immutable once submitted
for review; changed evidence requires a new bundle hash and review record.

## Durable bounded incident transactions

An incident transaction follows an explicit state machine such as:

```text
detected -> notified -> triaged -> candidate_ready -> review_pending
         -> approved -> applying -> resumed -> validating -> recovered
```

Any state may instead transition to `held_for_operator`, `rejected`,
`rollback_pending`, `rolled_back`, or `failed_closed` as applicable.

This is bounded rather than rigidly one-shot:

- The contract sets maximum model calls, wall time, transient retry count, and
  backoff before launch. A controller supplies conservative defaults and refuses
  an unbounded configuration.
- A transient retry is allowed only for classified infrastructure failures such
  as a model service timeout, rate limit, or notification transport error. It
  reuses the same immutable request and does not spend a retry revising code or
  widening authority.
- There is only one repair candidate revision per fingerprint. A failed test,
  malformed repair report, substantive reviewer `revise`/`reject`, exhausted
  budget, or uncertain classification closes automatic repair and holds for the
  operator. It must not start an iterative model debugging loop.
- Routine polling, duplicate fingerprints, and waiting states never invoke a
  model.

The purpose of the bound is to prevent repeated token spending and uncontrolled
experimentation on a live run while still tolerating a small number of harmless
transport failures.

## Pre-authorized repair envelope

Unattended recovery is permitted only inside a pre-authorized repair envelope
embedded in the experiment contract and approved by the responsible operator
before launch. The envelope must enumerate:

- writable repository paths or isolated worktree scope;
- permitted reversible, process-local configuration changes;
- exact command templates for tests, apply, rollback, and resume;
- allowed process identities and lock ownership checks;
- artifact and cache operations, including exact paths and ownership;
- model, time, retry, and cost budgets;
- required reviewer eligibility and approval evidence;
- conditions that always require an operator.

Automatic repair must not change data selection, corpus contents, seed,
hyperparameters, queue order, metric definitions, registered thresholds, or
other scientific-contract inputs. It must not delete checkpoints or shared
caches, act on another user's process, make system-wide changes, or start a
competing copy of the experiment. These actions are outside every automatic
envelope and fail closed pending explicit human authorization.

An absent, ambiguous, stale, or hash-mismatched envelope grants no authority.
For new runs, the envelope, approval bindings, incident ledger, and queue state
must also conform to the applicable `harn-schema-v1` branches. Schema or semantic
validation never authenticates the approval or authorizes repair execution.

## Model orchestration with Pilotfish-Codex

When an incident requires model judgment, Pilotfish-Codex is the preferred and
high-priority orchestration layer for role selection, model routing, context
isolation, approval boundaries, and fresh-context verification. Repository
policy should describe required outcomes rather than hard-code model product
names, tiers, or prompt choreography already governed by Pilotfish-Codex.

Pilotfish-Codex does not replace deterministic polling, the experiment contract,
the durable transaction ledger, controller locks, command allowlists, or runtime
enforcement. If Pilotfish-Codex is unavailable or not configured, the controller
must hold for an operator or use a contract-authorized equivalent workflow that
meets every policy requirement; it must not silently downgrade review.

A repair model receives bounded evidence and may produce one minimal candidate
in an isolated worktree or otherwise disjoint write scope. Its report must
include diagnosis, exact revision and diff hash, tests and results, scientific
contract check, rollback command, and exact proposed apply/resume commands.

## Independent review and approval

The reviewer must be independent of the repair context and eligible under the
contract. It receives the immutable evidence bundle, complete repair report,
exact candidate revision and diff hash, test evidence, rollback command, and
proposed apply/resume commands. It verifies safety, scope, contract preservation,
test sufficiency, and recovery criteria; it does not finish or iteratively debug
an incomplete repair.

No candidate may touch the live run, shared cache binding, or resume execution
until the controller records an explicit approval tied to the exact evidence
hash, revision, diff hash, and commands. A qualifying independent reviewer may
approve actions already inside the operator-approved repair envelope. Explicit
authorization by the responsible human operator is equivalent approval. Any
revision, command, evidence, or envelope change invalidates the approval.

After approval, the controller applies only the approved revision and commands,
while holding the appropriate experiment lock. Failure to prove exact identity
or lock ownership stops execution.

## Disk, stall, and Discord behavior

Disk measurements, warning and hard-stop behavior, gate notifications, terminal
coverage, promotion ordering, idempotency, and webhook handling must conform to
`docs/experiments/experiment_notification_policy.md`.

Additionally, the watcher must:

- monitor every filesystem registered for the current and next phase at the
  byte level;
- send one idempotent Discord event for a new hard incident, disk warning or
  hard stop, stalled phase, recovery start, recovery success, rollback, and
  terminal hold requiring an operator;
- include experiment/run ID, phase, fingerprint, decisive observation and
  threshold, report path/hash, current state, and next action;
- treat notification failure as a durable incident and retry it only within the
  contract's bounded transport policy;
- never allow a failed required notification to become an implicit promotion or
  successful recovery.

At a storage hard stop, unsafe stall, missing process identity, lost lock,
invalid gate, or unverifiable provenance, the controller blocks expansion of
compute or storage. Cleanup is limited to exact, recoverable actions registered
in the contract.

Notification severity and queue state are separate fields. `QUEUE HELD` is
reserved for a persisted controller/queue transition that verifiably stopped or
withheld the registered entry. If a storage gate blocks only the next registered
expansion while the current safe phase remains running, report an operational
advisory that names the blocked action and the observed queue state; do not call
the queue held. When the failed storage invariant later clears, deliver one
idempotent recovery event. That recovery event reports the cleared gate and must
not imply that a process was restarted or that the queue changed state.

## Recovery validation and rollback

Applying a patch or restarting a command is not recovery. After resume, the
controller reruns deterministic monitoring and requires evidence specific to
the failed phase and incident:

- training requires a new valid iteration or checkpoint with matching
  provenance after the resume point;
- evaluation requires newly valid metrics or a final report with matching
  provenance, not a stale training iteration;
- corpus or cache work requires the independent full validation gate and
  matching upstream/downstream hashes;
- notification recovery requires recorded delivery of the previously pending
  event;
- disk or stall recovery requires the failed invariant to clear for the
  contract's stability window.

The original hard incident must disappear and no new hard invariant may fail.
Validation results and artifact hashes are appended atomically to the ledger,
then a single recovery event is delivered.

If validation fails, the controller performs only the approved process-local
rollback when it is safe and contract-authorized. It then records the result and
fails closed or escalates a materially new fingerprint. It must not loop the
same repair, improvise a command, or accept partial progress as recovery.

## Fail-closed conditions

The controller stops automatic action and records `held_for_operator` or
`failed_closed` when any required contract, lock, state transition, evidence
hash, authorization, review, notification, provenance check, budget, or
recovery criterion is absent, inconsistent, expired, or exhausted.

Fail closed means preserving the assigned experiment and recoverable artifacts,
blocking the next expensive or destructive action, sending the required
deduplicated notification when possible, and waiting without model polling. It
does not mean killing an otherwise safe process unless that stop action was
explicitly preregistered and authorized.

## Acceptance criteria

Before a watcher/controller is assigned to a long run, no-GPU fixtures must
demonstrate:

1. healthy and duplicate polls make zero model calls and duplicate no alerts;
2. fingerprints and budgets survive controller restart;
3. a new incident creates one transaction and one bounded evidence bundle;
4. transient transport retries stop at their configured bound;
5. repair rejection, budget exhaustion, and missing authorization fail closed;
6. candidate identity and independent approval are checked before live apply;
7. disk warning, disk hard stop, phase stall, and notification failure branches;
8. resume neither duplicates the experiment nor skips a pending notification;
9. phase-specific recovery evidence is required and stale evidence is rejected;
10. failed validation follows the authorized rollback/hold path.

The acceptance report records fixture results and hashes of the policy,
experiment contract, watcher/controller, notifier, and model-orchestration
configuration used by the live run.
