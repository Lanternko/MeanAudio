# Experiment harness, storage, and notification policy

This is the canonical operational specification for every new long-running
MeanAudio corpus build, training run, evaluation, and conditional experiment
chain.  Experiment-specific shell or Python drivers implement this contract;
they do not override it.

## Required state machine

Before launch, the experiment contract must enumerate:

- a stable experiment and run ID;
- every phase and its immutable inputs, outputs, completion evidence, and resume
  behavior;
- every decision gate, its input reports, exact thresholds, and `pass`, `fail`,
  and `invalid` branches;
- every filesystem written by each phase and the peak additional byte estimate;
- every notification event and its idempotency key;
- authorized cleanup and cache-restoration behavior;
- terminal reports for success, failure, interruption, and post-run audit
  failure.

Every new contract, preflight report, event ledger, and queue state must conform
as one `harn-schema-v1` bundle under `docs/experiments/schemas/` and pass
`scripts/validate_experiment_harness_documents.py`. Validation success is not
authorization: the runtime controller must independently authenticate approval,
verify raw hashes and process/resource ownership, acquire the registered lock,
and repeat mutable checks immediately before launch.

The durable state is a machine-readable, atomically replaced JSON event ledger.
Log messages and `.done` markers are diagnostic hints only.  On resume, the
harness reconstructs its position from verified artifacts plus the ledger; it
must not repeat a completed notification or skip an undelivered one.

## Continuous experiment queue

Long-run utilization is managed by one durable queue controller, not by ad hoc
tmux launches. The machine-readable contract registers the ordered candidate
runs, dependencies, operator approval, resource requirements, expected duration,
preflight report, and terminal action for each queue entry. The controller owns
an experiment-scoped lock and records the assigned GPU and process identity.

The queue is executable state, not a planning list. Every nonterminal entry must
bind an approved experiment contract to a HARN launcher, durable state source,
preflight command, dependency set, and resume behavior. A JSON backlog whose
entries cannot be promoted by the running controller fails this policy.

Queue entries use two separate prerequisite sets:

- `ordering_dependencies` encode sequence only and are satisfied after the
  named entry reaches any verified terminal state (`completed`, `failed`, or
  `interrupted`) with its terminal notification delivered;
- `dependencies` encode scientific or artifact validity and are satisfied only
  after the named entry reaches verified `completed` state.

Failure or interruption must trigger a new eligibility scan across the entire
remaining queue. A science-blocked entry remains waiting with durable blocker
metadata while the controller considers later approved entries. The whole queue
may enter `dependency_hold` only when no remaining entry is eligible, and that
hold must be persisted and notified without stopping deterministic polling.

### Queue insertion and preemption

- Default insertion is append-only at the queue tail. This applies whenever an
  operator contributes a new experiment idea without an explicit priority
  instruction.
- Front or interior insertion requires an explicit operator instruction. The
  controller records the instruction hash, previous order, new order, insertion
  time, and affected entry IDs before the new order takes effect.
- Insertion never discards or implicitly cancels displaced entries. Their
  relative order remains stable unless the operator explicitly changes it.
- If an operator interrupts active work, the contract must define whether that
  entry resumes or terminates. After the inserted entry completes, the
  controller returns to the first eligible entry in the preserved queue.
- Ideas that are not yet scientifically approved may be recorded separately as
  proposals, but they are not launch-eligible and must not displace approved
  work.

Prepare and validate the next candidate before the current run ends. A queue
entry is launch-eligible only when its scientific contract is approved, inputs
and provenance are current, preflight and storage gates pass, required prior
notifications are delivered, its resources are available, and no matching run
is already active or complete. Queue order is immutable unless the responsible
operator approves and records a replacement contract.

After a terminal state, persist the terminal report, deliver its required
notification, perform only registered audit/restoration actions, release owned
resources, and atomically promote the next eligible entry. When approved
successors remain, the queue controller must stay alive across entry boundaries;
an experiment-scoped child controller may exit, but that exit cannot terminate
the top-level queue controller. The next launch happens without waiting for a
new model turn or operator reminder. A notification hold, invalid gate, missing
approval, stale provenance, storage hard stop, resource conflict, or uncertain
process identity blocks launch even when the GPU is idle; the controller must
persist and notify that exact hold rather than silently stop.

The handoff transaction is:

1. Validate and persist the current entry's terminal evidence.
2. Deliver its idempotent terminal notification.
3. Release only its owned resource lock.
4. Atomically mark it terminal and scan for the first entry whose ordering and
   scientific prerequisites are satisfied; retain blockers for skipped entries.
5. Revalidate the next entry's contract, launcher, approval, provenance,
   storage, notifications, duplicate guard, and resource availability.
6. Deliver an idempotent `queue_handoff`/start event and launch the next HARN.
7. If step 5 or 6 fails, persist `queue_hold`, notify the blocker and next safe
   action, and continue deterministic polling without model calls.

A child initialized into a resource-wait state remains a live queue entry. Its
controller may poll deterministically without claiming the GPU; after resources
become available it must repeat mutable preflight checks before recording
ownership. Controller restarts preserve this state and must not duplicate the
child or its notifications.

When an assigned GPU unexpectedly becomes idle and no launch-eligible process
owns it, persist one idempotent `gpu_idle` transition and send Discord with the
current queue state, blocking reason, and next action. Repeated polls do not
repeat the event. Resolve or escalate the blocker; never fill the GPU with an
unapproved, invalid, duplicate, or resource-conflicting run.

## Storage gates

Run a storage gate before initial launch and again before any phase that adds
training iterations, checkpoints, cache materialization, or generated audio.
For every affected filesystem, record:

- measured free bytes and timestamp;
- estimated peak additional bytes for the next phase;
- atomic-write/transient duplication;
- checkpoint, optimizer, EMA, evaluation-audio, cache, and log retention;
- recovery reserve;
- warning and hard-stop floors.

Unless an experiment contract registers a stricter value, the hard-stop floor
is the larger of 50 GiB and 1.25 times the estimated remaining peak write.  An
operator-approved exception must be explicit in the contract; a filesystem
showing `100%` by rounded `df -h` output is not itself an exception or a precise
capacity measurement.

The watcher repeats the byte-level measurement during execution.  It sends one
deduplicated warning when the warning floor is crossed.  At the hard-stop floor,
the harness blocks the next phase.  It may remove only exact transient artifacts
listed in the contract; shared caches, checkpoints, and unrelated experiments
are never opportunistically deleted.

## Notification events

Every new MeanAudio experiment must be launched through:

```text
scripts/run_with_experiment_report.sh
```

or an equivalent outer supervisor that survives the child process group. It
sends exactly one idempotent Discord report when the complete experiment
sequence:

- finishes successfully;
- exits with a failure;
- is interrupted by HUP, INT, TERM, or a child exit associated with an
  interruption.

That terminal report is necessary but not sufficient.  The harness or its
independent watcher must also send exactly one idempotent Discord event for:

- terminal success, failure, or interruption, keyed durably by experiment/run
  ID and terminal state across supervisor restarts and fresh invocations;
- each experiment start after process identity and resource ownership are
  recorded;
- every gate result: `pass`, `fail`, or `invalid`;
- every resulting `promote`, `stop`, or `hold` decision;
- every unexpected assigned-GPU idle transition and queue launch hold (Discord `--status held` / QUEUE HELD, never `--status failure`);
- disk warning and disk hard-stop transitions;
- post-run audit or cache-restoration failure;
- recovery after a previously notified operational incident.

Each event includes the experiment/run ID, phase, verdict, decisive values and
thresholds, report path/hash, and the action that will happen next.  Routine
progress polling stays silent unless the experiment contract explicitly
registers progress milestones.

### Promotion transaction

A promotion is committed in this order:

1. Validate all gate inputs and atomically persist the machine-readable report.
2. Append a `gate_result` event with a stable idempotency key to the ledger.
3. Deliver Discord and record the accepted response for that event.
4. Append the `promotion_started` transition.
5. Launch the promoted phase.

If step 3 fails, keep the state at `notification_pending`, retry with bounded
backoff, and do not begin the promoted expensive phase.  A human operator may
explicitly override this hold in the ledger.  Printing `[PROMOTE]` to a log does
not satisfy the transaction.

The report contains experiment name, time, host, Git revision, duration, exit
code, a bounded failure-log tail, and registered metrics when the final JSON
report exists. Every new Stage-2 training arm is unfinished until it has MusicCaps 5521 MeanFlow-25 CFG4.5 metrics (CLAP, AES CE/CU/PC/PQ), via caption10s_pipeline/eval_musiccaps_mf25.sh. NoQ arms pass --no_q; Q-conditioned arms report at least q0 and q9. This is the Caption 2.0 fair-compare protocol; MF1-only numbers are not sufficient. Discord mentions are disabled unless explicitly configured by
the operator.

The webhook URL is local-only:

```text
/home/kojiek/.config/meanaudio/discord_webhook_url
```

Required permissions are `0600`. The URL must never be put in Git, command-line
arguments, experiment contracts, logs, or final reports.

## Watcher and recovery requirements

- An independent watcher records phase, progress, process identity, log age,
  GPU health, free bytes on every registered filesystem, and new fatal
  fingerprints without spending model tokens on healthy polls.
- It persists notification fingerprints before/reliably around delivery so a
  restart cannot silently lose or repeatedly spam an event.
- Mutable shared caches require an owner lock, a detectable binding state, and
  a contract-authorized restore path on gate stop, crash, or interruption.
- A notification delivery error is itself a harness incident.  Record its
  response/error without exposing the webhook, retry with bounded backoff, and
  surface it through an operator-approved alternate channel when available.

## Pre-launch acceptance tests

No long run may launch until no-GPU harness fixtures pass for:

1. gate pass plus delivered promotion notification;
2. gate fail and gate invalid, both stopping before the promoted phase;
3. notifier failure holding promotion in `notification_pending`;
4. disk warning and hard-stop branches using injected free-byte values;
5. terminal success, ordinary failure, TERM/INT/HUP, and child exit 137,
   including exactly-once delivery across supervisor restart/reinvocation;
6. resume after interruption without duplicate notification or duplicate run;
7. mutable-cache detection and restoration, when applicable;
8. one eligible queued launch with exactly-once start notification;
9. duplicate, completed, invalid, unapproved, and resource-conflicting queue
   entries holding without launch;
10. promotion-notification failure holding the next queued phase;
11. exactly-once unexpected-idle notification across controller restart.
12. failed and interrupted ordering predecessors promoting the next eligible
    entry while completed-only scientific dependencies remain blocked;
13. scanning past a blocked entry to a later eligible entry, plus restart-safe
    resource waiting and exactly-once terminal-to-next handoff.

The acceptance report records fixture results and hashes of the contract,
controller, notifier, and watcher used by the live run.

Example:

```bash
scripts/run_with_experiment_report.sh \
  --experiment phase8_halfq_quarter_e2e \
  --report /home/kojiek/logs/phase8_halfq_quarter_e2e_FINAL_METRICS.json \
  --log /home/kojiek/logs/phase8_halfq_quarter_e2e_sequence.log \
  -- bash scripts/training_pipelines/sequence_phase8_halfq_quarter.sh
```
