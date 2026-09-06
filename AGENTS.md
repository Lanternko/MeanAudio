# MeanAudio Agent Operating Rules

These rules apply to every agent, watcher, supervisor, and experiment launched
from this repository.

## Operating priorities

The goal is to keep a preregistered sequence of scientifically valid experiments
running reliably with minimal avoidable GPU idle time. Priority is strict:

1. shared-host and user safety;
2. scientific validity, provenance, and explicit authorization;
3. recoverability, storage safety, and required notification delivery;
4. continuous execution and GPU utilization.

Never run an invalid, unapproved, duplicate, or resource-conflicting experiment
merely to keep a GPU busy. A policy-required hold is correct behavior; record it,
notify it, and preserve the next safe action.

## Shared-host safety

- Without explicit host-operator authorization, do not perform actions that
  affect shared host services, drivers, kernels, system packages, or another
  user's processes.
- Never reboot, suspend, stop shared GPU services, or signal an unrelated
  process as an experiment repair.
- Prefer reversible, process-local changes, workspace-local dependencies,
  isolated worktrees, unique ports, and resume-safe commands.
- Do not start a competing copy of an experiment or interfere with resources
  outside the assigned contract.

## Scientific contract

- Data, corpus contents, seeds, hyperparameters, queue order, metrics,
  thresholds, and comparison rules are immutable after launch unless the
  responsible operator explicitly approves a new contract.
- Every long-running corpus build, training run, evaluation, and experiment
  chain requires a preregistered machine-readable contract and state machine.
- Shell and Python drivers are implementations, not sources of truth. Logs and
  `.done` files are diagnostic hints, not proof of a completed gate.
- Generated-corpus work must fail closed and conform to
  `docs/experiments/generated_corpus_policy.md`.

### Canonical evaluation protocol

- Every new primary or fair-comparison evaluation uses MusicCaps 5,521,
  MeanFlow, 25 solver steps, literal `cfg_strength=3`, the fidelity negative
  prompt registered in `docs/experiments/evaluation_policy.md`, generation seed
  42, NoMask, and full precision unless the responsible operator explicitly
  approves a separately named secondary protocol.
- CFG 0 and CFG 4.5 results are historical protocol artifacts. Preserve their
  labels and provenance, but never use them as the new canonical comparator or
  silently relabel/reuse them as CFG 3 + negative prompt.
- Every result label, report, contract, and command manifest must encode the
  resolved CFG value and negative-prompt identity. A new canonical entry whose
  resolved protocol is not MusicCaps 5,521 / MeanFlow 25 / CFG 3 / registered
  fidelity negative prompt must fail closed before GPU launch.
- This protocol revision applies only to evaluations registered after
  2026-08-31T18:13:35+08:00. Already launched or queue-registered contracts keep
  their preregistered protocol and order unless the responsible operator
  explicitly authorizes a replacement contract.
- The detailed contract is `docs/experiments/evaluation_policy.md`.

## Harness and continuous queue

Every long run must conform to
`docs/experiments/experiment_notification_policy.md`. Before launch and before
each compute/storage expansion, verify the scientific design, commands and
configs, data and artifact provenance, harness branches, resume behavior,
notification delivery, and byte-level capacity on every writable filesystem.

Keep one durable, executable queue of operator-approved experiments. A backlog
file without a live controller, registered launcher, and terminal-to-next
transition is not an experiment queue. The controller must prepare the next
eligible run before the current run ends and launch it immediately after the
current terminal notification and resource release, subject to that next run's
own preflight, resource lock, storage, provenance, and notification gates.

Queue mutation rules are mandatory:

- A newly approved experiment is appended to the tail by default. New ideas do
  not replace, cancel, or silently reorder already approved work.
- An experiment moves ahead only when the responsible operator explicitly says
  to prioritize, insert, or interrupt. Record the instruction and resulting
  order durably before launch.
- Priority insertion is temporary ordering, not queue truncation. After the
  inserted work reaches a terminal state, the controller must immediately
  resume the preserved remainder in its recorded order.
- Completion, failure, or interruption of one entry must always cause an atomic
  queue transition: persist terminal evidence, deliver the required event,
  release owned resources, select the next eligible entry, and either launch it
  or persist and notify the exact hold reason. A controller may not exit merely
  because one entry completed while approved successors remain.
- Queue ordering and scientific dependency are different contracts. An
  `ordering_dependency` is satisfied by any verified terminal state, including
  failure or interruption; a scientific `dependency` is satisfied only by a
  verified successful completion. A failed entry must never deadlock the whole
  queue: skip science-blocked entries, launch the first later eligible entry,
  and keep every blocked entry durably visible with its exact reason.
- A child failure may terminate that child HARN, but it must not terminate the
  top-level queue controller. Resource waits and retryable launch holds remain
  pollable states and must not be converted into permanently ineligible entries.
- Every queued entry must have an approved scientific contract, an executable
  HARN launcher/state source, dependency metadata, and a tested resume path
  before it can be considered prepared. The controller must surface an
  unprepared next entry while the current run is still active.

Experiment start, completion, failure, interruption, queue handoff, unexpected
GPU idle, stall, disk warning/hard stop, every gate result, and every
promote/stop/hold decision require idempotent Discord events.

## Watchers and recovery

Every long run must conform to `docs/experiments/watcher_policy.md`. Healthy or
unchanged polling uses deterministic local monitoring and zero model calls.
Persist incident fingerprints, notification state, controller locks, and
phase-specific recovery evidence so restarts neither repeat actions nor lose
events.
