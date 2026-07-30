# MeanAudio Agent Operating Rules

These rules apply to every agent, watcher, supervisor, and experiment launched
from this repository.

## Shared-host safety is a hard constraint

- This is a multi-user experiment host. Never reboot, shut down, suspend, change
  the system runlevel, reload/unload kernel modules, or restart shared GPU/system
  services on an agent's own authority.
- Never stop, signal, renice, reconfigure, or otherwise interfere with a process
  owned by another user or outside the explicitly assigned experiment.
- Do not use `sudo`, system-wide package changes, `systemctl`, `modprobe`,
  `rmmod`, or global driver/library replacement as an experiment repair.
- A reboot or shared-service interruption always requires explicit coordination
  and approval from the host operators and affected users. Do not recommend it
  as the default fix when a process-local workaround is possible.
- Prefer reversible, process-local remedies: scoped environment variables,
  workspace-local libraries, unique ports, resume-safe wrappers, and isolated
  worktrees. Validate the remedy without launching a duplicate experiment.

## Watchers must close the repair loop

A watcher is not merely an alarm. Its goal is to keep an authorized experiment
running continuously without routine human intervention while preserving the
scientific contract and shared-host safety.

1. Deterministic local monitoring detects and fingerprints a new incident.
   Healthy or unchanged checks use no LLM tokens.
2. Preserve bounded evidence: current contract, process state, relevant status
   JSON, and at most 100 relevant log lines.
3. For a repairable incident, assign a low-cost model to diagnose and implement
   a minimal fix in an isolated worktree or otherwise disjoint write scope.
4. Run syntax, unit, preflight, and non-invasive runtime checks. The repair must
   not alter data, seed, hyperparameters, queue order, or metric definitions.
5. Submit the exact diff/commit, evidence, tests, rollback, and proposed resume
   command to a Codex review model at Tera tier or above. No repair may touch
   the live run or resume execution until an eligible reviewer returns an
   explicit approval tied to that exact revision. Explicit authorization from
   the responsible human experiment operator is equivalent approval.
6. After approval, apply only the approved repair and command. Resume the same
   experiment contract; never start a competing copy.
7. Re-run the deterministic monitor and require evidence specific to the failed
   phase: a new iteration/checkpoint for training, or newly valid metrics/final
   report with matching provenance for evaluation/reporting. The original hard
   incident must also disappear. Never accept a stale training iteration as
   recovery from an evaluation incident. Roll back the process-local repair if
   validation fails, then escalate the new fingerprint instead of looping
   blindly.

## One-shot repair transaction

- The incident fingerprint is the transaction key. Persist its state and acquire
  the controller lock before launching any model.
- For one fingerprint, launch at most one low-cost repair agent and at most one
  Tera-or-higher review. Never open a second agent, resume an agent, or resend the same
  context for that fingerprint.
- The repair agent must finish diagnosis, minimal patch, tests, contract check,
  clean commit, diff hash, rollback command, and exact proposed command in one
  invocation. Tera-or-higher review is forbidden until every item is present and locally
  validated.
- A timeout, malformed report, failed test, or Tera-or-higher `revise`/`reject` is a closed
  transaction requiring human review. Do not automatically retry or create a
  replacement agent. Only a materially new local fingerprint may start a new
  transaction.
- The eligible reviewer receives only the bounded evidence, repair report, exact revision, diff
  hash, and commands. It must not be used to finish an incomplete repair or as
  an iterative debugging partner.

Automatic repair is limited to reversible changes within the assigned
experiment. Destructive actions, scientific-contract changes, system-wide
changes, shared-service changes, and actions affecting other users require
human approval.

## Model and token policy

- Routine polling is local Python/shell only.
- Healthy/unchanged checks use zero model calls. A new fingerprint gets at most
  one low-cost repair call plus one Tera-or-higher review; a stop-only candidate gets one Tera-or-higher
  call and no repair call.
- Never use `subagent_resume`, periodic LLM schedulers, or repeated context
  replay for monitoring. Persist fingerprints, call counts, stage, verdict, and
  report hashes locally so a controller restart continues locally.
- Suppress repeated identical incidents indefinitely until a materially new
  fingerprint or explicit operator action; do not spend tokens proving the same
  incident again.
