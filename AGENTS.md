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
   command to Codex SOL. No repair may touch the live run or resume execution
   until SOL returns an explicit approval tied to that exact revision.
6. After approval, apply only the approved repair and command. Resume the same
   experiment contract; never start a competing copy.
7. Re-run the deterministic monitor and require evidence of forward progress.
   Roll back the process-local repair if validation fails, then escalate the new
   fingerprint instead of looping blindly.

Automatic repair is limited to reversible changes within the assigned
experiment. Destructive actions, scientific-contract changes, system-wide
changes, shared-service changes, and actions affecting other users require
human approval.

## Model and token policy

- Routine polling is local Python/shell only.
- Invoke a low-cost model only for a new or materially changed incident.
- Invoke SOL only to review a concrete repair/stop proposal with bounded
  evidence; never use SOL as a recurring poller.
- Suppress repeated identical incident calls and persist fingerprints/verdicts.

