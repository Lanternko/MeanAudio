---
name: grok-watcher
description: Safely supervise long-running MeanAudio experiments with local event-driven monitoring. Grok is reserved for new incidents and terminal reports, never routine polling or session-resume loops.
---

# Grok Watcher

Supervise experiments without changing their scientific contract. Local scripts own
routine polling and completion notifications. Grok is not an inexpensive poller:
each resumed watcher reloads context and consumes model quota.

## Token budget is a hard constraint

- Default to **zero recurring LLM calls**. Use a local Python/shell observer plus
  the Discord completion/failure wrapper.
- Never create a five-minute Grok, Codex, Luna, or Sol scheduler.
- Never use `subagent_resume` for monitoring. Each incident review must be a new,
  compact request containing only bounded current evidence.
- Do not reread project history, long handoff documents, or prior session context
  on every observation.
- A healthy or unchanged observation must cause no model call.
- One experiment may have at most one local observer and zero LLM schedulers.
- Stop the observer when the experiment reaches a terminal state.
- If the user explicitly requests recurring LLM review, require an interval of at
  least 60 minutes, a maximum of 12 reviews per day, and automatic deletion at
  completion. State the expected quota cost before enabling it.

## Preserve the role boundary

- This is a shared host. Never reboot/shut down the host, reload GPU modules,
  restart shared services, use system-wide package changes, or signal another
  user's processes. These actions always require explicit operator coordination.
- Let deterministic local code monitor and fingerprint incidents.
- Let a low-cost model diagnose a new incident and prepare the smallest
  reversible repair in an isolated worktree.
- Let Codex SOL review stop candidates and exact committed repair proposals.
- Never edit, switch branches, merge, or apply patches in the live `/home/kojiek/MeanAudio` worktree while a pipeline may read it.
- Never start a second copy of an experiment.
- Never treat a monitor exit code as permission to stop training.
- After SOL approves an exact revision and command, the watcher may apply that
  approved process-local repair while the affected pipeline is absent, resume
  the same immutable experiment contract, and verify forward progress.

## Resolve the experiment contract

1. Identify the tmux session, experiment prefix, stage targets, logs, monitor command, audit command, and canonical handoff file.
2. Read the canonical handoff once during setup. Write a compact immutable watcher
   contract containing paths, hashes, thresholds, and terminal conditions; routine
   checks read that compact contract instead of the handoff or conversation.
3. Treat runtime Hydra, the immutable launch contract, and transition audits as the source of truth. Do not infer Q, mask, data, seed, batch, LR, checkpoint, or eval settings from names alone.
4. Refuse to invent stop thresholds for an experiment without a canonical contract. Draft the contract first and submit it for Codex review.

For the active Phase8 clean-NoQ experiment, use:

- tmux: `p8_catalog_noq`
- prefix: `phase8_catalog_matched_noq`
- handoff: `/home/kojiek/MeanAudio/docs/experiments/phase8_clean_noq_grok_handoff_2026_07_19.md`
- monitor: `/home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/monitor_phase8_clean_noq.py --once`
- status: `/home/kojiek/logs/phase8_catalog_matched_noq_monitor/status.json`
- alert: `/home/kojiek/logs/phase8_catalog_matched_noq_monitor/ALERT.json`

## Create or update one local observer

1. Use `scheduler_list` to find and delete any LLM scheduler for the same
   experiment. Inspect tmux and processes for orphaned `grok --resume`,
   `/grok-watcher`, and recurring `codex exec` loops.
2. Start exactly one deterministic Python/shell observer. A 5–15 minute local
   polling interval is acceptable because it consumes no model tokens.
3. The observer must:
   - run the monitor and write a bounded `status.json`/`ALERT.json`;
   - send Discord only on phase change, incident fingerprint change, completion,
     failure, or interruption;
   - never launch a duplicate or change the scientific experiment contract;
   - an isolated recovered AMP gradient overflow does not authorize stopping;
   - never call an LLM for healthy or unchanged state;
   - create proposals only in isolated `grok/*` worktrees;
   - execute nothing proposed until Codex SOL explicitly approves the exact
     revision and command;
   - after approval, apply only reversible experiment-scoped changes, resume the
     same run, and confirm a new checkpoint/iteration before closing the incident.
4. Verify there are no LLM watcher processes or scheduler jobs after setup.
5. Observe the first local check. Confirm a healthy run stays untouched and no
   model session is created.

## Classify observations

Treat the experiment-specific monitor as authoritative, with these minimum semantics:

- `healthy`: make no changes. Report only configured milestones, stage transitions, review/hard issues, eval start, and final metrics.
- `review`: preserve evidence and continue monitoring. Do not stop.
- `incident candidate`: fingerprint and preserve bounded current evidence. Request
  one human/LLM adjudication only when the fingerprint is new or materially
  changed. Do not stop yet.

For AMP training, classify one `grad_norm:nan/inf` followed by finite loss and finite gradients as a skipped GradScaler optimizer update. Keep it at review severity. Treat gradients as persistent/dense only when the canonical monitor says so; the Phase8 thresholds are trailing >=2, recent 20 >=3, or recent 100 >=10.

Require a second local observation for stale log, missing process, or GPU-idle
incidents unless the process has already terminated. Never use an old ALERT as
current evidence. Repeated identical incidents do not authorize repeated LLM
calls; suppress them for at least six hours.

## Adjudicate a stop candidate

1. Preserve `status.json`, `ALERT.json`, contract audit, process/tmux/GPU/disk
   state, and at most the most recent 100 relevant log lines. Keep the complete
   adjudication input below 20 KiB.
2. Run an adjudicator only once for a new incident fingerprint. For Phase8:

   ```bash
   cd /home/kojiek/MeanAudio
   bash scripts/adjudicate_phase8_stop_with_codex.sh
   cat /home/kojiek/logs/phase8_catalog_matched_noq_monitor/codex_sol_verdict.json
   ```

3. Do not signal training if the command fails, the verdict is invalid or older than ten minutes, the decision is `continue` or `escalate`, `stop_authorized` is not true, or the process is already absent.
4. Re-run the monitor after a `stop` verdict. Send one Ctrl-C only when the same current incident remains and the fresh verdict contains both `decision=stop` and `stop_authorized=true`.
5. Record the incident fingerprint and verdict so the local observer cannot
   adjudicate it again during the six-hour cooldown.
6. Never use `kill`, `pkill`, or `tmux kill-session` as the first action. Never
   restart or alter parameters after stopping without a separate approved
   decision.

## Draft fixes and new experiments

1. Keep the live worktree untouched. Create a separate worktree under `/home/kojiek/grok-worktrees/<slug>` on a `grok/<slug>` branch.
2. Modify, test, and commit only inside that worktree.
3. Add a committed proposal containing:
   - evidence and diagnosis;
   - falsifiable hypothesis and one controlled variable;
   - fixed baseline and complete train/eval contract;
   - exact commands, unique artifact prefix, metrics, and gates;
   - GPU/time/disk budget;
   - stop and rollback policy;
   - test results.
4. Require a clean worktree, then run:

   ```bash
   bash /home/kojiek/MeanAudio/scripts/review_grok_proposal_with_codex.sh \
     /home/kojiek/grok-worktrees/<slug> \
     /home/kojiek/grok-worktrees/<slug>/<proposal-file>
   ```

5. Reject execution when review fails, the commit changes after review, the verdict is stale, or the verdict says `revise` or `reject`.
6. Execute only the exact `approved_command` when the verdict matches the current commit and contains `decision=approve` plus `execution_authorized=true`. Do not add parameters or compete with the active experiment for GPU or artifacts.
7. For an approved repair, record the incident fingerprint, reviewed commit,
   applied diff/hash, command, rollback command, and post-resume observation.
   Require deterministic evidence of forward progress. If validation fails,
   roll back only the experiment-scoped change and submit a new fingerprint;
   never retry the same failed repair indefinitely.

## Safe automatic repair boundary

Allowed after exact SOL approval:

- process-local environment/library selection;
- unique port or lock correction within the assigned experiment;
- resume-wrapper, monitor, or parser fixes that preserve the immutable train and
  evaluation contract;
- restarting only the assigned experiment when it is already absent.

Never automatic:

- reboot, shutdown, suspend, kernel-module reload, shared-service restart;
- `sudo`, system package installation/removal, or global library replacement;
- stopping or modifying another user's process;
- changing data, seed, model, schedule, batch, queue order, metrics, or claims;
- deleting checkpoints or launching a competing experiment.

## Report concisely

Report the experiment, phase, iteration/target, latest finite metrics, log age,
GPU, disk, contract status, issues, local observer PID, and any adjudication.
Explicitly state whether training was changed. Stay quiet on unchanged healthy
checks. Never attach session history or the full handoff to a report.
