# Grok watcher contract: Phase-8 Fixed-Q9 vs matched-NoQ

Use `/grok-watcher` and create exactly one durable recurring job, every five
minutes, for this experiment. The five-minute cadence is a deterministic local
observer that runs the monitor and reads its state; it is not an LLM scheduler
and must not spend an LLM call merely to decide whether to poll. The scheduler
must survive SSH/local disconnects.

Authoritative runtime contract:

- Live root: `/home/kojiek/MeanAudio`
- Queue tmux: `p8_fixedq_attm`
- Queue order: `fixedq9` → `matched noq` → paired CLAP → final comparison
- Monitor: `/home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/monitor_phase8_fixedq_attm_ft.py --once`
- Status: `/home/kojiek/logs/phase8_fixedq_attm_monitor/status.json`
- Alert: `/home/kojiek/logs/phase8_fixedq_attm_monitor/ALERT.json`
- Final: `/home/kojiek/logs/phase8_fixedq_attm_monitor/FINAL_COMPARISON.json`
- Stop adjudicator: `bash /home/kojiek/MeanAudio/scripts/adjudicate_phase8_fixedq_attm_stop_with_codex.sh`

Shared-host hard rule:

- Never reboot, shut down, reload GPU modules, restart shared services, use
  system-wide package changes, or signal another user's process.

Every local observation:

1. Run the monitor once and read the new status/alert. The five-minute trigger
   is only the deterministic local observer; use an LLM only for a new,
   materially changed incident or an explicitly bounded review.
2. Never duplicate or alter the scientific contract. A new incident may start
   one bounded repair workflow: low-cost model proposal in an isolated worktree,
   then Codex SOL review tied to the exact commit and command.
3. Stay quiet when healthy and unchanged. Report stage transitions, each 10k
   checkpoint, evaluation start, final metrics, review observations, and incidents.
4. An isolated AMP `grad_norm:nan/inf` followed by finite values is review-only.
   Loss near 0.98 is not a failure. Stale/process/GPU observations require two
   consecutive current observations.
5. If and only if the current monitor reports `incident`, run the stop adjudicator.
   Do not stop unless its fresh verdict says `decision=stop` and
   `stop_authorized=true`, then re-run the monitor and require the same incident
   before sending one Ctrl-C to the queue tmux. Never use kill/pkill.
6. If code or a new experiment is warranted, create a separate `grok/*` branch
   and worktree, commit a proposal and tests, and hand it to Codex for review.
   Execute nothing without a fresh SOL verdict containing `decision=approve`,
   `execution_authorized=true`, the matching commit, and an exact command.
   An approved reversible repair may be applied only while the affected run is
   absent, then resume that same run and verify forward iteration/checkpoint
   progress. Roll back and re-adjudicate if validation fails.
7. Never reinterpret `program_goal_met` as proof that Q helps. The Q claim is
   supported only by `fixedq_benefit_supported=true` (paired CI95 lower bound >0).

After creation, run `scheduler_list`, verify exactly one matching job, observe
the first fire, and report the scheduler job ID without changing training.
