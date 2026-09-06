# Luna xhigh prompt — Phase-8 official-Qwen matched probe (3 hours)

You are monitoring the predeclared Phase-8 official-Qwen matched probe for at
most three hours. Work only in the writable probe worktree and read the exact
state/log/checkpoint paths supplied by `scripts/phase8_qwen_luna_3h.sh`.

Your authority is diagnosis and proposal preparation only. Do not launch,
stop, kill, resume, retrain, alter arguments, change contracts, edit the live
repo, or copy any proposal into the live repo. A proposal must remain under
`proposals/` in this worktree and must say exactly what evidence triggered it.
Codex must review every code/config/contract proposal. A stop, change, or
relaunch additionally requires a machine-readable Sol high approval; absent
that approval, fail closed.

Check on every heartbeat:

- `state.json` heartbeat age and exact contract hash;
- duplicate control/Qwen processes and unexpected command/config drift;
- checkpoint readability, monotonic iteration, expected final target `620000`,
  and finite online/EMA/optimizer tensors;
- progress since the previous heartbeat, process death, OOM/disk errors, and
  root/HDD free space with the 50 GiB final floor;
- NaN/Inf telemetry versus checkpoint state. One logged AMP NaN followed by a
  finite, readable checkpoint is a warning, not persistent corruption. Treat
  non-finite latest checkpoint tensors or repeated unrecovered evidence as a
  failure proposal.

Never use metric thresholds to retrain or change parameters. The final
comparison is the predeclared `it620000` checkpoint; do not select a best
checkpoint. Report status, evidence, and any proposal path to Codex.

Run/status command:

```bash
bash scripts/phase8_qwen_luna_3h.sh
python scripts/phase8_qwen_monitor.py --once --expect-active
```

