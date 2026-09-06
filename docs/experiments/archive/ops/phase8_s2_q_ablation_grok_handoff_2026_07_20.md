# Phase8 S2-only Real-Q / Shuffled-Q sequence — Grok watcher contract

## Objective and immutable design

Test whether the Q **information** improves over the completed catalog-matched
NoQ baseline (MusicCaps CLAP 0.1888), rather than merely adding an embedding or
an in-support token. Both arms reuse the exact completed S1 NoQ checkpoint at
iteration 400,000. They train independent S2 models from 400,000 to 600,000.

1. `phase8_catalog_matched_s2_realq`: original aligned per-row `q_level`.
2. `phase8_catalog_matched_s2_shuffledq`: fixed seed 424242 permutation of
   `q_level` only; row order, id, caption, NPZ/cache mapping and Q histogram are
   unchanged.

Both arms are single-cap, NoMask, batch 8, accumulation 1, LR 1e-4, seed 42,
and `use_q_conditioning=true` in S2 only. MusicCaps q9 is the pre-registered
primary evaluation and q6 is secondary. The sequence must run Real-Q first and
Shuffled-Q second. A scientifically poor metric is still a valid result and
must not prevent the second arm. Only a failed contract/runtime may stop the
sequence.

## Runtime adapter

- tmux: `p8_s2_q_ablation`
- sequence log: `/home/kojiek/logs/phase8_s2_q_ablation_sequence.log`
- monitor: `python /home/kojiek/MeanAudio/scripts/monitor_phase8_s2_q_ablation.py --once`
- status: `/home/kojiek/logs/phase8_s2_q_ablation_monitor/status.json`
- alert: `/home/kojiek/logs/phase8_s2_q_ablation_monitor/ALERT.json`
- Codex stop review: `bash /home/kojiek/MeanAudio/scripts/adjudicate_phase8_s2_q_stop_with_codex.sh`
- final comparison: `/home/kojiek/logs/phase8_s2_q_ablation_monitor/FINAL_COMPARISON.json`

## Grok watcher instructions

Use `/grok-watcher`. Maintain exactly one durable recurring five-minute job
for this sequence and retire the completed clean-NoQ watcher. Every fire:

1. Run the monitor once and read status/alert.
2. Never launch another experiment, edit the live worktree, alter Q/data/mask/
   seed/LR/eval, delete artifacts, or skip/reorder an arm.
3. Stay quiet while healthy except at each 25k S2 iteration, arm transition,
   eval q9/q6 transition, review/incident, and final comparison.
4. An isolated recovered AMP gradient overflow is review-only.
5. For any incident candidate, preserve evidence and run the Codex stop review.
   A monitor nonzero exit is not stop authority.
6. Stop only if a verdict newer than ten minutes says both `decision=stop` and
   `stop_authorized=true`, then re-run the monitor and confirm the same issue.
   Send one Ctrl-C to `p8_s2_q_ablation`; never kill/restart automatically.
7. Grok may draft fixes only in `/home/kojiek/grok-worktrees/<slug>` on a
   `grok/<slug>` branch. It may not apply or execute them without Codex's
   structured approval through `review_grok_proposal_with_codex.sh`.

Interpret final q9 results without changing the preregistration:

- Real-Q >= 0.1998: historical-best target met.
- Real-Q >= 0.1938: meaningful +0.005 over NoQ.
- Evidence that Q information is useful requires Real-Q to beat **both** NoQ
  0.1888 and Shuffled-Q; report CLAP, CE, CU, PC and PQ for q9 and q6.
