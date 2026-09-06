# Phase8 Q-safe residual fine-tuning — Grok watcher contract

## Objective and falsifiable design

This sequence starts only after the current `phase8_s2_q_ablation_sequence`
has completed both arms and written its final comparison.  It does not repeat
the destructive S1-NoQ → random-q S2 transition.

Both new arms start from the exact completed NoQ MeanAudio S2 checkpoint
(`phase8_catalog_matched_noq_stage2_200000`, it=600000, MusicCaps CLAP
0.1888). Before training, q_embed rows 0..9 are copied exactly from the trained
null row q=10 in the online weights and both EMA tracks. Thus every q is a
function-preserving NoQ initialization at step zero.

1. `phase8_qsafe_realq_ft100k`: aligned per-row MeanSim q_level.
2. `phase8_qsafe_shuffledq_ft100k`: seed-424242 q-only permutation with the
   same histogram, audio, text, row order, initialization and compute.

Both continue MeanAudio for exactly 100k updates (600k→700k), LR 3e-5,
batch 8, accumulation 1, seed 14159265 (Hydra/NoQ default; earlier draft
wrote seed 42 but train.py never received it — Real-Q already ran under
14159265), single-cap, NoMask. Real-Q runs first.
MusicCaps q9 is primary; q6 is secondary. After both arms, a paired per-prompt
CLAP bootstrap compares Real-Q q9 against Shuffled-Q q9 and the existing NoQ
audio using 20,000 resamples.

Pre-registered interpretations:

- restoration: Real-Q q9 CLAP >= 0.1900;
- Q information supported: lower bound of paired 95% bootstrap CI for
  Real-Q minus Shuffled-Q is > 0;
- net Q gain supported: the above plus lower bound of Real-Q minus NoQ > 0;
- primary objective met if restoration or net Q gain is supported. Report a
  poor result unchanged; never move the q target or select a checkpoint on
  MusicCaps.

## Runtime

- queue/training tmux: `p8_qsafe_ft`
- watcher tmux: `p8_qsafe_grok_loop`
- queue log: `/home/kojiek/logs/phase8_qsafe_ft_queue.log`
- sequence log: `/home/kojiek/logs/phase8_qsafe_ft_sequence.log`
- monitor: `/home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/monitor_phase8_qsafe_ft.py --once`
- status/alert: `/home/kojiek/logs/phase8_qsafe_ft_monitor/{status,ALERT}.json`
- final: `/home/kojiek/logs/phase8_qsafe_ft_monitor/FINAL_COMPARISON.json`
- Codex SOL: `bash /home/kojiek/MeanAudio/scripts/adjudicate_phase8_qsafe_stop_with_codex.sh`

### 2026-07-21 seed-contract continuation

Real-Q finished train+eval (q9 CLAP 0.1823) but final audit failed because the
contract claimed `seed=42` while Hydra ran `seed=14159265`. Training data were
not corrupted (NoQ baseline uses the same default). Fix:

1. pipeline now passes `seed=14159265` to `train.py`; audit expects 14159265;
2. Real-Q contract corrected and re-audited to `passed` (artifacts retained);
3. continuation entrypoint (Shuffled-Q + paired bootstrap only):
   `bash scripts/training_pipelines/continue_phase8_qsafe_ft_from_shuffled.sh`
   under tmux `p8_qsafe_ft`.

## Grok watcher rules

Every five minutes, read this file completely, run the monitor once, and read
status plus ALERT if present.

1. Never start a second experiment, reorder/skip arms, edit the live worktree,
   alter Q/data/init/LR/seed/eval, or delete artifacts.
2. Healthy queued/training/eval states stay quiet. Report only queue→start,
   each 25k fine-tune milestone, arm/eval transition, incident review, paired
   bootstrap, and final comparison.
3. Isolated recovered AMP grad overflow and monitor nonzero exit are not stop
   authority.
4. For any incident, preserve evidence and call the Codex SOL script. Stop
   only when a verdict newer than ten minutes has both `decision=stop` and
   `stop_authorized=true`, then re-run the monitor and confirm the same issue.
   If authorized, send one Ctrl-C to `p8_qsafe_ft`; never kill/pkill/restart.
5. Grok proposals may be made only in
   `/home/kojiek/grok-worktrees/<slug>` on `grok/<slug>`, committed and sent
   through `review_grok_proposal_with_codex.sh`. No live execution before
   Codex approval.
6. Every report states `training changed: false` unless an explicitly
   authorized stop actually occurred.
