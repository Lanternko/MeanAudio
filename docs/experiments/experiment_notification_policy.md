# Experiment notification policy

Every new MeanAudio experiment must be launched through:

```text
scripts/run_with_experiment_report.sh
```

The wrapper sends exactly one Discord report when the complete experiment
sequence:

- finishes successfully;
- exits with a failure;
- is interrupted by HUP, INT, TERM, or a child exit associated with an
  interruption.

The report contains experiment name, time, host, Git revision, duration, exit
code, a bounded failure-log tail, and registered metrics when the final JSON
report exists. Discord mentions are disabled.

The webhook URL is local-only:

```text
/home/kojiek/.config/meanaudio/discord_webhook_url
```

Required permissions are `0600`. The URL must never be put in Git, command-line
arguments, experiment contracts, logs, or final reports.

Example:

```bash
scripts/run_with_experiment_report.sh \
  --experiment phase8_halfq_quarter \
  --report /home/kojiek/logs/phase8_halfq_qpilot_s2_50000_FINAL_METRICS.json \
  --log /home/kojiek/logs/phase8_halfq_quarter_sequence.log \
  -- bash scripts/training_pipelines/sequence_phase8_halfq_quarter.sh
```
