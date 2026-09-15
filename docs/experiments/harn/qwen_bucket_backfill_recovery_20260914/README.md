# 054 evaluation recovery

Operator instruction: “fix it then continue the experiment”.

The original guest exited 2 before generation because its checkpoint-root gate omitted `/mnt/HDD/kojiek/MeanAudio_exps`. Its NoQ action also supplied the nonexistent `noq_cfg0` arm instead of `noq_cfg0_noq`.

This recovery uses private evaluator copies and a report-aware guest. The original five checkpoint hashes, MusicCaps input, nine CFG0 cells, five CFG3 + fidelity8 cells, conditioning, seeds, solver, decision rules and sequence are preserved. The original contract is retained in `original_contract.json`. Shared evaluators and host services are unchanged.

Every cell validates exact 5,521-ID mono 16 kHz audio and finite CLAP/AES metrics before its final report. CFG3 output uses a protected, run-specific root and explicit fidelity8 labels. All 14 reports are required for success. Completed cells are validated and reused. Unreported partial output is retained in a unique sibling directory and the unfinished cell replays from row zero, avoiding the existing evaluator's skipped-clip RNG drift. No checkpoints or prior outputs are deleted.

The guest records process ownership, monitors storage/progress with zero model polling calls, handles P1 pause requests, and uses required durable Discord receipts. Recovery is marked validated only after the formerly blocked NoQ evaluation has a valid full report. The outer host retains responsibility for resource locks and terminal-to-next transitions.

CPU acceptance: experiment tests, notification tests, 33 queue fixtures, shell syntax checks, and harn-schema-v1 bundle validation. Live GPU generation is validated separately after seating; these fixtures do not claim completed scientific results.

Runtime ledger: `/home/kojiek/logs/qwen_bucket_backfill_recovery_harn/controller_ledger.json`.
