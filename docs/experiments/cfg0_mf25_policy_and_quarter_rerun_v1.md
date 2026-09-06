# Readiness unit: cfg0-mf25-policy-and-quarter-rerun-v1

## Outcome and boundary

This unit establishes the CFG0/MF25 policy and prepares exactly four Caption
2.0 quarter-scale evaluations. It ends in a durable held state. It does not
repair or mutate the authenticated top-level queue, change queue order, signal
an existing process, launch a GPU command, delete historical CFG4.5 artifacts,
or publish an AB site.

Owned existing paths and their pre-edit SHA-256 values:

- `AGENTS.md`: `f7717a65b19b5d2c677a827bd299d68e046771eac4d570b7da452bf83228ded6`
- `docs/experiments/experiment_notification_policy.md`: `358e02f7ae2f44a21f21e4189e117ebd9defcc4dec22bab8cb61efc0ee4dc0b1`
- `scripts/caption10s_pipeline/eval_musiccaps_mf25.sh`: `9fa92b2a232ed33dea61f7d6f19c4b7d3c3d201dacabfc5f9e1d3ffbb81e8d14`
- `eval.py`: `063faaa422f9012a41abb0385c606c3ce2bde117b844880a92b3324871d8ae58`
- `docs/experiments/results/phase8/post_bugfix_fair_25step_comparison_2026_08_13.md`: `ba85d423e38a56c3b53a73fd871acb535db8e96a44bfcf01d8b032e84085e831`
- `docs/experiments/caption_granularity_s1_s2_fair_ablation_contract.json`: `28ad8ad04178cf45c643826e4dcd856abd185a85ce9bb0ad28db15d39c4284bd`
- `docs/experiments/caption2p0_full_k3_s2q_contract.json`: `e09bafcf35dcd71344c7e8032fe6a526edff6467c2c615e6dbfb799a4d744cb3`
- `docs/experiments/caption2p0_full_k5_s2q_contract.json`: `482e672cc60730e7548be242322f038442122704e0ea5ca66b1e589b2f1a1be7`
- `scripts/training_pipelines/run_caption2p0_k3_s2q_action.sh`: `87240a77dd89227841158520624d39e602dfb442804d46119ec5dfb713bc0706`
- `scripts/training_pipelines/run_caption2p0_k5_s2q_action.sh`: `c0d190f479b53e4b112c5e329c39ee58d7614179296277e1cbd020cb57614546`
- `scripts/experiment_harness/caption2p0_k3_s2q_harn.py`: `dcafd5c3af2022be51c21c6632d2726da8b86d346a9324d6eedf50ae849fd0e3`
- `scripts/experiment_harness/caption2p0_k5_s2q_harn.py`: `80b34800f36cdc94250a0f82b5f9e275794fe85891e7df64cb580c48d6cda379`

New owned paths are the canonical evaluation policy, the four-cell contract,
fixed runner, HARN adapter, queue-entry proposal, and their no-GPU tests.

## Prerequisites, budget, and stops

Registration prerequisites are four exact checkpoints plus the MusicCaps TSV.
Launch prerequisites belong to a later readiness unit: authenticated queue
reconciliation, exact approval binding, delivered hold/recovery events, normal
release of the existing GPU owner, resource lock acquisition, duplicate check,
and repeated mutable preflight. Runtime audio, metrics, and reports use the
owner-only `/home/kojiek/cfg0_eval_runtime` tree; the shared `0777` HDD eval
tree is prohibited for this rerun.

The later run is sequential. Budget: at most four cells, 90 GPU-minutes per
cell and 360 GPU-minutes total; 8 GiB peak transient output, 2 GiB retained
metrics/logs, 10 GiB recovery reserve; root warning floor 180 GiB and hard floor
150 GiB. One cell failure stops the child HARN and leaves later cells visible
for resume. Hash drift, extra/missing IDs, corrupt audio, incomplete/non-finite
metrics, notification failure, resource conflict, disk floors, or budget expiry
all fail closed.

## Acceptance and rollback

No-GPU tests must prove exactly four unique cells, three NoQ plus one K3 q9,
and literal `--num_steps 25 --cfg_strength 0`. Existing output may skip only
after exact provenance validation. The held registration must not touch the
authenticated queue or GPU.

Rollback for this unit is a reverse patch limited to the owned policy/script
paths and removal of the new unlaunched registration files. Rollback must never
remove checkpoints, logs, metrics, queue history, approvals, or historical
CFG4.5 artifacts.

Security review disposition: registration remains pending and does not mutate
the queue; arbitrary labels and trailing evaluator arguments are rejected;
the later unit must create an exact HMAC approval after executable hashes
stabilize, retain lock/process identity, bind the notifier hash, and keep the
webhook secret out of argv, contracts, reports, logs, and version control.
