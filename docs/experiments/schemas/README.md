# Experiment harness schema bundle

`harn-schema-v1` is the structural and offline semantic-validation contract for
new MeanAudio experiment harness documents. It contains:

- `experiment-contract-v1.schema.json`;
- `preflight-report-v1.schema.json`;
- `event-ledger-v1.schema.json`; and
- `queue-state-v1.schema.json`.

Every schema uses JSON Schema Draft 2020-12, a stable `urn:meanaudio:` ID,
closed object shapes, and local-only references. The validator refuses unknown
document kinds, versions, schema bundle IDs, and remote references.

## Validation

Run the CPU-only self-test:

```bash
python3 scripts/tests/selftest_experiment_harness_schemas.py
```

Validate one complete document bundle:

```bash
python3 scripts/validate_experiment_harness_documents.py \
  --contract /absolute/path/contract.json \
  --preflight /absolute/path/preflight.json \
  --ledger /absolute/path/ledger.json \
  --queue /absolute/path/queue.json
```

Validation success means only that the four documents are structurally and
semantically consistent. The path-based CLI verifies the declared raw-byte
bindings of the four JSON files it reads, but it never authenticates an
approval, grants launch or repair authority, verifies referenced artifact or
executable bytes, or permits command execution.

## Enforcement boundary

The offline validator checks bounded JSON loading, duplicate keys, secret-like
material, closed schema shapes, applicability branches, approval bindings and
freshness, storage arithmetic, event order/hash links, notification ordering,
queue uniqueness/dependencies/resource conflicts, and cross-document identity.

The runtime controller remains responsible for trusted-channel authentication,
raw-byte and executable hashes, resolved path ownership, symlink/TOCTOU checks,
atomic fsync/rename persistence, locks, boot/process identity, live resource
ownership, approval consumption, and execution. It must repeat all mutable
checks while holding the experiment lock.

## Compatibility

Version 1 documents use `schema_bundle_id: harn-schema-v1` and the exact
`document_kind`/`schema_version` constants in their schemas. Once an approved
run adopts v1, its schema bundle is immutable for that run. A breaking field,
meaning, authority, or validation change requires new schema filenames, IDs,
document versions, compatibility rules, and fixtures. Unknown or mixed bundles
fail closed.

Legacy contracts are not implicitly compatible. They remain `review-needed`
until a separate migration creates all four v1 documents, binds their hashes,
passes the validator, and receives current runtime-authenticated approval.
