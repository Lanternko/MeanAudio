# Canonical evaluation policy

Effective 2026-08-31T18:13:35+08:00, the responsible operator defines the
canonical MeanAudio fair-comparison evaluation for newly registered runs as:

| Field | Required value |
|---|---|
| Benchmark | MusicCaps, exactly 5,521 unique IDs |
| Solver | MeanFlow |
| Steps | 25 |
| CFG | literal `cfg_strength=3` |
| Negative prompt | `low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi` |
| Generation seed | 42 |
| Text attention mask | disabled (`--no_text_attention_mask`) |
| Precision | full (`--full_precision`) |
| Metrics | CLAP, AES CE/CU/PC/PQ |

Canonical commands, labels, manifests, contracts, and reports must encode both
the literal CFG value 3 and the negative-prompt identity `fidelity8`. An absent,
empty, shortened, conditional, or otherwise altered negative prompt is a
different protocol and must not be accepted as completion of this protocol.

CFG 0 and CFG 4.5 are historical protocols. Existing audio, metrics, reports,
and contracts must be retained under their original labels. They must not be
renamed, silently reused as CFG 3 + `fidelity8`, or accepted as completion of a
new canonical evaluation.

Canonical CFG3 + `fidelity8` audio, metrics, and reports must be written beneath
a current-user-owned mode-0700 runtime root unique to that registered run. The
legacy HDD `eval_output` tree is group/world writable and is not an authorized
target for a new canonical run.

This revision is prospective. Evaluations launched or durably queue-registered
before the effective timestamp retain their preregistered CFG and negative
prompt, including `026_fake_random_full`, `028_random_quarter_neg_cfg1p5`, and
`029_c2p0_slot0_neg_cfg2p5_cfg4p0_full5521`. Their contracts and queue order are
not mutated by this policy revision. Their results remain labeled under their
actual protocols and are not canonical CFG3 + `fidelity8` results.

Every canonical evaluation must preregister the exact checkpoint path and
SHA-256, TSV path and SHA-256, conditioning mode, complete argv, unique output
label, report path, resource budget, cleanup behavior, and resume rules. Before
launch, the controller must fail closed on protocol drift, stale or mismatched
outputs, an unknown GPU process, missing resource ownership, insufficient
bytes, incomplete notification delivery, or unauthenticated approval.

Generated audio may be removed only after all 5,521 expected IDs have valid
mono 16 kHz audio, all five finite metrics are persisted, and the final report
binds the checkpoint, TSV, protocol, metrics file, and their hashes. Cleanup
never applies to historical results or checkpoints.
