# Readiness unit: slot3-cfg0-priority-interception-registration-v1

## Authorized outcome and boundary

The responsible operator instructed: `插在 slot3 之後。然後確保 slot3 相關的都用 cfg0`.
This unit interprets `slot3` as the active fair013 Caption 2.0 caption slot
(`temperature=1.0`, seed `31415926`) in PID 3413396. The requested durable
evaluation order is:

1. the already registered four-cell quarter CFG0 rerun;
2. fair013 K3 q9 CFG0;
3. fair013 best-of-3 NoQ CFG0;
4. fair013 worst-of-3 NoQ CFG0;
5. the preserved remainder of the previously launched fair013 chain.

Slot3 caption generation has already completed, so the earliest
non-interrupting seam is the first evaluator call after fair013 K3 training.
The registration/interception unit prepares a caller-bound, no-GPU deferral
hook at that pathname. It does not launch evaluation, mutate the authenticated
top queue, signal or restart PID 3413396, edit its already-open script inode,
alter training hyperparameters/data/seeds, or restore the unsafe CFG4.5
evaluator on rollback.

Actual GPU launch, authenticated top-queue reconciliation, exact transitive
GPU-evaluation runtime approval, and insertion commit are a later readiness
unit. The registration runtime is separately bound and approved before hook
activation. Until the later launch unit succeeds, the hook may only finalize
protected CFG0 contracts/proposals,
deliver a hold/registration event, and return success so the already approved
training/cache-restoration remainder is not truncated. It must never generate
audio or metrics and must never call the historical evaluator.

## Owned and observed paths

Owned existing paths and pre-edit SHA-256:

- `/home/kojiek/research/meanaudio_training/caption10s_pipeline/eval_musiccaps_mf25.sh`:
  `9fa92b2a232ed33dea61f7d6f19c4b7d3c3d201dacabfc5f9e1d3ffbb81e8d14`
- `docs/experiments/caption2p0_quarter_cfg0_queue_entry.pending.json`:
  `22d779768e992c7cb365581b695fb0158133a4cda7db370bfec5ddba69a49085`

New owned repository paths:

- `docs/experiments/slot3_fair013_cfg0_eval_template.json`
- `docs/experiments/slot3_cfg0_priority_order.pending.json`
- `scripts/eval/register_slot3_fair013_cfg0.py`
- `scripts/eval/validate_slot3_fair013_cfg0_registration.py`
- `scripts/tests/selftest_slot3_fair013_cfg0_registration.py`
- `docs/experiments/slot3_cfg0_registration_runtime_manifest.json`
- `docs/experiments/slot3_cfg0_interceptor_activation_approval.json`
- `scripts/eval/install_slot3_cfg0_interceptor.py`

New protected runtime root:

- `/home/kojiek/slot3_cfg0_interceptor` (current-user owned, mode 0700,
  beneath current-user-owned, group-non-writable `/home/kojiek`),
  containing a mode-0600 HMAC key, signed state, resolved per-arm contracts,
  notification outbox, and an exclusive mode-0600 coordinator lock.

Observed read-only identities:

- active chain script hash:
  `225a40eaee381482af8c803bfd3a9dc5bdd3d4270ea2c3a996432765ee52a02f`
- expected parent PID/UID/start ticks/boot ID:
  `3413396` / `1005` / `185217549` /
  `bdf33f4f-5fd4-45ea-8f7c-6281f73c367d`
- expected parent executable and argv:
  `/usr/bin/bash` /
  `/bin/bash /home/kojiek/research/meanaudio_training/caption10s_pipeline/run_c2p0_fair013_chain.sh`
- expected open script descriptor: FD 255, inode `9867918`, current hash equal
  to the active-chain script hash.
- top queue hash at planning:
  `cf8914f4fbd12808656bdf8b84be60a4b9e0f417d5472be439819df3b7d67c7d`
- top queue status hash at planning:
  `e19856f1b96b0d69eab4b9e2b628b5ea8fca0de45543449eadeee5459f1e5050`
  with state `queue_hold` and reason `top-level queue controller binding drift`.

## Exact three-call template

Every cell uses MusicCaps 5,521 (TSV SHA-256
`de567b13c39b6e7f7b3666f257817322ea119bcdece82fb5e8700b4a7470e51f`),
MeanFlow 25, literal CFG0, seed 42, NoMask, full precision, and CLAP plus AES
CE/CU/PC/PQ. The legacy caller tuples are accepted only as migration aliases;
they never reach the historical implementation.

| Arm | Legacy label suffix | Canonical label suffix | Conditioning | Exact checkpoint |
|---|---|---|---|---|
| fair013_k3_q9 | `_mf25_cfg4p5_q9` | `_mf25_cfg0_q9` | q9 | `exps/phase8_qwen_caption2p0_fair013_k3_balanced_quarter_stage2_50000/phase8_qwen_caption2p0_fair013_k3_balanced_quarter_stage2_50000_ema_final.pth` |
| fair013_best_noq | `_mf25_cfg4p5_noq` | `_mf25_cfg0_noq` | NoQ | `exps/phase8_qwen_caption2p0_fair013_bestof3_noq_quarter_stage2_50000/phase8_qwen_caption2p0_fair013_bestof3_noq_quarter_stage2_50000_ema_final.pth` |
| fair013_worst_noq | `_mf25_cfg4p5_noq` | `_mf25_cfg0_noq` | NoQ | `exps/phase8_qwen_caption2p0_fair013_worstof3_noq_quarter_stage2_50000/phase8_qwen_caption2p0_fair013_worstof3_noq_quarter_stage2_50000_ema_final.pth` |

At interception, the checkpoint must be a current-user-owned regular,
non-symlinked, single-link file at the exact path. Its SHA-256 is captured into
an atomically written, HMAC-signed resolved contract before registration is
accepted. Hashing uses an `O_NOFOLLOW` descriptor; device, inode, owner, mode,
link count, size, and mtime are captured from the same descriptor and the bytes
are read with `pread`. The later launcher must reopen and match all registered
identity and hash fields. The next arm cannot register before the previous arm. A repeated
identical call is idempotent; a changed checkpoint, tuple, caller, order, or
state is a hold.

## Budgets, stops, and continuation

The later seven-cell run has a maximum of 630 GPU minutes: 360 minutes for the
four existing cells plus 90 minutes for each of the three fair013 cells. It
retains the existing 8-GiB peak, 10-GiB recovery reserve, 180-GiB warning, and
150-GiB hard-stop floors for each sequential cell. This registration unit has
zero GPU minutes and may add at most 16 MiB of contracts/state/logs.

Registration stops before mutation on wrong PID, UID, start ticks, boot ID,
cmdline, executable, FD/inode/script hash, evaluator pathname/hash/mode/owner,
argv tuple, arm order, checkpoint identity, TSV hash, state HMAC, duplicate
drift, unexpected GPU launch attempt, top-queue mutation attempt, or notifier
failure. No branch may invoke CFG4.5.

Successful no-GPU registration returns zero to the existing chain only after
the resolved contract, priority-order state, and delivered idempotent hold event
are durable. This preserves later best/worst training and final slot0 cache
restoration. The installed shell hook owns an outer deterministic restart loop:
the Python coordinator returns zero only for an exact accepted/idempotent
registration; every rejection, crash injection, state error, notification
error, or retryable hold is recorded and then restarted/polled by the same hook
without returning to the `set -e` parent. Thus the parent remains blocked at
the exact evaluator call until registration succeeds, after which it continues
the untouched remainder. The loop performs no model calls and never invokes
the historical evaluator. A mode-0600 signed resume record names the current
arm and cache binding for every attempt.

Before activation, a registration-only transitive runtime manifest binds the
exact replacement hook, coordinator, registration validator, installer,
notifier, `/usr/bin/python3.12`, `/usr/bin/bash`, `/usr/bin/env`, templates,
schemas, fixed child environment, HMAC domains/state schema, and allowed write
roots. The activation approval binds that manifest hash, replacement hash,
original preimage hash, operator instruction hash, caller fingerprint, three
mappings, allowed writes, notification behavior, and the zero-GPU / zero-live-
queue-mutation boundary. Any bound-byte drift holds before state mutation or
notification. GPU evaluation code and dependencies are intentionally excluded
from this registration manifest and belong to the later launch unit.

## Installation, rollback, and acceptance

Activation is permitted only after plan/security approval binds the exact
replacement bytes and current caller fingerprint. Installation must refuse
symlink/wrong-owner/mode/hash drift and atomically replace the evaluator path.
The original CFG4.5 bytes are retained only as historical evidence, never as a
rollback executable. Rollback installs an owner-only fail-closed stub that
records `CFG0_INTERCEPTOR_ROLLED_BACK` and exits before GPU or filesystem
expansion.

No-GPU fixtures must prove:

- exactly the three ordered mappings above and the preceding four-cell entry;
- literal CFG0/MF25 and collision-free protected report identities;
- rejection of wrong parent, boot, FD, inode, script, argv, checkpoint path,
  order, duplicate drift, and fourth call;
- HMAC tampering and world-writable/symlink state rejection;
- no invocation of `eval.py`, the metric evaluator, or the historical wrapper;
- coordinator crash/replay/notification failure at each of the three calls is
  contained by the outer hook loop; the parent call does not return until one
  exact registration exists, and the next expected arm advances exactly once;
- an injected failure never loses or duplicates the parent remainder; a
  simulated parent proceeds through best, worst, and one final slot0 restore;
- live queue bytes remain unchanged and GPU process set gains no process;
- every accepted registration has a delivered idempotent hold event;
- rollback remains CFG0-only fail closed.

## Security-review findings and dispositions

Security review for this readiness unit required: exact scientific priority
authorization; atomic hook approval tied to the live caller; separation of
registration and GPU runtime approval; caller-scoped global-path replacement;
common GPU exclusion; and complete runtime containment. Disposition:

- The operator's current trusted-console instruction authorizes the scientific
  order intent. It does not by itself authorize executable activation; the
  activation record binds final bytes and caller identity after tests/review.
- The registration hook is zero-GPU and leaves top queue bytes unchanged, so no
  competing resource controller is created. Common GPU exclusion and queue
  reconciliation remain mandatory in the later launch unit.
- Caller identity includes PID, UID, start ticks, boot ID, executable, cmdline,
  FD 255 device/inode/hash, current evaluator preimage, invocation order, and
  exact argv. All are rechecked before atomic installation and on every call.
- The HMAC key is created once with `O_CREAT|O_EXCL`, mode 0600, 32 random
  bytes, under a mode-0700 current-user-owned root. Separate domains cover
  state, resolved contracts, and notification records. Sequence plus prior MAC
  prevents replay/reordering; duplicate exact calls are idempotent.
- Hook/coordinator child environments are fixed allowlists. The Discord secret
  remains a local mode-0600 file, never enters argv, contracts, state, logs, or
  version control. The notifier path/hash is part of registration runtime.
- Installation uses `O_NOFOLLOW`/owner/mode/preimage checks and atomic rename in
  the same directory. Rollback never restores the CFG4.5 executable.
- Checkpoint and contract TOCTOU controls use retained descriptors and atomic
  signed persistence; later GPU launch must independently reopen/revalidate.
