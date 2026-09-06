#!/usr/bin/env python3
"""Verify, atomically install, or fail-closed rollback the slot3 interceptor.

Verification and dry-run are read-only.  Activation is impossible while the
checked-in approval descriptor remains pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

import register_slot3_fair013_cfg0 as reg


APPROVAL = reg.APPROVAL
HISTORICAL_ARCHIVE = reg.RUNTIME / "historical_cfg4p5_evaluator.preimage"


def rollback_stub_bytes() -> bytes:
    return b"""#!/usr/bin/bash
set -uo pipefail
/usr/bin/env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 /usr/bin/python3.12 /home/kojiek/MeanAudio/scripts/eval/register_slot3_fair013_cfg0.py --record-rollback --hook-pid \"$$\" --caller-pid \"$PPID\" || true
/usr/bin/printf '%s\\n' CFG0_INTERCEPTOR_ROLLED_BACK >&2
exit 125
"""


def live_parent_observation(pid: int) -> dict[str, Any]:
    script, script_hash = reg.hash_proc_fd(pid, 255)
    return {
        "pid": pid,
        "uid": Path(f"/proc/{pid}").stat().st_uid,
        "start_ticks": reg.read_proc_start_ticks(pid),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
        "executable": os.readlink(f"/proc/{pid}/exe"),
        "argv": reg.read_proc_cmdline(pid),
        "script_fd": 255,
        "script_device": script["device"],
        "script_inode": script["inode"],
        "script_owner_uid": script["owner_uid"],
        "script_sha256": script_hash,
    }


def validate_parent(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for field in ("pid", "uid", "start_ticks", "boot_id", "executable", "argv", "script_fd", "script_device", "script_inode", "script_sha256"):
        if observed.get(field) != expected.get(field):
            raise reg.RegistrationError(f"activation caller mismatch: {field}")
    if observed.get("script_owner_uid") != expected.get("uid"):
        raise reg.RegistrationError("activation caller script owner mismatch")


def validate_live_target(expected_hash: str) -> dict[str, Any]:
    identity = reg.secure_file_identity(reg.LIVE_EVALUATOR, expected_uid=os.geteuid())
    if identity["mode"] != "0755" or identity["sha256"] != expected_hash:
        raise reg.RegistrationError("live evaluator owner/mode/preimage drift")
    return identity


def validate_descriptor(*, require_approved: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    template = reg.load_json(reg.TEMPLATE)
    reg.validate_template(template)
    plan_sha256 = reg.sha256_path(reg.PLAN)
    reg.validate_security_receipt(reg.load_json(reg.SECURITY_REVIEW), plan_sha256)
    manifest = reg.load_json(reg.MANIFEST)
    generated = reg.runtime_manifest_payload()
    if manifest != generated:
        raise reg.RegistrationError("runtime manifest does not match generated transitive bindings")
    approval = reg.load_json(APPROVAL)
    expected = {
        "approved_plan_sha256": plan_sha256,
        "security_review_sha256": reg.sha256_path(reg.SECURITY_REVIEW),
        "runtime_manifest_sha256": reg.sha256_path(reg.MANIFEST),
        "replacement_hook_sha256": reg.sha256_path(reg.HOOK),
        "rollback_failclosed_sha256": reg.sha256_bytes(rollback_stub_bytes()),
        "historical_preimage_sha256": template["interceptor"]["historical_preimage_sha256"],
        "template_file_sha256": reg.sha256_path(reg.TEMPLATE),
        "priority_order_file_sha256": reg.sha256_path(reg.ORDER),
        "operator_instruction_sha256": reg.sha256_bytes("插在 slot3 之後。然後確保 slot3 相關的都用 cfg0".encode("utf-8")),
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            raise reg.RegistrationError(f"activation descriptor binding drift: {key}")
    if approval.get("caller") != template["caller"] or approval.get("mappings") != template["arms"]:
        raise reg.RegistrationError("activation caller or mapping binding drift")
    if approval.get("allowed_writes") != [str(reg.RUNTIME), str(reg.LIVE_EVALUATOR)]:
        raise reg.RegistrationError("activation allowed-write scope drift")
    if approval.get("notification_behavior") != {
        "event": "held", "one_delivered_idempotent_event_required_per_exact_registration": True,
        "notifier": str(reg.NOTIFIER), "webhook_secret_in_argv_or_artifacts": False,
        "failure_returns_to_outer_hook_loop": True,
    }:
        raise reg.RegistrationError("activation notification behavior drift")
    boundary = approval.get("boundary", {})
    if boundary != {"gpu_minutes": 0, "live_queue_mutation": False, "historical_evaluator_execution": False}:
        raise reg.RegistrationError("activation boundary drift")
    if require_approved:
        if approval.get("approval_status") != "approved" or approval.get("operator_approval_sha256") is None:
            raise reg.RegistrationError("exact interceptor activation approval is pending")
        unsigned = dict(approval)
        supplied = unsigned.pop("operator_approval_sha256")
        expected_approval = reg.sha256_bytes(b"meanaudio-slot3-cfg0-activation-approval-v1\0" + reg.canonical(unsigned))
        if not isinstance(supplied, str) or not secrets.compare_digest(supplied, expected_approval):
            raise reg.RegistrationError("activation approval digest is invalid")
    elif approval.get("approval_status") != "pending" or approval.get("operator_approval_sha256") is not None:
        raise reg.RegistrationError("readiness descriptor must remain pending")
    return template, approval


def _atomic_replace_live(new_bytes: bytes, expected_hash: str, *, target: Path = reg.LIVE_EVALUATOR) -> None:
    parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    temp_name = f".{target.name}.slot3-cfg0.{os.getpid()}.{secrets.token_hex(8)}"
    try:
        before = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o755:
            raise reg.RegistrationError("unsafe live evaluator identity before replacement")
        current = reg.sha256_path(target)
        if current != expected_hash:
            raise reg.RegistrationError("live evaluator changed before atomic replacement")
        fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o700, dir_fd=parent_fd)
        try:
            view = memoryview(new_bytes)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            os.fchmod(fd, 0o755)
        finally:
            os.close(fd)
        current_stat = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        current_hash = reg.sha256_path(target)
        if any(getattr(current_stat, field) != getattr(before, field) for field in ("st_dev", "st_ino", "st_uid", "st_mode", "st_nlink", "st_size", "st_mtime_ns")) or current_hash != expected_hash:
            raise reg.RegistrationError("live evaluator raced during replacement")
        os.replace(temp_name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def archive_historical_preimage() -> None:
    reg.prepare_runtime(reg.RUNTIME)
    historical = reg.read_regular_bytes(reg.LIVE_EVALUATOR, expected_uid=os.geteuid(), expected_mode=0o755)
    if reg.sha256_bytes(historical) != reg.load_json(reg.TEMPLATE)["interceptor"]["historical_preimage_sha256"]:
        raise reg.RegistrationError("historical preimage drift before archive")
    if HISTORICAL_ARCHIVE.exists() or HISTORICAL_ARCHIVE.is_symlink():
        reg._check_owned_mode(HISTORICAL_ARCHIVE, 0o600, directory=False)
        if reg.sha256_path(HISTORICAL_ARCHIVE) != reg.sha256_bytes(historical):
            raise reg.RegistrationError("historical archive drift")
    else:
        reg.atomic_bytes(HISTORICAL_ARCHIVE, historical, 0o600)


def verify_readiness() -> tuple[dict[str, Any], dict[str, Any]]:
    template, approval = validate_descriptor(require_approved=False)
    reg.validate_runtime_ancestry(reg.RUNTIME, reg.RUNTIME_TRUST_ANCHOR)
    validate_parent(live_parent_observation(template["caller"]["pid"]), template["caller"])
    validate_live_target(template["interceptor"]["historical_preimage_sha256"])
    return template, approval


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--activate", action="store_true")
    modes.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.verify or args.dry_run:
        template, _ = verify_readiness()
        print(json.dumps({
            "status": "ACTIVATION_HELD_PENDING_EXACT_APPROVAL",
            "caller_pid": template["caller"]["pid"],
            "live_preimage_sha256": template["interceptor"]["historical_preimage_sha256"],
            "replacement_hook_sha256": reg.sha256_path(reg.HOOK),
            "mutation_performed": False,
        }, sort_keys=True))
        return 0
    template, approval = validate_descriptor(require_approved=True)
    validate_parent(live_parent_observation(template["caller"]["pid"]), template["caller"])
    if args.activate:
        validate_live_target(template["interceptor"]["historical_preimage_sha256"])
        archive_historical_preimage()
        hook_bytes = reg.read_regular_bytes(reg.HOOK, expected_uid=os.geteuid())
        if reg.sha256_bytes(hook_bytes) != approval["replacement_hook_sha256"]:
            raise reg.RegistrationError("replacement hook raced after approval validation")
        _atomic_replace_live(hook_bytes, template["interceptor"]["historical_preimage_sha256"])
        validate_live_target(approval["replacement_hook_sha256"])
        print("SLOT3_CFG0_INTERCEPTOR_ACTIVATED")
        return 0
    validate_live_target(approval["replacement_hook_sha256"])
    _atomic_replace_live(rollback_stub_bytes(), approval["replacement_hook_sha256"])
    validate_live_target(approval["rollback_failclosed_sha256"])
    print("CFG0_INTERCEPTOR_ROLLED_BACK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, reg.RegistrationError, json.JSONDecodeError) as exc:
        print(f"SLOT3_CFG0_INSTALL_HOLD {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(125)
