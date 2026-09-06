#!/usr/bin/env python3
"""Durable, fail-closed notification receipts for shared queue experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


RECEIPT_KIND = "shared_queue_notification_receipt_v1"
LEDGER_KIND = "shared_queue_notification_ledger_v1"
DEFAULT_ROOT = Path("/home/kojiek/gpu_queue/notification_receipts")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_secure_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    _fsync_dir(path.parent)


def secure_read_json(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode):
            raise ValueError(f"not a regular file: {path}")
        if meta.st_uid != os.geteuid():
            raise ValueError(f"wrong owner: {path}")
        if stat.S_IMODE(meta.st_mode) & 0o077:
            raise ValueError(f"insecure mode: {path}")
        raw = b""
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            raw += chunk
            if len(raw) > 4 << 20:
                raise ValueError(f"receipt too large: {path}")
    finally:
        os.close(fd)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not an object: {path}")
    return value


@contextmanager
def locked_root(root: Path) -> Iterator[None]:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock = root / ".lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def receipt_path(root: Path, experiment_id: str, event: str) -> Path:
    safe = lambda value: "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return root / safe(experiment_id) / f"{safe(event)}.json"


def event_binding(
    *, contract_path: Path, launcher_path: Path, event: str, status: str,
    summary: str, idempotency_key: str,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object")
    return {
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "experiment_id": str(contract["experiment_id"]),
        "run_id": str(contract["run_id"]),
        "launcher_path": str(launcher_path.resolve()),
        "launcher_sha256": sha256_file(launcher_path),
        "event": event,
        "status": status,
        "summary": summary,
        "idempotency_key": idempotency_key,
    }


def deliver_required(
    *, contract_path: Path, launcher_path: Path, event: str, status: str,
    summary: str, idempotency_key: str, notifier: Path,
    python: Path = Path("/home/kojiek/venvs/dac/bin/python"),
    root: Path = DEFAULT_ROOT, extra_args: list[str] | None = None,
    max_pre_request_failures: int = 2,
) -> Path:
    """Deliver once. An interrupted request is ambiguous and never replayed."""
    binding = event_binding(
        contract_path=contract_path, launcher_path=launcher_path, event=event,
        status=status, summary=summary, idempotency_key=idempotency_key,
    )
    binding["notifier_path"] = str(notifier.resolve())
    binding["notifier_sha256"] = sha256_file(notifier)
    binding["payload_sha256"] = canonical_hash({
        key: binding[key] for key in (
            "contract_path", "contract_sha256", "experiment_id", "run_id",
            "launcher_sha256", "event", "status", "summary", "idempotency_key",
            "notifier_sha256",
        )
    })
    path = receipt_path(root, binding["experiment_id"], event)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    with locked_root(root):
        if path.exists():
            old = secure_read_json(path)
            for key, value in binding.items():
                if old.get(key) != value:
                    raise RuntimeError(f"immutable receipt binding mismatch: {key}")
            if old.get("delivery_state") == "delivered":
                return path
            if old.get("delivery_state") in {"attempting", "ambiguous"}:
                raise RuntimeError(f"notification outcome ambiguous: {idempotency_key}")
            failures = int(old.get("pre_request_failures") or 0)
            if old.get("delivery_state") != "prepared" or failures >= max_pre_request_failures:
                raise RuntimeError(f"notification is not retryable: {idempotency_key}")
            receipt = old
        else:
            receipt = {
                "document_kind": RECEIPT_KIND, **binding,
                "delivery_state": "prepared", "prepared_at": utc_now(),
                "pre_request_failures": 0,
            }
            atomic_secure_json(path, receipt)
        # Persist attempting before crossing the network trust boundary.
        receipt.update({"delivery_state": "attempting", "attempting_at": utc_now()})
        atomic_secure_json(path, receipt)

    argv = [
        str(python), str(notifier), "--status", status,
        "--experiment", idempotency_key, "--summary", summary, "--exit-code", "0",
        "--receipt-managed",
    ]
    if extra_args:
        argv.extend(extra_args)
    try:
        completed = subprocess.run(argv, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        # The request might have reached Discord; fail closed without replay.
        with locked_root(root):
            receipt = secure_read_json(path)
            receipt.update({
                "delivery_state": "ambiguous", "ambiguous_at": utc_now(),
                "response_sha256": hashlib.sha256(repr(exc).encode()).hexdigest(),
            })
            atomic_secure_json(path, receipt)
        raise RuntimeError(f"notification outcome ambiguous: {idempotency_key}") from exc
    response = completed.stdout + completed.stderr
    with locked_root(root):
        receipt = secure_read_json(path)
        if receipt.get("delivery_state") != "attempting":
            raise RuntimeError("receipt state changed during notification")
        receipt.update({
            "discord_returncode": completed.returncode,
            "response_sha256": hashlib.sha256(response).hexdigest(),
            "completed_at": utc_now(),
        })
        if completed.returncode == 0:
            receipt["delivery_state"] = "delivered"
            receipt["delivered_at"] = utc_now()
        else:
            # A normal nonzero from the notifier is treated as known pre-request
            # only when it explicitly says so. All other failures are ambiguous.
            marker = b"[NOTIFY PRE-REQUEST FAIL]" in response
            if marker:
                receipt["delivery_state"] = "prepared"
                receipt["pre_request_failures"] = int(receipt.get("pre_request_failures") or 0) + 1
            else:
                receipt["delivery_state"] = "ambiguous"
                receipt["ambiguous_at"] = utc_now()
        atomic_secure_json(path, receipt)
    if completed.returncode != 0:
        raise RuntimeError(f"required notification failed: {idempotency_key}")
    return path


def validate_delivered_receipt(
    path: Path, *, contract_path: Path, launcher_path: Path, event: str,
    status: str | None = None,
) -> tuple[bool, str]:
    try:
        rec = secure_read_json(path)
        if rec.get("document_kind") != RECEIPT_KIND:
            return False, "receipt kind invalid"
        if rec.get("delivery_state") != "delivered":
            return False, "receipt is not delivered"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        # A delivered receipt is historical evidence about a run that already
        # finished. Bind its notifier to the sha this run's own contract declared,
        # not to whatever the notifier file happens to contain now: re-hashing live
        # disk state makes any later edit of the notifier retroactively invalidate
        # every past receipt, which deadlocks downstream jobs that gate on them.
        # Drift of the live notifier is still caught at accept time, where
        # lib_scheduler.accept_guest hashes it against the launching contract.
        declared_notifier = str(
            (contract.get("notification_receipts") or {}).get("notifier_sha256") or ""
        )
        if not declared_notifier:
            return False, "contract notifier binding missing"
        checks = {
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": sha256_file(contract_path),
            "experiment_id": str(contract["experiment_id"]),
            "run_id": str(contract["run_id"]),
            "launcher_sha256": sha256_file(launcher_path),
            "event": event,
            "notifier_sha256": declared_notifier,
        }
        if status is not None:
            checks["status"] = status
        for key, expected in checks.items():
            if rec.get(key) != expected:
                return False, f"receipt {key} mismatch"
        payload = {key: rec[key] for key in (
            "contract_path", "contract_sha256", "experiment_id", "run_id",
            "launcher_sha256", "event", "status", "summary", "idempotency_key",
            "notifier_sha256",
        )}
        if rec.get("payload_sha256") != canonical_hash(payload):
            return False, "receipt payload hash mismatch"
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"receipt invalid: {exc}"
    return True, "ok"
