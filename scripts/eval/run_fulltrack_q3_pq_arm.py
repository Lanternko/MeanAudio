#!/usr/bin/env python3
"""Execute one FTQ3-BMATRIX-v1 arm under the exact HARN authority boundary."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence


def _load_harn():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "experiment_harness/fulltrack_q3_pq_bmatrix_harn.py",
        Path("/home/kojiek/MeanAudio/scripts/experiment_harness/fulltrack_q3_pq_bmatrix_harn.py"),
    ]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError("registered HARN module unavailable")
    spec = importlib.util.spec_from_file_location("ftq3_bmatrix_harn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("registered HARN module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harn = _load_harn()


def _contract(path: Path, *, allow_candidate: bool) -> dict[str, Any]:
    mode = 0o600 if path == harn.FINAL_CONTRACT else None
    payload = harn.load_json_nofollow(path, require_uid=harn.EXPECTED_UID, require_mode=mode)
    if payload.get("plan_id") != harn.PLAN_ID or payload.get("plan_sha256") != harn.PLAN_SHA256:
        raise harn.SecurityHold("contract Plan binding mismatch")
    if payload.get("run_id") != harn.RUN_ID:
        raise harn.SecurityHold("contract run identity mismatch")
    if allow_candidate:
        if payload.get("launch_allowed") is not False or payload.get("queue_mutation_allowed") is not False:
            raise harn.SecurityHold("dry-run candidate is not fail-closed")
    else:
        if path != harn.FINAL_CONTRACT:
            raise harn.SecurityHold("runtime requires exact final contract path")
        if payload.get("launch_allowed") is not True or payload.get("approval_required") is not True:
            raise harn.SecurityHold("runtime contract is not launch-enabled/approval-required")
        capability = harn.STATE_ROOT / "gate2_capability.json"
        harn.validate_gate2_approval(capability, path, require_lifecycle="consumed")
    return payload


def _new_arm_tree(arm: str) -> dict[str, Path]:
    if arm not in harn.ARMS:
        raise harn.SecurityHold("arm must be B1-B6")
    root_fd = harn._open_dir_chain(harn.RESULT_ROOT, require_final_uid=harn.EXPECTED_UID,
                                   require_final_mode=0o700)
    try:
        try:
            os.mkdir(arm, 0o700, dir_fd=root_fd)
        except FileExistsError:
            raise harn.SecurityHold(f"arm directory already exists: {arm}") from None
        arm_fd = os.open(arm, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
                         getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
        try:
            for name in ("audio", "manifests", "metrics"):
                os.mkdir(name, 0o700, dir_fd=arm_fd)
            os.fsync(arm_fd)
        finally:
            os.close(arm_fd)
    finally:
        os.close(root_fd)
    base = harn.RESULT_ROOT / arm
    return {"base": base, "audio": base / "audio", "manifests": base / "manifests",
            "metrics": base / "metrics"}


def _progress_generation(audio: Path) -> int:
    try:
        names = os.listdir(audio)
    except OSError:
        return 0
    return len({name for name in names if name.endswith(".flac") and harn.SAFE_ID_RE.fullmatch(name[:-5])})


def _progress_scoring(metrics: Path) -> int:
    try:
        with metrics.open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def _run_monitored(argv: Sequence[str], *, pass_fds: Sequence[int], timeout: int,
                   stall_seconds: int, progress: Callable[[], int], require_gpu: bool) -> None:
    harn.validate_no_shell_argv(argv)
    started = time.monotonic()
    last_change = started
    last_value = progress()
    idle_started: float | None = None
    child = subprocess.Popen(
        list(argv), cwd=harn.SEALED_ROOT / "source/MeanAudio",
        env=harn.sanitized_child_environment({}), pass_fds=tuple(pass_fds),
        stdin=subprocess.DEVNULL,
    )
    try:
        while child.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise harn.SecurityHold("registered phase wall budget exhausted")
            current = progress()
            if current > last_value:
                last_value = current
                last_change = time.monotonic()
            elif current < last_value:
                raise harn.SecurityHold("phase progress regressed")
            if time.monotonic() - last_change >= stall_seconds:
                child.terminate()
                child.wait(timeout=30)
                raise harn.SecurityHold("registered progress signal stalled")
            if require_gpu:
                owned = harn.assert_p2_lease_identity()
                owned.add(child.pid)
                harn.assert_no_foreign_gpu_processes(owned)
                util = subprocess.run(
                    ["/usr/bin/nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, capture_output=True, timeout=15,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, check=False,
                )
                try:
                    active = util.returncode == 0 and int(util.stdout.splitlines()[0].strip()) > 0
                except (ValueError, IndexError):
                    active = False
                if active:
                    idle_started = None
                elif idle_started is None:
                    idle_started = time.monotonic()
                elif time.monotonic() - idle_started >= 300:
                    child.terminate()
                    child.wait(timeout=30)
                    raise harn.SecurityHold("unexpected GPU idle threshold reached")
            time.sleep(30)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    if child.returncode:
        raise harn.SecurityHold(f"scientific child exited {child.returncode}")


def _validate_audio(audio_dir: Path, ids: Sequence[str]) -> list[dict[str, Any]]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise harn.SecurityHold(f"audio validator dependency unavailable: {exc.name}") from None
    expected = {value + ".flac" for value in ids}
    dfd = harn._open_dir_chain(audio_dir, require_final_uid=harn.EXPECTED_UID,
                                require_final_mode=0o700)
    entries = []
    try:
        actual = set(os.listdir(dfd))
        if actual != expected:
            raise harn.SecurityHold(
                f"audio set mismatch missing={len(expected-actual)} extra={len(actual-expected)}"
            )
        for value in ids:
            name = value + ".flac"
            fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=dfd)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != harn.EXPECTED_UID or
                        info.st_nlink != 1 or info.st_size <= 0):
                    raise harn.SecurityHold(f"unsafe/empty audio: {name}")
                digest = harn.sha256_fd(fd)
                with sf.SoundFile(f"/proc/self/fd/{fd}", "r") as handle:
                    if handle.samplerate != 16000 or handle.channels != 1 or handle.frames <= 0:
                        raise harn.SecurityHold(f"audio format mismatch: {name}")
                    while True:
                        block = handle.read(65536, dtype="float32", always_2d=False)
                        if len(block) == 0:
                            break
                        if not np.isfinite(block).all():
                            raise harn.SecurityHold(f"nonfinite audio: {name}")
                entries.append({"id": value, "path": name, "bytes": info.st_size, "sha256": digest})
            finally:
                os.close(fd)
    finally:
        os.close(dfd)
    return entries


def _write_audio_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    raw = "id\tsha256\n" + "".join(
        f"{row['id']}\t{row['sha256']}\n" for row in rows
    )
    dfd = harn._open_dir_chain(path.parent, require_final_uid=harn.EXPECTED_UID,
                                require_final_mode=0o700)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                     getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=dfd)
        try:
            os.write(fd, raw.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_metrics(path: Path, ids: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    required = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != harn.EXPECTED_UID or info.st_nlink != 1:
            raise harn.SecurityHold("per-item metric file metadata unsafe")
        with os.fdopen(os.dup(fd), "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or set(reader.fieldnames) != {"id", *required}:
                raise harn.SecurityHold("per-item metric schema mismatch")
            rows = []
            for row in reader:
                value = harn.safe_id(str(row["id"]))
                parsed = {key: float(row[key]) for key in required}
                if not all(math.isfinite(item) for item in parsed.values()):
                    raise harn.SecurityHold("nonfinite per-item metric")
                rows.append({"id": value, **parsed})
    finally:
        os.close(fd)
    if [row["id"] for row in rows] != list(ids):
        raise harn.SecurityHold("per-item metric IDs/order mismatch")
    aggregate = {key: math.fsum(row[key] for row in rows) / len(rows) for key in required}
    if not all(math.isfinite(value) for value in aggregate.values()):
        raise harn.SecurityHold("nonfinite aggregate")
    return rows, aggregate


def execute_arm(contract_path: Path, arm: str) -> None:
    contract = _contract(contract_path, allow_candidate=False)
    sealed = harn.verify_sealed_receipt()
    if sealed["launch_blockers"]:
        raise harn.SecurityHold("sealed runtime has unresolved unapproved dependencies")
    if harn.storage_status()["verdict"] == "hard_stop":
        raise harn.SecurityHold("root filesystem below hard floor before arm")
    owned = harn.assert_p2_lease_identity()
    harn.assert_no_foreign_gpu_processes(owned)
    tree = _new_arm_tree(arm)
    tsv_path = harn.SEALED_ROOT / harn.REGISTERED_FILES["musiccaps_tsv"]["dest"]
    checkpoint_path = harn.SEALED_ROOT / harn.REGISTERED_FILES[harn.CHECKPOINT_BY_ARM[arm]]["dest"]
    ids = harn.musiccaps_ids(tsv_path)
    checkpoint_fd = os.open(checkpoint_path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    tsv_fd = os.open(tsv_path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        generation = harn.generation_argv(
            arm, checkpoint_path=f"/proc/self/fd/{checkpoint_fd}",
            tsv_path=f"/proc/self/fd/{tsv_fd}", output_path=str(tree["audio"]),
        )
        _run_monitored(generation, pass_fds=(checkpoint_fd, tsv_fd), timeout=harn.PER_ARM_WALL_SECONDS,
                       stall_seconds=900, progress=lambda: _progress_generation(tree["audio"]),
                       require_gpu=True)
    finally:
        os.close(checkpoint_fd)
        os.close(tsv_fd)
    audio_rows = _validate_audio(tree["audio"], ids)
    audio_manifest = tree["manifests"] / "audio_sha256.tsv"
    audio_manifest_sha = _write_audio_manifest(audio_manifest, audio_rows)
    if harn.storage_status()["verdict"] == "hard_stop":
        raise harn.SecurityHold("root filesystem below hard floor before scoring")
    scoring = harn.scoring_argv(arm)
    _run_monitored(scoring, pass_fds=(), timeout=harn.PER_ARM_WALL_SECONDS,
                   stall_seconds=1200,
                   progress=lambda: _progress_scoring(tree["metrics"] / "per_item.tsv"),
                   require_gpu=True)
    metrics_path = tree["metrics"] / "per_item.tsv"
    _, aggregate = _validate_metrics(metrics_path, ids)
    report = {
        "schema_version": 1, "document_kind": "fulltrack_q3_pq_arm_report",
        "status": "passed", "plan_id": harn.PLAN_ID, "plan_sha256": harn.PLAN_SHA256,
        "run_id": harn.RUN_ID, "arm": arm,
        "contract_sha256": harn.sha256_path_nofollow(contract_path),
        "checkpoint": {"path": str(checkpoint_path), "sha256": harn.sha256_path_nofollow(
            checkpoint_path, require_uid=harn.EXPECTED_UID, require_mode=0o400, require_one_link=True)},
        "tsv_sha256": harn.sha256_path_nofollow(tsv_path, require_uid=harn.EXPECTED_UID,
                                                require_mode=0o400, require_one_link=True),
        "argv": {"generation": harn.generation_argv(arm), "scoring": scoring},
        "environment": dict(harn.ALLOWED_CHILD_ENV), "sealed_copy_receipt": sealed,
        "audio_manifest": {"path": str(audio_manifest), "sha256": audio_manifest_sha},
        "per_item_metrics": {"path": str(metrics_path), "sha256": harn.sha256_path_nofollow(metrics_path)},
        "metrics": aggregate, "completed_at": harn.now(),
    }
    harn.atomic_write_json(tree["base"] / "report.json", report, replace=False)


def dry_run(contract_path: Path, arm: str) -> dict[str, Any]:
    _contract(contract_path, allow_candidate=True)
    if arm not in harn.ARMS:
        raise harn.SecurityHold("arm must be B1-B6")
    generation = harn.generation_argv(arm)
    scoring = harn.scoring_argv(arm)
    harn.validate_no_shell_argv(generation)
    harn.validate_no_shell_argv(scoring)
    return {"arm": arm, "launch_allowed": False, "generation": generation,
            "scoring": scoring, "environment": dict(harn.ALLOWED_CHILD_ENV)}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--arm", choices=harn.ARMS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.dry_run:
            print(json.dumps(dry_run(args.contract, args.arm), indent=2, sort_keys=True))
        else:
            execute_arm(args.contract, args.arm)
        return 0
    except harn.SecurityHold as exc:
        print(f"HOLD: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(_main())
