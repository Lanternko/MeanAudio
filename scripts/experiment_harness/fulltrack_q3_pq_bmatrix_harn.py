#!/usr/bin/env python3
"""Fail-closed HARN for FTQ3-BMATRIX-v1.

Gate 1 may use only ``stage`` and ``dry-run``.  ``register`` and ``run`` require
an immutable launch-enabled contract plus a one-use Gate-2 capability whose
fingerprints match the exact artifacts.  This module intentionally never
accepts scheduler/environment overrides.
"""
from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path("/home/kojiek/MeanAudio")
PLAN = ROOT / "docs/experiments/designs/fulltrack_q3_pq_bmatrix_v1.json"
PLAN_SHA256 = "e5d160d1e708fbd06318029b5525fa923a7f0a2219e5818acf133266d0495379"
PLAN_ID = "FTQ3-BMATRIX-v1"
RUN_ID = "run-20260828-ftq3-bmatrix-v1"
EXPERIMENT_ID = "fulltrack-q3-pq-bmatrix-v1"
CANDIDATE = ROOT / "docs/experiments/fulltrack_q3_pq_bmatrix_contract.candidate.json"
FINAL_CONTRACT = ROOT / "docs/experiments/fulltrack_q3_pq_bmatrix_contract.json"
RUNNER = ROOT / "scripts/eval/run_fulltrack_q3_pq_arm.py"
SCORER = ROOT / "scripts/eval/score_musiccaps_per_item.py"
ANALYZER = ROOT / "scripts/analysis/fulltrack_q3_paired_report.py"
VALIDATOR = ROOT / "scripts/validate_experiment_harness_documents.py"
QUEUE_CANDIDATE = ROOT / "scripts/queue_candidates/027_fulltrack_q3_pq_bmatrix.sh"
SEALED_ROOT = Path("/home/kojiek/cfg0_eval_runtime/fulltrack_q3_pq_bmatrix_v1/sealed_inputs")
RESULT_ROOT = SEALED_ROOT.parent
TEST_FIXTURES = RESULT_ROOT / "gate1_test_fixtures"
TEST_STATE = Path("/home/kojiek/logs/fulltrack_q3_pq_bmatrix_v1_gate1_test")
GATE1_APPROVAL = TEST_STATE / "gate1_approval_record.json"
STATE_ROOT = Path("/home/kojiek/logs/fulltrack_q3_pq_bmatrix_v1_harn")
STATE_KEY = STATE_ROOT / "state_hmac.key"
STATE_LEDGER = STATE_ROOT / "ledger.json"
STATE_OUTBOX = STATE_ROOT / "outbox"
STATE_LOCK_POINTER = STATE_ROOT / "controller_lock.json"
QUEUE_ROOT = Path("/home/kojiek/gpu_queue")
QUEUE_SOURCE = QUEUE_CANDIDATE
QUEUE_TARGET = QUEUE_ROOT / "p2/pending/027_fulltrack_q3_pq_bmatrix.sh"
QUEUE_LOCK = QUEUE_ROOT / "p2/runtime/queue-mutation.lock"
PYTHON = Path("/home/kojiek/venvs/dac/bin/python")
EXPECTED_UID = 1005
ARMS = ("B1", "B2", "B3", "B4", "B5", "B6")
PHASE1 = ("B1", "B2")
PHASE2 = ("B3", "B4", "B5", "B6")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
ZERO_SHA = "0" * 64
GIB = 1024**3
HARD_FLOOR = 150 * GIB
WARNING_FLOOR = 200 * GIB
PEAK_ADDITIONAL = 12 * GIB
IMPLEMENTATION_BUDGET_SECONDS = 7200
NO_GPU_TEST_BUDGET_SECONDS = 1800
PER_ARM_WALL_SECONDS = 21600
MATRIX_WALL_SECONDS = 129600
ALLOWED_CHILD_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": str(SEALED_ROOT / "hf_cache"),
    "PATH": "/home/kojiek/venvs/dac/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONNOUSERSITE": "1",
}
REJECTED_ENV = {
    "GPU_PREFLIGHT", "GPU_QUEUE_ROOT", "GPU_PYTHON", "GPU_NOTIFY",
    "GPU_SCHED_ACCEPT", "PYTHONPATH", "PYTHONHOME", "PYTHONINSPECT",
    "PYTHONSTARTUP", "LD_PRELOAD", "LD_LIBRARY_PATH",
    "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
}
REGISTERED_FILES: dict[str, dict[str, Any]] = {
    "musiccaps_tsv": {
        "source": "/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv",
        "dest": "inputs/musiccaps_test.tsv",
        "sha256": "de567b13c39b6e7f7b3666f257817322ea119bcdece82fb5e8700b4a7470e51f",
    },
    "fulltrack_q3_checkpoint": {
        "source": "/home/kojiek/MeanAudio/exps/phase8_qwen_s2q_from_noq_full_k3_balanced_stage2_200000/phase8_qwen_s2q_from_noq_full_k3_balanced_stage2_200000_ema_final.pth",
        "dest": "checkpoints/fulltrack_q3_ema.pth",
        "sha256": "b5fd22ee28d0d1c63f6ca66e446bf0142b2344ac74b0558949ad3b86ea4ab8ab",
        "resolved_roots": ["/home/kojiek/exps_nvme", "/mnt/HDD/kojiek/MeanAudio_exps"],
    },
    "segment_slot0_q3_checkpoint": {
        "source": "/home/kojiek/MeanAudio/exps/phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000/phase8_qwen_caption2p0_s2q_from_noq_full_k3_balanced_stage2_200000_ema_final.pth",
        "dest": "checkpoints/segment_slot0_q3_ema.pth",
        "sha256": "c72e26d1c46581f9bca4a179806849a29f799f6730c158782931dcb86ac9be14",
        "resolved_roots": ["/home/kojiek/exps_nvme", "/mnt/HDD/kojiek/MeanAudio_exps"],
    },
    "fulltrack_noq_checkpoint": {
        "source": "/home/kojiek/MeanAudio/exps/phase8_qwen_official_noq_full_stage2_200000/phase8_qwen_official_noq_full_stage2_200000_ema_final.pth",
        "dest": "checkpoints/fulltrack_noq_ema.pth",
        "sha256": "2519e83638c431aff006bf7690023ab53f17ff86b8894200f83d23c0ddceeeca",
        "resolved_roots": ["/home/kojiek/exps_nvme", "/mnt/HDD/kojiek/MeanAudio_exps"],
    },
    "segment_slot0_noq_checkpoint": {
        "source": "/home/kojiek/MeanAudio/exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth",
        "dest": "checkpoints/segment_slot0_noq_ema.pth",
        "sha256": "7c01ca5475293363ad92065dc0a89fb8e398d9f789bdff20453e0a751a0cf7f2",
        "resolved_roots": ["/home/kojiek/exps_nvme", "/mnt/HDD/kojiek/MeanAudio_exps"],
    },
    "clap_checkpoint": {
        "source": "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt",
        "dest": "weights/music_speech_audioset_epoch_15_esc_89.98.pt",
        "sha256": "51c68f12f9d7ea25fdaaccf741ec7f81e93ee594455410f3bca4f47f88d8e006",
    },
    "vae": {
        "source": "/home/kojiek/MeanAudio/weights/v1-16.pth",
        "dest": "weights/v1-16.pth",
        "sha256": "15ad082c714ccf3771898a771fc6eebdc1d9c8d5c6154726906a97f43603d62c",
        "bytes": 686652758,
    },
    "vocoder": {
        "source": "/home/kojiek/MeanAudio/weights/best_netG.pt",
        "dest": "weights/best_netG.pt",
        "sha256": "970ca75ee4d5ce583e9396a4534acb14971ea2b4f1c22e038f476680c868a789",
        "bytes": 449217313,
    },
    "empty_t5": {
        "source": "/home/kojiek/MeanAudio/weights/empty_string_t5.pth",
        "dest": "weights/empty_string_t5.pth",
        "sha256": "c7e9a3adce14701ceefb9d7bdd75fa5fcadad11002bab73fa793274d706b224f",
        "bytes": 316612,
    },
    "empty_clap": {
        "source": "/home/kojiek/MeanAudio/weights/empty_string_clap_c.pth",
        "dest": "weights/empty_string_clap_c.pth",
        "sha256": "1d17a4ac85c1c438b90c0b181a6bbfb6dd0da43fda828e1beca08e87323cd46c",
        "bytes": 3288,
    },
}
HF_SNAPSHOTS = {
    "flan_t5": {
        "model_cache": "/home/kojiek/.cache/huggingface/hub/models--google--flan-t5-large",
        "source": "/home/kojiek/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
        "dest_model": "hf_cache/hub/models--google--flan-t5-large",
        "revision": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
        "files": {
            "config.json": "bfa5beeb5a4630a97f043f071b9b5d858c842604cff5db874680f33b56090c8c",
            "model.safetensors": "a9dd06ce490f139af36e9eb77dd3758b4fd07a08a73d5a1abe5ff2591e2d388e",
            "special_tokens_map.json": "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f",
            "spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
            "tokenizer.json": "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a",
            "tokenizer_config.json": "5d19985330a9123285cc583fc60616d083aa9df7435812b5d8bb3e749f435d56",
        },
    },
    "audiobox": {
        "model_cache": "/home/kojiek/.cache/huggingface/hub/models--facebook--audiobox-aesthetics",
        "source": "/home/kojiek/.cache/huggingface/hub/models--facebook--audiobox-aesthetics/snapshots/9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        "dest_model": "hf_cache/hub/models--facebook--audiobox-aesthetics",
        "revision": "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        "files": {
            "config.json": "0b8eabc5ced92cefed116a3aca8f1e59d2a33dc4d376b9e86dfb9c072e1d280d",
            "model.safetensors": "a5a3c2412649cc2384ec525ffd5180ce6c4778f43bed6108e0a1303de04d014e",
        },
    },
}
CHECKPOINT_BY_ARM = {
    "B1": "fulltrack_q3_checkpoint", "B2": "segment_slot0_q3_checkpoint",
    "B3": "fulltrack_noq_checkpoint", "B4": "fulltrack_q3_checkpoint",
    "B5": "segment_slot0_q3_checkpoint", "B6": "segment_slot0_noq_checkpoint",
}
CONDITIONING_BY_ARM = {
    "B1": ("--quality_level", "9"), "B2": ("--quality_level", "9"),
    "B3": ("--no_q",), "B4": ("--quality_level", "0"),
    "B5": ("--quality_level", "0"), "B6": ("--no_q",),
}


class SecurityHold(RuntimeError):
    """A fail-closed security or authority hold."""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_fd(fd: int) -> str:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise SecurityHold("hash target is not a regular file")
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(fd, 8 << 20), b""):
        digest.update(block)
    after = os.fstat(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise SecurityHold("file changed while hashing")
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def sha256_path_nofollow(path: Path, *, require_uid: int | None = None,
                         require_mode: int | None = None, require_one_link: bool = False) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SecurityHold(f"not regular: {path}")
        if require_uid is not None and info.st_uid != require_uid:
            raise SecurityHold(f"wrong owner: {path}")
        if require_mode is not None and stat.S_IMODE(info.st_mode) != require_mode:
            raise SecurityHold(f"wrong mode: {path}")
        if require_one_link and info.st_nlink != 1:
            raise SecurityHold(f"extra hard link: {path}")
        return sha256_fd(fd)
    finally:
        os.close(fd)


def load_json_nofollow(path: Path, *, max_bytes: int = 4 << 20,
                       require_uid: int | None = None, require_mode: int | None = None) -> Any:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise SecurityHold(f"unsafe JSON object: {path}")
        if require_uid is not None and info.st_uid != require_uid:
            raise SecurityHold(f"wrong JSON owner: {path}")
        if require_mode is not None and stat.S_IMODE(info.st_mode) != require_mode:
            raise SecurityHold(f"wrong JSON mode: {path}")
        raw = b""
        while len(raw) <= max_bytes:
            block = os.read(fd, min(65536, max_bytes + 1 - len(raw)))
            if not block:
                break
            raw += block
        if len(raw) > max_bytes:
            raise SecurityHold("JSON size limit exceeded")
    finally:
        os.close(fd)
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise SecurityHold(f"invalid JSON: {path}: {type(exc).__name__}") from None


def _open_dir_chain(path: Path, *, require_final_uid: int | None = None,
                    require_final_mode: int | None = None) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise SecurityHold("directory path must be normalized absolute")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
                            getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if require_final_uid is not None and info.st_uid != require_final_uid:
            raise SecurityHold(f"wrong directory owner: {path}")
        if require_final_mode is not None and stat.S_IMODE(info.st_mode) != require_final_mode:
            raise SecurityHold(f"wrong directory mode: {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _mkdir_relative(parent_fd: int, name: str, mode: int = 0o700) -> int:
    if not SAFE_ID_RE.fullmatch(name):
        raise SecurityHold(f"unsafe directory component: {name!r}")
    try:
        os.mkdir(name, mode, dir_fd=parent_fd)
    except FileExistsError:
        pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
                 getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != mode:
        os.close(fd)
        raise SecurityHold(f"unsafe directory metadata: {name}")
    return fd


def ensure_private_dir(path: Path, *, must_be_new: bool = False) -> None:
    parent = path.parent
    if not parent.exists():
        ensure_private_dir(parent, must_be_new=False)
    pfd = _open_dir_chain(parent)
    try:
        if must_be_new:
            try:
                os.mkdir(path.name, 0o700, dir_fd=pfd)
            except FileExistsError:
                raise SecurityHold(f"required new directory already exists: {path}") from None
        else:
            try:
                os.mkdir(path.name, 0o700, dir_fd=pfd)
            except FileExistsError:
                pass
        fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
                     getattr(os, "O_NOFOLLOW", 0), dir_fd=pfd)
        try:
            info = os.fstat(fd)
            if info.st_uid != EXPECTED_UID or stat.S_IMODE(info.st_mode) != 0o700:
                raise SecurityHold(f"unsafe private directory: {path}")
        finally:
            os.close(fd)
    finally:
        os.close(pfd)


def atomic_write_json(path: Path, value: Any, *, mode: int = 0o600,
                      replace: bool = True) -> str:
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    dfd = _open_dir_chain(path.parent, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
    tmp = f".tmp-{secrets.token_hex(16)}"
    fd = -1
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                     getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=dfd)
        os.write(fd, raw)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if not replace:
            # link/rename without replacement where renameat2 is not portable.
            try:
                os.link(tmp, path.name, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False)
            except FileExistsError:
                raise SecurityHold(f"refusing to replace existing file: {path}") from None
            os.unlink(tmp, dir_fd=dfd)
        else:
            existing = None
            try:
                existing = os.open(path.name, os.O_RDONLY | os.O_CLOEXEC |
                                   getattr(os, "O_NOFOLLOW", 0), dir_fd=dfd)
            except FileNotFoundError:
                pass
            if existing is not None:
                info = os.fstat(existing)
                os.close(existing)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != EXPECTED_UID or info.st_nlink != 1:
                    raise SecurityHold(f"unsafe replace target: {path}")
            os.replace(tmp, path.name, src_dir_fd=dfd, dst_dir_fd=dfd)
        os.fsync(dfd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp, dir_fd=dfd)
        except FileNotFoundError:
            pass
        os.close(dfd)
    return sha256_bytes(raw)


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise SecurityHold("invalid approval timestamp") from None
    if parsed.tzinfo is None:
        raise SecurityHold("approval timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def validate_gate1_approval(path: Path = GATE1_APPROVAL) -> dict[str, Any]:
    record = load_json_nofollow(path, require_uid=EXPECTED_UID, require_mode=0o600)
    if record.get("document_kind") != "gate_1_implementation_approval_record":
        raise SecurityHold("not a Gate-1 approval record")
    if record.get("plan_id") != PLAN_ID or record.get("plan_sha256") != PLAN_SHA256:
        raise SecurityHold("Gate-1 Plan binding mismatch")
    if record.get("trusted_channel") != "operator_console" or record.get("execution_mode") != "AUTO":
        raise SecurityHold("Gate-1 trusted-channel or AUTO binding mismatch")
    if record.get("operator_approval_text") != "批准 Gate 1，AUTO":
        raise SecurityHold("Gate-1 approval text mismatch")
    if record.get("consumed") is not False or record.get("lifecycle") != "approved":
        raise SecurityHold("Gate-1 record is not approved/unconsumed")
    if _parse_utc(record["issued_at"]) > datetime.now(timezone.utc):
        raise SecurityHold("Gate-1 approval is from the future")
    if _parse_utc(record["expires_at"]) <= datetime.now(timezone.utc):
        raise SecurityHold("Gate-1 approval expired")
    if record.get("approved_sealed_staging_root") != str(SEALED_ROOT):
        raise SecurityHold("Gate-1 staging root mismatch")
    if not re.fullmatch(r"[a-f0-9]{64}", str(record.get("nonce", ""))):
        raise SecurityHold("Gate-1 nonce invalid")
    if sha256_path_nofollow(PLAN) != PLAN_SHA256:
        raise SecurityHold("approved Plan bytes drifted")
    return record


def sanitized_child_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    parent = os.environ if parent is None else parent
    leaked = REJECTED_ENV.intersection(parent)
    if leaked:
        # Reject rather than silently inherit or normalize an authority override.
        raise SecurityHold("rejected parent environment names present: " + ",".join(sorted(leaked)))
    return dict(ALLOWED_CHILD_ENV)


def safe_id(value: str) -> str:
    if not SAFE_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise SecurityHold(f"unsafe MusicCaps ID: {value!r}")
    return value


def musiccaps_ids(path: Path, expected: int = 5521) -> list[str]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SecurityHold("MusicCaps TSV is not regular")
        with os.fdopen(os.dup(fd), "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "id" not in reader.fieldnames or "caption" not in reader.fieldnames:
                raise SecurityHold("MusicCaps TSV schema mismatch")
            values = [safe_id(str(row["id"])) for row in reader]
    finally:
        os.close(fd)
    if len(values) != expected or len(set(values)) != expected:
        raise SecurityHold(f"MusicCaps ID count/uniqueness mismatch: {len(values)}")
    return values


def storage_status(free_bytes: int | None = None) -> dict[str, Any]:
    if free_bytes is None:
        free_bytes = shutil.disk_usage("/").free
    verdict = "hard_stop" if free_bytes < HARD_FLOOR else (
        "warning" if free_bytes < WARNING_FLOOR else "pass"
    )
    return {
        "path": "/", "free_bytes": int(free_bytes), "hard_floor_bytes": HARD_FLOOR,
        "warning_floor_bytes": WARNING_FLOOR, "peak_additional_bytes": PEAK_ADDITIONAL,
        "verdict": verdict,
    }


def _safe_source_fd(path: Path, *, allowed_symlink_root: Path | None = None,
                    allowed_resolved_roots: Sequence[Path] = ()) -> tuple[int, dict[str, Any]]:
    """Open a registered source; only a final relative HF cache symlink is allowed."""
    parent_fd = None
    try:
        try:
            parent_fd = _open_dir_chain(path.parent)
        except (NotADirectoryError, OSError) as exc:
            if not allowed_resolved_roots or getattr(exc, "errno", None) not in {errno.ENOTDIR, errno.ELOOP}:
                raise
            resolved = path.resolve(strict=True)
            approved = False
            for root_value in allowed_resolved_roots:
                root = root_value.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                    approved = True
                    break
                except ValueError:
                    continue
            if not approved:
                raise SecurityHold("registered source parent symlink escaped allowed roots") from None
            resolved_parent_fd = _open_dir_chain(resolved.parent)
            try:
                fd = os.open(resolved.name, os.O_RDONLY | os.O_CLOEXEC |
                             getattr(os, "O_NOFOLLOW", 0), dir_fd=resolved_parent_fd)
            finally:
                os.close(resolved_parent_fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                os.close(fd)
                raise SecurityHold(f"resolved source not regular: {path}")
            return fd, {
                "registered_path": str(path), "resolved_path": str(resolved),
                "source_parent_symlink_resolution": True, "source_symlink": None,
                "source_dev": info.st_dev, "source_ino": info.st_ino,
                "source_size": info.st_size,
            }
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            link_text = None
            resolved = path
        except OSError as exc:
            if exc.errno != errno.ELOOP or allowed_symlink_root is None:
                raise SecurityHold(f"source open failed without following links: {path}: {exc.strerror}") from None
            link_text = os.readlink(path.name, dir_fd=parent_fd)
            if os.path.isabs(link_text):
                raise SecurityHold("absolute HF source symlink forbidden")
            resolved = (path.parent / link_text).resolve(strict=True)
            root = allowed_symlink_root.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError:
                raise SecurityHold("HF source symlink escaped registered model cache") from None
            fd = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise SecurityHold(f"source not regular: {path}")
        return fd, {
            "registered_path": str(path), "resolved_path": str(resolved),
            "source_symlink": link_text, "source_dev": info.st_dev,
            "source_ino": info.st_ino, "source_size": info.st_size,
        }
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _destination_parent(root_fd: int, relative: PurePosixPath) -> tuple[int, str]:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SecurityHold("unsafe staging destination")
    fd = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            child = _mkdir_relative(fd, part)
            os.close(fd)
            fd = child
        name = relative.parts[-1]
        if not SAFE_ID_RE.fullmatch(name.replace(".", "_")):
            raise SecurityHold(f"unsafe staging basename: {name}")
        return fd, name
    except BaseException:
        os.close(fd)
        raise


def _copy_one(root_fd: int, source: Path, relative: str, expected_sha: str,
              *, expected_bytes: int | None = None,
              allowed_symlink_root: Path | None = None,
              allowed_resolved_roots: Sequence[Path] = ()) -> dict[str, Any]:
    sfd, source_meta = _safe_source_fd(
        source, allowed_symlink_root=allowed_symlink_root,
        allowed_resolved_roots=allowed_resolved_roots,
    )
    dfd, name = _destination_parent(root_fd, PurePosixPath(relative))
    out_fd = -1
    try:
        source_hash = sha256_fd(sfd)
        info = os.fstat(sfd)
        if source_hash != expected_sha:
            raise SecurityHold(f"registered source hash mismatch: {source}")
        if expected_bytes is not None and info.st_size != expected_bytes:
            raise SecurityHold(f"registered source size mismatch: {source}")
        out_fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                         getattr(os, "O_NOFOLLOW", 0), 0o400, dir_fd=dfd)
        os.lseek(sfd, 0, os.SEEK_SET)
        while True:
            block = os.read(sfd, 8 << 20)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(out_fd, view)
                view = view[written:]
        os.fsync(out_fd)
        os.fchmod(out_fd, 0o400)
        copied = os.fstat(out_fd)
        copied_hash = sha256_fd(out_fd)
        if copied_hash != expected_sha or copied.st_size != info.st_size:
            raise SecurityHold(f"copied bytes mismatch: {relative}")
        if copied.st_uid != EXPECTED_UID or stat.S_IMODE(copied.st_mode) != 0o400 or copied.st_nlink != 1:
            raise SecurityHold(f"copied file metadata mismatch: {relative}")
        os.fsync(dfd)
        return {
            **source_meta, "staged_path": str(SEALED_ROOT / relative),
            "staged_relative_path": relative, "sha256": copied_hash,
            "bytes": copied.st_size, "st_dev": copied.st_dev,
            "st_ino": copied.st_ino, "st_nlink": copied.st_nlink,
            "mode": "0400", "uid": copied.st_uid,
        }
    finally:
        if out_fd >= 0:
            os.close(out_fd)
        os.close(sfd)
        os.close(dfd)


def _copy_source_tree(root_fd: int) -> list[dict[str, Any]]:
    selected: list[tuple[Path, str]] = [(ROOT / "eval.py", "source/MeanAudio/eval.py")]
    for base_name in ("meanaudio", "config"):
        base = ROOT / base_name
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                selected.append((path, "source/MeanAudio/" + path.relative_to(ROOT).as_posix()))
    required_scripts = (
        Path(__file__).resolve(), RUNNER, SCORER, ANALYZER, QUEUE_CANDIDATE,
        CANDIDATE,
        ROOT / "scripts/tests/selftest_fulltrack_q3_pq_bmatrix_security.py",
        ROOT / "scripts/tests/selftest_fulltrack_q3_pq_bmatrix_science.py",
    )
    for path in required_scripts:
        if not path.is_file() or path.is_symlink():
            raise SecurityHold(f"required Gate-1 source is missing or linked: {path}")
        selected.append((path, "source/MeanAudio/" + path.relative_to(ROOT).as_posix()))
    entries = []
    for source, relative in selected:
        expected = sha256_path_nofollow(source)
        entries.append(_copy_one(root_fd, source, relative, expected))
    # Duplicate exact model-side weight bytes under the relative paths eval.py expects.
    for key in ("vae", "vocoder", "clap_checkpoint", "empty_t5", "empty_clap"):
        spec = REGISTERED_FILES[key]
        relative = "source/MeanAudio/weights/" + Path(spec["dest"]).name
        entries.append(_copy_one(root_fd, Path(spec["source"]), relative, spec["sha256"],
                                 expected_bytes=spec.get("bytes")))
    return entries


def _write_new_under_root(root_fd: int, relative: str, raw: bytes, mode: int = 0o400) -> dict[str, Any]:
    dfd, name = _destination_parent(root_fd, PurePosixPath(relative))
    fd = -1
    try:
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                     getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=dfd)
        os.write(fd, raw)
        os.fsync(fd)
        os.fchmod(fd, mode)
        info = os.fstat(fd)
        digest = sha256_fd(fd)
        os.fsync(dfd)
        return {"staged_path": str(SEALED_ROOT / relative), "staged_relative_path": relative,
                "sha256": digest, "bytes": info.st_size, "st_dev": info.st_dev,
                "st_ino": info.st_ino, "st_nlink": info.st_nlink,
                "mode": f"{mode:04o}", "uid": info.st_uid, "generated": True}
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(dfd)


def stage_inputs() -> dict[str, Any]:
    approval = validate_gate1_approval()
    if storage_status()["verdict"] == "hard_stop":
        raise SecurityHold("root filesystem below hard floor before staging")
    if SEALED_ROOT.exists() or SEALED_ROOT.is_symlink():
        raise SecurityHold("sealed root already exists; Gate-1 staging is one-shot")
    ensure_private_dir(RESULT_ROOT)
    ensure_private_dir(SEALED_ROOT, must_be_new=True)
    root_fd = _open_dir_chain(SEALED_ROOT, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
    entries: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for key, spec in REGISTERED_FILES.items():
            entry = _copy_one(root_fd, Path(spec["source"]), spec["dest"], spec["sha256"],
                              expected_bytes=spec.get("bytes"),
                              allowed_resolved_roots=tuple(Path(value) for value in spec.get("resolved_roots", ())))
            entry["key"] = key
            entries.append(entry)
            if time.monotonic() - started > IMPLEMENTATION_BUDGET_SECONDS:
                raise SecurityHold("Gate-1 implementation wall budget exhausted")
        for snapshot_key, snapshot in HF_SNAPSHOTS.items():
            for name, digest in snapshot["files"].items():
                relative = f"{snapshot['dest_model']}/snapshots/{snapshot['revision']}/{name}"
                entry = _copy_one(root_fd, Path(snapshot["source"]) / name, relative, digest,
                                  allowed_symlink_root=Path(snapshot["model_cache"]))
                entry["key"] = f"{snapshot_key}/{name}"
                entries.append(entry)
            ref_entry = _write_new_under_root(
                root_fd, f"{snapshot['dest_model']}/refs/main",
                (snapshot["revision"] + "\n").encode("ascii"),
            )
            ref_entry["key"] = f"{snapshot_key}/refs/main"
            entries.append(ref_entry)
        entries.extend(_copy_source_tree(root_fd))
        source_entries = [item for item in entries if item["staged_relative_path"].startswith("source/MeanAudio/")]
        manifest_rows = [
            {"path": item["staged_relative_path"], "bytes": item["bytes"], "sha256": item["sha256"]}
            for item in sorted(source_entries, key=lambda item: item["staged_relative_path"])
        ]
        source_manifest = _write_new_under_root(
            root_fd, "manifests/source_tree_manifest.json",
            json.dumps({"schema_version": 1, "files": manifest_rows}, indent=2, sort_keys=True).encode() + b"\n",
        )
        source_manifest["key"] = "source_tree_manifest"
        entries.append(source_manifest)
        freeze = subprocess.run([str(PYTHON), "-m", "pip", "freeze", "--all"],
                                text=True, capture_output=True, timeout=120, check=False,
                                env={"PATH": "/home/kojiek/venvs/dac/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
        if freeze.returncode:
            raise SecurityHold("pip-freeze manifest collection failed")
        freeze_raw = ("\n".join(sorted(line for line in freeze.stdout.splitlines() if line.strip())) + "\n").encode()
        freeze_entry = _write_new_under_root(root_fd, "manifests/pip_freeze.txt", freeze_raw)
        freeze_entry["key"] = "pip_freeze"
        entries.append(freeze_entry)
        receipt_unsigned = {
            "schema_version": 1, "document_kind": "sealed_input_copy_receipt",
            "plan_id": PLAN_ID, "plan_sha256": PLAN_SHA256,
            "gate1_approval_sha256": sha256_path_nofollow(GATE1_APPROVAL),
            "created_at": now(), "source_mutation": "none", "root": str(SEALED_ROOT),
            "launch_readiness": "blocked_missing_unapproved_dependency",
            "known_missing_dependencies": [{
                "dependency": "roberta-base assets transitively required by the pinned CLAP stack",
                "approved_registered_copy_source": False,
                "disposition": "fail_closed_before Gate-2 offline model/scorer load",
            }],
            "entries": sorted(entries, key=lambda item: item["staged_relative_path"]),
        }
        receipt_entry = _write_new_under_root(
            root_fd, "copy_receipt.json",
            json.dumps(receipt_unsigned, indent=2, sort_keys=True).encode() + b"\n",
        )
        os.fsync(root_fd)
    except BaseException:
        # Preserve partial staging as evidence; never recursively clean/retry in place.
        raise
    finally:
        os.close(root_fd)
    return {
        "status": "staged", "root": str(SEALED_ROOT), "entries": len(entries),
        "bytes": sum(int(item["bytes"]) for item in entries),
        "copy_receipt_sha256": receipt_entry["sha256"],
        "approval_nonce": approval["nonce"],
    }


def verify_sealed_receipt() -> dict[str, Any]:
    receipt_path = SEALED_ROOT / "copy_receipt.json"
    receipt = load_json_nofollow(receipt_path, require_uid=EXPECTED_UID, require_mode=0o400)
    if receipt.get("plan_sha256") != PLAN_SHA256 or receipt.get("root") != str(SEALED_ROOT):
        raise SecurityHold("sealed receipt binding mismatch")
    seen: set[str] = set()
    total = 0
    for entry in receipt.get("entries", []):
        relative = entry.get("staged_relative_path")
        if not isinstance(relative, str) or relative in seen:
            raise SecurityHold("duplicate/invalid sealed receipt entry")
        seen.add(relative)
        path = SEALED_ROOT / relative
        digest = sha256_path_nofollow(path, require_uid=EXPECTED_UID, require_mode=0o400,
                                      require_one_link=True)
        info = path.stat(follow_symlinks=False)
        if (digest != entry.get("sha256") or info.st_size != entry.get("bytes") or
                info.st_dev != entry.get("st_dev") or info.st_ino != entry.get("st_ino")):
            raise SecurityHold(f"sealed entry drift: {relative}")
        total += info.st_size
    blockers = receipt.get("known_missing_dependencies", [])
    if not isinstance(blockers, list):
        raise SecurityHold("sealed receipt blocker list malformed")
    return {"entries": len(seen), "bytes": total,
            "copy_receipt_sha256": sha256_path_nofollow(receipt_path),
            "launch_readiness": receipt.get("launch_readiness"),
            "launch_blockers": blockers}


def rollback_obsolete_staging(expected_receipt_sha256: str) -> dict[str, Any]:
    """Remove only an unreviewed Gate-1 seal named by its exact copy receipt.

    This is intentionally unavailable after a final contract or live HARN state
    exists.  Every file is rehashed and unlinked descriptor-relative; unexpected
    objects, links, or metadata hold the rollback.
    """
    validate_gate1_approval()
    if FINAL_CONTRACT.exists() or STATE_ROOT.exists() or QUEUE_TARGET.exists():
        raise SecurityHold("obsolete staging rollback forbidden after Gate 2/live state")
    if not SHA_RE.fullmatch(expected_receipt_sha256):
        raise SecurityHold("rollback receipt hash must be exact SHA-256")
    receipt_path = SEALED_ROOT / "copy_receipt.json"
    observed_receipt = sha256_path_nofollow(
        receipt_path, require_uid=EXPECTED_UID, require_mode=0o400, require_one_link=True
    )
    if observed_receipt != expected_receipt_sha256:
        raise SecurityHold("obsolete staging receipt hash mismatch")
    receipt = load_json_nofollow(receipt_path, require_uid=EXPECTED_UID, require_mode=0o400)
    entries = receipt.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SecurityHold("obsolete staging receipt has no entries")
    expected_files: dict[str, dict[str, Any]] = {}
    expected_dirs: set[str] = set()
    for entry in entries:
        relative = entry.get("staged_relative_path")
        if not isinstance(relative, str):
            raise SecurityHold("obsolete receipt path invalid")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in expected_files:
            raise SecurityHold("obsolete receipt path escaped/duplicated")
        expected_files[relative] = entry
        for parent in pure.parents:
            if str(parent) != ".":
                expected_dirs.add(str(parent))
    expected_files["copy_receipt.json"] = {
        "sha256": observed_receipt, "bytes": receipt_path.stat(follow_symlinks=False).st_size,
        "mode": "0400", "uid": EXPECTED_UID,
    }
    root_fd = _open_dir_chain(SEALED_ROOT, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
    try:
        actual_files: set[str] = set()
        actual_dirs: set[str] = set()
        for current, dirs, files, dirfd in os.fwalk(SEALED_ROOT, topdown=True, follow_symlinks=False):
            relative_root = Path(current).relative_to(SEALED_ROOT)
            for name in list(dirs):
                info = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != EXPECTED_UID or stat.S_IMODE(info.st_mode) != 0o700:
                    raise SecurityHold("unexpected/non-directory object in obsolete seal")
                value = (relative_root / name).as_posix()
                actual_dirs.add(value)
            for name in files:
                info = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise SecurityHold("non-regular object in obsolete seal")
                actual_files.add((relative_root / name).as_posix())
        if actual_files != set(expected_files) or actual_dirs != expected_dirs:
            raise SecurityHold("obsolete seal tree differs from copy receipt")
        removed_bytes = 0
        for relative in sorted(expected_files, key=lambda value: len(PurePosixPath(value).parts), reverse=True):
            entry = expected_files[relative]
            dfd, name = _destination_parent(root_fd, PurePosixPath(relative))
            try:
                fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=dfd)
                try:
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != EXPECTED_UID or
                            stat.S_IMODE(info.st_mode) != 0o400 or info.st_nlink != 1 or
                            info.st_size != int(entry["bytes"]) or sha256_fd(fd) != entry["sha256"]):
                        raise SecurityHold(f"obsolete sealed file drift: {relative}")
                    removed_bytes += info.st_size
                finally:
                    os.close(fd)
                os.unlink(name, dir_fd=dfd)
                os.fsync(dfd)
            finally:
                os.close(dfd)
        for relative in sorted(expected_dirs, key=lambda value: len(PurePosixPath(value).parts), reverse=True):
            pure = PurePosixPath(relative)
            dfd, name = _destination_parent(root_fd, pure)
            try:
                os.rmdir(name, dir_fd=dfd)
                os.fsync(dfd)
            finally:
                os.close(dfd)
    finally:
        os.close(root_fd)
    parent_fd = _open_dir_chain(SEALED_ROOT.parent, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
    try:
        os.rmdir(SEALED_ROOT.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {"status": "obsolete_unreviewed_seal_removed", "receipt_sha256": observed_receipt,
            "files": len(expected_files), "bytes": removed_bytes}


def generation_argv(arm: str, *, checkpoint_path: str | None = None,
                    tsv_path: str | None = None, output_path: str | None = None) -> list[str]:
    if arm not in ARMS:
        raise SecurityHold("arm must be B1-B6")
    checkpoint = checkpoint_path or str(SEALED_ROOT / REGISTERED_FILES[CHECKPOINT_BY_ARM[arm]]["dest"])
    tsv = tsv_path or str(SEALED_ROOT / REGISTERED_FILES["musiccaps_tsv"]["dest"])
    output = output_path or str(RESULT_ROOT / arm / "audio")
    return [
        str(PYTHON), str(SEALED_ROOT / "source/MeanAudio/eval.py"),
        "--variant", "meanaudio_s", "--model_path", checkpoint,
        "--output", output, "--tsv", tsv, "--use_meanflow",
        "--num_steps", "25", "--cfg_strength", "0", "--encoder_name", "t5_clap",
        "--text_c_dim", "512", "--seed", "42", "--no_text_attention_mask",
        "--full_precision", *CONDITIONING_BY_ARM[arm],
    ]


def scoring_argv(arm: str) -> list[str]:
    if arm not in ARMS:
        raise SecurityHold("arm must be B1-B6")
    return [
        str(PYTHON), str(SEALED_ROOT / "source/MeanAudio/scripts/eval/score_musiccaps_per_item.py"),
        "--tsv", str(SEALED_ROOT / REGISTERED_FILES["musiccaps_tsv"]["dest"]),
        "--audio-dir", str(RESULT_ROOT / arm / "audio"),
        "--out", str(RESULT_ROOT / arm / "metrics/per_item.tsv"),
        "--clap-checkpoint", str(SEALED_ROOT / REGISTERED_FILES["clap_checkpoint"]["dest"]),
        "--audiobox-snapshot", str(SEALED_ROOT / HF_SNAPSHOTS["audiobox"]["dest_model"] /
                                        "snapshots" / HF_SNAPSHOTS["audiobox"]["revision"]),
        "--local-files-only", "--require-exact-count", "5521",
    ]


def analysis_argv() -> list[str]:
    return [
        str(PYTHON), str(SEALED_ROOT / "source/MeanAudio/scripts/analysis/fulltrack_q3_paired_report.py"),
        "--contract", str(FINAL_CONTRACT), "--bootstrap-replicates", "10000",
        "--bootstrap-seed", "20260828", "--out", str(RESULT_ROOT / "matrix_report.json"),
    ]


def reproduction_argv() -> list[str]:
    return [
        str(PYTHON), str(SEALED_ROOT / "source/MeanAudio/scripts/analysis/fulltrack_q3_paired_report.py"),
        "--contract", str(FINAL_CONTRACT), "--bootstrap-replicates", "10000",
        "--bootstrap-seed", "20260828", "--out", str(RESULT_ROOT / "reproduction_gate.json"),
        "--reproduction-only",
    ]


def expanded_command_manifest() -> dict[str, Any]:
    return {
        "environment": dict(ALLOWED_CHILD_ENV),
        "arms": {arm: {"generation": generation_argv(arm), "scoring": scoring_argv(arm)} for arm in ARMS},
        "reproduction": reproduction_argv(), "analysis": analysis_argv(),
    }


def validate_no_shell_argv(argv: Sequence[str]) -> None:
    if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(v, str) and v for v in argv):
        raise SecurityHold("argv must be a nonempty string array")
    if argv[0] != str(PYTHON):
        raise SecurityHold("unregistered executable")
    forbidden = {"-c", "-m", "--eval", "--command"}
    if any(value in forbidden for value in argv[1:2]):
        raise SecurityHold("inline interpreter execution forbidden")
    if any("\x00" in value or "\n" in value or "\r" in value for value in argv):
        raise SecurityHold("control characters in argv")


def stable_event_key(phase: str, kind: str, evidence: Mapping[str, Any]) -> str:
    phase_id = safe_id(phase)
    kind_id = safe_id(kind)
    return f"{RUN_ID}:{phase_id}:{kind_id}:{sha256_bytes(canonical(evidence))}"


def redacted_notification(kind: str, phase: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"verdict", "reason_code", "free_bytes", "completed_ids", "next_phase"}
    fields = {key: evidence[key] for key in sorted(allowed.intersection(evidence))}
    raw = canonical(fields).decode("utf-8")
    if "webhook" in raw.casefold() or "token" in raw.casefold() or len(raw) > 2048:
        raise SecurityHold("notification disclosure violation")
    return {"event_key": stable_event_key(phase, kind, fields), "kind": kind,
            "phase": phase, "fields": fields, "allowed_mentions": {"parse": []}}


def sign_record(key: bytes, domain: bytes, record: Mapping[str, Any]) -> str:
    unsigned = {k: v for k, v in record.items() if k != "hmac_sha256"}
    return hmac.new(key, domain + b"\0" + canonical(unsigned), hashlib.sha256).hexdigest()


def verify_record(key: bytes, domain: bytes, record: Mapping[str, Any]) -> None:
    supplied = record.get("hmac_sha256")
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, sign_record(key, domain, record)):
        raise SecurityHold("authenticated state record mismatch")


def _runtime_key() -> bytes:
    fd = os.open(STATE_KEY, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != EXPECTED_UID or
                stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_size != 32):
            raise SecurityHold("runtime HMAC key metadata invalid")
        raw = os.read(fd, 33)
        if len(raw) != 32:
            raise SecurityHold("runtime HMAC key length invalid")
        return raw
    finally:
        os.close(fd)


def initialize_runtime_security_state() -> None:
    """Initialize authenticated runtime state only after Gate-2 created its root."""
    _open = _open_dir_chain(STATE_ROOT, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
    os.close(_open)
    ensure_private_dir(STATE_OUTBOX)
    if not STATE_KEY.exists():
        dfd = _open_dir_chain(STATE_ROOT, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
        try:
            fd = os.open(STATE_KEY.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                         getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=dfd)
            try:
                os.write(fd, os.urandom(32))
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(dfd)
        finally:
            os.close(dfd)
    key = _runtime_key()
    if not STATE_LOCK_POINTER.exists():
        lock_name = "controller-" + secrets.token_hex(24) + ".lock"
        atomic_write_json(STATE_LOCK_POINTER, {"lock_name": lock_name}, replace=False)
        dfd = _open_dir_chain(STATE_ROOT, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
        try:
            fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                         getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=dfd)
            os.fsync(fd)
            os.close(fd)
            os.fsync(dfd)
        finally:
            os.close(dfd)
    if not STATE_LEDGER.exists():
        ledger = {"schema_version": 1, "run_id": RUN_ID, "generation": 0,
                  "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  "events": [], "hmac_sha256": ""}
        ledger["hmac_sha256"] = sign_record(key, b"ftq3-ledger-v1", ledger)
        atomic_write_json(STATE_LEDGER, ledger, replace=False)


def acquire_runtime_lock() -> int:
    pointer = load_json_nofollow(STATE_LOCK_POINTER, require_uid=EXPECTED_UID, require_mode=0o600)
    name = pointer.get("lock_name")
    if not isinstance(name, str) or not re.fullmatch(r"controller-[a-f0-9]{48}\.lock", name):
        raise SecurityHold("runtime controller-lock pointer invalid")
    dfd = _open_dir_chain(STATE_ROOT, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
    try:
        fd = os.open(name, os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=dfd)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != EXPECTED_UID or
                stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            os.close(fd)
            raise SecurityHold("runtime controller lock metadata invalid")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise SecurityHold("second runtime controller refused") from None
        return fd
    finally:
        os.close(dfd)


def append_runtime_event(kind: str, phase: str, evidence: Mapping[str, Any],
                         *, notification_status: str = "not_applicable",
                         relates_to: str | None = None) -> dict[str, Any]:
    key = _runtime_key()
    ledger = load_json_nofollow(STATE_LEDGER, require_uid=EXPECTED_UID, require_mode=0o600)
    verify_record(key, b"ftq3-ledger-v1", ledger)
    events = ledger.get("events")
    if not isinstance(events, list):
        raise SecurityHold("runtime ledger events malformed")
    event_key = stable_event_key(phase, kind, evidence)
    matching = [item for item in events if item.get("event_key") == event_key]
    if matching:
        if len(matching) != 1:
            raise SecurityHold("duplicate runtime event key")
        return matching[0]
    sequence = len(events) + 1
    previous = events[-1]["event_sha256"] if events else None
    event = {"sequence": sequence, "event_key": event_key, "event_kind": kind,
             "phase": phase, "occurred_at": now(), "evidence_sha256": sha256_bytes(canonical(evidence)),
             "notification_status": notification_status, "relates_to": relates_to,
             "previous_event_sha256": previous}
    event["event_sha256"] = sha256_bytes(canonical(event))
    events.append(event)
    ledger["generation"] = int(ledger.get("generation", 0)) + 1
    ledger["hmac_sha256"] = sign_record(key, b"ftq3-ledger-v1", ledger)
    atomic_write_json(STATE_LEDGER, ledger)
    return event


def persist_outbox_event(payload: Mapping[str, Any]) -> Path:
    key = _runtime_key()
    event_key = str(payload.get("event_key", ""))
    if not event_key.startswith(RUN_ID + ":"):
        raise SecurityHold("outbox event key is not bound to this run")
    name = sha256_bytes(event_key.encode()) + ".json"
    path = STATE_OUTBOX / name
    record = {"schema_version": 1, "event_key": event_key, "status": "pending",
              "attempts": 0, "payload": dict(payload), "created_at": now(), "hmac_sha256": ""}
    record["hmac_sha256"] = sign_record(key, b"ftq3-outbox-v1", record)
    if path.exists():
        prior = load_json_nofollow(path, require_uid=EXPECTED_UID, require_mode=0o600)
        verify_record(key, b"ftq3-outbox-v1", prior)
        if prior.get("event_key") != event_key:
            raise SecurityHold("outbox hash collision")
        return path
    atomic_write_json(path, record, replace=False)
    return path


def deliver_notification(contract: Mapping[str, Any], kind: str, phase: str,
                         evidence: Mapping[str, Any]) -> str:
    payload = redacted_notification(kind, phase, evidence)
    outbox_path = persist_outbox_event(payload)
    key = _runtime_key()
    record = load_json_nofollow(outbox_path, require_uid=EXPECTED_UID, require_mode=0o600)
    verify_record(key, b"ftq3-outbox-v1", record)
    if record.get("status") == "delivered":
        return "delivered"
    if record.get("status") == "delivery_ambiguous":
        raise SecurityHold("notification delivery remains ambiguous")
    notifier = ROOT / "scripts/notify_experiment_webhook.py"
    declared = contract.get("security_bindings", {}).get("notifier_sha256")
    if not isinstance(declared, str) or sha256_path_nofollow(notifier) != declared:
        raise SecurityHold("notification executable is not contract-bound")
    summary = canonical(payload["fields"]).decode("utf-8")
    record["status"] = "attempting"
    record["attempts"] = int(record.get("attempts", 0)) + 1
    record["hmac_sha256"] = sign_record(key, b"ftq3-outbox-v1", record)
    atomic_write_json(outbox_path, record)
    try:
        completed = subprocess.run(
            [str(PYTHON), str(notifier), "--status", "held" if kind in {"hold", "failure"} else "test",
             "--experiment", EXPERIMENT_ID, "--summary", summary],
            cwd=ROOT, env={"PATH": "/home/kojiek/venvs/dac/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
                           "LC_ALL": "C.UTF-8", "PYTHONNOUSERSITE": "1"},
            text=True, capture_output=True, timeout=30, check=False,
        )
    except subprocess.TimeoutExpired:
        completed = None
    record = load_json_nofollow(outbox_path, require_uid=EXPECTED_UID, require_mode=0o600)
    verify_record(key, b"ftq3-outbox-v1", record)
    if completed is not None and completed.returncode == 0:
        record["status"] = "delivered"
        record["accepted_evidence_sha256"] = sha256_bytes(completed.stdout.encode())
    else:
        # The notifier owns the HTTP transaction; nonzero/timeout cannot prove
        # the request was not accepted, so retry is forbidden without reconciliation.
        record["status"] = "delivery_ambiguous"
    record["updated_at"] = now()
    record["hmac_sha256"] = sign_record(key, b"ftq3-outbox-v1", record)
    atomic_write_json(outbox_path, record)
    if record["status"] != "delivered":
        raise SecurityHold("notification delivery ambiguous; operator reconciliation required")
    return "delivered"


def classify_notification_transition(prior: str | None, outcome: str) -> str:
    if prior in {"delivered", "delivery_ambiguous"}:
        raise SecurityHold("notification retry forbidden after acceptance/ambiguity")
    if outcome == "accepted":
        return "delivered"
    if outcome == "known_pre_request_failure":
        return "retryable"
    if outcome in {"timeout_after_send", "connection_lost_after_send", "unknown"}:
        return "delivery_ambiguous"
    raise SecurityHold("unknown notification outcome")


def recovery_promotable(incident: Mapping[str, Any], recovery: Mapping[str, Any] | None) -> bool:
    return bool(incident.get("notification_status") == "delivered" and recovery and
                recovery.get("relates_to") == incident.get("event_key") and
                recovery.get("notification_status") == "delivered")


def budget_status(started_monotonic: float, now_monotonic: float, limit_seconds: int) -> str:
    if limit_seconds <= 0 or not all(math.isfinite(value) for value in (started_monotonic, now_monotonic)):
        return "invalid"
    if now_monotonic < started_monotonic:
        return "invalid"
    return "exhausted" if now_monotonic - started_monotonic >= limit_seconds else "within_budget"


def progress_status(last_value: int, current_value: int, seconds_since_change: float,
                    stall_seconds: int) -> str:
    if min(last_value, current_value, stall_seconds) < 0 or not math.isfinite(seconds_since_change):
        return "invalid"
    if current_value < last_value:
        return "regressed"
    if current_value > last_value:
        return "progressed"
    return "stalled" if seconds_since_change >= stall_seconds else "unchanged_healthy"


def queue_dependency_eligible(status: str, terminal_notification_status: str) -> bool:
    return status in {"completed", "failed", "interrupted"} and terminal_notification_status == "delivered"


def validate_cleanup_names(expected_ids: Iterable[str], journal: Mapping[str, Any]) -> list[str]:
    expected = [safe_id(value) + ".flac" for value in expected_ids]
    intent = journal.get("intent")
    if intent != expected or len(intent) != len(set(intent)):
        raise SecurityHold("cleanup journal is not exact ordered ID intent")
    progress = journal.get("completed", [])
    if not isinstance(progress, list) or any(value not in set(expected) for value in progress):
        raise SecurityHold("cleanup progress escaped exact intent")
    return expected


def cleanup_audio_exact(audio_dir: Path, expected_ids: Sequence[str], journal_path: Path) -> None:
    journal = load_json_nofollow(journal_path, require_uid=EXPECTED_UID, require_mode=0o600)
    names = validate_cleanup_names(expected_ids, journal)
    completed = set(journal.get("completed", []))
    dfd = _open_dir_chain(audio_dir, require_final_uid=EXPECTED_UID, require_final_mode=0o700)
    try:
        for name in names:
            if name in completed:
                continue
            fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=dfd)
            info = os.fstat(fd)
            os.close(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != EXPECTED_UID or info.st_nlink != 1:
                raise SecurityHold(f"unsafe cleanup object: {name}")
            os.unlink(name, dir_fd=dfd)
            completed.add(name)
            journal["completed"] = [item for item in names if item in completed]
            atomic_write_json(journal_path, journal)
        extras = [name for name in os.listdir(dfd) if name not in set(names)]
        if extras:
            raise SecurityHold("extra audio-directory objects remain after exact cleanup")
    finally:
        os.close(dfd)


def validate_gate2_payload(approval: Mapping[str, Any], contract: Mapping[str, Any],
                           bindings: Mapping[str, str], *, require_lifecycle: str,
                           at: datetime | None = None) -> None:
    if contract.get("launch_allowed") is not True or contract.get("approval_required") is not True:
        raise SecurityHold("contract is not launch-enabled and approval-required")
    if approval.get("document_kind") != "gate_2_exact_capability":
        raise SecurityHold("not a Gate-2 capability")
    for key, value in bindings.items():
        if approval.get("bindings", {}).get(key) != value:
            raise SecurityHold(f"Gate-2 binding mismatch: {key}")
    expected_consumed = require_lifecycle == "consumed"
    if approval.get("lifecycle") != require_lifecycle or approval.get("consumed") is not expected_consumed:
        raise SecurityHold("Gate-2 lifecycle/replay mismatch")
    if _parse_utc(str(approval.get("expires_at", ""))) <= (at or datetime.now(timezone.utc)):
        raise SecurityHold("Gate-2 capability expired")
    if approval.get("auth_root_decision") not in {"external_signature_verified", "operator_accepted_same_uid_residual"}:
        raise SecurityHold("Gate-2 AUTH_ROOT decision missing")


def validate_gate2_approval(path: Path, contract_path: Path = FINAL_CONTRACT,
                            *, require_lifecycle: str = "approved") -> dict[str, Any]:
    contract = load_json_nofollow(contract_path, require_uid=EXPECTED_UID, require_mode=0o600)
    if contract_path != FINAL_CONTRACT:
        raise SecurityHold("only exact final contract path may authorize runtime")
    approval = load_json_nofollow(path, require_uid=EXPECTED_UID, require_mode=0o600)
    required = {
        "plan_sha256": PLAN_SHA256,
        "contract_sha256": sha256_path_nofollow(contract_path),
        "launcher_sha256": sha256_path_nofollow(QUEUE_CANDIDATE),
    }
    validate_gate2_payload(approval, contract, required, require_lifecycle=require_lifecycle)
    return approval


def reserve_gate2(path: Path) -> dict[str, Any]:
    approval = validate_gate2_approval(path)
    approval["lifecycle"] = "reserved"
    approval["reserved_at"] = now()
    approval["reserved_run_id"] = RUN_ID
    atomic_write_json(path, approval)
    return approval


def _queue_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"directories": {}}
    for category in ("running", "pending", "held", "failed", "done"):
        directory = QUEUE_ROOT / "p2" / category
        dfd = _open_dir_chain(directory)
        try:
            names = sorted(name for name in os.listdir(dfd) if name.endswith(".sh"))
            snapshot["directories"][category] = names
        finally:
            os.close(dfd)
    snapshot["sha256"] = sha256_bytes(canonical(snapshot["directories"]))
    return snapshot


def _assert_queue_append_preconditions(approval: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
    all_names = [name for names in snapshot["directories"].values() for name in names]
    if "027_fulltrack_q3_pq_bmatrix.sh" in all_names:
        raise SecurityHold("027 collision exists")
    pending = snapshot["directories"]["pending"]
    running = snapshot["directories"]["running"]
    predecessors_terminal = not running and not any(name.startswith(("024_", "025_", "026_")) for name in pending)
    exact_active = running == ["024_fair013_k3_full.sh"] and pending[-2:] == [
        "025_true_random_full.sh", "026_fake_random_full.sh"
    ]
    if not (exact_active or predecessors_terminal):
        raise SecurityHold("p2 append-tail predecessor snapshot mismatch")
    if approval.get("bindings", {}).get("queue_snapshot_sha256") != snapshot["sha256"]:
        raise SecurityHold("Gate-2 queue snapshot drift")


def register_queue_entry(gate2_path: Path) -> None:
    # Never accept a queue-root override and never run from an unregistered acceptance bypass.
    if REJECTED_ENV.intersection(os.environ):
        raise SecurityHold("scheduler/acceptance environment override rejected")
    if gate2_path != STATE_ROOT / "gate2_capability.json":
        raise SecurityHold("Gate-2 capability must use the exact registered state path")
    lock_parent = _open_dir_chain(QUEUE_LOCK.parent)
    lock_fd = os.open(QUEUE_LOCK.name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC |
                      getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=lock_parent)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        approval = validate_gate2_approval(gate2_path)
        initialize_runtime_security_state()
        snapshot = _queue_snapshot()
        _assert_queue_append_preconditions(approval, snapshot)
        reserve_gate2(gate2_path)
        source_fd = os.open(QUEUE_SOURCE, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
        target_parent = _open_dir_chain(QUEUE_TARGET.parent)
        temp_name = ".027-stage-" + secrets.token_hex(16)
        target_fd = -1
        try:
            source_info = os.fstat(source_fd)
            if not stat.S_ISREG(source_info.st_mode) or source_info.st_uid != EXPECTED_UID:
                raise SecurityHold("queue candidate source unsafe")
            target_fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                                getattr(os, "O_NOFOLLOW", 0), 0o700, dir_fd=target_parent)
            os.lseek(source_fd, 0, os.SEEK_SET)
            while block := os.read(source_fd, 65536):
                os.write(target_fd, block)
            os.fsync(target_fd)
            if sha256_fd(source_fd) != sha256_fd(target_fd):
                raise SecurityHold("queue entry copy mismatch")
            os.close(target_fd)
            target_fd = -1
            try:
                os.link(temp_name, QUEUE_TARGET.name, src_dir_fd=target_parent,
                        dst_dir_fd=target_parent, follow_symlinks=False)
            except FileExistsError:
                raise SecurityHold("queue target collision during commit") from None
            os.unlink(temp_name, dir_fd=target_parent)
            os.fsync(target_parent)
        finally:
            if target_fd >= 0:
                os.close(target_fd)
            try:
                os.unlink(temp_name, dir_fd=target_parent)
            except FileNotFoundError:
                pass
            os.close(source_fd)
            os.close(target_parent)
        append_runtime_event(
            "queue_registration", "queue_handoff",
            {"prior_order_sha256": snapshot["sha256"],
             "entry_sha256": sha256_path_nofollow(QUEUE_TARGET),
             "position": len(snapshot["directories"]["pending"]) + 1},
        )
        approval = validate_gate2_approval(gate2_path, require_lifecycle="reserved")
        approval["lifecycle"] = "consumed"
        approval["consumed"] = True
        approval["consumed_at"] = now()
        atomic_write_json(gate2_path, approval)
    finally:
        os.close(lock_fd)
        os.close(lock_parent)


def gpu_compute_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["/usr/bin/nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, check=False,
    )
    if completed.returncode:
        raise SecurityHold("GPU process enumeration failed")
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit() or not parts[2].isdigit():
            raise SecurityHold("malformed GPU process row")
        rows.append({"pid": int(parts[0]), "process_name": parts[1], "used_memory_mib": int(parts[2])})
    return rows


def assert_no_foreign_gpu_processes(owned_pids: set[int]) -> None:
    foreign = [row for row in gpu_compute_processes() if row["pid"] not in owned_pids]
    if foreign:
        raise SecurityHold("foreign GPU process detected at zero threshold")


def assert_p2_lease_identity() -> set[int]:
    """Return owned process IDs only when the existing p2 lease can be proven.

    The current scheduler does not pass its lease FD to the guest.  This check is
    deliberately strict: missing/ambiguous /proc facts hold rather than assuming
    ownership.  Gate-2 review must validate this against registered fixtures/live
    state before any launch approval.
    """
    running_path = QUEUE_ROOT / "p2.running.json"
    owner_path = QUEUE_ROOT / "gpu0.owner.json"
    if not running_path.is_file() or not owner_path.is_file():
        raise SecurityHold("p2 running/owner identity records missing")
    running = load_json_nofollow(running_path)
    owner = load_json_nofollow(owner_path)
    pid = int(running.get("pid", 0))
    if pid <= 1 or int(owner.get("pid", 0)) != pid:
        raise SecurityHold("p2 owner/running PID mismatch")
    if not Path(f"/proc/{pid}").is_dir():
        raise SecurityHold("p2 host PID is not live")
    # An inherited guest may not claim the host PID as its own tree without ancestry proof.
    ancestry: set[int] = {os.getpid()}
    current = os.getppid()
    for _ in range(64):
        if current <= 1 or current in ancestry:
            break
        ancestry.add(current)
        try:
            fields = Path(f"/proc/{current}/stat").read_text().split()
            current = int(fields[3])
        except (OSError, ValueError, IndexError):
            raise SecurityHold("process ancestry proof failed") from None
    if pid not in ancestry:
        raise SecurityHold("p2 host PID is not in guest ancestry")
    # The /proc/locks holder proof is Linux-specific and must find exactly one holder.
    lock_info = os.stat(QUEUE_ROOT / "gpu0.lock", follow_symlinks=False)
    candidates = []
    device_major = os.major(lock_info.st_dev)
    device_minor = os.minor(lock_info.st_dev)
    for line in Path("/proc/locks").read_text().splitlines():
        if "FLOCK" not in line or "WRITE" not in line:
            continue
        fields = line.split()
        if len(fields) < 6:
            continue
        devino = fields[5].split(":")
        if len(devino) == 3:
            try:
                major, minor, ino = int(devino[0], 16), int(devino[1], 16), int(devino[2])
            except ValueError:
                continue
            if (major, minor, ino) == (device_major, device_minor, lock_info.st_ino):
                candidates.append(int(fields[4]))
    if len(candidates) != 1:
        raise SecurityHold("gpu0.lock holder is missing or ambiguous")
    holder = candidates[0]
    cmdline = Path(f"/proc/{holder}/cmdline").read_bytes().split(b"\0")
    if not any(part.endswith(b"hold_lock.py") for part in cmdline):
        raise SecurityHold("gpu0.lock holder executable identity mismatch")
    if str(pid).encode() not in cmdline or str(QUEUE_ROOT / "gpu0.lock").encode() not in cmdline:
        raise SecurityHold("gpu0.lock holder argv binding mismatch")
    return ancestry | {holder}


def dry_run() -> dict[str, Any]:
    validate_gate1_approval()
    manifest = expanded_command_manifest()
    for arm in ARMS:
        validate_no_shell_argv(manifest["arms"][arm]["generation"])
        validate_no_shell_argv(manifest["arms"][arm]["scoring"])
    validate_no_shell_argv(manifest["analysis"])
    validate_no_shell_argv(manifest["reproduction"])
    return {
        "status": "implementation_complete_awaiting_exact_review",
        "launch_allowed": False, "queue_mutation_allowed": False,
        "plan_sha256": PLAN_SHA256, "commands": manifest,
        "sealed": verify_sealed_receipt() if SEALED_ROOT.is_dir() else None,
    }


def run_matrix() -> None:
    # Runtime implementation exists for exact review, but Gate 1 cannot satisfy it.
    if not FINAL_CONTRACT.is_file():
        raise SecurityHold("final contract absent; Gate 2 has not been reached")
    gate2 = STATE_ROOT / "gate2_capability.json"
    contract = load_json_nofollow(FINAL_CONTRACT, require_uid=EXPECTED_UID, require_mode=0o600)
    validate_gate2_approval(gate2, require_lifecycle="consumed")
    controller_lock = acquire_runtime_lock()
    try:
        append_runtime_event("experiment_started", "B1", {"verdict": "start"},
                             notification_status="pending")
        deliver_notification(contract, "start", "B1", {"verdict": "start", "next_phase": "B1"})
        sealed = verify_sealed_receipt()
        if sealed["launch_blockers"]:
            raise SecurityHold("sealed runtime has unresolved unapproved dependencies")
        if storage_status()["verdict"] == "hard_stop":
            raise SecurityHold("root filesystem below hard floor")
        owned = assert_p2_lease_identity()
        assert_no_foreign_gpu_processes(owned)
        # The arm runner separately revalidates the consumed capability and all artifacts.
        started = time.monotonic()
        for arm in PHASE1:
            if time.monotonic() - started >= MATRIX_WALL_SECONDS:
                raise SecurityHold("matrix wall budget exhausted")
            completed = subprocess.run(
                [str(PYTHON), str(RUNNER), "--contract", str(FINAL_CONTRACT), "--arm", arm],
                cwd=SEALED_ROOT / "source/MeanAudio", env=sanitized_child_environment({}),
                timeout=PER_ARM_WALL_SECONDS, check=False,
            )
            if completed.returncode:
                raise SecurityHold(f"phase-1 arm failed/held: {arm}")
            evidence = {"verdict": "pass", "next_phase": "B2" if arm == "B1" else "reproduction_gate"}
            append_runtime_event("gate_result", arm, evidence, notification_status="pending")
            deliver_notification(contract, "gate", arm, evidence)
        reproduction_report = RESULT_ROOT / "reproduction_gate.json"
        reproduction_completed = subprocess.run(
            reproduction_argv(), cwd=SEALED_ROOT / "source/MeanAudio",
            env=sanitized_child_environment({}), timeout=3600, check=False,
        )
        if reproduction_completed.returncode or not reproduction_report.is_file():
            raise SecurityHold("reproduction gate execution failed")
        reproduction = load_json_nofollow(reproduction_report)
        if reproduction.get("decision") != "passed":
            evidence = {"verdict": "fail", "reason_code": str(reproduction.get("decision", "invalid"))}
            append_runtime_event("queue_hold", "reproduction_gate", evidence, notification_status="pending")
            deliver_notification(contract, "hold", "reproduction_gate", evidence)
            raise SecurityHold("reproduction gate did not pass; B3-B6 held")
        append_runtime_event("gate_result", "reproduction_gate", {"verdict": "pass"},
                             notification_status="pending")
        deliver_notification(contract, "gate", "reproduction_gate",
                             {"verdict": "pass", "next_phase": "B3"})
        for arm in PHASE2:
            owned = assert_p2_lease_identity()
            assert_no_foreign_gpu_processes(owned)
            if storage_status()["verdict"] == "hard_stop":
                raise SecurityHold("root filesystem below hard floor")
            completed = subprocess.run(
                [str(PYTHON), str(RUNNER), "--contract", str(FINAL_CONTRACT), "--arm", arm],
                cwd=SEALED_ROOT / "source/MeanAudio", env=sanitized_child_environment({}),
                timeout=PER_ARM_WALL_SECONDS, check=False,
            )
            if completed.returncode:
                raise SecurityHold(f"phase-2 arm failed/held: {arm}")
            evidence = {"verdict": "pass", "next_phase": "paired_analysis" if arm == "B6" else f"B{int(arm[1])+1}"}
            append_runtime_event("gate_result", arm, evidence, notification_status="pending")
            deliver_notification(contract, "gate", arm, evidence)
        completed = subprocess.run(analysis_argv(), cwd=SEALED_ROOT / "source/MeanAudio",
                                   env=sanitized_child_environment({}), timeout=3600, check=False)
        if completed.returncode:
            raise SecurityHold("paired analysis failed")
        append_runtime_event("experiment_completed", "queue_handoff", {"verdict": "pass"},
                             notification_status="pending")
        deliver_notification(contract, "completion", "queue_handoff", {"verdict": "pass"})
    except BaseException as exc:
        try:
            append_runtime_event("experiment_failed", "queue_handoff",
                                 {"verdict": "fail", "reason_code": type(exc).__name__},
                                 notification_status="pending")
            deliver_notification(contract, "failure", "queue_handoff",
                                 {"verdict": "fail", "reason_code": type(exc).__name__})
        except BaseException:
            pass
        raise
    finally:
        os.close(controller_lock)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage")
    rollback = sub.add_parser("rollback-obsolete-stage")
    rollback.add_argument("--receipt-sha256", required=True)
    sub.add_parser("dry-run")
    register = sub.add_parser("register")
    register.add_argument("--gate2-approval", type=Path, required=True)
    sub.add_parser("run")
    args = parser.parse_args()
    try:
        if args.command == "stage":
            print(json.dumps(stage_inputs(), sort_keys=True))
        elif args.command == "rollback-obsolete-stage":
            print(json.dumps(rollback_obsolete_staging(args.receipt_sha256), sort_keys=True))
        elif args.command == "dry-run":
            print(json.dumps(dry_run(), indent=2, sort_keys=True))
        elif args.command == "register":
            register_queue_entry(args.gate2_approval)
            print("registered")
        else:
            run_matrix()
        return 0
    except SecurityHold as exc:
        print(f"HOLD: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(_main())
