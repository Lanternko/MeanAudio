#!/usr/bin/env python3
"""Exact RNG-replay repair for one preregistered R-Matched FLAC artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/rmatched_matrix_repair_harn")
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
S2 = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
CELL = "s2_mf25_cfg0p5"
TARGET_ID = "5xIBQGMjiX4_30"
TARGET_INDEX = 707
EXPECTED_COUNT = 5521
EXPECTED_FRAMES = 159744
EXPECTED_SAMPLE_RATE = 16000
ORIGINAL_SHA256 = "c3472953aa061d327979ab97cc736dcf772e6d98a4148c815b28495a865d947e"
OUT_ROOT = ROOT / "eval_output/rmatched_s1_s2_steps_cfg_matrix_seed14159265"
AUDIO = OUT_ROOT / CELL / "audio"
TARGET = AUDIO / f"{TARGET_ID}.flac"
METRICS = ROOT / "eval_output/metrics/rmatched_s1_s2_steps_cfg_matrix_seed14159265_s2_mf25_cfg0p5/metrics.txt"
PRE_MANIFEST = STATE / "repair_pre_manifest.json"
REPORT = STATE / "repair_report.json"
SCRATCH = STATE / "rng_replay"
REPLAY_AUDIO = SCRATCH / "audio"
REPLAY_TSV = SCRATCH / "musiccaps_through_target.tsv"
REPLAY_LOG = STATE / "rng_replay.log"
QUARANTINE = STATE / "quarantine"
QUARANTINED_TARGET = QUARANTINE / f"{TARGET_ID}.{ORIGINAL_SHA256}.flac"
QUARANTINED_METRICS = QUARANTINE / "invalid_metrics.txt"
PYTHON = Path("/home/kojiek/venvs/dac/bin/python")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_rows() -> tuple[list[str], list[dict[str, str]]]:
    with TSV.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    ids = [row.get("id", "") for row in rows]
    if len(rows) != EXPECTED_COUNT or len(ids) != len(set(ids)) or ids[TARGET_INDEX] != TARGET_ID:
        raise RuntimeError("benchmark row identity/order drift")
    return fields, rows


def audio_shape(path: Path) -> dict[str, int]:
    with sf.SoundFile(path) as handle:
        return {"frames": len(handle), "sample_rate": handle.samplerate, "channels": handle.channels}


def validate_audio(path: Path) -> None:
    shape = audio_shape(path)
    expected = {"frames": EXPECTED_FRAMES, "sample_rate": EXPECTED_SAMPLE_RATE, "channels": 1}
    if shape != expected:
        raise RuntimeError(f"invalid audio shape: {path}: {shape} expected={expected}")
    with sf.SoundFile(path) as handle:
        for block in handle.blocks(blocksize=16384, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise RuntimeError(f"non-finite audio samples: {path}")


def audit() -> dict[str, Any]:
    _, rows = read_rows()
    files = sorted(AUDIO.glob("*.flac"))
    if len(files) != EXPECTED_COUNT:
        raise RuntimeError(f"audio count drift: {len(files)} != {EXPECTED_COUNT}")
    expected_paths = {AUDIO / f"{row['id']}.flac" for row in rows}
    if set(files) != expected_paths:
        raise RuntimeError("audio IDs do not map one-to-one to the benchmark TSV")
    hashes = {path.name: sha256(path) for path in files}
    if hashes[TARGET.name] != ORIGINAL_SHA256:
        raise RuntimeError("corrupt target hash drift")
    invalid: list[dict[str, Any]] = []
    for path in files:
        try:
            validate_audio(path)
        except Exception as exc:
            invalid.append({"path": str(path), "sha256": hashes[path.name], "reason": str(exc)})
    if len(invalid) != 1 or invalid[0]["path"] != str(TARGET):
        raise RuntimeError(f"repair envelope requires exactly the registered corrupt target: {invalid}")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "created_at": now(),
        "tsv": str(TSV),
        "tsv_sha256": sha256(TSV),
        "expected_count": EXPECTED_COUNT,
        "target_index_zero_based": TARGET_INDEX,
        "target_id": TARGET_ID,
        "invalid": invalid,
        "audio_sha256": hashes,
        "invalid_prior_metrics": (
            {"path": str(METRICS), "sha256": sha256(METRICS)} if METRICS.is_file() else None
        ),
    }
    atomic_json(PRE_MANIFEST, payload)
    return payload


def write_replay_tsv(fields: list[str], rows: list[dict[str, str]]) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
    with REPLAY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[: TARGET_INDEX + 1])


def replay_command() -> list[str]:
    return [
        str(PYTHON), "eval.py", "--variant", "meanaudio_s", "--model_path", str(S2),
        "--output", str(REPLAY_AUDIO), "--tsv", str(REPLAY_TSV), "--num_steps", "25",
        "--cfg_strength", "0.5", "--encoder_name", "t5_clap", "--text_c_dim", "512",
        "--seed", "42", "--no_q", "--no_text_attention_mask", "--full_precision",
        "--use_meanflow",
    ]


def apply_repair() -> None:
    if REPORT.is_file() and json.loads(REPORT.read_text()).get("status") == "passed":
        validate_audio(TARGET)
        return
    manifest = audit()
    fields, rows = read_rows()
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    write_replay_tsv(fields, rows)
    REPLAY_AUDIO.mkdir(parents=True, exist_ok=True, mode=0o700)
    with REPLAY_LOG.open("wb") as log:
        completed = subprocess.run(
            replay_command(), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        raise RuntimeError(f"RNG replay failed with exit {completed.returncode}")
    replay_files = sorted(REPLAY_AUDIO.glob("*.flac"))
    if len(replay_files) != TARGET_INDEX + 1:
        raise RuntimeError(f"RNG replay count mismatch: {len(replay_files)}")
    mismatches = []
    for row in rows[:TARGET_INDEX]:
        name = f"{row['id']}.flac"
        replay_hash = sha256(REPLAY_AUDIO / name)
        if replay_hash != manifest["audio_sha256"][name]:
            mismatches.append({"name": name, "existing": manifest["audio_sha256"][name], "replay": replay_hash})
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise RuntimeError(f"RNG replay equivalence failed: {mismatches}")
    replacement = REPLAY_AUDIO / TARGET.name
    validate_audio(replacement)
    replacement_hash = sha256(replacement)
    if replacement_hash == ORIGINAL_SHA256:
        raise RuntimeError("replacement unexpectedly matches corrupt bytes")

    QUARANTINE.mkdir(parents=True, exist_ok=True, mode=0o700)
    if QUARANTINED_TARGET.exists() or QUARANTINED_METRICS.exists():
        raise RuntimeError("repair quarantine is not empty")
    os.replace(TARGET, QUARANTINED_TARGET)
    metrics_moved = False
    try:
        if METRICS.is_file():
            os.replace(METRICS, QUARANTINED_METRICS)
            metrics_moved = True
        os.replace(replacement, TARGET)
    except BaseException:
        if TARGET.exists() and sha256(TARGET) != ORIGINAL_SHA256:
            os.replace(TARGET, replacement)
        if metrics_moved and QUARANTINED_METRICS.exists():
            os.replace(QUARANTINED_METRICS, METRICS)
        if QUARANTINED_TARGET.exists():
            os.replace(QUARANTINED_TARGET, TARGET)
        raise

    validate_audio(TARGET)
    current = {path.name: sha256(path) for path in sorted(AUDIO.glob("*.flac"))}
    changed = [name for name, old_hash in manifest["audio_sha256"].items() if current.get(name) != old_hash]
    if changed != [TARGET.name] or current[TARGET.name] != replacement_hash:
        raise RuntimeError(f"post-repair mutation boundary failed: {changed}")
    if sha256(QUARANTINED_TARGET) != ORIGINAL_SHA256:
        raise RuntimeError("quarantined source hash mismatch")
    atomic_json(REPORT, {
        "schema_version": 1,
        "status": "passed",
        "completed_at": now(),
        "operator_authorization_sha256": "6d93c976c3319491d291d8b829ddd958116b5a01782ae4839ee2b6b737e96a77",
        "pre_manifest": str(PRE_MANIFEST),
        "pre_manifest_sha256": sha256(PRE_MANIFEST),
        "target": str(TARGET),
        "target_index_zero_based": TARGET_INDEX,
        "original_sha256": ORIGINAL_SHA256,
        "replacement_sha256": replacement_hash,
        "quarantine": str(QUARANTINED_TARGET),
        "prefix_replay_hash_matches": TARGET_INDEX,
        "changed_audio_files": changed,
        "invalidated_metrics": str(QUARANTINED_METRICS) if metrics_moved else None,
        "replay_command": replay_command(),
    })


def rollback() -> None:
    report = json.loads(REPORT.read_text())
    if report.get("status") != "passed" or not QUARANTINED_TARGET.is_file():
        raise RuntimeError("no completed repair is available to roll back")
    replacement = REPLAY_AUDIO / TARGET.name
    if replacement.exists():
        raise RuntimeError("replacement rollback destination already exists")
    os.replace(TARGET, replacement)
    os.replace(QUARANTINED_TARGET, TARGET)
    if QUARANTINED_METRICS.exists():
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        os.replace(QUARANTINED_METRICS, METRICS)
    if sha256(TARGET) != ORIGINAL_SHA256:
        raise RuntimeError("rollback did not restore original bytes")
    report.update({"status": "rolled_back", "rolled_back_at": now()})
    atomic_json(REPORT, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "apply", "rollback"))
    args = parser.parse_args()
    if args.action == "audit":
        print(json.dumps(audit(), indent=2, sort_keys=True))
    elif args.action == "apply":
        apply_repair()
        print(REPORT.read_text())
    else:
        rollback()


if __name__ == "__main__":
    main()
