#!/usr/bin/env python3
"""Strict completion validator shared by the CFG0 wrapper and HARN."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

METRIC_KEYS = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
PROTOCOL = "MusicCaps 5521; MeanFlow 25; CFG 0; seed 42; NoMask; full precision"
# A cell's evaluated S2 EMA does not exist when the contract is preregistered,
# so its sha256 is authored as this placeholder and bound on the first run.
PENDING_SHA = "pending_runtime_output"
BINDING_KIND = "cfg0_checkpoint_binding"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def binding_path(contract: dict, cell: dict) -> Path:
    """Sidecar path for one contract cell.

    The name must be unique per cell across every CFG0 contract: they all share
    one reports/ directory and most reuse the cell_id "canonical_noq", so a name
    built from the directory plus the arm collided and made each arm after the
    first compare its own EMA against a previous experiment's bound sha256.
    The cell label is the per-experiment eval identity, so it disambiguates.
    """
    root = Path(contract["runtime_storage"]["metrics_root"]).parent / "bindings"
    return root / f"{cell['label']}_{cell['cell_id']}.checkpoint.json"


def same_path(a: str | Path, b: str | Path) -> bool:
    """Compare two paths by their resolved location.

    exps/ is a symlink farm (MeanAudio/exps -> exps_nvme, and individual
    experiments may be symlinks onto the HDD archive), so the logical path a
    contract registers and the real path a runtime records name the same file
    while differing as strings. NVMe and HDD backing store are equivalent.
    """
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def validate(contract_path: Path, arm: str, report_path: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    matches = [cell for cell in contract["cells"] if cell["cell_id"] == arm]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate arm")
    cell = matches[0]
    protocol = contract["fixed_protocol"]
    if not same_path(report_path, cell["report"]) or not report_path.is_file() or report_path.is_symlink():
        raise ValueError("report identity mismatch")
    report_info = report_path.lstat()
    if (not stat.S_ISREG(report_info.st_mode) or report_info.st_uid != os.geteuid()
            or report_info.st_nlink != 1):
        raise ValueError("unsafe report file")
    checkpoint = Path(cell["checkpoint"])
    tsv = Path(protocol["tsv"])
    if digest(tsv) != protocol["tsv_sha256"]:
        raise ValueError("registered input hash drift")

    # The evaluated checkpoint's sha256 cannot be preregistered (the EMA does not
    # exist until training finishes), so a placeholder cell binds on first run to
    # a sidecar and is compared strictly from then on. The preregistered contract
    # file is never mutated.
    expected_sha = cell["checkpoint_sha256"]
    actual_sha = digest(checkpoint)
    sidecar = binding_path(contract, cell)
    if expected_sha == PENDING_SHA:
        bound = json.loads(sidecar.read_text()) if sidecar.is_file() else None
        if bound is not None:
            if bound.get("document_kind") != BINDING_KIND or bound.get("arm") != arm:
                raise ValueError("checkpoint binding sidecar invalid")
            expected_sha = str(bound.get("checkpoint_sha256") or "")
        else:
            expected_sha = actual_sha
            atomic_json(sidecar, {
                "document_kind": BINDING_KIND,
                "contract": str(contract_path),
                "experiment_id": contract.get("experiment_id"),
                "arm": arm,
                "label": cell["label"],
                "checkpoint": str(checkpoint),
                "checkpoint_realpath": os.path.realpath(str(checkpoint)),
                "checkpoint_sha256": actual_sha,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "bound_at": datetime.now(timezone.utc).isoformat(),
            })
    if actual_sha != expected_sha:
        raise ValueError("registered input hash drift")
    payload = json.loads(report_path.read_text())
    expected_metrics = Path(contract["runtime_storage"]["metrics_root"]) / cell["label"] / "metrics.txt"
    expected = {
        "status": "passed", "label": cell["label"], "protocol": PROTOCOL,
        "cfg_strength": 0, "num_steps": 25, "seed": 42,
        "conditioning": cell["conditioning"],
        "checkpoint_sha256": expected_sha,
        "tsv_sha256": protocol["tsv_sha256"],
    }
    drift = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    # Path-valued fields are compared by resolved location, not by string, so a
    # report that recorded the real NVMe/HDD path still matches a contract that
    # registered the logical exps/ path (and vice versa).
    for key, value in (("checkpoint", checkpoint), ("tsv", tsv),
                       ("metrics_path", expected_metrics)):
        if not same_path(payload.get(key) or "", value):
            drift[key] = (payload.get(key), str(value))
    if drift:
        raise ValueError(f"report protocol drift: {drift}")
    audio = payload.get("audio_validation")
    if audio != {"rows": 5521, "unique_ids": 5521, "sample_rate": 16000, "channels": 1}:
        raise ValueError("audio validation evidence mismatch")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise ValueError("metric key mismatch")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite metric")
    if not expected_metrics.is_file() or expected_metrics.is_symlink():
        raise ValueError("missing or unsafe metrics file")
    metrics_info = expected_metrics.lstat()
    if (not stat.S_ISREG(metrics_info.st_mode) or metrics_info.st_uid != os.geteuid()
            or metrics_info.st_nlink != 1):
        raise ValueError("unsafe metrics file")
    if payload.get("metrics_sha256") != digest(expected_metrics):
        raise ValueError("metrics hash mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    validate(args.contract, args.arm, args.report)
    print(f"STRICT_REPORT_OK arm={args.arm} report={args.report}")


if __name__ == "__main__":
    main()
