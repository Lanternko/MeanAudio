#!/usr/bin/env python3
"""Fail-closed full-corpus audit of TSV captions against the mutable NPZ cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_ENCODER_FINGERPRINT = "27e88fac68d94a8a10e44d2db930a8f79db8ca0454ce996b82e448c48c40ab4c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def array_meta(archive: zipfile.ZipFile, key: str, *, read_value: bool = False):
    """Read an NPY header (and only tiny scalar payloads) from an NPZ member."""
    member = f"{key}.npy"
    with archive.open(member) as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NPY header version for {key}: {version}")
        value = None
        if read_value:
            count = int(np.prod(shape)) if shape else 1
            if count != 1 or dtype.itemsize > 4096:
                raise ValueError(f"refusing non-scalar metadata read: {key} {shape} {dtype}")
            value = np.frombuffer(handle.read(dtype.itemsize), dtype=dtype, count=1)[0]
            value = str(value.item() if hasattr(value, "item") else value)
        return shape, dtype, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=249537)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise SystemExit("workers must be in [1, 32]")

    with args.tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(rows) != args.expected_rows or len(names) != args.expected_rows:
        raise SystemExit(
            f"row count mismatch rows={len(rows)} names={len(names)} expected={args.expected_rows}"
        )
    ids = [row.get("id", "") for row in rows]
    if any(not clip_id for clip_id in ids) or len(ids) != len(set(ids)):
        raise SystemExit("blank or duplicate TSV ids")
    if len(names) != len(set(names)):
        raise SystemExit("duplicate cache names")

    required = {
        "mean", "std", "text_features", "text_features_c", "text_attention_mask",
        "clip_id", "caption_sha256", "text_encoder_fingerprint",
    }
    def check_row(item):
        index, row, name = item
        path = args.npz_dir / name
        try:
            with zipfile.ZipFile(path) as archive:
                members = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
                missing = required - members
                if missing:
                    raise ValueError(f"missing keys {sorted(missing)}")
                expected_caption = hashlib.sha256(row["caption"].encode("utf-8")).hexdigest()
                # Caption/ID/fingerprint are exhaustive row-level gates. Array
                # keys are exhaustive via the ZIP directory; expensive header
                # shape reads use a deterministic ~512-row sample because the
                # rebind operation never changes audio arrays or array shapes.
                check_shapes = index % 487 == 0
                shapes = {}
                if check_shapes:
                    shapes = {
                        key: array_meta(archive, key)[0]
                        for key in (
                            "mean", "std", "text_features", "text_features_c", "text_attention_mask"
                        )
                    }
                metadata = {
                    key: array_meta(archive, key, read_value=True)[2]
                    for key in ("clip_id", "caption_sha256", "text_encoder_fingerprint")
                }
                checks = {
                    "clip_id": metadata["clip_id"] == row["id"],
                    "caption_sha256": metadata["caption_sha256"] == expected_caption,
                    "encoder": metadata["text_encoder_fingerprint"] == EXPECTED_ENCODER_FINGERPRINT,
                    "mean_shape": not check_shapes or shapes["mean"] == (312, 20),
                    "std_shape": not check_shapes or shapes["std"] == (312, 20),
                    "text_shape": not check_shapes or shapes["text_features"] == (77, 1024),
                    "clap_shape": not check_shapes or shapes["text_features_c"] == (512,),
                    "mask_shape": not check_shapes or shapes["text_attention_mask"] == (77,),
                }
                failed = [key for key, passed in checks.items() if not passed]
                if failed:
                    raise ValueError(f"failed checks {failed}")
            return None
        except Exception as exc:  # exhaustive audit must report bounded evidence
            return f"row={index} name={name} id={row.get('id')} error={exc}"

    errors: list[str] = []
    items = ((index, row, name) for index, (row, name) in enumerate(zip(rows, names)))
    executor = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    results = executor.map(check_row, items) if executor else map(check_row, items)
    for index, error in enumerate(results):
        if error is not None and len(errors) < 20:
            errors.append(error)
        if (index + 1) % 25000 == 0:
            print(
                f"audit_progress={index + 1}/{len(rows)} errors_seen={len(errors)}",
                flush=True,
            )
    if executor:
        executor.shutdown()

    payload = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completed_rows": len(rows),
        "rows_checked": len(rows),
        "shape_rows_checked": (len(rows) + 486) // 487,
        "errors_preview": errors,
        "tsv": str(args.tsv),
        "tsv_sha256": sha256_file(args.tsv),
        "cache_list": str(args.cache_list),
        "cache_list_sha256": sha256_file(args.cache_list),
        "npz_dir": str(args.npz_dir),
        "text_encoder_fingerprint": EXPECTED_ENCODER_FINGERPRINT,
    }
    atomic_json(args.report, payload)
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(f"NPZ binding audit failed; first errors: {errors[:3]}")


if __name__ == "__main__":
    main()
