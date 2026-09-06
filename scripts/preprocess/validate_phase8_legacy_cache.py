#!/usr/bin/env python3
"""Validate a rebuilt Phase-8 legacy cache before expensive training.

The cheap checks are exhaustive over every row and filename.  The expensive
NPZ/audio comparisons use a deterministic, full-range sample large enough to
catch positional or partial-copy failures without rereading the entire cache
twice from disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DATA_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data")
DEFAULT_CATALOG = DATA_DIR / "_QUARANTINED_npz.tsv"
DEFAULT_CACHE = DATA_DIR / "npz_cache_train.txt"
DEFAULT_Q_TSV = DATA_DIR / "_QUARANTINED_phase7_v1_train.tsv"
DEFAULT_SOURCE = Path("/home/kojiek/research/meanaudio_training/npz_phase7_clean")
DEFAULT_OUTPUT = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz")
DEFAULT_TSV = DATA_DIR / "phase8_legacy_catalog_train.tsv"
DEFAULT_REPORT = DEFAULT_OUTPUT / "FULL_VALIDATION.json"
EXPECTED_ROWS = 251_599


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--q-tsv", type=Path, default=DEFAULT_Q_TSV)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--deep-sample-size", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260717)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_cache(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def caption_sha256(caption: str) -> str:
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def read_catalog_for_names(
    path: Path, names: list[str]
) -> dict[int, dict[str, str]]:
    requested = {int(Path(name).stem) for name in names}
    selected: dict[int, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if index in requested:
                selected[index] = row
    missing = requested - selected.keys()
    if missing:
        raise SystemExit(f"Catalog is missing requested row {min(missing)}")
    return selected


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.report.unlink(missing_ok=True)
    required = [
        args.catalog,
        args.cache,
        args.q_tsv,
        args.source_dir,
        args.output_dir,
        args.output_tsv,
        args.output_dir / "MANIFEST.tsv",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing validation input: {path}")

    names = read_cache(args.cache)
    rows = read_tsv(args.output_tsv)
    q_rows = read_tsv(args.q_tsv)
    manifest_path = args.output_dir / "MANIFEST.tsv"
    manifest = read_tsv(manifest_path)
    if not (
        len(names)
        == len(set(names))
        == len(rows)
        == len(q_rows)
        == len(manifest)
        == args.expected_rows
    ):
        raise SystemExit(
            "Row-count/uniqueness failure: "
            f"cache={len(names):,}, unique={len(set(names)):,}, "
            f"tsv={len(rows):,}, q={len(q_rows):,}, "
            f"manifest={len(manifest):,}, expected={args.expected_rows:,}"
        )

    catalog = read_catalog_for_names(args.catalog, names)
    for position, (name, row, q_row, item) in enumerate(
        zip(names, rows, q_rows, manifest)
    ):
        catalog_index = int(Path(name).stem)
        catalog_row = catalog[catalog_index]
        caption = catalog_row["caption"].strip()
        clip_id = catalog_row["id"].strip()
        q_level = q_row["q_level"].strip()
        expected_item = {
            "row_index": str(position),
            "catalog_index": str(catalog_index),
            "clip_id": clip_id,
            "npz_fname": name,
            "caption_sha256": caption_sha256(caption),
            "historical_q_level": q_level,
        }
        expected_row = {"id": clip_id, "caption": caption, "q_level": q_level}
        if item != expected_item:
            raise SystemExit(f"Manifest mismatch at position {position}: {name}")
        if row != expected_row:
            raise SystemExit(f"Output TSV mismatch at position {position}: {name}")

    expected_files = set(names)
    actual_files = {path.name for path in args.output_dir.glob("*.npz")}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise SystemExit(
            f"NPZ filename-set mismatch: missing={len(missing):,} "
            f"first_missing={missing[:1]}, extra={len(extra):,} first_extra={extra[:1]}"
        )
    print(f"[OK] exhaustive rows/files: {len(names):,}/{len(names):,}")

    if not 0 < args.deep_sample_size <= len(names):
        raise SystemExit("Invalid --deep-sample-size")
    rng = random.Random(args.sample_seed)
    positions = set(rng.sample(range(len(names)), args.deep_sample_size))
    positions.update(
        position
        for position in (0, 1, 100, 1000, 10000, len(names) - 1)
        if position < len(names)
    )
    positions = sorted(positions)
    quartiles = [0, 0, 0, 0]
    for position in positions:
        quartiles[min(3, position * 4 // len(names))] += 1
        name = names[position]
        row = rows[position]
        item = manifest[position]
        with np.load(args.source_dir / name) as source, np.load(
            args.output_dir / name
        ) as output:
            if not np.array_equal(source["mean"], output["mean"]):
                raise SystemExit(f"Audio mean changed: {name}")
            if not np.array_equal(source["std"], output["std"]):
                raise SystemExit(f"Audio std changed: {name}")
            if output["text_features"].shape != (77, 1024):
                raise SystemExit(f"Bad T5 feature shape: {name}")
            if output["text_features_c"].shape != (512,):
                raise SystemExit(f"Bad CLAP feature shape: {name}")
            if output["text_attention_mask"].shape != (77,):
                raise SystemExit(f"Bad text attention-mask shape: {name}")
            if str(output["clip_id"].item()) != row["id"]:
                raise SystemExit(f"Embedded clip ID mismatch: {name}")
            if int(output["catalog_index"].item()) != int(item["catalog_index"]):
                raise SystemExit(f"Embedded catalog index mismatch: {name}")
            if str(output["caption_sha256"].item()) != item["caption_sha256"]:
                raise SystemExit(f"Embedded caption hash mismatch: {name}")
    print(
        f"[OK] deep NPZ/audio sample: {len(positions):,}; "
        f"quartiles={quartiles}"
    )

    payload: dict[str, object] = {
        "status": "passed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "expected_rows": args.expected_rows,
        "deep_sample_size": len(positions),
        "sample_seed": args.sample_seed,
        "quartile_counts": quartiles,
        "paths": {
            "cache": str(args.cache.resolve()),
            "catalog": str(args.catalog.resolve()),
            "manifest": str(manifest_path.resolve()),
            "output_tsv": str(args.output_tsv.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        },
        "sha256": {
            "cache": sha256_file(args.cache),
            "catalog": sha256_file(args.catalog),
            "manifest": sha256_file(manifest_path),
            "output_tsv": sha256_file(args.output_tsv),
        },
    }
    atomic_json(args.report, payload)
    print(f"[PASS] validation report: {args.report}")


if __name__ == "__main__":
    main()
