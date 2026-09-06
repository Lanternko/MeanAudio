#!/usr/bin/env python3
"""Apply a preregistered Q assignment to a caption TSV without changing captions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-tsv", type=Path, required=True)
    parser.add_argument("--assignment-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=251_599)
    parser.add_argument("--expected-q-codes", default="0,2,5,7,9")
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        raise SystemExit("[FAIL] output or manifest already exists; derived input is fresh-only")

    with args.caption_tsv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"id", "caption", "q_level"} <= set(reader.fieldnames):
            raise SystemExit("[FAIL] caption TSV must contain id, caption, and q_level")
        fieldnames = list(reader.fieldnames)
        caption_rows = list(reader)

    with args.assignment_tsv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"id", "q_level"} <= set(reader.fieldnames):
            raise SystemExit("[FAIL] assignment TSV must contain id and q_level")
        assignment_rows = list(reader)

    if len(caption_rows) != args.expected_rows or len(assignment_rows) != args.expected_rows:
        raise SystemExit(
            f"[FAIL] row count mismatch: captions={len(caption_rows)} "
            f"assignments={len(assignment_rows)} expected={args.expected_rows}"
        )

    expected_codes = {int(item) for item in args.expected_q_codes.split(",")}
    q_values: list[int] = []
    for index, (caption, assignment) in enumerate(zip(caption_rows, assignment_rows)):
        if caption["id"] != assignment["id"]:
            raise SystemExit(f"[FAIL] ID/order mismatch at row {index}")
        for key in ("official_path", "track_id"):
            if key in caption and key in assignment and caption[key] != assignment[key]:
                raise SystemExit(f"[FAIL] {key} mismatch at row {index}")
        q_values.append(int(assignment["q_level"]))

    histogram = Counter(q_values)
    if set(histogram) != expected_codes:
        raise SystemExit(
            f"[FAIL] Q support mismatch: actual={sorted(histogram)} expected={sorted(expected_codes)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row, q_level in zip(caption_rows, q_values):
            output = dict(row)
            output["q_level"] = str(q_level)
            writer.writerow(output)
    os.replace(tmp, args.output)

    assignment_hash = hashlib.sha256(
        "\n".join(map(str, q_values)).encode("ascii")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "preserve caption TSV; replace q_level by row-aligned assignment TSV",
        "rows": len(caption_rows),
        "caption_tsv": str(args.caption_tsv),
        "caption_tsv_sha256": sha256(args.caption_tsv),
        "assignment_tsv": str(args.assignment_tsv),
        "assignment_tsv_sha256": sha256(args.assignment_tsv),
        "output_tsv": str(args.output),
        "output_tsv_sha256": sha256(args.output),
        "q_assignment_sha256": assignment_hash,
        "q_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "invariants": {
            "row_order_and_ids_unchanged": True,
            "captions_and_metadata_unchanged": True,
            "only_q_level_replaced": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = args.manifest.with_suffix(args.manifest.suffix + f".tmp.{os.getpid()}")
    manifest_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
