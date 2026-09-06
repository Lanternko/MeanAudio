#!/usr/bin/env python3
"""Create the deterministic shuffled-Q control TSV for Phase-8 S2-only Q.

Only the q_level column is permuted.  Row order, ids, captions, cache mapping,
and every other TSV field remain byte-for-field values from the input rows.
The output and its manifest are written atomically and are immutable inputs to
the scheduled experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--expected-rows", type=int, default=251599)
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "q_level" not in reader.fieldnames:
            raise SystemExit(f"input TSV lacks q_level: {args.input}")
        fieldnames = list(reader.fieldnames)
        original_q = [int(row["q_level"]) for row in reader]

    if len(original_q) != args.expected_rows:
        raise SystemExit(
            f"row count mismatch: actual={len(original_q)} expected={args.expected_rows}"
        )
    invalid = sorted({q for q in original_q if not 0 <= q <= 9})
    if invalid:
        raise SystemExit(f"q_level values outside 0..9: {invalid}")

    shuffled_q = original_q.copy()
    random.Random(args.seed).shuffle(shuffled_q)
    changed_rows = sum(a != b for a, b in zip(original_q, shuffled_q))
    if Counter(original_q) != Counter(shuffled_q):
        raise SystemExit("internal error: shuffled histogram changed")
    if changed_rows < len(original_q) // 2:
        raise SystemExit(f"shuffle changed too few rows: {changed_rows}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open(newline="") as src, output_tmp.open("w", newline="") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(
            dst, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for index, row in enumerate(reader):
            row["q_level"] = str(shuffled_q[index])
            writer.writerow(row)
    os.replace(output_tmp, args.output)

    assignment_hash = hashlib.sha256(
        "\n".join(map(str, shuffled_q)).encode("ascii")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "fixed-seed permutation of q_level only",
        "seed": args.seed,
        "rows": len(original_q),
        "changed_rows": changed_rows,
        "unchanged_rows": len(original_q) - changed_rows,
        "q_histogram": {str(k): v for k, v in sorted(Counter(original_q).items())},
        "input_tsv": str(args.input),
        "input_sha256": sha256(args.input),
        "output_tsv": str(args.output),
        "output_sha256": sha256(args.output),
        "q_assignment_sha256": assignment_hash,
        "invariants": {
            "row_order_unchanged": True,
            "id_caption_and_other_fields_unchanged": True,
            "q_histogram_unchanged": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
