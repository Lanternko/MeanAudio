#!/usr/bin/env python3
"""Build the deterministic Fixed-Q=9 training TSV for Phase-8 Fixed-Q / ATTM.

Preserves every original row, id, caption, field, and row order from the
catalog train TSV.  The only mutation is forcing q_level=9 (adding the column
if it is absent).  Output and manifest are written atomically.
"""

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
    parser = argparse.ArgumentParser(
        description="Create Fixed-Q=9 TSV from catalog train TSV (row-order preserving)."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=251_599)
    parser.add_argument("--fixed-q", type=int, default=9)
    args = parser.parse_args()

    if args.fixed_q != 9:
        raise SystemExit(f"[FAIL] this experiment requires fixed-q=9, got {args.fixed_q}")
    if args.output.exists() or args.manifest.exists():
        raise SystemExit(
            f"[FAIL] output/manifest already exists (fresh-only): "
            f"{args.output} / {args.manifest}"
        )

    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"[FAIL] empty/invalid TSV header: {args.input}")
        fieldnames = list(reader.fieldnames)
        if "id" not in fieldnames or "caption" not in fieldnames:
            raise SystemExit(
                f"[FAIL] input must have id+caption columns, got {fieldnames}"
            )
        rows = list(reader)

    if len(rows) != args.expected_rows:
        raise SystemExit(
            f"[FAIL] row count mismatch: actual={len(rows)} expected={args.expected_rows}"
        )

    had_q = "q_level" in fieldnames
    if not had_q:
        fieldnames = fieldnames + ["q_level"]

    original_q: list[int | None] = []
    for row in rows:
        raw = row.get("q_level")
        if raw is None or raw == "":
            original_q.append(None)
        else:
            original_q.append(int(raw))

    changed_rows = sum(1 for q in original_q if q != args.fixed_q)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with output_tmp.open("w", newline="") as dst:
        writer = csv.DictWriter(
            dst, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["q_level"] = str(args.fixed_q)
            # Preserve id/caption byte-for-field values.
            out["id"] = row["id"]
            out["caption"] = row["caption"]
            writer.writerow(out)
    os.replace(output_tmp, args.output)

    with args.output.open(newline="") as handle:
        out_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(out_rows) != args.expected_rows:
        raise SystemExit(f"[FAIL] rewritten TSV row count {len(out_rows)}")
    out_q = [int(row["q_level"]) for row in out_rows]
    support = sorted(set(out_q))
    if support != [args.fixed_q]:
        raise SystemExit(f"[FAIL] unique Q support must be [{args.fixed_q}], got {support}")
    for index, (src, dst) in enumerate(zip(rows, out_rows)):
        if src["id"] != dst["id"] or src["caption"] != dst["caption"]:
            raise SystemExit(f"[FAIL] id/caption drift at row {index}")

    assignment_hash = hashlib.sha256(
        "\n".join(str(q) for q in out_q).encode("ascii")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "force every q_level to fixed constant; preserve row order/id/caption",
        "fixed_q": args.fixed_q,
        "rows": len(out_rows),
        "changed_rows": changed_rows,
        "unchanged_rows": len(out_rows) - changed_rows,
        "input_had_q_level": had_q,
        "q_histogram_input": {
            str(k if k is not None else "missing"): v
            for k, v in sorted(Counter(original_q).items(), key=lambda x: (x[0] is None, x[0]))
        },
        "q_histogram_output": {
            str(k): v for k, v in sorted(Counter(out_q).items())
        },
        "unique_q_support": support,
        "input_tsv": str(args.input),
        "input_sha256": sha256(args.input),
        "output_tsv": str(args.output),
        "output_sha256": sha256(args.output),
        "q_assignment_sha256": assignment_hash,
        "invariants": {
            "row_order_unchanged": True,
            "id_and_caption_unchanged": True,
            "unique_q_support_exactly_fixed": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
