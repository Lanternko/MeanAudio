#!/usr/bin/env python3
"""Report which caption TSV a mutable NPZ text cache is currently bound to.

The caption10s family rebinds text features in place inside one shared NPZ
directory, so the directory alone does not say whose captions it holds.  This
samples rows on which the candidate TSVs actually disagree and compares the
embedded ``caption_sha256`` against each candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_captions(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row["id"] for row in rows], [row["caption"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="candidate caption TSV; repeat for each binding to test",
    )
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    candidates: dict[str, Path] = {}
    for item in args.candidate:
        name, _, raw = item.partition("=")
        if not name or not raw:
            raise SystemExit(f"--candidate must be NAME=PATH, got: {item}")
        candidates[name] = Path(raw)
    if len(candidates) < 2:
        raise SystemExit("need at least two candidates to discriminate a binding")

    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    loaded = {name: read_captions(path) for name, path in candidates.items()}
    lengths = {len(ids) for ids, _ in loaded.values()} | {len(names)}
    if len(lengths) != 1:
        raise SystemExit(f"row count mismatch across inputs: {lengths}")
    id_sets = {tuple(ids) for ids, _ in loaded.values()}
    if len(id_sets) != 1:
        raise SystemExit("candidate TSVs do not share an identical id order")

    captions = {name: caps for name, (_, caps) in loaded.items()}
    order = list(candidates)
    # Only rows where every candidate disagrees can discriminate a binding.
    discriminating = [
        index
        for index in range(len(names))
        if len({captions[name][index] for name in order}) == len(order)
    ]
    if not discriminating:
        raise SystemExit("candidates are identical on every row; nothing to discriminate")

    step = max(1, len(discriminating) // args.samples)
    probes = discriminating[::step][: args.samples]
    hits = {name: 0 for name in order}
    unknown = 0
    for index in probes:
        path = args.npz_dir / names[index]
        with np.load(path, allow_pickle=False) as data:
            if "caption_sha256" not in data.files:
                raise SystemExit(f"NPZ lacks caption provenance: {path}")
            stored = str(data["caption_sha256"].item())
        for name in order:
            digest = hashlib.sha256(captions[name][index].encode("utf-8")).hexdigest()
            if stored == digest:
                hits[name] += 1
                break
        else:
            unknown += 1

    bound = [name for name, count in hits.items() if count == len(probes)]
    binding = bound[0] if len(bound) == 1 else "unknown"
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "npz_dir": str(args.npz_dir),
        "cache_list": str(args.cache_list),
        "rows": len(names),
        "discriminating_rows": len(discriminating),
        "probes": len(probes),
        "matches": hits,
        "unmatched_probes": unknown,
        "binding": binding,
        "candidates": {name: str(path) for name, path in candidates.items()},
    }
    if args.report:
        atomic_json(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    # Exit non-zero on an indeterminate cache so callers fail closed.
    raise SystemExit(0 if binding != "unknown" else 4)


if __name__ == "__main__":
    main()
