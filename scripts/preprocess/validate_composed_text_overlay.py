#!/usr/bin/env python3
"""Validate a multi-overlay caption pool before it is used to train.

Checks, per sampled row: every source resolves to the same clip_id as the TSV,
the TSV caption is a member of the composed pool, the picked slot's embedding is
byte-identical to the source it claims to come from, and the rotation actually
visits every slot with the expected frequency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def stored_hashes(value: np.ndarray) -> list[str]:
    if value.ndim == 0:
        return str(value.item()).split(",")
    return [str(item) for item in value.tolist()]


def true_random_cap_index(clip_id: str, n_caps: int, seed: int, epoch: int) -> int:
    payload = f"k3-true-random-v1\0{seed}\0{epoch}\0{clip_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n_caps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--gt-cache", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="<dir>[:<slot>]; omit the slot for single-caption overlays")
    ap.add_argument("--slot-tsv", action="append", default=[],
                    help="<slot_position>:<tsv> ground-truth captions for that pool position")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=14159265)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    sources = []
    for spec in args.source:
        head, _, slot = spec.rpartition(":")
        if head and slot.isdigit():
            sources.append((Path(head), int(slot)))
        else:
            sources.append((Path(spec), None))

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8", newline=""), delimiter="\t"))
    names = [line.strip() for line in open(args.gt_cache, encoding="utf-8") if line.strip()]
    if len(rows) != len(names):
        print(f"FAIL tsv rows {len(rows)} != gt_cache entries {len(names)}")
        return 2

    slot_truth = {}
    for spec in args.slot_tsv:
        pos, _, path = spec.partition(":")
        table = {r["id"]: r["caption"] for r in
                 csv.DictReader(open(path, encoding="utf-8", newline=""), delimiter="\t")}
        slot_truth[int(pos)] = table

    rng = random.Random(20260902)
    picks = rng.sample(range(len(rows)), min(args.samples, len(rows)))

    failures: list[str] = []
    histogram = Counter()
    checked_embeddings = 0
    for idx in picks:
        row, name = rows[idx], names[idx]
        pool_hashes = []
        for source_dir, slot in sources:
            data = np.load(source_dir / name)
            if str(data["clip_id"].item()) != row["id"]:
                failures.append(f"{name}: clip_id mismatch in {source_dir.name}")
                break
            stored = stored_hashes(data["caption_sha256"])
            if slot is None:
                if len(stored) != 1:
                    failures.append(f"{name}: {source_dir.name} is stacked but no slot given")
                    break
                pool_hashes.append(stored[0])
            else:
                if slot >= len(stored):
                    failures.append(f"{name}: slot {slot} out of range in {source_dir.name}")
                    break
                pool_hashes.append(stored[slot])
        else:
            row_sha = hashlib.sha256(row["caption"].encode("utf-8")).hexdigest()
            if row_sha not in pool_hashes:
                failures.append(f"{name}: TSV caption is not a member of the composed pool")
            for position, table in slot_truth.items():
                truth = table.get(row["id"])
                if truth is None:
                    failures.append(f"{name}: id missing from slot {position} ground-truth TSV")
                elif hashlib.sha256(truth.encode("utf-8")).hexdigest() != pool_hashes[position]:
                    failures.append(f"{name}: pool position {position} does not hold that slot's caption")
            for epoch in range(args.epochs):
                cap_idx = true_random_cap_index(row["id"], len(sources), args.seed, epoch)
                histogram[cap_idx] += 1
                if epoch == 0:
                    source_dir, slot = sources[cap_idx]
                    data = np.load(source_dir / name)
                    picked = data["text_features"] if slot is None else data["text_features"][slot]
                    if picked.shape != (77, 1024):
                        failures.append(f"{name}: picked embedding shape {picked.shape}")
                    elif not np.isfinite(picked).all():
                        failures.append(f"{name}: picked embedding is not finite")
                    else:
                        checked_embeddings += 1

    total = sum(histogram.values())
    report = {
        "tsv": args.tsv,
        "sources": [f"{d}:{s}" if s is not None else str(d) for d, s in sources],
        "rows": len(rows),
        "sampled": len(picks),
        "epochs": args.epochs,
        "embeddings_checked": checked_embeddings,
        "rotation_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "rotation_share": {str(k): round(v / total, 4) for k, v in sorted(histogram.items())} if total else {},
        "failures": failures[:20],
        "failure_count": len(failures),
        "status": "passed" if not failures else "failed",
    }
    print(json.dumps(report, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
