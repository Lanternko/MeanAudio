#!/usr/bin/env python3
"""Build a Phase7-like static random 3-caption TSV for Music Flamingo runs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_OUT_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data")


def load_tsv(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "id" not in reader.fieldnames or "caption" not in reader.fieldnames:
            raise SystemExit(f"[FAIL] expected id/caption columns: {path}")
        for row in reader:
            sid = row["id"]
            caption = (row.get("caption") or "").strip()
            if sid and caption:
                rows[sid] = caption
    return rows


def write_train_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "caption", "caption_source"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_clips_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "name", "start_sample", "end_sample"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "name": row["id"],
                    "start_sample": 0,
                    "end_sample": 160000,
                }
            )


def write_holdout(path: Path, phase4_test: Path, train_ids: set[str], n: int, seed: int) -> None:
    lpmc = load_tsv(phase4_test)
    candidates = [
        {"id": sid, "caption": caption}
        for sid, caption in lpmc.items()
        if sid not in train_ids
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    rows = candidates[:n]
    if len(rows) != n:
        raise SystemExit(f"[FAIL] only found {len(rows)} holdout rows, requested {n}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "caption"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-tsv", type=Path, required=True)
    parser.add_argument("--short-direct-tsv", type=Path, required=True)
    parser.add_argument("--short-aesthetic-tsv", type=Path, required=True)
    parser.add_argument("--phase4-test-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="music_flamingo_slice10_10k_static_random_3cap")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--holdout-n", type=int, default=2048)
    parser.add_argument("--holdout-seed", type=int, default=20260523)
    args = parser.parse_args()

    sources = {
        "original": load_tsv(args.original_tsv),
        "short_direct": load_tsv(args.short_direct_tsv),
        "short_aesthetic": load_tsv(args.short_aesthetic_tsv),
    }
    id_sets = {name: set(rows) for name, rows in sources.items()}
    common_ids = set.intersection(*id_sets.values())
    missing = {name: len(common_ids ^ ids) for name, ids in id_sets.items()}
    if any(v for v in missing.values()):
        print(f"[warn] source id sets differ; symmetric diffs vs common: {missing}")
    if not common_ids:
        raise SystemExit("[FAIL] no shared ids across the three caption sources")

    base_order = [sid for sid in sources["short_direct"] if sid in common_ids]
    rng = random.Random(args.seed)
    rows = []
    counts = {name: 0 for name in sources}
    for sid in base_order:
        source_name = rng.choice(tuple(sources))
        rows.append(
            {
                "id": sid,
                "caption": sources[source_name][sid],
                "caption_source": source_name,
            }
        )
        counts[source_name] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tsv = args.out_dir / f"{args.prefix}_train.tsv"
    clips_tsv = args.out_dir / f"{args.prefix}_clips.tsv"
    holdout_tsv = args.out_dir / f"{args.prefix}_jamendo_holdout2048.tsv"

    write_train_tsv(train_tsv, rows)
    write_clips_tsv(clips_tsv, rows)
    write_holdout(holdout_tsv, args.phase4_test_tsv, {row["id"] for row in rows}, args.holdout_n, args.holdout_seed)

    print(f"train_rows={len(rows)}")
    print(f"source_counts={counts}")
    print(f"train_tsv={train_tsv}")
    print(f"clips_tsv={clips_tsv}")
    print(f"holdout_tsv={holdout_tsv}")


if __name__ == "__main__":
    main()
