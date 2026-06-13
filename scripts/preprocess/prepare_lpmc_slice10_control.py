#!/usr/bin/env python3
"""Prepare an LP-MC slice10 control using the same ids as a Music Flamingo run."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_IDS_JSONL = Path("/home/kojiek/eval_output/music_flamingo_slice10_10k/caption.jsonl")
DEFAULT_LPMC_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv")
DEFAULT_OUT_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data")


def load_ids_from_flamingo(path: Path, n: int) -> list[str]:
    ids = []
    seen = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            segment_id = str(rec["id"])
            if segment_id in seen:
                continue
            ids.append(segment_id)
            seen.add(segment_id)
            if len(ids) >= n:
                break
    if len(ids) < n:
        raise SystemExit(f"Only found {len(ids)} usable ids, requested {n}: {path}")
    return ids


def load_lpmc(path: Path) -> dict[str, str]:
    out = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[row["id"]] = row["caption"]
    return out


def write_train_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "caption"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_clips_tsv(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "name", "start_sample", "end_sample"],
            delimiter="\t",
        )
        writer.writeheader()
        for segment_id in ids:
            writer.writerow(
                {
                    "id": segment_id,
                    "name": segment_id,
                    "start_sample": 0,
                    "end_sample": 160000,
                }
            )


def write_holdout(path: Path, lpmc: dict[str, str], train_ids: set[str], n: int, seed: int) -> None:
    candidates = [
        {"id": segment_id, "caption": caption}
        for segment_id, caption in lpmc.items()
        if segment_id not in train_ids
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    rows = candidates[:n]
    if len(rows) < n:
        raise SystemExit(f"Only found {len(rows)} holdout rows, requested {n}")
    write_train_tsv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-jsonl", type=Path, default=DEFAULT_IDS_JSONL)
    parser.add_argument("--lpmc-tsv", type=Path, default=DEFAULT_LPMC_TSV)
    parser.add_argument("--holdout-tsv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--holdout-n", type=int, default=2048)
    parser.add_argument("--holdout-seed", type=int, default=20260523)
    parser.add_argument("--prefix", default="lpmc_slice10_10k_control")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = load_ids_from_flamingo(args.ids_jsonl, args.n)
    lpmc = load_lpmc(args.lpmc_tsv)
    holdout_lpmc = load_lpmc(args.holdout_tsv or args.lpmc_tsv)
    missing = [segment_id for segment_id in ids if segment_id not in lpmc]
    if missing:
        raise SystemExit(f"Missing {len(missing)} LP-MC captions; first missing id: {missing[0]}")

    rows = [{"id": segment_id, "caption": lpmc[segment_id]} for segment_id in ids]
    train_ids = set(ids)

    train_tsv = args.out_dir / f"{args.prefix}_train.tsv"
    clips_tsv = args.out_dir / f"{args.prefix}_clips.tsv"
    holdout_tsv = args.out_dir / f"{args.prefix}_jamendo_holdout2048.tsv"

    write_train_tsv(train_tsv, rows)
    write_clips_tsv(clips_tsv, ids)
    write_holdout(holdout_tsv, holdout_lpmc, train_ids, args.holdout_n, args.holdout_seed)

    print(f"train_rows={len(rows)}")
    print(f"train_tsv={train_tsv}")
    print(f"clips_tsv={clips_tsv}")
    print(f"holdout_tsv={holdout_tsv}")


if __name__ == "__main__":
    main()
