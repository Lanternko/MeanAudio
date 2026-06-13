#!/usr/bin/env python3
"""Prepare Music Flamingo slice10 captions for MeanAudio training."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_CAPTIONS = Path("/home/kojiek/eval_output/music_flamingo_slice10_10k/caption.jsonl")
DEFAULT_PHASE4_TEST = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv")
DEFAULT_OUT_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data")


def load_music_flamingo(path: Path, n: int) -> list[dict]:
    rows = []
    seen = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            segment_id = rec["id"]
            caption = ((rec.get("output") or {}).get("text") or rec.get("raw_text") or "").strip()
            if not caption or segment_id in seen:
                continue
            rows.append({"id": segment_id, "caption": caption})
            seen.add(segment_id)
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise SystemExit(f"Only found {len(rows)} usable captions, requested {n}")
    return rows


def load_lpmc(path: Path) -> dict[str, str]:
    out = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[row["id"]] = row["caption"]
    return out


def write_train_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "caption"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_clips_tsv(path: Path, rows: list[dict]) -> None:
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
    lpmc = load_lpmc(phase4_test)
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
    parser.add_argument("--captions-jsonl", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--phase4-test-tsv", type=Path, default=DEFAULT_PHASE4_TEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--holdout-n", type=int, default=2048)
    parser.add_argument("--holdout-seed", type=int, default=20260523)
    parser.add_argument("--prefix", default="music_flamingo_slice10_10k")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_music_flamingo(args.captions_jsonl, args.n)
    train_ids = {row["id"] for row in rows}

    train_tsv = args.out_dir / f"{args.prefix}_train.tsv"
    clips_tsv = args.out_dir / f"{args.prefix}_clips.tsv"
    holdout_tsv = args.out_dir / f"{args.prefix}_jamendo_holdout2048.tsv"

    write_train_tsv(train_tsv, rows)
    write_clips_tsv(clips_tsv, rows)
    write_holdout(holdout_tsv, args.phase4_test_tsv, train_ids, args.holdout_n, args.holdout_seed)

    print(f"train_rows={len(rows)}")
    print(f"train_tsv={train_tsv}")
    print(f"clips_tsv={clips_tsv}")
    print(f"holdout_tsv={holdout_tsv}")


if __name__ == "__main__":
    main()
