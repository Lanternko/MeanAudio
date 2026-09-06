#!/usr/bin/env python3
"""Build ID-aligned K=3 sources for true-random vs fixed-random training.

The true-random extraction TSV contains three rows per audio in slot order
0/1/3.  The true-random training TSV contains one row per audio and is only the
index/metadata side of a future stacked-caption NPZ cache.  The fixed-random
TSV chooses one slot per audio using a stable SHA-256 mapping, so rebuilding or
resuming cannot change an audio's assigned caption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SLOTS = ("slot0", "slot1", "slot3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            audio_id = str(record.get("id", ""))
            caption = record.get("caption")
            if not audio_id or not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"invalid record at {path}:{line_number}")
            if audio_id in result:
                raise ValueError(f"duplicate id {audio_id!r} in {path}")
            result[audio_id] = caption
    return result


def fixed_slot(seed: int, audio_id: str) -> str:
    payload = f"k3-fixed-random-v1\0{seed}\0{audio_id}".encode()
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(SLOTS)
    return SLOTS[index]


def write_tsv(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tsv", type=Path, required=True)
    parser.add_argument("--slot0", type=Path, required=True)
    parser.add_argument("--slot1", type=Path, required=True)
    parser.add_argument("--slot3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=14159265)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "slot0": load_jsonl(args.slot0),
        "slot1": load_jsonl(args.slot1),
        "slot3": load_jsonl(args.slot3),
    }
    with args.official_tsv.open(encoding="utf-8", newline="") as handle:
        official_rows = list(csv.DictReader(handle, delimiter="\t"))
    ids = [str(row["id"]) for row in official_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("official TSV contains duplicate ids")
    for slot, captions in sources.items():
        missing = [audio_id for audio_id in ids if audio_id not in captions]
        extra = set(captions).difference(ids)
        if missing or extra:
            raise ValueError(
                f"{slot} coverage mismatch: missing={len(missing)} extra={len(extra)}"
            )

    extraction_rows = []
    true_index_rows = []
    fixed_rows = []
    histogram: Counter[str] = Counter()
    for source_row in official_rows:
        audio_id = str(source_row["id"])
        for slot in SLOTS:
            extraction_rows.append(
                {"id": audio_id, "caption": sources[slot][audio_id], "source_slot": slot}
            )
        # Caption is informational only for the stacked-caption loader. The NPZ
        # caption sampler, not this field, determines true-random conditioning.
        true_index_rows.append(
            {"id": audio_id, "caption": sources["slot0"][audio_id]}
        )
        slot = fixed_slot(args.selection_seed, audio_id)
        histogram[slot] += 1
        fixed_rows.append(
            {"id": audio_id, "caption": sources[slot][audio_id], "source_slot": slot}
        )

    output_dir = args.output_dir
    true_extraction = output_dir / "k3_true_random_extraction.tsv"
    true_index = output_dir / "k3_true_random_train.tsv"
    fixed_train = output_dir / "k3_fake_random_fixed_train.tsv"
    write_tsv(true_extraction, ["id", "caption", "source_slot"], extraction_rows)
    write_tsv(true_index, ["id", "caption"], true_index_rows)
    write_tsv(fixed_train, ["id", "caption", "source_slot"], fixed_rows)

    manifest = {
        "schema_version": 1,
        "document_kind": "k3_true_fake_random_source_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_algorithm": "sha256(k3-fixed-random-v1\\0seed\\0audio_id)[:8] mod 3",
        "selection_seed": args.selection_seed,
        "slot_order": list(SLOTS),
        "audio_count": len(ids),
        "true_random_extraction_rows": len(extraction_rows),
        "fixed_random_slot_histogram": dict(sorted(histogram.items())),
        "inputs": {
            "official_tsv": {"path": str(args.official_tsv), "sha256": sha256_file(args.official_tsv)},
            "slot0": {"path": str(args.slot0), "sha256": sha256_file(args.slot0)},
            "slot1": {"path": str(args.slot1), "sha256": sha256_file(args.slot1)},
            "slot3": {"path": str(args.slot3), "sha256": sha256_file(args.slot3)},
        },
        "outputs": {
            "true_random_extraction": {"path": str(true_extraction), "sha256": sha256_file(true_extraction)},
            "true_random_train_index": {"path": str(true_index), "sha256": sha256_file(true_index)},
            "fake_random_fixed_train": {"path": str(fixed_train), "sha256": sha256_file(fixed_train)},
        },
    }
    manifest_path = output_dir / "k3_true_fake_random_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
