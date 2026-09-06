#!/usr/bin/env python3
"""Build one-rich-caption-per-track supervision on the full 251,599-row corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path


ID_RE = re.compile(r"^(?P<prefix>\d{2})_(?P<track>\d+)_segment_(?P<segment>\d+)_0$")
SELECTION_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def choose_position(track_id: str, count: int) -> int:
    token = f"{SELECTION_SEED}:{track_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich-tsv", type=Path, required=True)
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--out-mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with args.rich_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(rows) != 251599 or len(names) != 251599:
        raise SystemExit(f"expected 251599 rows/cache names; got {len(rows)}/{len(names)}")
    ids = [row.get("id", "") for row in rows]
    if any(not clip_id or not row.get("caption") for clip_id, row in zip(ids, rows)):
        raise SystemExit("blank id or caption")
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise SystemExit("duplicate id or cache name")

    by_track: dict[str, list[tuple[int, int]]] = defaultdict(list)
    track_for: list[str] = []
    segment_for: list[int] = []
    for index, clip_id in enumerate(ids):
        match = ID_RE.fullmatch(clip_id)
        if not match:
            raise SystemExit(f"unexpected clip id: {clip_id}")
        track_id = f"{match.group('prefix')}/{match.group('track')}"
        segment = int(match.group("segment"))
        track_for.append(track_id)
        segment_for.append(segment)
        by_track[track_id].append((segment, index))

    source_for_track: dict[str, int] = {}
    singleton_tracks = 0
    for track_id, entries in by_track.items():
        ordered = sorted(entries)
        if len({segment for segment, _ in ordered}) != len(ordered):
            raise SystemExit(f"duplicate segment index for {track_id}")
        if len(ordered) == 1:
            singleton_tracks += 1
        source_for_track[track_id] = ordered[choose_position(track_id, len(ordered))][1]

    output_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        source_index = source_for_track[track_for[index]]
        source = rows[source_index]
        output_rows.append(
            {"id": row["id"], "caption": source["caption"], "q_level": row.get("q_level", "")}
        )
        mapping_rows.append(
            {
                "id": row["id"],
                "track_id": track_for[index],
                "segment_index": str(segment_for[index]),
                "cache_name": names[index],
                "shared_caption_source_id": source["id"],
            }
        )

    source_ids = {rows[index]["id"] for index in source_for_track.values()}
    fixed_source_rows = sum(
        row["id"] == mapping["shared_caption_source_id"]
        for row, mapping in zip(rows, mapping_rows)
    )
    exact_text_unchanged = sum(
        before["caption"] == after["caption"] for before, after in zip(rows, output_rows)
    )
    observed = (len(by_track), singleton_tracks, len(source_ids), fixed_source_rows)
    expected = (36985, 2062, 36985, 36985)
    if observed != expected:
        raise SystemExit(f"corpus drift: observed={observed} expected={expected}")
    for track_id, entries in by_track.items():
        selected = {output_rows[index]["caption"] for _, index in entries}
        if len(selected) != 1:
            raise AssertionError(f"track does not share exactly one caption: {track_id}")

    out = io.StringIO(newline="")
    writer = csv.DictWriter(
        out, fieldnames=["id", "caption", "q_level"], delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(output_rows)
    atomic_text(args.out_tsv, out.getvalue())

    mapping = io.StringIO(newline="")
    mapping_writer = csv.DictWriter(
        mapping,
        fieldnames=["id", "track_id", "segment_index", "cache_name", "shared_caption_source_id"],
        delimiter="\t",
        lineterminator="\n",
    )
    mapping_writer.writeheader()
    mapping_writer.writerows(mapping_rows)
    atomic_text(args.out_mapping, mapping.getvalue())

    payload = {
        "schema_version": 1,
        "design": "one deterministic rich 10s caption source per track, broadcast to all retained rows",
        "selection": {
            "algorithm": "sha256(seed:track_id) first 64 bits modulo retained segment count",
            "seed": SELECTION_SEED,
            "avoids_first-segment_position_bias": True,
        },
        "inputs": {
            "rich_tsv": str(args.rich_tsv),
            "rich_tsv_sha256": sha256_file(args.rich_tsv),
            "cache_list": str(args.cache_list),
            "cache_list_sha256": sha256_file(args.cache_list),
        },
        "invariants": {
            "rows": len(rows),
            "tracks": len(by_track),
            "multi_segment_tracks": len(by_track) - singleton_tracks,
            "singleton_tracks": singleton_tracks,
            "unique_caption_source_ids": len(source_ids),
            "source_id_fixed_rows": fixed_source_rows,
            "exact_text_unchanged_rows": exact_text_unchanged,
            "one_shared_caption_per_track": True,
            "source_caption_always_from_same_track": True,
            "row_id_cache_order_preserved": True,
            "mean_rows_per_shared_caption_source": len(rows) / len(source_ids),
        },
        "outputs": {
            "tsv": str(args.out_tsv),
            "tsv_sha256": sha256_file(args.out_tsv),
            "mapping": str(args.out_mapping),
            "mapping_sha256": sha256_file(args.out_mapping),
        },
    }
    atomic_text(args.manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
