#!/usr/bin/env python3
"""Build or verify the aligned, balanced two-bin MeanSimilarity-Q dataset.

The lower half of actual-clip MeanSimilarity ranks is mapped to q=0 and the
upper half to q=9.  Ranking by ``(mean_similarity, source_id)`` makes ties
deterministic while keeping the two bins balanced.  The input must already be
the fully aligned historical MeanSimilarity-Q TSV, and every historical label
is rechecked from the original five-caption JSONL before the binary label is
written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(relative_path: str) -> str:
    return relative_path.removesuffix(".mp3").replace("/", "_")


def historical_q(mean_similarity: float) -> int:
    if not math.isfinite(mean_similarity):
        raise ValueError(f"non-finite mean_similarity: {mean_similarity}")
    return min(9, max(0, math.floor(mean_similarity * 10)))


def resolve(tsv_id: str, source: dict[str, dict[str, Any]]) -> tuple[str, str]:
    exact = tsv_id if tsv_id in source else None
    stripped = (
        tsv_id[:-2]
        if tsv_id.endswith("_0") and tsv_id[:-2] in source
        else None
    )
    if exact is not None and stripped is not None:
        raise ValueError(
            f"ambiguous source id: both {exact!r} and {stripped!r} exist"
        )
    if exact is not None:
        return exact, "exact"
    if stripped is not None:
        return stripped, "stripped_final_partition_suffix"
    raise KeyError(tsv_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=251_599)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("[FAIL] refusing to overwrite the aligned input TSV")
    if args.output.exists() != args.manifest.exists():
        raise SystemExit("[FAIL] output and manifest must either both exist or both be absent")

    aligned_contract = json.loads(args.aligned_manifest.read_text(encoding="utf-8"))
    input_hash = sha256(args.input)
    source_hash = sha256(args.source_jsonl)
    if aligned_contract.get("output_sha256") != input_hash:
        raise SystemExit("[FAIL] aligned input hash disagrees with its manifest")
    if aligned_contract.get("source_jsonl_sha256") != source_hash:
        raise SystemExit("[FAIL] source JSONL hash disagrees with aligned manifest")
    if aligned_contract.get("rows") != args.expected_rows:
        raise SystemExit("[FAIL] aligned manifest row count is not expected")
    if aligned_contract.get("formula") != (
        "clamp(floor(mean_similarity * 10), 0, 9)"
    ):
        raise SystemExit("[FAIL] aligned manifest has an unexpected Q formula")

    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"id", "caption", "q_level"} <= set(
            reader.fieldnames
        ):
            raise SystemExit("[FAIL] input must contain id, caption, and q_level")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if len(rows) != args.expected_rows:
        raise SystemExit(
            f"[FAIL] input rows={len(rows)}, expected={args.expected_rows}"
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise SystemExit(f"[FAIL] duplicate input ids; first={duplicates[:10]}")

    needed = set(ids)
    source: dict[str, dict[str, Any]] = {}
    with args.source_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            key = source_id(str(item["relative_path"]))
            if key not in needed and f"{key}_0" not in needed:
                continue
            if key in source:
                raise SystemExit(
                    f"[FAIL] duplicate source id at JSONL row {line_number}: {key}"
                )
            analysis = item.get("credibility_analysis") or {}
            value = analysis.get("mean_similarity")
            if value is None:
                raise SystemExit(
                    f"[FAIL] source row {line_number} lacks mean_similarity: {key}"
                )
            caption_count = len(item.get("caption_details") or [])
            successful = item.get("successful_captions")
            if caption_count != 5 or successful != 5:
                raise SystemExit(
                    f"[FAIL] incomplete five-caption source at row "
                    f"{line_number}: {key} captions={caption_count} "
                    f"successful={successful}"
                )
            source[key] = {
                "mean_similarity": float(value),
                "relative_path": str(item["relative_path"]),
            }

    resolved: list[tuple[str, float]] = []
    resolution_hist: Counter[str] = Counter()
    used_source_ids: set[str] = set()
    for index, row in enumerate(rows):
        try:
            key, resolution = resolve(row["id"], source)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"[FAIL] row {index} id resolution: {exc}") from exc
        if key in used_source_ids:
            raise SystemExit(f"[FAIL] source id reused by multiple rows: {key}")
        used_source_ids.add(key)
        resolution_hist[resolution] += 1
        mean_similarity = source[key]["mean_similarity"]
        expected_historical_q = historical_q(mean_similarity)
        if int(row["q_level"]) != expected_historical_q:
            raise SystemExit(
                f"[FAIL] aligned historical Q mismatch at row {index}: "
                f"id={row['id']} stored={row['q_level']} "
                f"expected={expected_historical_q}"
            )
        resolved.append((key, mean_similarity))

    low_count = len(rows) // 2
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (resolved[index][1], resolved[index][0]),
    )
    half_q = [9] * len(rows)
    for index in ranked[:low_count]:
        half_q[index] = 0
    histogram = Counter(half_q)
    expected_histogram = {0: low_count, 9: len(rows) - low_count}
    if dict(sorted(histogram.items())) != expected_histogram:
        raise SystemExit(f"[FAIL] internal half-Q histogram error: {histogram}")

    assignment_hash = hashlib.sha256(
        "\n".join(map(str, half_q)).encode("ascii")
    ).hexdigest()
    immutable_payload = {
        "schema_version": 1,
        "method": (
            "balanced rank split of actual-clip mean_similarity; "
            "rank key=(mean_similarity, source_id)"
        ),
        "rows": len(rows),
        "low_q": 0,
        "high_q": 9,
        "low_rows": low_count,
        "high_rows": len(rows) - low_count,
        "q_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "lower_max_mean_similarity": resolved[ranked[low_count - 1]][1],
        "upper_min_mean_similarity": resolved[ranked[low_count]][1],
        "tie_break": "source_id ascending",
        "input_tsv": str(args.input),
        "input_sha256": input_hash,
        "aligned_manifest": str(args.aligned_manifest),
        "aligned_manifest_sha256": sha256(args.aligned_manifest),
        "source_jsonl": str(args.source_jsonl),
        "source_jsonl_sha256": source_hash,
        "source_signal": "credibility_analysis.mean_similarity",
        "source_caption_count": 5,
        "historical_q_formula_rechecked": (
            "clamp(floor(mean_similarity * 10), 0, 9)"
        ),
        "historical_q_rows_verified": len(rows),
        "resolution_histogram": dict(sorted(resolution_hist.items())),
        "unique_source_rows": len(used_source_ids),
        "q_assignment_sha256": assignment_hash,
        "invariants": {
            "row_order_unchanged": True,
            "id_caption_and_other_fields_unchanged": True,
            "balanced_bins": True,
            "actual_clip_alignment_rechecked": True,
        },
    }

    if args.output.exists():
        with args.output.open(encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(existing_rows) != len(rows):
            raise SystemExit("[FAIL] existing output row count changed")
        for index, (source_row, output_row) in enumerate(zip(rows, existing_rows)):
            expected = dict(source_row)
            expected["q_level"] = str(half_q[index])
            if output_row != expected:
                raise SystemExit(
                    f"[FAIL] existing half-Q output mismatch at row {index}"
                )
        existing_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        for key, value in immutable_payload.items():
            if existing_manifest.get(key) != value:
                raise SystemExit(f"[FAIL] existing half-Q manifest drift: {key}")
        if existing_manifest.get("output_sha256") != sha256(args.output):
            raise SystemExit("[FAIL] existing output hash disagrees with manifest")
        print(
            json.dumps(
                {
                    "status": "verified",
                    "rows": len(rows),
                    "q_histogram": immutable_payload["q_histogram"],
                    "output_sha256": existing_manifest["output_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with output_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            output_row = dict(row)
            output_row["q_level"] = str(half_q[index])
            writer.writerow(output_row)
    os.replace(output_tmp, args.output)

    payload = {
        **immutable_payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_tsv": str(args.output),
        "output_sha256": sha256(args.output),
    }
    manifest_tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, args.manifest)
    print(json.dumps({"status": "created", **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
