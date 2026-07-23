#!/usr/bin/env python3
"""Audit and repair MeanSimilarity-Q labels against the actual catalog clip.

The historical Phase-8 cache is indexed through a cache list whose row order is
not the Phase-7 TSV row order.  Captions/audio can therefore be catalog-aligned
while a copied row-position ``q_level`` still belongs to another clip.  This
tool resolves every TSV id against the original five-caption JSONL, recomputes
the historical MeanSimilarity-Q rule, and writes a provenance manifest.

Historical rule (verified exhaustively on the recoverable Phase-7 rows):

    q_level = clamp(floor(mean_similarity * 10), 0, 9)
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


def clip_id_from_relative_path(relative_path: str) -> str:
    return relative_path.removesuffix(".mp3").replace("/", "_")


def meansim_q(mean_similarity: float) -> int:
    if not math.isfinite(mean_similarity):
        raise ValueError(f"non-finite mean_similarity: {mean_similarity}")
    return min(9, max(0, math.floor(mean_similarity * 10)))


def resolve_source_id(tsv_id: str, source: dict[str, dict[str, Any]]) -> str:
    """Resolve one id and fail closed if ``_0`` normalization is ambiguous."""
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
        return exact
    if stripped is not None:
        return stripped
    raise KeyError(tsv_id)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Catalog-aligned TSV")
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        required=True,
        help="Original five-caption JSONL containing credibility_analysis.mean_similarity",
    )
    parser.add_argument("--output", type=Path, help="Write repaired TSV here")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--examples",
        type=int,
        default=30,
        help="Maximum changed rows retained for manual caption/Q inspection",
    )
    parser.add_argument(
        "--require-current-match",
        action="store_true",
        help="Audit-only guard: fail unless every current q_level is already correct",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None and args.output.resolve() == args.input.resolve():
        raise SystemExit("[FAIL] refusing to overwrite the input TSV")
    if args.output is not None and args.output.exists():
        raise SystemExit(f"[FAIL] output already exists: {args.output}")
    if args.manifest.exists():
        raise SystemExit(f"[FAIL] manifest already exists: {args.manifest}")

    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"id", "caption", "q_level"} <= set(reader.fieldnames):
            raise SystemExit("[FAIL] input TSV must contain id, caption, and q_level")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    input_ids = [row["id"] for row in rows]
    if len(input_ids) != len(set(input_ids)):
        duplicates = [
            clip_id for clip_id, count in Counter(input_ids).items() if count > 1
        ]
        raise SystemExit(
            f"[FAIL] input TSV contains duplicate ids; first={duplicates[:10]}"
        )

    needed = {row["id"] for row in rows}
    source: dict[str, dict[str, Any]] = {}
    with args.source_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            source_id = clip_id_from_relative_path(str(item["relative_path"]))
            if source_id not in needed and f"{source_id}_0" not in needed:
                continue
            if source_id in source:
                raise SystemExit(
                    f"[FAIL] source JSONL contains duplicate id at row "
                    f"{line_number}: {source_id}"
                )
            analysis = item.get("credibility_analysis") or {}
            if analysis.get("mean_similarity") is None:
                raise SystemExit(
                    f"[FAIL] source row {line_number} lacks mean_similarity: {source_id}"
                )
            captions = [
                str(cap.get("caption", "")).replace("\n", " ").replace("\r", " ").strip()
                for cap in item.get("caption_details", [])
            ]
            source[source_id] = {
                "mean_similarity": float(analysis["mean_similarity"]),
                "candidate_captions": captions,
            }

    missing: list[str] = []
    changed_examples: list[dict[str, Any]] = []
    current_hist: Counter[int] = Counter()
    corrected_hist: Counter[int] = Counter()
    transition_hist: Counter[tuple[int, int]] = Counter()
    abs_differences: list[int] = []
    corrected_rows: list[dict[str, str]] = []
    resolved_source_ids: set[str] = set()

    for index, row in enumerate(rows):
        try:
            source_id = resolve_source_id(row["id"], source)
        except KeyError:
            missing.append(row["id"])
            continue
        except ValueError as exc:
            raise SystemExit(f"[FAIL] row {index}: {exc}") from exc
        if source_id in resolved_source_ids:
            raise SystemExit(
                f"[FAIL] multiple TSV rows resolve to source id {source_id!r}"
            )
        resolved_source_ids.add(source_id)
        mean_similarity = source[source_id]["mean_similarity"]
        corrected_q = meansim_q(mean_similarity)
        try:
            current_q = int(row["q_level"])
        except ValueError as exc:
            raise SystemExit(
                f"[FAIL] row {index} has non-integer q_level={row['q_level']!r}"
            ) from exc
        if not 0 <= current_q <= 9:
            raise SystemExit(f"[FAIL] row {index} q_level outside 0..9: {current_q}")

        current_hist[current_q] += 1
        corrected_hist[corrected_q] += 1
        transition_hist[(current_q, corrected_q)] += 1
        abs_differences.append(abs(current_q - corrected_q))

        repaired = dict(row)
        repaired["q_level"] = str(corrected_q)
        corrected_rows.append(repaired)
        if current_q != corrected_q and len(changed_examples) < args.examples:
            changed_examples.append(
                {
                    "row": index,
                    "tsv_id": row["id"],
                    "source_id": source_id,
                    "current_q": current_q,
                    "corrected_q": corrected_q,
                    "mean_similarity": mean_similarity,
                    "training_caption": row["caption"],
                    "candidate_captions": source[source_id]["candidate_captions"],
                }
            )

    if missing:
        raise SystemExit(
            f"[FAIL] {len(missing)} TSV ids have no MeanSimilarity source; "
            f"first={missing[:10]}"
        )
    if len(corrected_rows) != len(rows):
        raise SystemExit("[FAIL] internal row-count mismatch")

    changed = sum(count for (old, new), count in transition_hist.items() if old != new)
    exact = len(rows) - changed
    diff_ge_2 = sum(value >= 2 for value in abs_differences)

    if args.require_current_match and changed:
        raise SystemExit(
            f"[FAIL] current q_level is not catalog-aligned: changed={changed}/{len(rows)}"
        )

    output_hash = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(corrected_rows)
        os.replace(tmp, args.output)
        output_hash = sha256(args.output)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "source_jsonl": str(args.source_jsonl),
        "source_jsonl_sha256": sha256(args.source_jsonl),
        "output": str(args.output) if args.output is not None else None,
        "output_sha256": output_hash,
        "rows": len(rows),
        "matched_source_rows": len(corrected_rows),
        "formula": "clamp(floor(mean_similarity * 10), 0, 9)",
        "signal": "credibility_analysis.mean_similarity from the actual catalog clip id",
        "exact_current_rows": exact,
        "changed_rows": changed,
        "current_match_rate": exact / len(rows) if rows else 1.0,
        "mean_absolute_q_difference": (
            sum(abs_differences) / len(abs_differences) if abs_differences else 0.0
        ),
        "difference_ge_2_rows": diff_ge_2,
        "difference_ge_2_rate": diff_ge_2 / len(rows) if rows else 0.0,
        "current_histogram": {str(k): v for k, v in sorted(current_hist.items())},
        "corrected_histogram": {str(k): v for k, v in sorted(corrected_hist.items())},
        "transition_histogram": {
            f"{old}->{new}": count
            for (old, new), count in sorted(transition_hist.items())
            if count
        },
        "manual_review_examples": changed_examples,
    }
    atomic_write_text(args.manifest, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "exact_current_rows": exact,
                "changed_rows": changed,
                "current_match_rate": payload["current_match_rate"],
                "mean_absolute_q_difference": payload["mean_absolute_q_difference"],
                "difference_ge_2_rate": payload["difference_ge_2_rate"],
                "corrected_histogram": payload["corrected_histogram"],
                "output": payload["output"],
                "manifest": str(args.manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
