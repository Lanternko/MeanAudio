#!/usr/bin/env python3
"""Build exhaustively aligned Qwen-caption Q-bucket TSVs and a pilot prompt set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


KS = (2, 3, 5, 10)
STRATEGIES = ("balanced", "fixed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"[FAIL] TSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def tsv_text(fields: list[str], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_or_verify(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    rewrite_derived: bool = False,
) -> str:
    text = tsv_text(fields, rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            if not rewrite_derived:
                raise SystemExit(f"[FAIL] existing output drift: {path}")
            atomic_text(path, text)
            return "rewritten"
        return "verified"
    atomic_text(path, text)
    return "created"


def source_id(relative_path: str) -> str:
    return relative_path.removesuffix(".mp3").replace("/", "_")


def resolve(tsv_id: str, source: dict[str, float]) -> str:
    exact = tsv_id if tsv_id in source else None
    stripped = tsv_id[:-2] if tsv_id.endswith("_0") and tsv_id[:-2] in source else None
    if exact and stripped:
        raise ValueError(f"ambiguous id {tsv_id!r}")
    if exact or stripped:
        return exact or stripped  # type: ignore[return-value]
    raise KeyError(tsv_id)


def q_codes(k: int) -> list[int]:
    # Keep q0/q9 as comparable low/high endpoints. Intermediate integers are
    # merely categorical embedding labels; ROUND_HALF_UP makes the map explicit.
    return [
        int(
            (Decimal(index * 9) / Decimal(k - 1)).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
        for index in range(k)
    ]


def assignments(
    scores: list[float], source_ids: list[str], k: int, strategy: str
) -> list[int]:
    if strategy == "fixed":
        return [min(k - 1, max(0, math.floor(score * k))) for score in scores]
    ranked = sorted(range(len(scores)), key=lambda i: (scores[i], source_ids[i]))
    base, remainder = divmod(len(scores), k)
    result = [-1] * len(scores)
    cursor = 0
    # Put one extra row in each of the highest ``remainder`` bins. This keeps
    # every bucket within one row and exactly preserves the already-trained
    # K=2 Half-Q definition (125,799 low / 125,800 high).
    for bucket in range(k):
        size = base + (bucket >= k - remainder)
        for row_index in ranked[cursor : cursor + size]:
            result[row_index] = bucket
        cursor += size
    if cursor != len(scores) or min(result) < 0:
        raise RuntimeError("balanced assignment failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-tsv", type=Path, required=True)
    parser.add_argument("--qwen-combined-manifest", type=Path, required=True)
    parser.add_argument("--qwen-npz-manifest", type=Path, required=True)
    parser.add_argument("--qwen-cache-audit", type=Path, required=True)
    parser.add_argument("--qwen-cache-list", type=Path, required=True)
    parser.add_argument("--official-json", type=Path, required=True)
    parser.add_argument("--aligned-tsv", type=Path, required=True)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--existing-k2-balanced", type=Path, required=True)
    parser.add_argument("--existing-k10-fixed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--musiccaps", type=Path, required=True)
    parser.add_argument("--pilot-prompts", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=251599)
    parser.add_argument("--pilot-size", type=int, default=512)
    parser.add_argument("--pilot-seed", type=int, default=14159265)
    parser.add_argument(
        "--rewrite-derived",
        action="store_true",
        help="Atomically replace only output-dir grid TSVs and the grid manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = (
        args.qwen_tsv,
        args.qwen_combined_manifest,
        args.qwen_npz_manifest,
        args.qwen_cache_audit,
        args.qwen_cache_list,
        args.official_json,
        args.aligned_tsv,
        args.aligned_manifest,
        args.source_jsonl,
        args.existing_k2_balanced,
        args.existing_k10_fixed,
        args.musiccaps,
    )
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing input: {path}")

    qwen_fields, qwen_rows = read_tsv(args.qwen_tsv)
    aligned_fields, aligned_rows = read_tsv(args.aligned_tsv)
    if len(qwen_rows) != args.expected_rows or len(aligned_rows) != args.expected_rows:
        raise SystemExit("[FAIL] input cardinality mismatch")
    if not {"id", "caption", "q_level"} <= set(qwen_fields):
        raise SystemExit("[FAIL] Qwen TSV lacks id/caption/q_level")
    aligned_manifest = json.loads(args.aligned_manifest.read_text(encoding="utf-8"))
    if (
        aligned_manifest.get("rows") != args.expected_rows
        or aligned_manifest.get("matched_source_rows") != args.expected_rows
        or aligned_manifest.get("output_sha256") != sha256(args.aligned_tsv)
    ):
        raise SystemExit("[FAIL] aligned manifest is not bound to the aligned TSV")
    combined = json.loads(args.qwen_combined_manifest.read_text(encoding="utf-8"))
    npz_manifest = json.loads(args.qwen_npz_manifest.read_text(encoding="utf-8"))
    cache_audit = json.loads(args.qwen_cache_audit.read_text(encoding="utf-8"))
    qwen_hash = sha256(args.qwen_tsv)
    cache_hash = sha256(args.qwen_cache_list)
    official_hash = sha256(args.official_json)
    if (
        combined.get("status") != "passed"
        or combined.get("rows") != args.expected_rows
        or combined.get("inputs", {}).get(str(args.qwen_tsv)) != qwen_hash
        or combined.get("inputs", {}).get(str(args.official_json)) != official_hash
        or combined.get("outputs", {}).get("halfq_tsv_sha256")
        != sha256(args.existing_k2_balanced)
        or combined.get("outputs", {}).get("fullq_tsv_sha256")
        != sha256(args.existing_k10_fixed)
        or npz_manifest.get("status") != "passed"
        or npz_manifest.get("completed_rows") != args.expected_rows
        or npz_manifest.get("tsv_sha256") != qwen_hash
        or npz_manifest.get("cache_list_sha256") != cache_hash
        or cache_audit.get("status") != "passed"
        or cache_audit.get("rows") != args.expected_rows
        or cache_audit.get("tsv_sha256") != qwen_hash
        or cache_audit.get("cache_list_sha256") != cache_hash
        or cache_audit.get("semantic_gate", {}).get("status") != "passed"
    ):
        raise SystemExit("[FAIL] Qwen caption/NPZ/cache provenance is not self-consistent")

    needed = {row["id"] for row in qwen_rows}
    source: dict[str, float] = {}
    with args.source_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            key = source_id(str(item["relative_path"]))
            if key not in needed and f"{key}_0" not in needed:
                continue
            if key in source:
                raise SystemExit(f"[FAIL] duplicate source id at JSONL row {line_number}")
            value = (item.get("credibility_analysis") or {}).get("mean_similarity")
            if value is None or not math.isfinite(float(value)):
                raise SystemExit(f"[FAIL] invalid MeanSimilarity for {key}")
            source[key] = float(value)

    scores: list[float] = []
    source_ids: list[str] = []
    seen: set[str] = set()
    for index, (qwen, aligned) in enumerate(zip(qwen_rows, aligned_rows)):
        if qwen["id"] != aligned["id"]:
            raise SystemExit(f"[FAIL] Qwen/aligned row mismatch at {index}")
        if qwen["id"] in seen:
            raise SystemExit(f"[FAIL] duplicate Qwen id: {qwen['id']}")
        seen.add(qwen["id"])
        try:
            key = resolve(qwen["id"], source)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"[FAIL] source resolution row {index}: {exc}") from exc
        value = source[key]
        historical = min(9, max(0, math.floor(value * 10)))
        if int(aligned["q_level"]) != historical:
            raise SystemExit(f"[FAIL] actual-clip Q mismatch at row {index}")
        scores.append(value)
        source_ids.append(key)

    outputs: dict[str, object] = {}
    write_statuses: dict[str, str] = {}
    for k in KS:
        codes = q_codes(k)
        for strategy in STRATEGIES:
            bins = assignments(scores, source_ids, k, strategy)
            mapped = [codes[bucket] for bucket in bins]
            rows = []
            for source_row, q in zip(qwen_rows, mapped):
                row = dict(source_row)
                row["q_level"] = str(q)
                rows.append(row)
            path = args.output_dir / f"phase8_qwen_meansim_k{k}_{strategy}.tsv"
            key = f"k{k}_{strategy}"
            status = write_or_verify(
                path, qwen_fields, rows, rewrite_derived=args.rewrite_derived
            )
            write_statuses[key] = status
            bin_hist = Counter(bins)
            q_hist = Counter(mapped)
            occupied_codes = [q for q in codes if q_hist[q] > 0]
            supported_codes = [
                q for q in codes if q_hist[q] / args.expected_rows >= 0.01
            ]
            supported_low = next(
                (q for q in supported_codes if q != codes[-1]), None
            )
            outputs[key] = {
                "path": str(path),
                "sha256": sha256(path),
                "status": "passed",
                "q_codes": codes,
                "bin_histogram": {str(i): bin_hist[i] for i in range(k)},
                "q_histogram": {str(q): q_hist[q] for q in codes},
                "nominal_k": k,
                "occupied_k": len(occupied_codes),
                "occupied_q_codes": occupied_codes,
                "supported_k_at_1pct": len(supported_codes),
                "supported_q_codes_at_1pct": supported_codes,
                "diagnostic_low_q": supported_low,
                "nominal_low_q": codes[0],
                "high_q": codes[-1],
            }

    generated_k2 = Path(outputs["k2_balanced"]["path"])  # type: ignore[index]
    generated_k10 = Path(outputs["k10_fixed"]["path"])  # type: ignore[index]
    if sha256(generated_k2) != sha256(args.existing_k2_balanced):
        raise SystemExit("[FAIL] K=2 balanced does not reproduce existing Half-Q")
    if sha256(generated_k10) != sha256(args.existing_k10_fixed):
        raise SystemExit("[FAIL] K=10 fixed does not reproduce existing Full-Q")

    music_fields, music_rows = read_tsv(args.musiccaps)
    if music_fields != ["id", "caption"] or any("q_level" in row for row in music_rows):
        raise SystemExit("[FAIL] MusicCaps pilot source must have only id/caption")
    music_ids = [row["id"] for row in music_rows]
    if len(music_ids) != len(set(music_ids)):
        raise SystemExit("[FAIL] MusicCaps contains duplicate ids")
    if args.pilot_size <= 0 or args.pilot_size > len(music_rows):
        raise SystemExit("[FAIL] invalid pilot size")
    ranked_prompts = sorted(
        range(len(music_rows)),
        key=lambda i: hashlib.sha256(
            f"{args.pilot_seed}:{music_rows[i]['id']}".encode("utf-8")
        ).digest(),
    )
    selected = set(ranked_prompts[: args.pilot_size])
    pilot_rows = [row for i, row in enumerate(music_rows) if i in selected]
    pilot_status = write_or_verify(
        args.pilot_prompts,
        music_fields,
        pilot_rows,
        rewrite_derived=args.rewrite_derived,
    )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "rows": args.expected_rows,
        "signal": "actual-clip credibility_analysis.mean_similarity",
        "caption": "official aligned Qwen caption TSV",
        "strategies": {
            "balanced": (
                "equal-frequency rank bins; key=(mean_similarity, source_id); "
                "one extra row assigned to each of the highest remainder bins"
            ),
            "fixed": "equal-width bins on [0,1]; floor(mean_similarity*K), clamped",
        },
        "q_code_policy": "round-half-up(index*9/(K-1)); endpoints always q0/q9",
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": outputs,
        "reuse_equivalence": {
            "k2_balanced_equals_existing_halfq": True,
            "k10_fixed_equals_existing_fullq": True,
        },
        "pilot_prompts": {
            "path": str(args.pilot_prompts),
            "sha256": sha256(args.pilot_prompts),
            "status": "passed",
            "rows": len(pilot_rows),
            "seed": args.pilot_seed,
            "selection": "smallest sha256(seed:id), emitted in original MusicCaps order",
        },
    }
    immutable = {key: value for key, value in payload.items() if key != "created_at"}
    if args.manifest.exists():
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))
        if {key: value for key, value in previous.items() if key != "created_at"} != immutable:
            if not args.rewrite_derived:
                raise SystemExit("[FAIL] existing grid manifest drift")
            atomic_text(
                args.manifest,
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
            )
            manifest_status = "rewritten"
        else:
            manifest_status = "verified"
    else:
        atomic_text(
            args.manifest,
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        manifest_status = "created"
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest": manifest_status,
                "write_statuses": {
                    **write_statuses,
                    "pilot_prompts": pilot_status,
                },
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
