#!/usr/bin/env python3
"""Build Qwen-caption Full-Q and Half-Q TSVs with exhaustive provenance gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCAL_ID_RE = re.compile(r"^(?P<prefix>\d{2})_(?P<track>\d+)_segment_.+$")
OFFICIAL_PATH_RE = re.compile(
    r"^(?P<prefix>\d{2})/(?P<track>\d+)\.(?P<extension>[A-Za-z0-9]+)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"[FAIL] TSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"[FAIL] expected JSON object: {path}")
    return payload


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def track_from_clip(clip_id: str) -> tuple[str, str]:
    match = LOCAL_ID_RE.fullmatch(clip_id)
    if not match:
        raise SystemExit(f"[FAIL] cannot parse clip id: {clip_id!r}")
    return match.group("prefix"), match.group("track")


def track_from_path(path: str) -> tuple[str, str]:
    match = OFFICIAL_PATH_RE.fullmatch(path.strip().lstrip("./"))
    if not match:
        raise SystemExit(f"[FAIL] cannot parse official path: {path!r}")
    return match.group("prefix"), match.group("track")


def load_official(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("[FAIL] official Qwen JSON must be a list")
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not {"path", "caption"} <= set(item):
            raise SystemExit(f"[FAIL] malformed official Qwen row {index}")
        official_path = str(item["path"]).strip()
        caption = str(item["caption"])
        if not caption.strip():
            raise SystemExit(f"[FAIL] blank official caption at row {index}")
        key = track_from_path(official_path)
        if key in result:
            raise SystemExit(f"[FAIL] duplicate official Qwen track: {key}")
        result[key] = (official_path, caption)
    return result


def validate_manifest_hash(
    manifest: dict[str, Any], dotted: tuple[str, ...], actual: str, label: str
) -> None:
    value: Any = manifest
    for key in dotted:
        if not isinstance(value, dict):
            value = None
            break
        value = value.get(key)
    if value != actual:
        raise SystemExit(
            f"[FAIL] {label} hash drift: manifest={value!r}, actual={actual}"
        )


def write_or_verify(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> str:
    if path.exists():
        existing_fields, existing = read_tsv(path)
        if existing_fields != fieldnames or existing != rows:
            raise SystemExit(f"[FAIL] existing output content drift: {path}")
        return "verified"
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-tsv", type=Path, required=True)
    parser.add_argument("--qwen-mapper-manifest", type=Path, required=True)
    parser.add_argument("--qwen-npz-manifest", type=Path, required=True)
    parser.add_argument("--qwen-cache-audit", type=Path, required=True)
    parser.add_argument("--qwen-cache-list", type=Path, required=True)
    parser.add_argument("--official-json", type=Path, required=True)
    parser.add_argument("--aligned-tsv", type=Path, required=True)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--halfq-tsv", type=Path, required=True)
    parser.add_argument("--halfq-manifest", type=Path, required=True)
    parser.add_argument("--fullq-output", type=Path, required=True)
    parser.add_argument("--halfq-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=251599)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = (
        args.qwen_tsv,
        args.qwen_mapper_manifest,
        args.qwen_npz_manifest,
        args.qwen_cache_audit,
        args.qwen_cache_list,
        args.official_json,
        args.aligned_tsv,
        args.aligned_manifest,
        args.halfq_tsv,
        args.halfq_manifest,
    )
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing input: {path}")

    qwen_fields, qwen_rows = read_tsv(args.qwen_tsv)
    aligned_fields, aligned_rows = read_tsv(args.aligned_tsv)
    half_fields, half_rows = read_tsv(args.halfq_tsv)
    required_qwen = {"id", "caption", "q_level", "official_path", "track_id"}
    if not required_qwen <= set(qwen_fields):
        raise SystemExit(f"[FAIL] Qwen TSV fields changed: {qwen_fields}")
    if not {"id", "caption", "q_level"} <= set(aligned_fields):
        raise SystemExit("[FAIL] aligned TSV lacks id/caption/q_level")
    if aligned_fields != half_fields:
        raise SystemExit("[FAIL] aligned and Half-Q TSV schemas differ")
    if not (
        len(qwen_rows)
        == len(aligned_rows)
        == len(half_rows)
        == args.expected_rows
    ):
        raise SystemExit(
            "[FAIL] cardinality mismatch: "
            f"qwen={len(qwen_rows)} full={len(aligned_rows)} "
            f"half={len(half_rows)} expected={args.expected_rows}"
        )

    mapper = read_json(args.qwen_mapper_manifest)
    npz_manifest = read_json(args.qwen_npz_manifest)
    cache_audit = read_json(args.qwen_cache_audit)
    aligned_manifest = read_json(args.aligned_manifest)
    half_manifest = read_json(args.halfq_manifest)
    hashes = {path: sha256(path) for path in input_paths}
    validate_manifest_hash(
        mapper, ("outputs", "tsv_sha256"), hashes[args.qwen_tsv], "Qwen mapper TSV"
    )
    validate_manifest_hash(
        mapper,
        ("outputs", "cache_list_sha256"),
        hashes[args.qwen_cache_list],
        "Qwen mapper cache list",
    )
    validate_manifest_hash(
        mapper,
        ("inputs", "official_json_sha256"),
        hashes[args.official_json],
        "official Qwen JSON",
    )
    validate_manifest_hash(
        npz_manifest, ("tsv_sha256",), hashes[args.qwen_tsv], "Qwen NPZ TSV"
    )
    validate_manifest_hash(
        npz_manifest,
        ("cache_list_sha256",),
        hashes[args.qwen_cache_list],
        "Qwen NPZ cache list",
    )
    validate_manifest_hash(
        cache_audit, ("tsv_sha256",), hashes[args.qwen_tsv], "Qwen cache audit TSV"
    )
    validate_manifest_hash(
        cache_audit,
        ("cache_list_sha256",),
        hashes[args.qwen_cache_list],
        "Qwen cache audit list",
    )
    validate_manifest_hash(
        aligned_manifest,
        ("output_sha256",),
        hashes[args.aligned_tsv],
        "aligned Full-Q TSV",
    )
    validate_manifest_hash(
        half_manifest, ("output_sha256",), hashes[args.halfq_tsv], "Half-Q TSV"
    )
    for label, payload in (
        ("Qwen NPZ", npz_manifest),
        ("Qwen cache audit", cache_audit),
    ):
        if payload.get("status") != "passed":
            raise SystemExit(f"[FAIL] {label} status is not passed")
    for label, payload in (
        ("mapper", mapper),
        ("NPZ", npz_manifest),
        ("cache audit", cache_audit),
    ):
        rows = (
            payload.get("stats", {}).get("mapped_rows")
            if label == "mapper"
            else payload.get("completed_rows", payload.get("rows"))
        )
        if rows != args.expected_rows:
            raise SystemExit(f"[FAIL] {label} row count={rows}")
    if cache_audit.get("semantic_gate", {}).get("status") != "passed":
        raise SystemExit("[FAIL] Qwen cache semantic gate is not passed")
    if (
        aligned_manifest.get("matched_source_rows") != args.expected_rows
        or aligned_manifest.get("rows") != args.expected_rows
        or aligned_manifest.get("formula")
        != "clamp(floor(mean_similarity * 10), 0, 9)"
    ):
        raise SystemExit("[FAIL] aligned Full-Q manifest did not repair all rows")
    if half_manifest.get("historical_q_rows_verified") != args.expected_rows:
        raise SystemExit("[FAIL] Half-Q manifest did not reverify all historical Q")

    official = load_official(args.official_json)
    seen_ids: set[str] = set()
    full_out: list[dict[str, str]] = []
    half_out: list[dict[str, str]] = []
    full_hist: Counter[int] = Counter()
    half_hist: Counter[int] = Counter()
    full_changes = 0
    half_changes = 0
    for index, (qwen, full, half) in enumerate(
        zip(qwen_rows, aligned_rows, half_rows)
    ):
        ids = (qwen["id"], full["id"], half["id"])
        if len(set(ids)) != 1:
            raise SystemExit(f"[FAIL] row order/id mismatch at {index}: {ids}")
        clip_id = ids[0]
        if clip_id in seen_ids:
            raise SystemExit(f"[FAIL] duplicate clip id at row {index}: {clip_id}")
        seen_ids.add(clip_id)
        key = track_from_clip(clip_id)
        if key not in official:
            raise SystemExit(f"[FAIL] Qwen official track missing at row {index}: {key}")
        expected_path, expected_caption = official[key]
        expected_track = f"{key[0]}/{key[1]}"
        if (
            qwen["official_path"] != expected_path
            or qwen["track_id"] != expected_track
            or qwen["caption"] != expected_caption
        ):
            raise SystemExit(f"[FAIL] Qwen caption/track mismatch at row {index}")
        for field in aligned_fields:
            if field == "q_level":
                continue
            if full[field] != half[field]:
                raise SystemExit(
                    f"[FAIL] aligned Full-Q/Half-Q drift row={index} field={field}"
                )
        full_q = int(full["q_level"])
        half_q = int(half["q_level"])
        if not 0 <= full_q <= 9 or half_q not in {0, 9}:
            raise SystemExit(f"[FAIL] invalid Q at row {index}: {full_q}, {half_q}")
        full_row = dict(qwen)
        half_row = dict(qwen)
        full_row["q_level"] = str(full_q)
        half_row["q_level"] = str(half_q)
        full_out.append(full_row)
        half_out.append(half_row)
        full_hist[full_q] += 1
        half_hist[half_q] += 1
        full_changes += qwen["q_level"] != str(full_q)
        half_changes += qwen["q_level"] != str(half_q)

    expected_full_hist = {
        int(key): value
        for key, value in aligned_manifest.get("corrected_histogram", {}).items()
    }
    if dict(sorted(full_hist.items())) != expected_full_hist:
        raise SystemExit(f"[FAIL] Full-Q histogram drift: {full_hist}")
    if half_hist != Counter({0: 125799, 9: 125800}):
        raise SystemExit(f"[FAIL] Half-Q histogram drift: {half_hist}")

    full_status = write_or_verify(args.fullq_output, qwen_fields, full_out)
    half_status = write_or_verify(args.halfq_output, qwen_fields, half_out)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "method": (
            "official Qwen caption rows joined by exact catalog clip id/order to "
            "independently actual-clip-aligned MeanSimilarity Full-Q and Half-Q"
        ),
        "rows": args.expected_rows,
        "unique_clip_ids": len(seen_ids),
        "unique_qwen_tracks_used": len({row["track_id"] for row in qwen_rows}),
        "official_caption_rows_verified": args.expected_rows,
        "qwen_npz_caption_hash_rows_previously_verified": cache_audit.get("rows"),
        "qwen_npz_semantic_gate": cache_audit.get("semantic_gate"),
        "fullq_histogram": {
            str(key): value for key, value in sorted(full_hist.items())
        },
        "halfq_histogram": {
            str(key): value for key, value in sorted(half_hist.items())
        },
        "source_q_changes": {"fullq": full_changes, "halfq": half_changes},
        "inputs": {str(path): digest for path, digest in hashes.items()},
        "outputs": {
            "fullq_tsv": str(args.fullq_output),
            "fullq_tsv_sha256": sha256(args.fullq_output),
            "halfq_tsv": str(args.halfq_output),
            "halfq_tsv_sha256": sha256(args.halfq_output),
        },
        "invariants": {
            "row_order_unchanged": True,
            "fullq_halfq_differ_only_in_q_level": True,
            "clip_to_official_qwen_track_exhaustively_verified": True,
            "official_qwen_caption_exhaustively_verified": True,
            "actual_clip_meansim_q_exhaustively_verified_upstream": True,
            "qwen_npz_caption_hash_exhaustively_verified_upstream": True,
        },
    }
    if args.manifest.exists():
        previous = read_json(args.manifest)
        for key, value in payload.items():
            if key == "created_at":
                continue
            if previous.get(key) != value:
                raise SystemExit(f"[FAIL] existing combined manifest drift: {key}")
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
                "rows": args.expected_rows,
                "fullq": full_status,
                "halfq": half_status,
                "manifest": manifest_status,
                "fullq_histogram": payload["fullq_histogram"],
                "halfq_histogram": payload["halfq_histogram"],
                "fullq_sha256": payload["outputs"]["fullq_tsv_sha256"],
                "halfq_sha256": payload["outputs"]["halfq_tsv_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
