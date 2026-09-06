#!/usr/bin/env python3
"""Map the local Phase-8 catalog rows to official ATTM Qwen captions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase8_qwen_probe_lib import (
    EXPECTED_ROWS,
    ContractError,
    canonical_json,
    read_cache_list,
    read_tsv,
    sha256_file,
    validate_row_cache_alignment,
    write_immutable_json,
    write_text_atomic,
)


LOCAL_ID_RE = re.compile(r"^(?P<prefix>\d{2})_(?P<track>\d+)_segment_.+$")
OFFICIAL_PATH_RE = re.compile(
    r"^(?P<prefix>\d{2})/(?P<track>\d+)\.(?P<extension>[A-Za-z0-9]+)$"
)


DEFAULT_LOCAL_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train.tsv")
DEFAULT_CACHE_LIST = Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt")
DEFAULT_QWEN_JSON = Path(
    "/home/kojiek/reference-repos/ICME26-ATTM-GC-FluxAudio/data/captions/jamendo_qwen.json"
)
DEFAULT_OUT_TSV = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched.tsv"
)
DEFAULT_OUT_CACHE = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt"
)
DEFAULT_MANIFEST = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_manifest.json"
)


def local_track_id(clip_id: str) -> tuple[str, str]:
    match = LOCAL_ID_RE.fullmatch(clip_id)
    if not match:
        raise ContractError(f"cannot parse local clip id: {clip_id!r}")
    return match.group("prefix"), match.group("track")


def official_track_id(path: str) -> tuple[str, str]:
    normalized = path.strip().lstrip("./")
    match = OFFICIAL_PATH_RE.fullmatch(normalized)
    if not match:
        raise ContractError(f"cannot parse official Qwen path: {path!r}")
    return match.group("prefix"), match.group("track")


def load_official(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid official JSON: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ContractError("official Qwen JSON must be a list")
    by_track: dict[tuple[str, str], dict[str, str]] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(("path", "caption")) - set(item):
            raise ContractError(f"official row {index} lacks path/caption")
        official_path = str(item["path"]).strip()
        caption = str(item["caption"])
        if not caption.strip():
            raise ContractError(f"blank official caption at row {index}: {official_path}")
        key = official_track_id(official_path)
        if key in by_track:
            raise ContractError(f"duplicate/ambiguous official track: {key}")
        by_track[key] = {"path": official_path, "caption": caption}
    return by_track


def build_mapping(local_tsv: Path, cache_list: Path, official_json: Path) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    rows = read_tsv(local_tsv)
    names = read_cache_list(cache_list)
    validate_row_cache_alignment(rows, names)
    official = load_official(official_json)

    seen_local: set[str] = set()
    mapped: list[dict[str, str]] = []
    missing: list[str] = []
    for index, row in enumerate(rows):
        clip_id = str(row["id"])
        if clip_id in seen_local:
            raise ContractError(f"duplicate local clip id at row {index}: {clip_id}")
        seen_local.add(clip_id)
        prefix, track = local_track_id(clip_id)
        item = official.get((prefix, track))
        if item is None:
            missing.append(f"row={index} clip_id={clip_id} track={prefix}/{track}")
            continue
        mapped.append(
            {
                "id": clip_id,
                "caption": item["caption"],
                # Kept only as provenance.  The queue explicitly sets NoQ.
                "q_level": str(row.get("q_level", "")),
                "official_path": item["path"],
                "track_id": f"{prefix}/{track}",
            }
        )
    if missing:
        preview = "; ".join(missing[:5])
        raise ContractError(f"official coverage missing {len(missing)} rows: {preview}")
    if len(mapped) != len(rows):
        raise ContractError(f"mapped rows={len(mapped)}, local rows={len(rows)}")

    stats = {
        "local_rows": len(rows),
        "cache_rows": len(names),
        "official_rows": len(official),
        "mapped_rows": len(mapped),
        "unique_local_tracks": len({local_track_id(row["id"]) for row in rows}),
        "unique_official_tracks_used": len({row["track_id"] for row in mapped}),
        "coverage_exact": True,
    }
    return mapped, names, stats


def write_outputs(
    mapped: list[dict[str, str]],
    names: list[str],
    stats: dict[str, Any],
    *,
    local_tsv: Path,
    cache_list: Path,
    official_json: Path,
    out_tsv: Path,
    out_cache: Path,
    manifest: Path,
) -> dict[str, Any]:
    if len(mapped) != EXPECTED_ROWS:
        raise ContractError(
            f"refusing non-full official mapping: {len(mapped)} rows, expected {EXPECTED_ROWS}"
        )
    if out_tsv.exists() or out_cache.exists() or manifest.exists():
        raise ContractError("mapper outputs are immutable and already exist")

    fields = ["id", "caption", "q_level", "official_path", "track_id"]
    lines: list[str] = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(mapped)
    lines.append(buffer.getvalue())
    write_text_atomic(out_tsv, "".join(lines))
    write_text_atomic(out_cache, "".join(f"{name}\n" for name in names))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "phase8_qwen_official_matched_metadata",
        "inputs": {
            "local_tsv": str(local_tsv),
            "local_tsv_sha256": sha256_file(local_tsv),
            "cache_list": str(cache_list),
            "cache_list_sha256": sha256_file(cache_list),
            "official_json": str(official_json),
            "official_json_sha256": sha256_file(official_json),
        },
        "outputs": {
            "tsv": str(out_tsv),
            "tsv_sha256": sha256_file(out_tsv),
            "cache_list": str(out_cache),
            "cache_list_sha256": sha256_file(out_cache),
        },
        "row_order": "local TSV order; cache list order unchanged",
        "training_q_policy": "NoQ; q_level retained only as local provenance",
        "stats": stats,
    }
    write_immutable_json(manifest, payload)
    return payload


def verify_existing(
    mapped: list[dict[str, str]],
    names: list[str],
    stats: dict[str, Any],
    *,
    local_tsv: Path,
    cache_list: Path,
    official_json: Path,
    out_tsv: Path,
    out_cache: Path,
    manifest: Path,
) -> dict[str, Any]:
    for path in (out_tsv, out_cache, manifest):
        if not path.is_file():
            raise ContractError(f"resume verification missing mapper output: {path}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected_inputs = {
        "local_tsv": str(local_tsv),
        "local_tsv_sha256": sha256_file(local_tsv),
        "cache_list": str(cache_list),
        "cache_list_sha256": sha256_file(cache_list),
        "official_json": str(official_json),
        "official_json_sha256": sha256_file(official_json),
    }
    if payload.get("inputs") != expected_inputs:
        raise ContractError("existing mapper manifest input provenance drift")
    if payload.get("stats") != stats or payload.get("row_order") != "local TSV order; cache list order unchanged":
        raise ContractError("existing mapper manifest stats/order drift")
    if payload.get("training_q_policy") != "NoQ; q_level retained only as local provenance":
        raise ContractError("existing mapper manifest training policy drift")
    if sha256_file(out_tsv) != payload.get("outputs", {}).get("tsv_sha256"):
        raise ContractError("existing Qwen TSV hash drift")
    if sha256_file(out_cache) != payload.get("outputs", {}).get("cache_list_sha256"):
        raise ContractError("existing Qwen cache-list hash drift")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-tsv", type=Path, default=DEFAULT_LOCAL_TSV)
    parser.add_argument("--cache-list", type=Path, default=DEFAULT_CACHE_LIST)
    parser.add_argument("--official-json", type=Path, default=DEFAULT_QWEN_JSON)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_OUT_TSV)
    parser.add_argument("--out-cache", type=Path, default=DEFAULT_OUT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--write", action="store_true", help="write immutable outputs")
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sum(bool(value) for value in (args.validate_only, args.write, args.verify_existing)) != 1:
        raise SystemExit("choose exactly one of --validate-only, --write, --verify-existing")
    mapped, names, stats = build_mapping(args.local_tsv, args.cache_list, args.official_json)
    if args.validate_only:
        print(canonical_json({"status": "passed", **stats}))
        return 0
    if args.verify_existing:
        payload = verify_existing(
            mapped,
            names,
            stats,
            local_tsv=args.local_tsv,
            cache_list=args.cache_list,
            official_json=args.official_json,
            out_tsv=args.out_tsv,
            out_cache=args.out_cache,
            manifest=args.manifest,
        )
        print(canonical_json({"status": "passed", "manifest": payload}))
        return 0
    payload = write_outputs(
        mapped,
        names,
        stats,
        local_tsv=args.local_tsv,
        cache_list=args.cache_list,
        official_json=args.official_json,
        out_tsv=args.out_tsv,
        out_cache=args.out_cache,
        manifest=args.manifest,
    )
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
