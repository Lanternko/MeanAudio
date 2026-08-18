#!/usr/bin/env python3
"""CPU-only official-caption inventory for Jamendo / ATTM prep.

Compares local Jamendo track coverage and caption hashes against official
Qwen and/or MusicFlamingo JSON files supplied as arguments.

Hard guarantees:
  - does not encode captions (no T5/CLAP/text encoder calls)
  - does not use GPU
  - does not load audio
  - pure coverage / hash inventory for later ATTM official work
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_id(raw: str) -> str:
    text = str(raw).strip()
    # Common Jamendo forms: bare id, track_<id>, <id>_segment_*, path tails.
    if text.startswith("track_"):
        text = text[len("track_") :]
    if "_segment_" in text:
        text = text.split("_segment_", 1)[0]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if text.endswith(".mp3") or text.endswith(".flac") or text.endswith(".wav"):
        text = text.rsplit(".", 1)[0]
    # Local Phase-8 clip ids carry the two-digit Jamendo shard as ``00_``;
    # official ATTM paths encode the same shard as a directory (``00/``).
    # Track identity is the numeric suffix in both cases.
    match = re.fullmatch(r"\d{2}_(\d+)", text)
    if match:
        text = match.group(1)
    return text


def extract_pairs(obj: Any, *, source_name: str) -> list[dict[str, str]]:
    """Best-effort extraction of (track_id, caption) from heterogeneous JSON."""
    pairs: list[dict[str, str]] = []

    def push(track_id: Any, caption: Any) -> None:
        if track_id is None or caption is None:
            return
        cap = caption if isinstance(caption, str) else json.dumps(caption, sort_keys=True)
        if not str(track_id).strip() or not cap.strip():
            return
        pairs.append(
            {
                "track_id": normalize_id(str(track_id)),
                "caption": cap,
                "caption_sha256": sha256_text(cap),
                "source": source_name,
            }
        )

    if isinstance(obj, dict):
        # Dict of id -> caption or id -> {caption: ...}
        sample_vals = list(obj.values())[:5]
        if sample_vals and all(isinstance(v, (str, dict, list)) for v in sample_vals):
            # Could still be a single record; detect id-like keys.
            id_keys = {
                "id", "track_id", "jamendo_id", "audio_id", "clip_id", "track", "path"
            }
            if id_keys.intersection(obj.keys()) and (
                "caption" in obj or "text" in obj or "captions" in obj
            ):
                tid = next((obj[k] for k in id_keys if k in obj), None)
                cap = obj.get("caption", obj.get("text"))
                if cap is None and isinstance(obj.get("captions"), list) and obj["captions"]:
                    cap = obj["captions"][0]
                push(tid, cap)
            else:
                for key, value in obj.items():
                    if isinstance(value, str):
                        push(key, value)
                    elif isinstance(value, dict):
                        cap = value.get("caption", value.get("text"))
                        if cap is None and isinstance(value.get("captions"), list):
                            for c in value["captions"]:
                                push(key, c)
                        else:
                            push(key, cap)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                push(key, item)
                            elif isinstance(item, dict):
                                push(key, item.get("caption", item.get("text")))
        return pairs

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                tid = None
                for key in (
                    "id",
                    "track_id",
                    "jamendo_id",
                    "audio_id",
                    "clip_id",
                    "track",
                    "path",
                ):
                    if key in item:
                        tid = item[key]
                        break
                cap = item.get("caption", item.get("text"))
                if cap is None and isinstance(item.get("captions"), list):
                    for c in item["captions"]:
                        if isinstance(c, str):
                            push(tid, c)
                        elif isinstance(c, dict):
                            push(tid, c.get("caption", c.get("text")))
                else:
                    push(tid, cap)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                push(item[0], item[1])
        return pairs

    raise SystemExit(f"[FAIL] unsupported JSON root type: {type(obj).__name__}")


def load_official_json(path: Path, name: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"[FAIL] official JSON not found: {path}")
    # Refuse accidental GPU / torch import paths by design: only stdlib JSON.
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        pairs: list[dict[str, str]] = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[FAIL] {path}:{line_no} JSONL parse error: {exc}") from exc
            pairs.extend(extract_pairs(obj, source_name=name))
        return pairs
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[FAIL] {path} JSON parse error: {exc}") from exc
    return extract_pairs(obj, source_name=name)


def load_local_tsv(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    out: list[dict[str, str]] = []
    for row in rows:
        tid = normalize_id(row["id"])
        cap = row["caption"]
        out.append(
            {
                "track_id": tid,
                "clip_id": row["id"],
                "caption": cap,
                "caption_sha256": sha256_text(cap),
                "source": "local_tsv",
            }
        )
    return out


def summarize_coverage(
    local: list[dict[str, str]], official: list[dict[str, str]], name: str
) -> dict[str, Any]:
    local_tracks = {row["track_id"] for row in local}
    official_tracks = {row["track_id"] for row in official}
    local_by_track: dict[str, set[str]] = {}
    for row in local:
        local_by_track.setdefault(row["track_id"], set()).add(row["caption_sha256"])
    official_by_track: dict[str, set[str]] = {}
    for row in official:
        official_by_track.setdefault(row["track_id"], set()).add(row["caption_sha256"])

    both = local_tracks & official_tracks
    hash_match = 0
    hash_mismatch = 0
    for track in both:
        if local_by_track[track] & official_by_track[track]:
            hash_match += 1
        else:
            hash_mismatch += 1

    return {
        "name": name,
        "official_rows": len(official),
        "official_unique_tracks": len(official_tracks),
        "local_unique_tracks": len(local_tracks),
        "intersection_tracks": len(both),
        "local_only_tracks": len(local_tracks - official_tracks),
        "official_only_tracks": len(official_tracks - local_tracks),
        "intersection_caption_hash_overlap_tracks": hash_match,
        "intersection_caption_hash_mismatch_tracks": hash_mismatch,
        "coverage_of_local": (
            len(both) / len(local_tracks) if local_tracks else 0.0
        ),
        "coverage_of_official": (
            len(both) / len(official_tracks) if official_tracks else 0.0
        ),
    }


def main() -> None:
    # Hard CPU-only guard: refuse if CUDA is forced on.
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        # Still OK as long as we never import torch/cuda; record note only.
        pass
    # Ensure no accidental GPU framework import later in this process.
    forbidden = {"torch", "tensorflow", "jax", "laion_clap"}
    already = forbidden.intersection(sys.modules)
    if already:
        raise SystemExit(
            f"[FAIL] GPU/encoder modules already imported: {sorted(already)}; "
            "this inventory must stay CPU-only and encoding-free"
        )

    parser = argparse.ArgumentParser(
        description="CPU-only Jamendo vs official caption coverage inventory"
    )
    parser.add_argument(
        "--local-tsv",
        type=Path,
        required=True,
        help="Local Jamendo training/eval TSV with id+caption columns",
    )
    parser.add_argument(
        "--official-qwen-json",
        type=Path,
        default=None,
        help="Path to official Qwen caption JSON/JSONL",
    )
    parser.add_argument(
        "--official-musicflamingo-json",
        type=Path,
        default=None,
        help="Path to official MusicFlamingo caption JSON/JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-mismatch-limit",
        type=int,
        default=20,
        help="Max example track ids to include for mismatches",
    )
    args = parser.parse_args()

    if not args.official_qwen_json and not args.official_musicflamingo_json:
        raise SystemExit(
            "[FAIL] provide at least one of --official-qwen-json or "
            "--official-musicflamingo-json"
        )

    local = load_local_tsv(args.local_tsv)
    local_track_counts = Counter(row["track_id"] for row in local)

    inventories: list[dict[str, Any]] = []
    sources_meta: dict[str, Any] = {
        "local_tsv": str(args.local_tsv),
        "local_tsv_sha256": sha256_file(args.local_tsv),
        "local_rows": len(local),
        "local_unique_tracks": len(local_track_counts),
    }

    for label, path in (
        ("qwen", args.official_qwen_json),
        ("musicflamingo", args.official_musicflamingo_json),
    ):
        if path is None:
            continue
        pairs = load_official_json(path, name=label)
        summary = summarize_coverage(local, pairs, name=label)
        sources_meta[f"official_{label}_json"] = str(path)
        sources_meta[f"official_{label}_sha256"] = sha256_file(path)

        local_tracks = {row["track_id"] for row in local}
        official_tracks = {row["track_id"] for row in pairs}
        only_official = sorted(official_tracks - local_tracks)[: args.sample_mismatch_limit]
        only_local = sorted(local_tracks - official_tracks)[: args.sample_mismatch_limit]
        summary["sample_official_only_track_ids"] = only_official
        summary["sample_local_only_track_ids"] = only_local
        inventories.append(summary)

    # Final encoder/ML-framework guard after work (stdlib + json/csv only).
    leaked = forbidden.intersection(sys.modules)
    if leaked:
        raise SystemExit(
            f"[FAIL] forbidden modules imported during inventory: "
            f"{sorted(leaked)}; inventory must stay CPU-only and encoding-free"
        )
    # Reject torch.cuda / tensorflow.python.framework specifically if present.
    for name in sys.modules:
        if name == "torch" or name.startswith("torch.") or name.startswith("tensorflow"):
            raise SystemExit(
                f"[FAIL] ML framework module imported: {name}; inventory is CPU-only"
            )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "official_caption_inventory_cpu_only_no_encoding",
        "gpu_used": False,
        "captions_encoded": False,
        "sources": sources_meta,
        "inventories": inventories,
        "notes": [
            "ATTM official 90.14 evaluator remains blocked until exact 100-prompt CSV.",
            "This inventory only compares track coverage and caption hashes.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
