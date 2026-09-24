"""Shared manifest and alignment helpers for multi-caption NPZ caches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORMAT_VERSION = "multicap-v2-canonical-map"
MANIFEST_NAME = "MANIFEST.tsv"
MANIFEST_FIELDS = (
    "format_version",
    "row_index",
    "clip_id",
    "npz_fname",
    "caption_sha256",
    "slot_count",
)


def normalize_caption(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").replace("\r", " ").split())


def relative_path_to_id(rel: str) -> str:
    return rel.removesuffix(".mp3").replace("/", "_")


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or "id" not in rows[0] or "caption" not in rows[0]:
        raise ValueError(f"TSV must contain id and caption columns: {path}")
    return rows


def load_gt_cache(path: Path) -> list[str]:
    with path.open() as handle:
        names = [line.strip() for line in handle if line.strip()]
    invalid = [name for name in names if Path(name).name != name or not name.endswith(".npz")]
    if invalid:
        raise ValueError(f"gt_cache contains unsafe/non-NPZ names, first={invalid[0]!r}")
    if len(set(names)) != len(names):
        raise ValueError("gt_cache contains duplicate NPZ filenames")
    return names


def load_caption_lookup(path: Path) -> dict[str, list[str]]:
    """Load LP-MusicCaps or merged Qwen JSONL as clip_id -> caption slots."""
    lookup: dict[str, list[str]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "captions" in item and "id" in item:
                clip_id = str(item["id"])
                raw = list(item["captions"])
                valid = [normalize_caption(c) for c in raw if c and normalize_caption(c)]
                if not valid:
                    raise ValueError(f"No valid captions at {path}:{line_number}")
                captions = [normalize_caption(c) if c and normalize_caption(c) else valid[-1]
                            for c in raw]
            else:
                clip_id = relative_path_to_id(str(item["relative_path"]))
                captions = [normalize_caption(detail["caption"])
                            for detail in item["caption_details"]]
            if not captions or any(not caption for caption in captions):
                raise ValueError(f"Empty caption at {path}:{line_number}")
            if clip_id in lookup:
                raise ValueError(f"Duplicate caption JSONL id {clip_id!r}")
            lookup[clip_id] = captions
    return lookup


def normalize_slots(captions: Iterable[str], slot_count: int) -> tuple[str, ...]:
    slots = [normalize_caption(caption) for caption in captions]
    if not slots:
        raise ValueError("Caption slot list is empty")
    while len(slots) < slot_count:
        slots.append(slots[-1])
    return tuple(slots[:slot_count])


def caption_sha256(captions: Iterable[str]) -> str:
    payload = json.dumps(list(captions), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheSpec:
    row_index: int
    clip_id: str
    npz_fname: str
    captions: tuple[str, ...]

    def manifest_row(self) -> dict[str, str]:
        return {
            "format_version": FORMAT_VERSION,
            "row_index": str(self.row_index),
            "clip_id": self.clip_id,
            "npz_fname": self.npz_fname,
            "caption_sha256": caption_sha256(self.captions),
            "slot_count": str(len(self.captions)),
        }


def build_specs(
    rows: list[dict[str, str]],
    npz_names: list[str],
    caption_lookup: dict[str, list[str]],
    *,
    slot_count: int,
    allow_missing_captions: bool,
    allow_tsv_caption_outside_pool: bool,
) -> list[CacheSpec]:
    if len(rows) != len(npz_names):
        raise ValueError(f"TSV/gt_cache row mismatch: {len(rows)} != {len(npz_names)}")

    specs: list[CacheSpec] = []
    missing: list[str] = []
    outside: list[str] = []
    for index, (row, npz_name) in enumerate(zip(rows, npz_names)):
        clip_id = str(row["id"])
        captions = caption_lookup.get(clip_id)
        if captions is None:
            missing.append(clip_id)
            if not allow_missing_captions:
                continue
            captions = [row["caption"]] * slot_count
        slots = normalize_slots(captions, slot_count)
        selected = normalize_caption(row["caption"])
        if selected not in slots:
            outside.append(clip_id)
        specs.append(CacheSpec(index, clip_id, npz_name, slots))

    if missing and not allow_missing_captions:
        raise ValueError(
            f"Caption JSONL is missing {len(missing)} TSV ids; first={missing[:5]}. "
            "Use --allow-missing-captions only for an explicitly documented fallback run."
        )
    if outside and not allow_tsv_caption_outside_pool:
        raise ValueError(
            f"TSV caption is outside the JSONL caption pool for {len(outside)} rows; "
            f"first={outside[:5]}. This usually means the wrong TSV/JSONL pair."
        )
    return specs


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def expected_manifest(specs: Iterable[CacheSpec]) -> list[dict[str, str]]:
    return [spec.manifest_row() for spec in specs]


def assert_manifest_matches(path: Path, specs: list[CacheSpec]) -> None:
    actual = read_manifest(path)
    expected = expected_manifest(specs)
    if len(actual) != len(expected):
        raise ValueError(f"Manifest row mismatch: {len(actual)} != {len(expected)}")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if any(got.get(field) != want[field] for field in MANIFEST_FIELDS):
            raise ValueError(f"Manifest mismatch at row {index}: got={got}, expected={want}")


def write_manifest_atomic(path: Path, specs: list[CacheSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(expected_manifest(specs))
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
