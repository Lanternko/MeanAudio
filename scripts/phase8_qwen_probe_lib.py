#!/usr/bin/env python3
"""Small, dependency-light primitives shared by the Phase-8 Qwen probe.

The large-data commands deliberately use the cache-list as their only source
of NPZ names.  In particular, no command in this module enumerates an NPZ
directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


EXPECTED_ROWS = 251_599
MIN_FREE_BYTES = 50 * 1024**3
REQUIRED_KEYS = (
    "mean",
    "std",
    "text_features",
    "text_features_c",
    "text_attention_mask",
    "clip_id",
    "catalog_index",
    "caption_sha256",
)
TEXT_KEYS = {
    "text_features",
    "text_features_c",
    "text_attention_mask",
    "caption_sha256",
}
DEFAULT_SCHEMA = {
    "mean": ((312, 20), "float32"),
    "std": ((312, 20), "float32"),
    "text_features": ((77, 1024), "float32"),
    "text_features_c": ((512,), "float32"),
    "text_attention_mask": ((77,), "bool"),
}


class ContractError(RuntimeError):
    """An input, invariant, or immutable-contract failure."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes(order="C")


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(array_bytes(array))


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ContractError(f"expected scalar array, got shape={array.shape}")
    item = array.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ContractError(f"expected scalar integer array, got shape={array.shape}")
    return int(array.item())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ContractError(f"empty TSV: {path}")
    if not rows[0].get("id") or "caption" not in rows[0]:
        raise ContractError(f"TSV must contain id and caption columns: {path}")
    return rows


def read_cache_list(path: Path, expected_rows: int | None = None) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            name = raw.strip()
            if not name:
                continue
            candidate = Path(name)
            if candidate.name != name or candidate.suffix != ".npz":
                raise ContractError(f"unsafe/non-NPZ cache name: {name!r}")
            names.append(name)
    if not names:
        raise ContractError(f"empty cache list: {path}")
    if len(set(names)) != len(names):
        raise ContractError(f"duplicate cache names: {path}")
    if expected_rows is not None and len(names) != expected_rows:
        raise ContractError(
            f"cache rows={len(names)}, expected={expected_rows} for {path}"
        )
    return names


def validate_row_cache_alignment(rows: Sequence[Mapping[str, str]], names: Sequence[str]) -> None:
    if len(rows) != len(names):
        raise ContractError(f"TSV/cache count mismatch: {len(rows)} != {len(names)}")
    for index, name in enumerate(names):
        if not name.endswith(".npz"):
            raise ContractError(f"cache name is not NPZ: {name}")
        # Cache filenames are deliberately not assumed to be sequential.
        # Their order is the only index mapping used by the loader.
        if not rows[index].get("id"):
            raise ContractError(f"blank clip id at TSV row {index}")


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_atomic(path, canonical_json(payload))


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a manifest once; an existing manifest must be byte-equivalent."""
    encoded = canonical_json(payload)
    if path.exists():
        actual = path.read_text(encoding="utf-8")
        if actual != encoded:
            raise ContractError(f"immutable manifest drift: {path}")
        return
    write_text_atomic(path, encoded)


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one NPZ completely, fsync it, then publish it by rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def validate_schema(data: Mapping[str, np.ndarray], *, expected: Mapping[str, tuple[tuple[int, ...], str]] = DEFAULT_SCHEMA) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ContractError(f"NPZ missing required keys: {missing}")
    for key, (shape, dtype_name) in expected.items():
        array = data[key]
        if tuple(array.shape) != tuple(shape):
            raise ContractError(f"{key} shape={array.shape}, expected={shape}")
        if str(array.dtype) != dtype_name:
            raise ContractError(f"{key} dtype={array.dtype}, expected={dtype_name}")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ContractError(f"{key} contains NaN/Inf")


def compare_arrays_exact(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
        raise ContractError(f"array mismatch: {label}")


def projected_free_space(output_dir: Path, additional_bytes: int) -> dict[str, int]:
    """Return free-space facts and fail if either final margin is below 50 GiB."""
    hdd_free = int(shutil.disk_usage(output_dir).free)
    root_free = int(shutil.disk_usage(Path("/")).free)
    return check_projected_free(root_free, hdd_free, additional_bytes)


def check_projected_free(root_free: int, hdd_free: int, additional_bytes: int) -> dict[str, int]:
    projected_hdd = int(hdd_free) - int(additional_bytes)
    if int(root_free) < MIN_FREE_BYTES:
        raise ContractError(f"root free space below 50 GiB: {root_free}")
    if projected_hdd < MIN_FREE_BYTES:
        raise ContractError(
            f"projected HDD free space below 50 GiB: {projected_hdd} "
            f"(additional={additional_bytes})"
        )
    return {
        "root_free_bytes": int(root_free),
        "hdd_free_bytes": int(hdd_free),
        "projected_hdd_free_bytes": projected_hdd,
        "additional_bytes": int(additional_bytes),
    }


def ensure_fresh(paths: Iterable[Path]) -> None:
    conflicts = [str(path) for path in paths if path.exists()]
    if conflicts:
        raise ContractError("fresh run has existing artifacts: " + ", ".join(conflicts))


def iter_rows(rows: Sequence[Mapping[str, str]], limit: int | None) -> Iterator[tuple[int, Mapping[str, str]]]:
    count = len(rows) if limit is None else min(limit, len(rows))
    for index in range(count):
        yield index, rows[index]
