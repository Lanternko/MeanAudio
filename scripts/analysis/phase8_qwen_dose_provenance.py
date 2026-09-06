#!/usr/bin/env python3
"""Shared nested cache-provenance checks for the Phase-8 Qwen dose chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_nested_cache_provenance(
    arm: str, npz_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    if arm == "control":
        validation = Path(str(manifest.get("validation_report", "")))
        if (
            not validation.is_file()
            or sha256_file(validation) != manifest.get("validation_report_sha256")
        ):
            raise RuntimeError("control cache validation report hash drift")
        payload = json.loads(validation.read_text(encoding="utf-8"))
        canonical = Path(str(payload.get("paths", {}).get("manifest", "")))
        if (
            payload.get("status") != "passed"
            or payload.get("paths", {}).get("output_dir") != str(npz_dir)
            or not canonical.is_file()
            or sha256_file(canonical) != payload.get("sha256", {}).get("manifest")
        ):
            raise RuntimeError("control NPZ validation/canonical manifest provenance drift")
        return {
            "validation_report_sha256": sha256_file(validation),
            "canonical_manifest_sha256": sha256_file(canonical),
        }
    if arm == "qwen":
        mapper = Path(str(manifest.get("mapper_manifest", "")))
        boundary = manifest.get("resume_boundary", {})
        boundary_path = npz_dir / str(boundary.get("name", ""))
        if (
            not mapper.is_file()
            or sha256_file(mapper) != manifest.get("mapper_manifest_sha256")
            or not boundary_path.is_file()
            or sha256_file(boundary_path) != boundary.get("sha256")
        ):
            raise RuntimeError("qwen NPZ mapper/boundary provenance drift")
        return {
            "mapper_manifest_sha256": sha256_file(mapper),
            "boundary_sha256": sha256_file(boundary_path),
        }
    raise RuntimeError(f"unsupported arm for nested provenance: {arm}")
