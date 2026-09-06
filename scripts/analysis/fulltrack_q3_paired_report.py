#!/usr/bin/env python3
"""Build the sealed Fulltrack-Q3 paired MusicCaps analysis report.

The report is intentionally independent from generation and model code.  It
validates exact ID joins, finite per-item metric rows, declared file hashes,
the B1/B2 historical reproduction gate, and then computes the predeclared
paired contrasts.  All threshold decisions are made after the validation
gate, with invalid evidence taking precedence over any positive result.

The module exposes small pure functions used by the CPU-only science
self-test.  The command line entry point never downloads models and never
uses CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment failure
    raise RuntimeError("numpy is required for paired bootstrap analysis") from exc


METRIC_KEYS = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")
ARM_IDS = ("B1", "B2", "B3", "B4", "B5", "B6")
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUANTUM = Decimal("0.0001")
TARGET_PQ = Decimal("6.9")
DEFAULT_COUNT = 5521


class AnalysisInputError(RuntimeError):
    """Raised when an analysis input cannot be trusted."""


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisInputError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisInputError(f"{label} is non-finite: {value!r}")
    return result


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError, ValueError) as exc:
        raise AnalysisInputError(f"{label} is not a Decimal number: {value!r}") from exc
    if not result.is_finite():
        raise AnalysisInputError(f"{label} is non-finite: {value!r}")
    return result


def quantize_metric(value: Any) -> Decimal:
    """Quantize a finite metric using the preregistered half-up rule."""

    return _decimal(value, "metric").quantize(QUANTUM, rounding=ROUND_HALF_UP)


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise AnalysisInputError(f"{label} must be a regular non-symlink file: {path}")


def _resolve_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AnalysisInputError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def sha256_file(path: Path) -> str:
    _regular_file(path, "hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_file(path: Path, label: str) -> Mapping[str, Any]:
    _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"cannot parse {label}: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AnalysisInputError(f"{label} must contain a JSON object: {path}")
    return payload


def _declared_hash(obj: Mapping[str, Any], path_key: str) -> str | None:
    candidates = (
        f"{path_key}_sha256",
        f"{path_key.removesuffix('_path')}_sha256",
        "sha256",
    )
    for key in candidates:
        value = obj.get(key)
        if value is not None:
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise AnalysisInputError(f"invalid declared SHA-256 for {path_key}: {value!r}")
            return value
    return None


def verify_declared_hash(path: Path, declared: str | None, label: str) -> str:
    observed = sha256_file(path)
    if declared is not None and observed != declared:
        raise AnalysisInputError(
            f"{label} hash mismatch: declared {declared}, observed {observed}"
        )
    return observed


def _find_mapping_path(obj: Any, keys: Sequence[str]) -> tuple[Path, str, str | None] | None:
    """Find the first path-bearing mapping entry in a nested contract object."""

    if isinstance(obj, Mapping):
        for key in keys:
            if key not in obj:
                continue
            value = obj[key]
            if isinstance(value, str):
                return Path(value), key, _declared_hash(obj, key)
            if isinstance(value, Mapping):
                nested = _find_mapping_path(value, ("path", "file", "value"))
                if nested is not None:
                    return nested
        for value in obj.values():
            nested = _find_mapping_path(value, keys)
            if nested is not None:
                return nested
    elif isinstance(obj, list):
        for value in obj:
            nested = _find_mapping_path(value, keys)
            if nested is not None:
                return nested
    return None


def _find_path_entry(obj: Any, keys: Sequence[str], *, base: Path, label: str) -> dict[str, Any] | None:
    found = _find_mapping_path(obj, keys)
    if found is None:
        return None
    raw_path, key, declared = found
    path = raw_path if raw_path.is_absolute() else base / raw_path
    return {
        "key": key,
        "path": str(path),
        "sha256_declared": declared,
        "sha256": verify_declared_hash(path, declared, label),
    }


def _read_ids_from_tsv(path: Path, expected_count: int) -> list[str]:
    _regular_file(path, "MusicCaps TSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise AnalysisInputError("MusicCaps TSV is missing id column")
        ids: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            item_id = (row.get("id") or "").strip()
            if ID_RE.fullmatch(item_id) is None:
                raise AnalysisInputError(f"unsafe MusicCaps ID at row {row_number}: {item_id!r}")
            if item_id in seen:
                raise AnalysisInputError(f"duplicate MusicCaps ID in TSV: {item_id}")
            seen.add(item_id)
            ids.append(item_id)
    if len(ids) != expected_count:
        raise AnalysisInputError(f"MusicCaps TSV has {len(ids)} IDs, expected {expected_count}")
    return ids


def _expected_ids(contract: Mapping[str, Any], *, base: Path, expected_count: int) -> list[str]:
    direct = contract.get("expected_ids")
    if direct is None:
        for key in ("musiccaps_ids", "frozen_ids", "ids"):
            direct = contract.get(key)
            if direct is not None:
                break
    if direct is not None:
        if not isinstance(direct, list) or len(direct) != expected_count:
            raise AnalysisInputError("contract expected_ids is not an exact list")
        ids = [str(item) for item in direct]
        if any(ID_RE.fullmatch(item) is None for item in ids) or len(set(ids)) != len(ids):
            raise AnalysisInputError("contract expected_ids contains unsafe or duplicate IDs")
        return ids

    entry = _find_path_entry(
        contract,
        ("musiccaps_tsv", "tsv", "evaluation_tsv", "musiccaps_test_tsv"),
        base=base,
        label="MusicCaps TSV",
    )
    if entry is None:
        raise AnalysisInputError("contract does not bind expected MusicCaps IDs")
    return _read_ids_from_tsv(Path(entry["path"]), expected_count)


def _arm_specs(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = contract.get("arms")
    if raw is None:
        raw = contract.get("arm_artifacts", contract.get("results"))
    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        for arm_id, spec in raw.items():
            if arm_id in ARM_IDS and isinstance(spec, Mapping):
                result[arm_id] = spec
    elif isinstance(raw, list):
        for spec in raw:
            if not isinstance(spec, Mapping):
                continue
            arm_id = spec.get("id", spec.get("arm"))
            if arm_id in ARM_IDS:
                result[str(arm_id)] = spec
    return result


def _result_root(contract: Mapping[str, Any], *, base: Path) -> Path | None:
    """Resolve the immutable runtime result root used by the HARN.

    The executable contract may carry arm artifact paths explicitly, but the
    candidate/final document also permits a single result root because the
    arm layout is fixed by the Plan.  Keeping this fallback here makes the
    analyzer compatible with both forms without guessing from the current
    working directory.
    """

    candidates: list[Any] = []
    for key in ("result_root", "results_root", "runtime_result_root"):
        if key in contract:
            candidates.append(contract[key])
    for parent_key in ("runtime_storage", "outputs", "runtime", "artifact_layout"):
        parent = contract.get(parent_key)
        if isinstance(parent, Mapping):
            for key in ("result_root", "results_root", "runtime_result_root"):
                if key in parent:
                    candidates.append(parent[key])
    for value in candidates:
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else base / path
    return None


def _arm_path_spec(spec: Mapping[str, Any], *, base: Path, keys: Sequence[str], label: str) -> dict[str, Any] | None:
    entry = _find_path_entry(spec, keys, base=base, label=label)
    return entry


def _parse_metric_rows(path: Path, expected_ids: Sequence[str]) -> tuple[dict[str, dict[str, Decimal]], str]:
    _regular_file(path, "per-item metrics TSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        required_fields = ("id",) + METRIC_KEYS
        if fields is None or set(fields) != set(required_fields) or len(fields) != len(required_fields):
            raise AnalysisInputError(
                f"per-item metrics header must be exactly {required_fields}: {path}"
            )
        rows: dict[str, dict[str, Decimal]] = {}
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise AnalysisInputError(f"malformed metrics row {row_number}: {path}")
            item_id = (row.get("id") or "").strip()
            if ID_RE.fullmatch(item_id) is None:
                raise AnalysisInputError(f"unsafe metric ID at row {row_number}: {item_id!r}")
            if item_id in rows:
                raise AnalysisInputError(f"duplicate metric ID: {item_id}")
            values: dict[str, Decimal] = {}
            for key in METRIC_KEYS:
                values[key] = _decimal(row.get(key), f"{item_id} {key}")
            rows[item_id] = values
    expected = set(expected_ids)
    actual = set(rows)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AnalysisInputError(
            f"metrics IDs do not exactly match expected IDs; missing={missing[:5]}, extra={extra[:5]}"
        )
    if len(rows) != len(expected_ids):
        raise AnalysisInputError("metrics row count is not exact")
    return rows, sha256_file(path)


def _quantized_aggregate(rows: Mapping[str, Mapping[str, Decimal]]) -> dict[str, Decimal]:
    if not rows:
        raise AnalysisInputError("cannot aggregate zero metric rows")
    count = Decimal(len(rows))
    return {
        key: (sum((row[key] for row in rows.values()), Decimal(0)) / count).quantize(
            QUANTUM, rounding=ROUND_HALF_UP
        )
        for key in METRIC_KEYS
    }


def _aggregate_report_metrics(path: Path, expected: Mapping[str, Decimal]) -> dict[str, Any]:
    payload = _json_file(path, "aggregate report")
    if payload.get("status") != "passed":
        raise AnalysisInputError(f"aggregate report is not status=passed: {path}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_KEYS):
        raise AnalysisInputError(f"aggregate report has non-exact metrics: {path}")
    parsed = {key: quantize_metric(metrics[key]) for key in METRIC_KEYS}
    if parsed != dict(expected):
        raise AnalysisInputError(
            f"aggregate report disagrees with per-item quantized means: {path}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metrics": {key: float(parsed[key]) for key in METRIC_KEYS},
        "metrics_decimal": {key: str(parsed[key]) for key in METRIC_KEYS},
    }


def _audio_manifest_binding(
    path: Path, expected_ids: Sequence[str]
) -> dict[str, Any]:
    _regular_file(path, "audio hash manifest")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        # The HARN manifest also records the basename and byte count.  These
        # fields are informational here, but the ID and digest columns are
        # mandatory and the complete row set is still checked.
        if reader.fieldnames is None or not {"id", "sha256"}.issubset(set(reader.fieldnames)):
            raise AnalysisInputError(f"audio hash manifest header is invalid: {path}")
        seen: set[str] = set()
        for row in reader:
            item_id = (row.get("id") or "").strip()
            digest = (row.get("sha256") or "").strip()
            if item_id in seen or item_id not in set(expected_ids):
                raise AnalysisInputError(f"audio hash manifest has invalid/duplicate ID: {item_id}")
            if SHA256_RE.fullmatch(digest) is None:
                raise AnalysisInputError(f"audio hash manifest has invalid digest: {item_id}")
            seen.add(item_id)
    if seen != set(expected_ids):
        raise AnalysisInputError("audio hash manifest does not contain the exact expected IDs")
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(seen)}


def validate_arm(
    arm_id: str,
    spec: Mapping[str, Any],
    *,
    base: Path,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    metrics_entry = _arm_path_spec(
        spec,
        base=base,
        keys=("per_item_metrics", "per_item_tsv", "per_item_metrics_path", "metrics_tsv"),
        label=f"{arm_id} per-item metrics",
    )
    if metrics_entry is None:
        raise AnalysisInputError(f"{arm_id} has no per-item metrics path")
    metrics_path = Path(metrics_entry["path"])
    rows, observed_metrics_hash = _parse_metric_rows(metrics_path, expected_ids)
    if observed_metrics_hash != metrics_entry["sha256"]:
        raise AnalysisInputError(f"{arm_id} per-item metrics hash changed during read")
    aggregate = _quantized_aggregate(rows)

    report_entry = _arm_path_spec(
        spec,
        base=base,
        keys=("aggregate_report", "report_path", "report", "aggregate_json"),
        label=f"{arm_id} aggregate report",
    )
    aggregate_report: dict[str, Any] | None = None
    report_payload: Mapping[str, Any] | None = None
    if report_entry is not None:
        aggregate_report = _aggregate_report_metrics(Path(report_entry["path"]), aggregate)
        if report_entry["sha256"] != aggregate_report["sha256"]:
            raise AnalysisInputError(f"{arm_id} aggregate report hash changed during read")
        report_payload = _json_file(Path(report_entry["path"]), f"{arm_id} aggregate report")

    audio_entry = _arm_path_spec(
        spec,
        base=base,
        keys=("audio_hash_manifest", "audio_sha256_manifest", "audio_manifest"),
        label=f"{arm_id} audio hash manifest",
    )
    audio_manifest = None
    if audio_entry is not None:
        audio_manifest = _audio_manifest_binding(Path(audio_entry["path"]), expected_ids)
        if audio_entry["sha256"] != audio_manifest["sha256"]:
            raise AnalysisInputError(f"{arm_id} audio manifest hash changed during read")

    if report_entry is not None:
        # The per-arm report is the provenance boundary written by the HARN.
        # If it carries nested artifact bindings, they must point at the exact
        # files just parsed; a report that merely contains matching aggregate
        # numbers is not sufficient evidence of source identity.
        assert report_payload is not None
        declared_metrics = report_payload.get("per_item_metrics")
        if isinstance(declared_metrics, Mapping):
            if declared_metrics.get("path") != str(metrics_path):
                raise AnalysisInputError(f"{arm_id} report per-item path binding mismatch")
            if declared_metrics.get("sha256") != observed_metrics_hash:
                raise AnalysisInputError(f"{arm_id} report per-item hash binding mismatch")
        declared_audio = report_payload.get("audio_manifest")
        if isinstance(declared_audio, Mapping) and audio_manifest is not None:
            if declared_audio.get("path") != audio_manifest["path"]:
                raise AnalysisInputError(f"{arm_id} report audio-manifest path mismatch")
            if declared_audio.get("sha256") != audio_manifest["sha256"]:
                raise AnalysisInputError(f"{arm_id} report audio-manifest hash mismatch")

    binding: dict[str, Any] = {
        "per_item_metrics": {
            "path": str(metrics_path),
            "sha256": observed_metrics_hash,
            "sha256_declared": metrics_entry["sha256_declared"],
        },
        "aggregate_report": aggregate_report,
        "audio_hash_manifest": audio_manifest,
    }
    if report_payload is not None:
        # Carry only bounded provenance fields into the matrix report; the
        # complete sealed-copy receipt remains owned by the HARN report.
        report_fields = {}
        for key in ("plan_id", "plan_sha256", "run_id", "contract_sha256", "tsv_sha256"):
            if key in report_payload:
                report_fields[key] = report_payload[key]
        for key in ("checkpoint", "argv", "environment"):
            if key in report_payload:
                report_fields[key] = report_payload[key]
        binding["arm_report_fields"] = report_fields
    # Preserve and verify optional scientific input bindings in the arm
    # descriptor.  The harness owns descriptor-safe opening; this layer checks
    # that the bytes named by the report are the bytes that were analyzed.
    for label, keys in (
        ("checkpoint", ("checkpoint", "checkpoint_path", "model_checkpoint")),
        ("tsv", ("tsv", "musiccaps_tsv", "evaluation_tsv")),
        ("source_code", ("source_code", "source_tree_manifest")),
    ):
        entry = _arm_path_spec(spec, base=base, keys=keys, label=f"{arm_id} {label}")
        if entry is not None:
            binding[label] = entry

    return {
        "status": "passed",
        "rows": len(rows),
        "aggregate": {key: float(value) for key, value in aggregate.items()},
        "aggregate_decimal": {key: str(value) for key, value in aggregate.items()},
        "bindings": binding,
        # Internal Decimal rows are not serialized; contrasts use this field.
        "_rows": rows,
    }


def _historical_entry(contract: Mapping[str, Any], arm_id: str) -> Mapping[str, Any] | None:
    for parent_key in ("historical_reports", "historical_sources"):
        parent = contract.get(parent_key)
        if isinstance(parent, Mapping) and isinstance(parent.get(arm_id), Mapping):
            return parent[arm_id]
    for parent_key in ("reproduction_gate", "reproduction", "historical_reproduction"):
        parent = contract.get(parent_key)
        if not isinstance(parent, Mapping):
            continue
        for child_key in ("historical_reports", "authoritative_sources", "sources"):
            child = parent.get(child_key)
            if isinstance(child, Mapping) and isinstance(child.get(arm_id), Mapping):
                return child[arm_id]
    frozen = contract.get("frozen_existing_inputs")
    if isinstance(frozen, Mapping):
        value = frozen.get(f"{arm_id}_historical_report")
        if isinstance(value, Mapping):
            return value
    return None


def _historical_vectors(contract: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if isinstance(contract.get("historical_vectors"), Mapping):
        return contract["historical_vectors"]
    for parent_key in ("reproduction_gate", "reproduction", "historical_reproduction"):
        parent = contract.get(parent_key)
        if isinstance(parent, Mapping) and isinstance(parent.get("historical_vectors"), Mapping):
            return parent["historical_vectors"]
    return None


def _audit_source_entry(contract: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("audit_source", "b1_audit_source"):
        if isinstance(contract.get(key), Mapping):
            return contract[key]
    for parent_key in ("reproduction_gate", "reproduction", "historical_reproduction"):
        parent = contract.get(parent_key)
        if isinstance(parent, Mapping):
            for key in ("audit_source", "b1_audit_source"):
                if isinstance(parent.get(key), Mapping):
                    return parent[key]
    frozen = contract.get("frozen_existing_inputs")
    if isinstance(frozen, Mapping) and isinstance(frozen.get("audit_source"), Mapping):
        return frozen["audit_source"]
    return None


def _source_metrics(payload: Mapping[str, Any], label: str) -> dict[str, Decimal]:
    if payload.get("status") != "passed":
        raise AnalysisInputError(f"{label} status is not passed")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_KEYS):
        raise AnalysisInputError(f"{label} does not contain exactly five metrics")
    return {key: _decimal(metrics[key], f"{label} {key}") for key in METRIC_KEYS}


def validate_reproduction(
    contract: Mapping[str, Any],
    current_arms: Mapping[str, Mapping[str, Any]],
    *,
    base: Path,
) -> dict[str, Any]:
    """Validate B1/B2 historical sources and classify reproduction."""

    vectors = _historical_vectors(contract)
    if not isinstance(vectors, Mapping):
        return {"verdict": "reproduction_invalid", "reason": "historical vectors missing"}
    parsed_vectors: dict[str, dict[str, Decimal]] = {}
    sources: dict[str, Any] = {}
    try:
        for arm_id in ("B1", "B2"):
            expected = vectors.get(arm_id)
            if not isinstance(expected, Mapping) or set(expected) != set(METRIC_KEYS):
                raise AnalysisInputError(f"historical vector {arm_id} is incomplete")
            parsed_vectors[arm_id] = {
                key: _decimal(expected[key], f"historical vector {arm_id} {key}")
                for key in METRIC_KEYS
            }
            entry = _historical_entry(contract, arm_id)
            if not isinstance(entry, Mapping):
                raise AnalysisInputError(f"historical source {arm_id} is missing")
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                raise AnalysisInputError(f"historical source {arm_id} path is missing")
            path = _resolve_path(raw_path, base=base, label=f"historical source {arm_id}")
            declared = entry.get("sha256")
            if not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None:
                raise AnalysisInputError(f"historical source {arm_id} hash is missing/invalid")
            observed = verify_declared_hash(path, declared, f"historical source {arm_id}")
            source_payload = _json_file(path, f"historical source {arm_id}")
            source_values = _source_metrics(source_payload, f"historical source {arm_id}")
            if source_values != parsed_vectors[arm_id]:
                raise AnalysisInputError(f"historical vector disagrees with source {arm_id}")
            sources[arm_id] = {
                "path": str(path),
                "sha256": observed,
                "metrics": {key: float(value) for key, value in source_values.items()},
                "metrics_decimal": {key: str(value) for key, value in source_values.items()},
            }

        audit_entry = _audit_source_entry(contract)
        if not isinstance(audit_entry, Mapping):
            raise AnalysisInputError("B1 audit source is missing")
        raw_audit_path = audit_entry.get("path")
        declared_audit = audit_entry.get("sha256")
        if not isinstance(raw_audit_path, str) or not isinstance(declared_audit, str):
            raise AnalysisInputError("B1 audit source path/hash is missing")
        audit_path = _resolve_path(raw_audit_path, base=base, label="B1 audit source")
        audit_hash = verify_declared_hash(audit_path, declared_audit, "B1 audit source")
        audit_payload = _json_file(audit_path, "B1 audit source")
        audit_metrics = audit_payload.get("historical_metrics")
        if not isinstance(audit_metrics, Mapping):
            raise AnalysisInputError("B1 audit source has no historical_metrics")
        audit_values = {
            key: _decimal(audit_metrics.get(key), f"B1 audit source {key}") for key in METRIC_KEYS
        }
        if audit_values != parsed_vectors["B1"]:
            raise AnalysisInputError("B1 audit/report metric fields are not exactly equal")
        sources["audit_source"] = {"path": str(audit_path), "sha256": audit_hash}
    except AnalysisInputError as exc:
        return {"verdict": "reproduction_invalid", "reason": str(exc), "sources": sources}

    unequal: list[str] = []
    for arm_id in ("B1", "B2"):
        current = current_arms.get(arm_id)
        if not current or current.get("status") != "passed":
            return {
                "verdict": "reproduction_invalid",
                "reason": f"current {arm_id} arm is invalid",
                "sources": sources,
            }
        current_values = {
            key: quantize_metric(current["aggregate_decimal"][key]) for key in METRIC_KEYS
        }
        if current_values != {key: quantize_metric(value) for key, value in parsed_vectors[arm_id].items()}:
            unequal.append(arm_id)
    verdict = "historical_repeat_failed" if unequal else "passed"
    result: dict[str, Any] = {
        "verdict": verdict,
        "sources": sources,
        "historical_vectors": {
            arm_id: {key: float(value) for key, value in values.items()}
            for arm_id, values in parsed_vectors.items()
        },
        "historical_vectors_decimal": {
            arm_id: {key: str(value) for key, value in values.items()}
            for arm_id, values in parsed_vectors.items()
        },
    }
    if unequal:
        result["unequal_arms"] = unequal
    return result


def paired_bootstrap(
    delta: Sequence[float],
    *,
    replicates: int = 10000,
    seed: int = 20260828,
) -> tuple[float, float]:
    """Return a deterministic paired percentile 95% bootstrap interval."""

    if replicates <= 0:
        raise AnalysisInputError("bootstrap replicates must be positive")
    values = np.asarray([_finite_float(item, "bootstrap delta") for item in delta], dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise AnalysisInputError("bootstrap delta must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    # Keep peak memory bounded while consuming the generator in deterministic
    # row-major order.  This is equivalent to drawing one n-sized index vector
    # per replicate and is practical for 5521 IDs × 10000 replicates.
    chunk = 256
    cursor = 0
    while cursor < replicates:
        size = min(chunk, replicates - cursor)
        indices = rng.integers(0, values.size, size=(size, values.size))
        means[cursor : cursor + size] = values[indices].mean(axis=1)
        cursor += size
    interval = np.percentile(means, [2.5, 97.5], method="linear")
    return _finite_float(interval[0], "bootstrap CI low"), _finite_float(interval[1], "bootstrap CI high")


def classify_contrast(
    mean_delta: Any,
    ci95_low: Any,
    ci95_high: Any,
    practical_threshold: Any,
) -> dict[str, Any]:
    """Classify one contrast; invalid evidence has no directional label."""

    try:
        mean = _finite_float(mean_delta, "contrast mean")
        low = _finite_float(ci95_low, "contrast CI low")
        high = _finite_float(ci95_high, "contrast CI high")
        threshold = _finite_float(practical_threshold, "contrast threshold")
    except AnalysisInputError as exc:
        return {
            "classification": "invalid",
            "reason": str(exc),
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "practical_threshold": None,
        }
    if threshold < 0 or low > high:
        return {
            "classification": "invalid",
            "reason": "invalid threshold or CI ordering",
            "mean_delta": mean,
            "ci95_low": low,
            "ci95_high": high,
            "practical_threshold": threshold,
        }
    positive = mean >= threshold and low > 0
    negative = mean <= -threshold and high < 0
    if positive and negative:
        # This is mathematically impossible for a non-negative threshold, but
        # retaining an explicit guard prevents a future rule edit from making
        # contradictory claims.
        classification = "invalid"
        reason = "contradictory positive and negative support"
    elif positive:
        classification, reason = "positive_supported", None
    elif negative:
        classification, reason = "negative_supported", None
    else:
        classification, reason = "small_or_uncertain", None
    result: dict[str, Any] = {
        "classification": classification,
        "mean_delta": mean,
        "ci95_low": low,
        "ci95_high": high,
        "practical_threshold": threshold,
    }
    if reason is not None:
        result["reason"] = reason
    return result


CONTRASTS: tuple[tuple[str, tuple[tuple[str, int], ...], float], ...] = (
    ("Q_inference_fulltrack", (("B1", 1), ("B4", -1)), 0.05),
    ("Q_inference_segment_slot0", (("B2", 1), ("B5", -1)), 0.05),
    ("checkpoint_family_q9", (("B1", 1), ("B2", -1)), 0.15),
    ("checkpoint_family_q0", (("B4", 1), ("B5", -1)), 0.15),
    ("checkpoint_family_noq", (("B3", 1), ("B6", -1)), 0.15),
    (
        "Q_by_family_interaction",
        (("B1", 1), ("B4", -1), ("B2", -1), ("B5", 1)),
        0.05,
    ),
)


def _contrast_values(
    arms: Mapping[str, Mapping[str, Any]],
    terms: Sequence[tuple[str, int]],
    metric: str,
    expected_ids: Sequence[str],
) -> list[float]:
    rows = [arms[arm_id]["_rows"] for arm_id, _ in terms]
    values: list[float] = []
    for item_id in expected_ids:
        delta = 0.0
        for row, (_, sign) in zip(rows, terms):
            delta += sign * float(row[item_id][metric])
        values.append(_finite_float(delta, f"{metric} contrast delta"))
    return values


def compute_contrasts(
    arms: Mapping[str, Mapping[str, Any]],
    expected_ids: Sequence[str],
    *,
    replicates: int = 10000,
    seed: int = 20260828,
) -> dict[str, dict[str, Any]]:
    """Compute all six predeclared contrasts for all five metrics."""

    result: dict[str, dict[str, Any]] = {}
    for contrast_index, (name, terms, pq_threshold) in enumerate(CONTRASTS):
        per_metric: dict[str, Any] = {}
        # Use a deterministic independent stream per contrast while preserving
        # the one declared base seed in the output.
        contrast_seed = int(np.random.SeedSequence([seed, contrast_index]).generate_state(1)[0])
        for metric in METRIC_KEYS:
            delta = _contrast_values(arms, terms, metric, expected_ids)
            mean_delta = _finite_float(np.mean(np.asarray(delta, dtype=np.float64)), "contrast mean")
            low, high = paired_bootstrap(delta, replicates=replicates, seed=contrast_seed)
            threshold = pq_threshold if metric == "aes_PQ" else 0.0
            classified = classify_contrast(mean_delta, low, high, threshold)
            classified["n"] = len(delta)
            classified["bootstrap_replicates"] = replicates
            classified["bootstrap_seed"] = seed
            classified["contrast_seed"] = contrast_seed
            per_metric[metric] = classified
        result[name] = {
            "terms": [{"arm": arm_id, "sign": sign} for arm_id, sign in terms],
            "metrics": per_metric,
        }
    return result


def _pq_classification(contrasts: Mapping[str, Any], name: str) -> str:
    try:
        return str(contrasts[name]["metrics"]["aes_PQ"]["classification"])
    except (KeyError, TypeError):
        return "invalid"


def _family_decision(contrasts: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    labels = [_pq_classification(contrasts, name) for name in names]
    if any(label == "invalid" for label in labels):
        return {"status": "invalid", "contrast_classifications": labels}
    supported = {label for label in labels if label in {"positive_supported", "negative_supported"}}
    if supported == {"positive_supported"}:
        status = "positive_replicated"
    elif supported == {"negative_supported"}:
        status = "negative_replicated"
    elif len(supported) > 1:
        status = "heterogeneous_opposite"
    elif all(label == "small_or_uncertain" for label in labels):
        status = "no_practically_supported_direction"
    else:
        status = "family_specific_or_insufficient"
    return {"status": status, "contrast_classifications": labels}


def classify_decisions(
    arms: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the exhaustive decision table with invalid precedence."""

    reproduction_verdict = reproduction.get("verdict")
    if reproduction_verdict == "reproduction_invalid":
        reproduction_decision = "hold_invalid_reproduction"
    elif reproduction_verdict == "historical_repeat_failed":
        reproduction_decision = "hold_historical_repeat_failed"
    elif reproduction_verdict == "passed":
        reproduction_decision = "proceed_to_B3_B6"
    else:
        reproduction_decision = "hold_invalid_reproduction"

    q_decision = _family_decision(
        contrasts, ("Q_inference_fulltrack", "Q_inference_segment_slot0")
    )
    family_decision = _family_decision(
        contrasts, ("checkpoint_family_q9", "checkpoint_family_q0", "checkpoint_family_noq")
    )

    canonical_qualifiers: list[str] = []
    secondary_qualifiers: list[str] = []
    target_arms = ("B2", "B5", "B6")
    if reproduction_verdict in {"reproduction_invalid", "historical_repeat_failed"}:
        target = {
            "status": "held_by_reproduction",
            "qualifying_canonical_arms": [],
            "qualifying_secondary_arms": [],
        }
    else:
        for arm_id in target_arms:
            arm = arms.get(arm_id)
            if not arm or arm.get("status") != "passed":
                continue
            try:
                pq = _decimal(arm["aggregate_decimal"]["aes_PQ"], f"{arm_id} PQ")
            except (KeyError, AnalysisInputError):
                continue
            if pq >= TARGET_PQ:
                if arm_id in {"B2", "B6"}:
                    canonical_qualifiers.append(arm_id)
                elif arm_id == "B5":
                    secondary_qualifiers.append(arm_id)
        if canonical_qualifiers:
            target = {
                "status": "canonical_non_fulltrack_pq_ge_6_9_achieved",
                "qualifying_canonical_arms": canonical_qualifiers,
                "qualifying_secondary_arms": secondary_qualifiers,
            }
        elif secondary_qualifiers:
            target = {
                "status": "secondary_q0_non_fulltrack_pq_ge_6_9_only",
                "qualifying_canonical_arms": [],
                "qualifying_secondary_arms": secondary_qualifiers,
            }
        elif all(arms.get(arm_id, {}).get("status") == "passed" for arm_id in target_arms):
            target = {
                "status": "target_open_all_valid_below_6_9",
                "qualifying_canonical_arms": [],
                "qualifying_secondary_arms": [],
            }
        else:
            target = {
                "status": "invalid_or_incomplete",
                "qualifying_canonical_arms": [],
                "qualifying_secondary_arms": [],
            }

    # Explicit consistency guard: an invalid required contrast cannot coexist
    # with a directional family claim, and an achieved target must name only
    # valid arms.  This prevents future decision-table edits from emitting two
    # incompatible conclusions for the same scope.
    if q_decision["status"] == "invalid" and q_decision.get("status") in {
        "positive_replicated",
        "negative_replicated",
    }:
        raise AnalysisInputError("contradictory Q decision")
    if target["status"] == "canonical_non_fulltrack_pq_ge_6_9_achieved":
        if any(arms[arm_id].get("status") != "passed" for arm_id in canonical_qualifiers):
            raise AnalysisInputError("target decision names an invalid canonical arm")

    return {
        "reproduction": {"status": reproduction_decision, "verdict": reproduction_verdict},
        "q_inference_association": q_decision,
        "checkpoint_family_association": family_decision,
        "non_fulltrack_target": target,
        "causal_follow_up": {
            "status": (
                "prepare_separate_matched_training_replication"
                if family_decision["status"] in {"positive_replicated", "negative_replicated"}
                else "none"
            ),
            "auto_launch": False,
        },
    }


def _serialize_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arm.items() if not key.startswith("_")}


def build_report(
    contract_path: Path,
    *,
    replicates: int = 10000,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Build a JSON-serializable report; invalid evidence is represented explicitly."""

    _regular_file(contract_path, "analysis contract")
    contract = _json_file(contract_path, "analysis contract")
    base = contract_path.parent
    contract_hash = sha256_file(contract_path)
    raw_count = contract.get("protocol", {}).get("unique_ids", contract.get("unique_ids", DEFAULT_COUNT))
    try:
        expected_count = int(raw_count)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError("contract unique ID count is invalid") from exc
    if expected_count <= 0:
        raise AnalysisInputError("contract unique ID count must be positive")
    ids = _expected_ids(contract, base=base, expected_count=expected_count)
    specs = _arm_specs(contract)
    result_root = _result_root(contract, base=base)
    if result_root is not None:
        # A fixed result root is an output binding, not a mutable discovery
        # operation.  Fill only absent artifact paths; explicit contract paths
        # always remain authoritative and are validated below.
        for arm_id in ARM_IDS:
            if arm_id not in specs:
                specs[arm_id] = {}
            spec = dict(specs[arm_id])
            spec.setdefault(
                "per_item_metrics",
                str(result_root / arm_id / "metrics" / "per_item.tsv"),
            )
            spec.setdefault("aggregate_report", str(result_root / arm_id / "report.json"))
            spec.setdefault(
                "audio_hash_manifest",
                str(result_root / arm_id / "manifests" / "audio_sha256.tsv"),
            )
            specs[arm_id] = spec
    arms: dict[str, dict[str, Any]] = {}
    arm_errors: dict[str, str] = {}
    for arm_id in ARM_IDS:
        spec = specs.get(arm_id)
        if spec is None:
            arms[arm_id] = {"status": "invalid", "error": "arm descriptor missing"}
            arm_errors[arm_id] = "arm descriptor missing"
            continue
        try:
            arms[arm_id] = validate_arm(arm_id, spec, base=base, expected_ids=ids)
        except AnalysisInputError as exc:
            arms[arm_id] = {"status": "invalid", "error": str(exc)}
            arm_errors[arm_id] = str(exc)

    current_valid = not arm_errors
    reproduction = validate_reproduction(contract, arms, base=base)
    contrasts: dict[str, Any]
    if current_valid and reproduction.get("verdict") != "reproduction_invalid":
        contrasts = compute_contrasts(arms, ids, replicates=replicates, seed=seed)
    else:
        contrasts = {}
    decisions = classify_decisions(arms, contrasts, reproduction)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if current_valid and reproduction.get("verdict") != "reproduction_invalid" else "invalid",
        "analysis_id": "FTQ3-BMATRIX-v1",
        "contract": {"path": str(contract_path), "sha256": contract_hash},
        "source_bindings": {
            "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "contract": {"path": str(contract_path), "sha256": contract_hash},
            "environment": contract.get("environment_observed_for_plan", contract.get("environment")),
        },
        "protocol": contract.get("protocol", {}),
        "expected_ids": {"count": len(ids), "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "arms": {arm_id: _serialize_arm(arms[arm_id]) for arm_id in ARM_IDS},
        "reproduction": reproduction,
        "contrasts": contrasts,
        "decisions": decisions,
        "bootstrap": {"replicates": replicates, "seed": seed, "interval": "paired percentile 95%"},
        "invalid_arms": arm_errors,
        "prohibited_claims": [
            "historical_byte_level_reproduction",
            "fulltrack_causal_advantage",
            "caption_granularity_is_the_cause",
            "non_fulltrack_pq_over_6_9_achieved",
            "queue_registration_or_launch_authorized",
        ],
    }
    # Keep the report self-contained but never serialize all per-item rows.
    return report


def build_reproduction_report(contract_path: Path) -> dict[str, Any]:
    """Validate only B1/B2 for the HARN's phase transition gate.

    B3--B6 are intentionally not inspected here: the HARN must be able to
    decide whether to enter phase 2 before those arm directories exist.  The
    top-level ``decision`` is retained for compatibility with the controller's
    phase gate; the richer ``reproduction.verdict`` remains the authoritative
    scientific label.
    """

    _regular_file(contract_path, "analysis contract")
    contract = _json_file(contract_path, "analysis contract")
    base = contract_path.parent
    contract_hash = sha256_file(contract_path)
    raw_count = contract.get("protocol", {}).get("unique_ids", contract.get("unique_ids", DEFAULT_COUNT))
    try:
        expected_count = int(raw_count)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError("contract unique ID count is invalid") from exc
    ids = _expected_ids(contract, base=base, expected_count=expected_count)
    specs = _arm_specs(contract)
    result_root = _result_root(contract, base=base)
    if result_root is not None:
        for arm_id in ("B1", "B2"):
            spec = dict(specs.get(arm_id, {}))
            spec.setdefault("per_item_metrics", str(result_root / arm_id / "metrics" / "per_item.tsv"))
            spec.setdefault("aggregate_report", str(result_root / arm_id / "report.json"))
            spec.setdefault("audio_hash_manifest", str(result_root / arm_id / "manifests" / "audio_sha256.tsv"))
            specs[arm_id] = spec
    arms: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for arm_id in ("B1", "B2"):
        try:
            arms[arm_id] = validate_arm(arm_id, specs[arm_id], base=base, expected_ids=ids)
        except (KeyError, AnalysisInputError) as exc:
            arms[arm_id] = {"status": "invalid", "error": str(exc)}
            errors[arm_id] = str(exc)
    reproduction = validate_reproduction(contract, arms, base=base)
    verdict = reproduction.get("verdict", "reproduction_invalid")
    return {
        "schema_version": 1,
        "status": "passed" if verdict == "passed" else "invalid",
        "decision": "passed" if verdict == "passed" else verdict,
        "analysis_id": "FTQ3-BMATRIX-v1-reproduction-gate",
        "contract": {"path": str(contract_path), "sha256": contract_hash},
        "expected_ids": {"count": len(ids), "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "arms": {arm_id: _serialize_arm(arms[arm_id]) for arm_id in ("B1", "B2")},
        "reproduction": reproduction,
        "invalid_arms": errors,
        "prohibited_claims": [
            "historical_byte_level_reproduction",
            "fulltrack_causal_advantage",
            "non_fulltrack_pq_over_6_9_achieved",
        ],
    }


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AnalysisInputError(f"report output already exists; refusing stale overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reproduction-only",
        action="store_true",
        help="validate only B1/B2 and emit the phase-2 reproduction gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise AnalysisInputError("bootstrap replicates must be positive")
    if args.reproduction_only:
        report = build_reproduction_report(args.contract)
    else:
        report = build_report(
            args.contract,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
    _atomic_json_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisInputError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(2)
