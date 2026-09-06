#!/usr/bin/env python3
"""Score a complete MusicCaps generation directory item by item.

This is the scientific scorer for the Fulltrack-Q3 matrix.  It deliberately
has a narrower contract than the historical aggregate evaluator:

* the TSV and generated directory must contain exactly the same safe IDs;
* every audio file is checked before a model is called;
* CLAP and Audiobox scores are persisted for every ID, in TSV order;
* local model paths are required and the Audiobox loader is always invoked
  with ``local_files_only=True``.

The expensive model loaders are normal module functions so CPU-only tests can
replace them with deterministic fakes.  No model is imported at module import
time.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRIC_KEYS = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")
AES_KEYS = ("CE", "CU", "PC", "PQ")
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_EXPECTED_COUNT = 5521


class ScoringInputError(RuntimeError):
    """Raised when a scientific input or model result fails closed."""


@dataclass(frozen=True)
class MusicCapsRecord:
    id: str
    caption: str


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ScoringInputError(f"{label} must be a regular non-symlink file: {path}")


def _directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ScoringInputError(f"{label} must be a regular non-symlink directory: {path}")


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScoringInputError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ScoringInputError(f"{label} is non-finite: {value!r}")
    return result


def read_musiccaps_tsv(path: Path, expected_count: int | None = DEFAULT_EXPECTED_COUNT) -> list[MusicCapsRecord]:
    """Read and validate the frozen MusicCaps ID/caption table."""

    _regular_file(path, "MusicCaps TSV")
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ScoringInputError(f"cannot open MusicCaps TSV {path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        if fields is None or "id" not in fields or "caption" not in fields:
            raise ScoringInputError("MusicCaps TSV must contain id and caption columns")
        if len(fields) != len(set(fields)):
            raise ScoringInputError("MusicCaps TSV has duplicate column names")

        records: list[MusicCapsRecord] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ScoringInputError(f"malformed TSV row {row_number}")
            item_id = (row.get("id") or "").strip()
            if not item_id or ID_RE.fullmatch(item_id) is None:
                raise ScoringInputError(f"unsafe or empty MusicCaps ID at row {row_number}: {item_id!r}")
            if item_id in seen:
                raise ScoringInputError(f"duplicate MusicCaps ID: {item_id}")
            caption = row.get("caption")
            if caption is None:
                raise ScoringInputError(f"missing caption at row {row_number}")
            seen.add(item_id)
            records.append(MusicCapsRecord(item_id, caption))

    if expected_count is not None and len(records) != expected_count:
        raise ScoringInputError(
            f"MusicCaps row count {len(records)} != required exact count {expected_count}"
        )
    if not records:
        raise ScoringInputError("MusicCaps TSV is empty")
    return records


def validate_audio_directory(audio_dir: Path, records: Sequence[MusicCapsRecord]) -> dict[str, Path]:
    """Return the exact ID-to-FLAC mapping after validating the directory."""

    _directory(audio_dir, "audio directory")
    expected = {record.id for record in records}
    actual: dict[str, Path] = {}
    try:
        entries = list(audio_dir.iterdir())
    except OSError as exc:
        raise ScoringInputError(f"cannot enumerate audio directory {audio_dir}: {exc}") from exc

    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ScoringInputError(f"audio directory contains non-regular entry: {entry.name}")
        if entry.suffix != ".flac":
            raise ScoringInputError(f"audio directory contains unexpected file: {entry.name}")
        item_id = entry.stem
        if ID_RE.fullmatch(item_id) is None or item_id not in expected:
            raise ScoringInputError(f"audio file is not an expected safe ID: {entry.name}")
        if item_id in actual:
            raise ScoringInputError(f"duplicate audio ID: {item_id}")
        if entry.stat().st_size <= 0:
            raise ScoringInputError(f"audio file is empty: {entry}")
        actual[item_id] = entry

    missing = sorted(expected - set(actual))
    if missing:
        raise ScoringInputError(f"missing audio IDs (first 5): {missing[:5]}")
    if len(actual) != len(expected):
        raise ScoringInputError(
            f"audio count {len(actual)} != expected ID count {len(expected)}"
        )

    # Validate the format and all samples before invoking either scorer.  The
    # import is lazy so parser and mock tests stay independent of torchaudio.
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - runtime environment issue
        raise ScoringInputError("soundfile and numpy are required for audio validation") from exc

    for item_id in sorted(actual):
        path = actual[item_id]
        try:
            info = sf.info(str(path))
            if info.frames <= 0 or info.samplerate != 16000 or info.channels != 1:
                raise ScoringInputError(
                    f"audio format for {item_id} must be non-empty mono 16 kHz; "
                    f"got frames={info.frames}, rate={info.samplerate}, channels={info.channels}"
                )
            samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        except ScoringInputError:
            raise
        except Exception as exc:
            raise ScoringInputError(f"cannot read audio {path}: {exc}") from exc
        if sample_rate != 16000 or np.asarray(samples).size == 0:
            raise ScoringInputError(f"audio has invalid sample data: {path}")
        if not np.isfinite(np.asarray(samples)).all():
            raise ScoringInputError(f"audio contains non-finite samples: {path}")

    return {record.id: actual[record.id] for record in records}


def _resolve_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ScoringInputError(f"unsupported device: {requested}")
    if requested != "auto":
        if requested == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ScoringInputError("CUDA was explicitly requested but is unavailable")
        if requested == "mps":
            import torch

            if not torch.backends.mps.is_available():
                raise ScoringInputError("MPS was explicitly requested but is unavailable")
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _validate_local_model_inputs(clap_checkpoint: Path, audiobox_snapshot: Path) -> None:
    _regular_file(clap_checkpoint, "CLAP checkpoint")
    _directory(audiobox_snapshot, "Audiobox snapshot")
    required = (audiobox_snapshot / "config.json", audiobox_snapshot / "model.safetensors")
    for path in required:
        _regular_file(path, f"Audiobox snapshot file {path.name}")


def load_clap_model(checkpoint: Path, *, device: str, local_files_only: bool) -> Any:
    """Load the pinned local CLAP model.

    ``CLAP_Module`` itself constructs a tokenizer from its local Hugging Face
    cache.  Offline environment flags plus a direct checkpoint path ensure a
    missing dependency fails instead of downloading during an evaluation.
    """

    if not local_files_only:
        raise ScoringInputError("local-only scoring is mandatory")
    _regular_file(checkpoint, "CLAP checkpoint")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import laion_clap
    except ImportError as exc:  # pragma: no cover - runtime environment issue
        raise ScoringInputError("laion_clap is required for CLAP scoring") from exc
    try:
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device, amodel="HTSAT-base")
        model.load_ckpt(str(checkpoint), verbose=False)
        model.eval()
        return model
    except Exception as exc:
        raise ScoringInputError(f"failed to load local CLAP model: {exc}") from exc


class LocalAesPredictor:
    """Audiobox predictor whose model is loaded from a pinned local snapshot."""

    def __init__(self, snapshot: Path, *, device: str, batch_size: int = 16) -> None:
        if batch_size <= 0:
            raise ScoringInputError("AES batch size must be positive")
        _directory(snapshot, "Audiobox snapshot")
        self.batch_size = batch_size
        self.sample_rate = 16000
        self.device_name = device
        try:
            import torch
            from audiobox_aesthetics.infer import AXES_NAME, make_inference_batch
            from audiobox_aesthetics.model.aes import AesMultiOutput, Normalize
        except ImportError as exc:  # pragma: no cover - runtime environment issue
            raise ScoringInputError("audiobox_aesthetics is required for AES scoring") from exc

        self._torch = torch
        self._axes = tuple(AXES_NAME)
        self._make_inference_batch = make_inference_batch
        try:
            self.device = torch.device(device)
            # Path input and local_files_only are intentional: a model ID is
            # not accepted, and no hub fallback is possible.
            self.model = AesMultiOutput.from_pretrained(
                str(snapshot), local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise ScoringInputError(f"failed to load local Audiobox model: {exc}") from exc
        self._target_transform = {
            axis: Normalize(
                mean=self.model.target_transform[axis]["mean"],
                std=self.model.target_transform[axis]["std"],
            )
            for axis in self._axes
        }

    def _load_audio(self, path: Path):
        import numpy as np
        import soundfile as sf

        wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        # soundfile returns [T, C], while Audiobox expects [C, T].
        tensor = self._torch.from_numpy(np.asarray(wav).T.copy())
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            import torchaudio

            tensor = torchaudio.functional.resample(
                tensor, orig_freq=sample_rate, new_freq=self.sample_rate
            )
        return tensor

    def forward(self, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
        """Score path dictionaries in the same shape as AesPredictor.forward."""

        if not batch:
            return []
        torch = self._torch
        wavs = [self._load_audio(Path(str(item["path"]))) for item in batch]
        padded, masks, weights, bids = self._make_inference_batch(
            wavs, 10, 10, sample_rate=self.sample_rate
        )
        wavs_tensor = torch.stack(padded).to(self.device)
        masks_tensor = torch.stack(masks).to(self.device)
        weights_tensor = torch.tensor(weights, device=self.device)
        bids_tensor = torch.tensor(bids, device=self.device)
        with torch.inference_mode():
            outputs = self.model({"wav": wavs_tensor, "mask": masks_tensor})
        results: list[dict[str, float]] = []
        for item_index in range(len(batch)):
            result: dict[str, float] = {}
            selected = bids_tensor == item_index
            for axis in AES_KEYS:
                if axis not in outputs or axis not in self._target_transform:
                    raise ScoringInputError(f"Audiobox output is missing axis {axis}")
                values = self._target_transform[axis].inverse(outputs[axis])
                value = (values[selected] * weights_tensor[selected]).sum() / weights_tensor[selected].sum()
                result[axis] = _finite_float(value.item(), f"AES {axis}")
            results.append(result)
        return results


def load_aes_predictor(snapshot: Path, *, device: str, batch_size: int = 16) -> LocalAesPredictor:
    return LocalAesPredictor(snapshot, device=device, batch_size=batch_size)


def _as_rows(value: Any) -> list[list[float]]:
    """Convert a tensor/array-like batch to finite Python rows."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        rows = value.tolist()
    except AttributeError:
        rows = value
    if not isinstance(rows, list) or (rows and not isinstance(rows[0], (list, tuple))):
        raise ScoringInputError("model embedding output must be a 2-D batch")
    converted: list[list[float]] = []
    for row_index, row in enumerate(rows):
        converted.append(
            [_finite_float(component, f"embedding row {row_index}") for component in row]
        )
    return converted


def _cosine(left: Sequence[float], right: Sequence[float], label: str) -> float:
    if len(left) == 0 or len(left) != len(right):
        raise ScoringInputError(f"embedding dimensions do not match for {label}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ScoringInputError(f"zero-norm embedding for {label}")
    return _finite_float(dot / (left_norm * right_norm), f"CLAP {label}")


def _clap_batch(model: Any, paths: Sequence[Path], captions: Sequence[str]) -> list[float]:
    try:
        audio = model.get_audio_embedding_from_filelist(
            [str(path) for path in paths], use_tensor=True
        )
        text = model.get_text_embedding(list(captions), use_tensor=True)
    except Exception as exc:
        raise ScoringInputError(f"CLAP scoring failed: {exc}") from exc
    audio_rows = _as_rows(audio)
    text_rows = _as_rows(text)
    if len(audio_rows) != len(paths) or len(text_rows) != len(paths):
        raise ScoringInputError(
            f"CLAP returned {len(audio_rows)} audio/{len(text_rows)} text rows for {len(paths)} IDs"
        )
    return [_cosine(a, t, str(index)) for index, (a, t) in enumerate(zip(audio_rows, text_rows))]


def _aes_batch(predictor: Any, paths: Sequence[Path]) -> list[dict[str, float]]:
    try:
        result = predictor.forward([{"path": str(path)} for path in paths])
    except Exception as exc:
        raise ScoringInputError(f"Audiobox scoring failed: {exc}") from exc
    if not isinstance(result, list) or len(result) != len(paths):
        raise ScoringInputError(
            f"Audiobox returned {len(result) if isinstance(result, list) else 'non-list'} "
            f"rows for {len(paths)} IDs"
        )
    normalized: list[dict[str, float]] = []
    for index, row in enumerate(result):
        if not isinstance(row, Mapping):
            raise ScoringInputError(f"Audiobox row {index} is not a mapping")
        current: dict[str, float] = {}
        for axis in AES_KEYS:
            key = axis if axis in row else f"aes_{axis}"
            if key not in row:
                raise ScoringInputError(f"Audiobox row {index} is missing {axis}")
            current[axis] = _finite_float(row[key], f"AES {axis} row {index}")
        normalized.append(current)
    return normalized


def score_records(
    records: Sequence[MusicCapsRecord],
    audio_by_id: Mapping[str, Path],
    clap_model: Any,
    aes_predictor: Any,
    *,
    batch_size: int = 16,
    on_batch: Any | None = None,
) -> list[dict[str, float | str]]:
    """Score all records with strict one-row-in/one-row-out semantics."""

    if batch_size <= 0:
        raise ScoringInputError("batch size must be positive")
    rows: list[dict[str, float | str]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        paths = [audio_by_id[record.id] for record in batch_records]
        clap_values = _clap_batch(clap_model, paths, [record.caption for record in batch_records])
        aes_values = _aes_batch(aes_predictor, paths)
        if len(clap_values) != len(batch_records) or len(aes_values) != len(batch_records):
            raise ScoringInputError("model batch cardinality mismatch")
        for record, clap, aes in zip(batch_records, clap_values, aes_values):
            row: dict[str, float | str] = {"id": record.id, "clap_score": clap}
            row.update({f"aes_{axis}": aes[axis] for axis in AES_KEYS})
            # Validate every value again at the persistence boundary.
            for key in METRIC_KEYS:
                row[key] = _finite_float(row[key], f"{record.id} {key}")
            rows.append(row)
        if on_batch is not None:
            on_batch(rows[-len(batch_records) :])
    if len(rows) != len(records) or [row["id"] for row in rows] != [r.id for r in records]:
        raise ScoringInputError("scored row IDs do not exactly preserve TSV order")
    return rows


def _open_metrics_stream(path: Path):
    """Open the final metrics path exclusively and return a CSV writer.

    The HARN uses the persisted row count as the scoring heartbeat.  Therefore
    this file intentionally becomes visible with its header before model
    scoring starts and receives one flushed/fsynced batch at a time.  A failed
    scorer leaves a partial file as evidence; the arm is then quarantined and
    can never be resumed in place.
    """

    if path.exists() or path.is_symlink():
        raise ScoringInputError(f"metrics output already exists; refusing stale overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
    except OSError as exc:
        raise ScoringInputError(f"cannot create metrics output exclusively: {path}: {exc}") from exc
    handle = os.fdopen(fd, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=("id",) + METRIC_KEYS,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    handle.flush()
    os.fsync(handle.fileno())
    return handle, writer


def _write_metric_batch(handle: Any, writer: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        writer.writerow(
            {
                "id": row["id"],
                **{key: format(float(row[key]), ".17g") for key in METRIC_KEYS},
            }
        )
    handle.flush()
    os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise ScoringInputError(f"metrics output already exists; refusing stale overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("id",) + METRIC_KEYS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        **{key: format(float(row[key]), ".17g") for key in METRIC_KEYS},
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ScoringInputError("cannot aggregate zero scored rows")
    return {
        key: _finite_float(sum(float(row[key]) for row in rows) / len(rows), f"aggregate {key}")
        for key in METRIC_KEYS
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clap-checkpoint", type=Path, required=True)
    parser.add_argument("--audiobox-snapshot", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--require-exact-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--aggregate-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.local_files_only:
        raise ScoringInputError("--local-files-only is mandatory for the scientific scorer")
    if args.require_exact_count <= 0:
        raise ScoringInputError("--require-exact-count must be positive")
    if args.batch_size <= 0:
        raise ScoringInputError("--batch-size must be positive")

    _validate_local_model_inputs(args.clap_checkpoint, args.audiobox_snapshot)
    records = read_musiccaps_tsv(args.tsv, args.require_exact_count)
    audio_by_id = validate_audio_directory(args.audio_dir, records)
    device = _resolve_device(args.device)
    clap_model = load_clap_model(
        args.clap_checkpoint, device=device, local_files_only=args.local_files_only
    )
    aes_predictor = load_aes_predictor(
        args.audiobox_snapshot, device=device, batch_size=args.batch_size
    )
    try:
        metrics_handle, metrics_writer = _open_metrics_stream(args.out)
    except ScoringInputError:
        raise
    rows: list[dict[str, float | str]] = []
    try:
        rows = score_records(
            records,
            audio_by_id,
            clap_model,
            aes_predictor,
            batch_size=args.batch_size,
            on_batch=lambda batch: _write_metric_batch(metrics_handle, metrics_writer, batch),
        )
    finally:
        metrics_handle.close()
    aggregate = _aggregate(rows)
    if args.aggregate_json is not None:
        if args.aggregate_json.exists() or args.aggregate_json.is_symlink():
            raise ScoringInputError(
                f"aggregate output already exists; refusing stale overwrite: {args.aggregate_json}"
            )
        args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "status": "passed",
            "rows": len(rows),
            "metrics": aggregate,
            "per_item_tsv": str(args.out),
            "per_item_tsv_sha256": _sha256(args.out),
        }
        args.aggregate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": len(rows),
                "metrics": aggregate,
                "device": device,
                "per_item_tsv": str(args.out),
                "per_item_tsv_sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScoringInputError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(2)
