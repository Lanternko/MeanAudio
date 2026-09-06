#!/usr/bin/env python3
"""Fail-closed structural, identity, audio, hash, and semantic cache audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase8_qwen_probe_lib import (  # noqa: E402
    DEFAULT_SCHEMA,
    EXPECTED_ROWS,
    ContractError,
    compare_arrays_exact,
    load_npz,
    read_cache_list,
    read_tsv,
    scalar_int,
    scalar_text,
    sha256_file,
    validate_row_cache_alignment,
    validate_schema,
    write_json_atomic,
)


DEFAULT_BASE_NPZ = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz")
DEFAULT_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train.tsv")
DEFAULT_CACHE = Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt")
DEFAULT_CLAP_CHECKPOINT = Path(
    "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


def semantic_gate(
    *,
    captions: Sequence[str],
    cached_text_c: np.ndarray,
    clap_checkpoint: Path,
    seed: int,
    audio_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Compare deterministic matched and shuffled CLAP scores.

    With ``audio_paths`` this is audio-to-cached-text CLAP.  Without audio,
    it is a text-to-cached-text CLAP sanity check, which still catches row
    permutation and wrong-caption encoding while avoiding an audio overlay.
    """
    if not clap_checkpoint.is_file():
        raise ContractError(f"CLAP checkpoint missing for semantic gate: {clap_checkpoint}")
    import torch
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(clap_checkpoint), verbose=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.inference_mode():
        if audio_paths is not None:
            audio = model.get_audio_embedding_from_filelist(
                [str(path) for path in audio_paths], use_tensor=True
            )
            audio_np = audio.detach().float().cpu().numpy()
            left = audio_np
            method = "audio_to_cached_text"
        else:
            encoded = model.get_text_embedding(list(captions), use_tensor=True)
            left = encoded.detach().float().cpu().numpy()
            method = "text_to_cached_text"
    matched = _cosine_rows(left, cached_text_c)
    shuffled_indices = list(range(len(captions)))
    random.Random(seed).shuffle(shuffled_indices)
    shuffled = _cosine_rows(left, cached_text_c[np.asarray(shuffled_indices)])
    matched_mean = float(np.mean(matched))
    shuffled_mean = float(np.mean(shuffled))
    margin = matched_mean - shuffled_mean
    passed = bool(matched_mean > 0.0 and margin > 0.01)
    if not passed:
        raise ContractError(
            f"semantic matched-vs-shuffled CLAP gate failed: matched={matched_mean:.6f}, "
            f"shuffled={shuffled_mean:.6f}, margin={margin:.6f}"
        )
    return {
        "status": "passed",
        "method": method,
        "seed": seed,
        "sample_count": len(captions),
        "matched_mean": matched_mean,
        "shuffled_mean": shuffled_mean,
        "margin": margin,
        "minimum_margin": 0.01,
    }


def audit(
    *,
    tsv: Path,
    cache_list: Path,
    npz_dir: Path,
    reference_npz_dir: Path | None,
    limit: int | None,
    clap_checkpoint: Path | None,
    audio_dir: Path | None,
    seed: int,
    skip_semantic: bool,
    allow_test_rows: bool = False,
) -> dict[str, Any]:
    rows = read_tsv(tsv)
    names = read_cache_list(cache_list)
    validate_row_cache_alignment(rows, names)
    if len(rows) != EXPECTED_ROWS and not allow_test_rows:
        raise ContractError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if not npz_dir.is_dir():
        raise ContractError(f"NPZ directory missing: {npz_dir}")
    sample_count = len(rows) if limit is None else min(limit, len(rows))
    if sample_count > 512 and limit is not None:
        raise ContractError("sample audit limit must be <=512")

    checks: dict[str, Any] = {
        "rows": len(rows),
        "cache_names": len(names),
        "sample_count": sample_count,
        "npz_dir": str(npz_dir),
        "reference_npz_dir": str(reference_npz_dir) if reference_npz_dir else None,
    }
    semantic_indices = set(
        int(value) for value in np.linspace(0, sample_count - 1, min(512, sample_count), dtype=int)
    )
    cached_c: list[np.ndarray] = []
    captions: list[str] = []
    semantic_rows: list[dict[str, str]] = []
    content_digest = hashlib.sha256()
    seen_ids: set[str] = set()
    for index in range(sample_count):
        row = rows[index]
        name = names[index]
        path = npz_dir / name
        if not path.is_file():
            raise ContractError(f"cache-listed output NPZ missing: {path}")
        data = load_npz(path)
        validate_schema(data)
        content_digest.update(name.encode("utf-8") + b"\0")
        for key in sorted(data):
            array = np.ascontiguousarray(data[key])
            content_digest.update(key.encode("utf-8") + b"\0")
            content_digest.update(str(array.dtype).encode("ascii") + b"\0")
            content_digest.update(repr(tuple(array.shape)).encode("ascii") + b"\0")
            content_digest.update(array.tobytes(order="C"))
        clip_id = scalar_text(data["clip_id"])
        if clip_id != str(row["id"]):
            raise ContractError(f"clip-id drift row={index}: npz={clip_id!r}, tsv={row['id']!r}")
        if clip_id in seen_ids:
            raise ContractError(f"duplicate NPZ clip id at row {index}: {clip_id}")
        seen_ids.add(clip_id)
        expected_hash = __import__("hashlib").sha256(str(row["caption"]).encode("utf-8")).hexdigest()
        if scalar_text(data["caption_sha256"]) != expected_hash:
            raise ContractError(f"caption hash drift row={index}, clip={clip_id}")
        if scalar_int(data["catalog_index"]) < 0:
            raise ContractError(f"negative catalog index row={index}")
        if reference_npz_dir is not None:
            ref_path = reference_npz_dir / name
            if not ref_path.is_file():
                raise ContractError(f"cache-listed reference NPZ missing: {ref_path}")
            reference = load_npz(ref_path)
            for key in ("mean", "std", "clip_id", "catalog_index"):
                compare_arrays_exact(data[key], reference[key], f"{name}:{key}")
        if index in semantic_indices:
            cached_c.append(np.asarray(data["text_features_c"], dtype=np.float32))
            captions.append(str(row["caption"]))
            semantic_rows.append(row)

    semantic: dict[str, Any]
    if skip_semantic:
        semantic = {"status": "skipped", "reason": "explicit self-test/diagnostic option"}
    else:
        if clap_checkpoint is None:
            raise ContractError("--clap-checkpoint is required unless --skip-semantic")
        audio_paths: list[Path] | None = None
        if audio_dir is not None:
            audio_paths = []
            for row in semantic_rows:
                path = audio_dir / f"{row['id']}.flac"
                if not path.is_file():
                    raise ContractError(f"semantic audio is not paired: {path}")
                audio_paths.append(path)
        semantic = semantic_gate(
            captions=captions,
            cached_text_c=np.stack(cached_c),
            clap_checkpoint=clap_checkpoint,
            seed=seed,
            audio_paths=audio_paths,
        )
    checks["semantic_gate"] = semantic
    checks["semantic_sample_count"] = len(captions)
    checks["semantic_sample_policy"] = "deterministic evenly spaced across audited rows"
    checks["output_content_sha256"] = content_digest.hexdigest()
    checks["status"] = "passed"
    checks["tsv_sha256"] = sha256_file(tsv)
    checks["cache_list_sha256"] = sha256_file(cache_list)
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--reference-npz-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--clap-checkpoint", type=Path, default=DEFAULT_CLAP_CHECKPOINT)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--seed", type=int, default=14159265)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--allow-test-rows", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(
        tsv=args.tsv,
        cache_list=args.cache_list,
        npz_dir=args.npz_dir,
        reference_npz_dir=args.reference_npz_dir,
        limit=args.limit,
        clap_checkpoint=None if args.skip_semantic else args.clap_checkpoint,
        audio_dir=args.audio_dir,
        seed=args.seed,
        skip_semantic=args.skip_semantic,
        allow_test_rows=args.allow_test_rows,
    )
    if args.json_out:
        write_json_atomic(args.json_out, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
