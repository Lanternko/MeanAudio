"""Strict semantic validator for canonical multi-caption NPZ caches.

Unlike the historical shape-only preflight, this validator requires the TSV,
caption JSONL, canonical gt_cache, source audio NPZ directory, and v2 manifest.
By default it checks every output file and requires exact mapped ``mean/std``
equality.  A sampled alignment mode exists only for development smoke tests.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from multicap_cache_common import (
    CacheSpec,
    MANIFEST_NAME,
    assert_manifest_matches,
    build_specs,
    caption_sha256,
    load_caption_lookup,
    load_gt_cache,
    load_tsv,
)


REQUIRED_KEYS = {
    "mean", "std", "text_features", "text_features_c", "text_attention_mask",
    "clip_id", "row_index", "caption_sha256",
}
EXPECTED_TEXT_SHAPES = {
    "text_features": (5, 77, 1024),
    "text_features_c": (5, 512),
    "text_attention_mask": (5, 77),
}


def parse_check_count(value: str) -> int | None:
    if value.lower() == "all":
        return None
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("alignment checks must be 'all' or a positive integer")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--gt-cache", required=True, type=Path)
    parser.add_argument("--src-npz", required=True, type=Path)
    parser.add_argument("--npz-dir", "--npz_dir", dest="npz_dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--alignment-checks",
        type=parse_check_count,
        default=None,
        metavar="all|N",
        help="Mapped mean/std exact-equality checks (default: all files).",
    )
    parser.add_argument(
        "--deep",
        type=int,
        default=200,
        help="Additional random schema checks when alignment is sampled.",
    )
    parser.add_argument("--slot-count", type=int, default=5)
    parser.add_argument("--allow-missing-captions", action="store_true")
    parser.add_argument("--allow-tsv-caption-outside-pool", action="store_true")
    return parser.parse_args()


def choose_indices(total: int, alignment_checks: int | None, deep: int) -> list[int]:
    if alignment_checks is None or alignment_checks >= total:
        return list(range(total))
    rng = random.Random(42)
    mandatory = {0, 1, total - 1} if total > 1 else {0}
    requested = min(total, max(alignment_checks, len(mandatory)))
    alignment = set(rng.sample(range(total), requested)) | mandatory
    if deep > 0:
        alignment |= set(rng.sample(range(total), min(total, deep)))
    return sorted(alignment)


def validate_one(output_path: Path, source_path: Path, spec: CacheSpec) -> str | None:
    try:
        with np.load(output_path) as output, np.load(source_path) as source:
            missing = REQUIRED_KEYS - set(output.files)
            if missing:
                return f"missing keys {sorted(missing)}"
            if str(output["clip_id"].item()) != spec.clip_id:
                return f"clip_id {output['clip_id'].item()!r} != {spec.clip_id!r}"
            if int(output["row_index"].item()) != spec.row_index:
                return f"row_index {output['row_index'].item()} != {spec.row_index}"
            expected_hash = caption_sha256(spec.captions)
            if str(output["caption_sha256"].item()) != expected_hash:
                return "caption_sha256 differs from the TSV/JSONL caption pool"
            for key, expected_shape in EXPECTED_TEXT_SHAPES.items():
                if output[key].shape != expected_shape:
                    return f"{key} shape {output[key].shape} != {expected_shape}"
            if output["text_attention_mask"].dtype != np.bool_:
                return f"text_attention_mask dtype {output['text_attention_mask'].dtype} != bool"
            if not output["text_attention_mask"].any(axis=1).all():
                return "one or more caption masks contain no valid tokens"
            for key in ("text_features", "text_features_c"):
                if not np.isfinite(output[key]).all():
                    return f"{key} contains NaN/Inf"
            for key in ("mean", "std"):
                if key not in source.files:
                    return f"mapped source missing {key!r}"
                if output[key].shape != source[key].shape:
                    return f"{key} shape {output[key].shape} != source {source[key].shape}"
                if not np.array_equal(output[key], source[key]):
                    return f"{key} differs from canonical mapped source"
    except FileNotFoundError as error:
        return f"file missing: {error.filename}"
    except (OSError, ValueError, KeyError) as error:
        return f"load error: {error}"
    return None


def main() -> int:
    args = parse_args()
    manifest = args.manifest or args.npz_dir / MANIFEST_NAME
    for path in (args.tsv, args.jsonl, args.gt_cache, args.src_npz, args.npz_dir, manifest):
        if not path.exists():
            print(f"[FAIL] required path not found: {path}")
            return 1

    try:
        rows = load_tsv(args.tsv)
        names = load_gt_cache(args.gt_cache)
        lookup = load_caption_lookup(args.jsonl)
        specs = build_specs(
            rows,
            names,
            lookup,
            slot_count=args.slot_count,
            allow_missing_captions=args.allow_missing_captions,
            allow_tsv_caption_outside_pool=args.allow_tsv_caption_outside_pool,
        )
        assert_manifest_matches(manifest, specs)
    except (ValueError, KeyError) as error:
        print(f"[FAIL] metadata alignment: {error}")
        return 1
    print(f"[1/4] TSV/JSONL/gt_cache/manifest alignment: {len(specs):,} rows OK")

    expected_names = set(names)
    actual_names = {path.name for path in args.npz_dir.glob("*.npz")}
    missing = sorted(expected_names - actual_names)
    extras = sorted(actual_names - expected_names)
    if missing or extras:
        print(f"[FAIL] inventory mismatch: missing={len(missing)}, extra={len(extras)}")
        if missing:
            print(f"       first missing: {missing[:5]}")
        if extras:
            print(f"       first extra: {extras[:5]}")
        return 1
    print(f"[2/4] canonical NPZ inventory: {len(actual_names):,} files OK")

    indices = choose_indices(len(specs), args.alignment_checks, args.deep)
    mode = "full" if len(indices) == len(specs) else f"sampled ({len(indices):,})"
    print(f"[3/4] schema + mapped audio exact-equality audit: {mode}")
    failures: list[tuple[int, str, str]] = []
    for position, index in enumerate(indices, 1):
        spec = specs[index]
        message = validate_one(
            args.npz_dir / spec.npz_fname,
            args.src_npz / spec.npz_fname,
            spec,
        )
        if message:
            failures.append((index, spec.npz_fname, message))
            if len(failures) >= 20:
                break
        if position % 25000 == 0:
            print(f"       checked {position:,}/{len(indices):,}")
    if failures:
        print(f"[FAIL] {len(failures)} validation failure(s), first entries:")
        for index, name, message in failures:
            print(f"       row {index}, {name}: {message}")
        return 1
    print(f"       {len(indices):,} files passed schema and mapped mean/std checks")

    if len(indices) != len(specs):
        print("[4/4] [WARN] sampled mode is for development only; training preflight must use all")
    else:
        print("[4/4] full semantic alignment audit complete")
    print(f"[OK] cache is safe for canonical gt_cache training: {args.npz_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
