"""Generate a canonical, manifest-backed multi-caption NPZ cache.

Each output filename comes from ``--gt-cache``.  For TSV row ``i``, both the
source audio statistics and destination filename are the mapped name at row
``i``; positional ``i.npz`` lookup is intentionally forbidden.

Example (use a NEW output directory for the clean rerun):

  python gen_multicap_npz.py \
    --tsv /mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv \
    --jsonl ~/research/music_cleaning/results_20260119_043407.jsonl \
    --gt-cache /mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt \
    --src-npz ~/research/meanaudio_training/npz_phase7_clean \
    --out-npz /mnt/HDD/kojiek/phase9_multicap_npz_clean \
    --resume

Outputs contain ``mean``, ``std``, five T5/CLAP caption embeddings, and five
T5 attention masks.  ``MANIFEST.tsv`` binds TSV id, canonical NPZ filename,
and the ordered-caption SHA-256 for every row.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from multicap_cache_common import (
    MANIFEST_NAME,
    assert_manifest_matches,
    build_specs,
    caption_sha256,
    load_caption_lookup,
    load_gt_cache,
    load_tsv,
    write_manifest_atomic,
)


CLAP_CKPT = Path.home() / "MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
TEXT_SHAPE = (5, 77, 1024)
CLAP_SHAPE = (5, 512)
MASK_SHAPE = (5, 77)
ESTIMATED_BYTES_PER_FILE = 1_640_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument(
        "--gt-cache",
        required=True,
        type=Path,
        help="Canonical row-to-NPZ map; generation refuses positional lookup.",
    )
    parser.add_argument("--src-npz", "--src_npz", dest="src_npz", required=True, type=Path)
    parser.add_argument("--out-npz", "--out_npz", dest="out_npz", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate metadata/write manifest/disk-check, but do not load encoders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing cache. Cannot be combined with --resume.",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--slot-count", type=int, default=5)
    parser.add_argument("--allow-missing-captions", action="store_true")
    parser.add_argument("--allow-tsv-caption-outside-pool", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.slot_count != 5:
        parser.error("MeanAudio multi-cap cache currently requires --slot-count 5")
    return args


def encode_captions(
    captions_per_slot: list[list[str]],
    tokenizer,
    t5_model,
    clap_model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    slot_count = len(captions_per_slot)
    batch_size = len(captions_per_slot[0])
    flat = [caption for slot in captions_per_slot for caption in slot]

    tokens = tokenizer(
        flat,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        t5_flat = t5_model(
            input_ids=tokens.input_ids.cuda(),
            attention_mask=tokens.attention_mask.cuda(),
        )[0]
    clap_flat = clap_model.get_text_embedding(flat, use_tensor=True)

    text = t5_flat.cpu().numpy().reshape(slot_count, batch_size, 77, 1024)
    text = text.transpose(1, 0, 2, 3)
    clap = clap_flat.detach().cpu().numpy().reshape(slot_count, batch_size, 512)
    clap = clap.transpose(1, 0, 2)
    masks = tokens.attention_mask.cpu().numpy().reshape(slot_count, batch_size, 77)
    masks = masks.transpose(1, 0, 2).astype(bool)
    return text, clap, masks


def output_is_complete(output_path: Path, source_path: Path, spec) -> bool:
    """A resume hit is valid only when schema and mapped audio are exact."""
    try:
        with np.load(output_path) as output, np.load(source_path) as source:
            required = {
                "mean", "std", "text_features", "text_features_c", "text_attention_mask",
                "clip_id", "row_index", "caption_sha256",
            }
            if not required.issubset(output.files):
                return False
            if str(output["clip_id"].item()) != spec.clip_id:
                return False
            if int(output["row_index"].item()) != spec.row_index:
                return False
            if str(output["caption_sha256"].item()) != caption_sha256(spec.captions):
                return False
            if output["text_features"].shape != TEXT_SHAPE:
                return False
            if output["text_features_c"].shape != CLAP_SHAPE:
                return False
            if output["text_attention_mask"].shape != MASK_SHAPE:
                return False
            return np.array_equal(output["mean"], source["mean"]) and np.array_equal(
                output["std"], source["std"]
            )
    except (OSError, ValueError, KeyError):
        return False


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez(temp_name, **arrays)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = parse_args()
    for path in (args.tsv, args.jsonl, args.gt_cache, args.src_npz):
        if not path.exists():
            raise SystemExit(f"[FAIL] required input not found: {path}")

    rows = load_tsv(args.tsv)
    npz_names = load_gt_cache(args.gt_cache)
    lookup = load_caption_lookup(args.jsonl)
    specs = build_specs(
        rows,
        npz_names,
        lookup,
        slot_count=args.slot_count,
        allow_missing_captions=args.allow_missing_captions,
        allow_tsv_caption_outside_pool=args.allow_tsv_caption_outside_pool,
    )
    print(f"TSV={len(rows):,}, gt_cache={len(npz_names):,}, JSONL ids={len(lookup):,}")

    args.out_npz.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_npz / MANIFEST_NAME
    existing_npz = list(args.out_npz.glob("*.npz"))
    if existing_npz and not manifest_path.exists() and not args.overwrite:
        raise SystemExit(
            "[FAIL] output contains NPZ files but no v2 manifest. This may be the historical "
            "misaligned cache; use a new output directory (recommended), or --overwrite."
        )
    if args.resume:
        if manifest_path.exists():
            assert_manifest_matches(manifest_path, specs)
        else:
            write_manifest_atomic(manifest_path, specs)
    elif existing_npz and not args.overwrite:
        raise SystemExit("[FAIL] output is non-empty; pass --overwrite or choose a new directory")
    else:
        write_manifest_atomic(manifest_path, specs)

    pending = []
    repaired = 0
    if args.resume:
        print("Checking resume files against canonical mapped mean/std...")
        for spec in tqdm(specs, desc="resume audit"):
            output_path = args.out_npz / spec.npz_fname
            source_path = args.src_npz / spec.npz_fname
            if output_path.exists() and output_is_complete(output_path, source_path, spec):
                continue
            if output_path.exists():
                repaired += 1
            pending.append(spec)
    else:
        pending = specs
    print(f"pending={len(pending):,}, invalid-existing-to-repair={repaired:,}")
    if not pending:
        print("Cache is already complete. Run validate_multicap_npz.py for the full audit.")
        return

    new_file_count = sum(
        not (args.out_npz / spec.npz_fname).exists() for spec in pending
    )
    estimated_bytes = max(2 * ESTIMATED_BYTES_PER_FILE,
                          new_file_count * ESTIMATED_BYTES_PER_FILE)
    free_bytes = shutil.disk_usage(args.out_npz).free
    if free_bytes < int(estimated_bytes * 1.05):
        raise SystemExit(
            f"[FAIL] insufficient output space: free={free_bytes / 1e9:.1f} GB, "
            f"estimated additional space+5%={estimated_bytes * 1.05 / 1e9:.1f} GB"
        )
    print(
        f"disk preflight: free={free_bytes / 1e9:.1f} GB, "
        f"new files={new_file_count:,}, additional space estimate={estimated_bytes / 1e9:.1f} GB"
    )
    if args.prepare_only:
        print(f"[OK] metadata and disk preflight complete; manifest={manifest_path}")
        return

    print("Loading FLAN-T5-large + CLAP on CUDA...")
    import laion_clap
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5EncoderModel.from_pretrained("google/flan-t5-large").eval().cuda()
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval().cuda()
    clap_model.load_ckpt(str(CLAP_CKPT), verbose=False)

    for start in tqdm(range(0, len(pending), args.batch_size), desc="multi-cap NPZ"):
        batch = pending[start : start + args.batch_size]
        captions_per_slot = [
            [spec.captions[slot] for spec in batch] for slot in range(args.slot_count)
        ]
        text, clap, masks = encode_captions(
            captions_per_slot, tokenizer, t5_model, clap_model
        )
        for batch_index, spec in enumerate(batch):
            source_path = args.src_npz / spec.npz_fname
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Mapped source missing at TSV row {spec.row_index}: {source_path}"
                )
            with np.load(source_path) as source:
                save_npz_atomic(
                    args.out_npz / spec.npz_fname,
                    mean=source["mean"],
                    std=source["std"],
                    text_features=text[batch_index],
                    text_features_c=clap[batch_index],
                    text_attention_mask=masks[batch_index],
                    clip_id=np.asarray(spec.clip_id),
                    row_index=np.asarray(spec.row_index, dtype=np.int64),
                    caption_sha256=np.asarray(caption_sha256(spec.captions)),
                )

    print(f"[OK] generated {len(pending):,} canonical files in {args.out_npz}")
    print("Mandatory next step: run validate_multicap_npz.py with the same inputs.")


if __name__ == "__main__":
    main()
