"""Rebuild clean Phase-7/P8 NPZs from existing Phase-8V4 audio latents.

The Phase-8V4 cache has the same clip order and audio latents as Phase 7, but
its text features encode a ``[consistency=...]`` prefix.  This script keeps the
audio ``mean``/``std`` arrays and re-encodes the original, unprefixed Phase-7
captions with the historical 77-token FLAN-T5 + CLAP recipe.

The output is resumable and uses atomic per-file writes.  A manifest check
prevents silently pairing text features with the wrong audio latent.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import laion_clap
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel


DEFAULT_TSV = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv"
)
DEFAULT_CACHE = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt"
)
DEFAULT_SOURCE = Path(
    "/home/kojiek/research/meanaudio_training/npz_phase8v4"
)
DEFAULT_OUTPUT = Path(
    "/home/kojiek/research/meanaudio_training/npz_phase7_clean"
)
DEFAULT_CLAP_CKPT = Path("weights/music_speech_audioset_epoch_15_esc_89.98.pt")
T5_MODEL_NAME = "google/flan-t5-large"
TEXT_SEQ_LEN = 77
MANIFEST_NAME = "MANIFEST.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clap-ckpt", type=Path, default=DEFAULT_CLAP_CKPT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (use a separate output directory).",
    )
    return parser.parse_args()


def read_rows(tsv: Path) -> list[dict[str, str]]:
    with tsv.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or not {"id", "caption"}.issubset(rows[0]):
        raise SystemExit(f"Invalid training TSV: {tsv}")
    return rows


def read_cache(cache: Path) -> list[str]:
    with cache.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def verify_source_manifest(
    source_dir: Path, rows: list[dict[str, str]], npz_files: list[str]
) -> None:
    manifest_path = source_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"Missing source manifest: {manifest_path}")

    with manifest_path.open(newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    if len(rows) != len(npz_files) or len(rows) != len(manifest):
        raise SystemExit(
            "Alignment length mismatch: "
            f"tsv={len(rows):,}, cache={len(npz_files):,}, "
            f"manifest={len(manifest):,}"
        )

    for index, (row, npz_name, item) in enumerate(zip(rows, npz_files, manifest)):
        expected = (str(index), row["id"], npz_name)
        actual = (item["idx"], item["clip_id"], item["npz_fname"])
        if expected != actual:
            raise SystemExit(
                f"Source alignment mismatch at row {index}: "
                f"expected={expected}, manifest={actual}"
            )
        source_path = source_dir / npz_name
        if index in (0, 1, 100, 1000, 10000, len(rows) - 1) and not source_path.exists():
            raise SystemExit(f"Missing sampled source NPZ: {source_path}")


def write_or_verify_output_manifest(
    output_dir: Path, rows: list[dict[str, str]], npz_files: list[str]
) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    expected_header = "idx\tclip_id\tnpz_fname\n"

    if manifest_path.exists():
        with manifest_path.open(newline="") as handle:
            manifest = list(csv.DictReader(handle, delimiter="\t"))
        if len(manifest) != len(rows):
            raise SystemExit(
                f"Output manifest has {len(manifest):,} rows; expected {len(rows):,}"
            )
        for index, (row, npz_name, item) in enumerate(zip(rows, npz_files, manifest)):
            if (item["idx"], item["clip_id"], item["npz_fname"]) != (
                str(index),
                row["id"],
                npz_name,
            ):
                raise SystemExit(f"Output manifest mismatch at row {index}")
        print(f"[Manifest] verified {len(rows):,} output rows")
        return

    tmp_path = manifest_path.with_suffix(".tsv.tmp")
    with tmp_path.open("w", newline="") as handle:
        handle.write(expected_header)
        for index, (row, npz_name) in enumerate(zip(rows, npz_files)):
            handle.write(f"{index}\t{row['id']}\t{npz_name}\n")
    os.replace(tmp_path, manifest_path)
    print(f"[Manifest] wrote {manifest_path} ({len(rows):,} rows)")


@torch.inference_mode()
def encode_t5(
    tokenizer: AutoTokenizer,
    model: T5EncoderModel,
    captions: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = tokenizer(
        captions,
        max_length=TEXT_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    features = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return (
        features.float().cpu().numpy(),
        attention_mask.bool().cpu().numpy(),
    )


@torch.inference_mode()
def encode_clap(model: laion_clap.CLAP_Module, captions: list[str]) -> np.ndarray:
    features = model.get_text_embedding(captions, use_tensor=True)
    return features.float().cpu().numpy()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp_path, path)


def validate_output(
    output_dir: Path, source_dir: Path, npz_files: list[str]
) -> None:
    sample_indices = sorted({0, 1, min(100, len(npz_files) - 1), len(npz_files) - 1})
    for index in sample_indices:
        npz_name = npz_files[index]
        source = np.load(source_dir / npz_name)
        output = np.load(output_dir / npz_name)
        if not np.array_equal(source["mean"], output["mean"]):
            raise SystemExit(f"Audio mean changed: {npz_name}")
        if not np.array_equal(source["std"], output["std"]):
            raise SystemExit(f"Audio std changed: {npz_name}")
        if output["text_features"].shape != (77, 1024):
            raise SystemExit(f"Bad T5 shape in {npz_name}")
        if output["text_features_c"].shape != (512,):
            raise SystemExit(f"Bad CLAP shape in {npz_name}")
        if output["text_attention_mask"].shape != (77,):
            raise SystemExit(f"Bad mask shape in {npz_name}")
    print(f"[Validation] {len(sample_indices)} sampled outputs passed")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    for path in (args.tsv, args.cache, args.source_dir, args.clap_ckpt):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    rows = read_rows(args.tsv)
    npz_files = read_cache(args.cache)
    verify_source_manifest(args.source_dir, rows, npz_files)

    if args.limit is not None:
        rows = rows[: args.limit]
        npz_files = npz_files[: args.limit]
        print(f"[Limit] processing first {len(rows):,} rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_verify_output_manifest(args.output_dir, rows, npz_files)

    missing = sum(not (args.output_dir / name).exists() for name in npz_files)
    print(
        f"[Input] rows={len(rows):,}, missing={missing:,}, "
        f"source={args.source_dir}, output={args.output_dir}"
    )
    if missing == 0:
        validate_output(args.output_dir, args.source_dir, npz_files)
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for text re-encoding")
    device = torch.device("cuda")
    print(f"[Models] loading {T5_MODEL_NAME} on {torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    t5_model = T5EncoderModel.from_pretrained(T5_MODEL_NAME).eval().to(device)
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap_model.load_ckpt(str(args.clap_ckpt), verbose=False)
    clap_model = clap_model.to(device)

    for start in tqdm(range(0, len(rows), args.batch_size), desc="Re-encoding text"):
        end = min(start + args.batch_size, len(rows))
        batch_names = npz_files[start:end]
        missing_offsets = [
            offset
            for offset, name in enumerate(batch_names)
            if not (args.output_dir / name).exists()
        ]
        if not missing_offsets:
            continue

        captions = [rows[start + offset]["caption"] for offset in missing_offsets]
        text_features, masks = encode_t5(tokenizer, t5_model, captions, device)
        text_features_c = encode_clap(clap_model, captions)

        for encoded_index, offset in enumerate(missing_offsets):
            npz_name = batch_names[offset]
            source_path = args.source_dir / npz_name
            if not source_path.exists():
                raise SystemExit(f"Missing source NPZ: {source_path}")
            source = np.load(source_path)
            atomic_savez(
                args.output_dir / npz_name,
                mean=source["mean"],
                std=source["std"],
                text_features=text_features[encoded_index],
                text_features_c=text_features_c[encoded_index],
                text_attention_mask=masks[encoded_index],
            )

    validate_output(args.output_dir, args.source_dir, npz_files)
    print(f"[Done] rebuilt {len(rows):,} clean Phase-7 NPZs")


if __name__ == "__main__":
    main()
