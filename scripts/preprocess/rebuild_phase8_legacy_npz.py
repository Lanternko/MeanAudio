#!/usr/bin/env python3
"""Rebuild the audio/text-matched cache actually used by historical Phase 8.

Historical Phase 8 supplied ``phase7_v1_train.tsv`` to the training loader, but
the loader consumed cached text features from the NPZ files named by
``npz_cache_train.txt``.  That cache list is not aligned to the Phase-7 TSV.
The authoritative caption for ``N.npz`` is therefore row N of the extraction
catalog (``npz.tsv``), not the same-position row of the Phase-7 TSV.

This tool copies only the audio statistics from an extant cache and re-encodes
the caption recorded in the original extraction catalog.  It can consume a
small subset cache list for a cheap causal probe before any full rebuild.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
from pathlib import Path

import laion_clap
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel


DATA_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data")
DEFAULT_CATALOG = DATA_DIR / "_QUARANTINED_npz.tsv"
DEFAULT_CACHE = DATA_DIR / "npz_cache_train.txt"
DEFAULT_Q_TSV = DATA_DIR / "_QUARANTINED_phase7_v1_train.tsv"
DEFAULT_Q_CACHE = DATA_DIR / "npz_cache_train.txt"
DEFAULT_SOURCE = Path(
    "/home/kojiek/research/meanaudio_training/npz_phase7_clean"
)
DEFAULT_OUTPUT = Path(
    "/home/kojiek/research/meanaudio_training/npz_phase8_legacy_matched"
)
DEFAULT_OUTPUT_TSV = DATA_DIR / "phase8_legacy_catalog_train.tsv"
DEFAULT_CLAP_CKPT = Path("weights/music_speech_audioset_epoch_15_esc_89.98.pt")
T5_MODEL_NAME = "google/flan-t5-large"
TEXT_SEQ_LEN = 77
MANIFEST_NAME = "MANIFEST.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--q-tsv",
        type=Path,
        default=DEFAULT_Q_TSV,
        help="Historical row-position q labels (the old runner ignored use_q_conditioning=false).",
    )
    parser.add_argument(
        "--q-cache",
        type=Path,
        default=DEFAULT_Q_CACHE,
        help="Cache list pairing --q-tsv rows to NPZ filenames.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_OUTPUT_TSV)
    parser.add_argument(
        "--output-cache",
        type=Path,
        help="Optional cache-list output matching the selected/rebuilt rows.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Uniformly sample this many filenames from --cache before rebuilding.",
    )
    parser.add_argument("--sample-seed", type=int, default=20260717)
    parser.add_argument("--clap-ckpt", type=Path, default=DEFAULT_CLAP_CKPT)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def read_cache(path: Path) -> list[str]:
    with path.open() as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        raise SystemExit(f"Empty cache list: {path}")
    if len(names) != len(set(names)):
        raise SystemExit(f"Duplicate NPZ filenames in cache list: {path}")
    for name in names:
        if Path(name).name != name or Path(name).suffix != ".npz":
            raise SystemExit(f"Unsafe NPZ filename: {name!r}")
        try:
            int(Path(name).stem)
        except ValueError as exc:
            raise SystemExit(f"Non-numeric historical NPZ filename: {name!r}") from exc
    return names


def select_names(
    names: list[str], sample_size: int | None, sample_seed: int
) -> list[str]:
    if sample_size is None:
        return names
    if not 0 < sample_size <= len(names):
        raise SystemExit(
            f"--sample-size must be in [1, {len(names):,}], got {sample_size}"
        )
    rng = random.Random(sample_seed)
    # Keep the sampled positions in their original cache order.  This makes the
    # subset deterministic while exercising filenames spread across the full
    # historical cache rather than only a convenient contiguous prefix.
    positions = sorted(rng.sample(range(len(names)), sample_size))
    selected = [names[position] for position in positions]
    print(
        f"[Sampling] selected={len(selected):,}/{len(names):,}, "
        f"seed={sample_seed}, source-position-range={positions[0]}..{positions[-1]}"
    )
    return selected


def read_catalog_rows(catalog: Path, names: list[str]) -> list[dict[str, str]]:
    positions = {int(Path(name).stem): pos for pos, name in enumerate(names)}
    rows: list[dict[str, str] | None] = [None] * len(names)
    max_index = max(positions)
    with catalog.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"id", "caption"}.issubset(reader.fieldnames):
            raise SystemExit(f"Catalog must contain id/caption columns: {catalog}")
        for catalog_index, row in enumerate(reader):
            position = positions.get(catalog_index)
            if position is not None:
                caption = row["caption"].strip()
                clip_id = row["id"].strip()
                if not clip_id or not caption:
                    raise SystemExit(
                        f"Empty catalog id/caption at catalog row {catalog_index}"
                    )
                rows[position] = {
                    "id": clip_id,
                    "caption": caption,
                    "catalog_index": str(catalog_index),
                }
            if catalog_index >= max_index and all(row is not None for row in rows):
                break
    missing = [names[i] for i, row in enumerate(rows) if row is None]
    if missing:
        raise SystemExit(
            f"Catalog does not contain {len(missing)} requested indices; first={missing[0]}"
        )
    return [row for row in rows if row is not None]


def attach_historical_q_levels(
    rows: list[dict[str, str]],
    names: list[str],
    q_tsv: Path,
    q_cache: Path,
) -> None:
    q_names = read_cache(q_cache)
    with q_tsv.open(newline="") as handle:
        q_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(q_names) != len(q_rows):
        raise SystemExit(
            f"Historical q row mismatch: TSV={len(q_rows):,}, cache={len(q_names):,}"
        )
    if not q_rows or "q_level" not in q_rows[0]:
        raise SystemExit(f"Historical q TSV lacks q_level: {q_tsv}")
    q_by_name: dict[str, str] = {}
    for name, row in zip(q_names, q_rows):
        value = row["q_level"].strip()
        try:
            level = int(value)
        except ValueError as exc:
            raise SystemExit(f"Invalid historical q_level {value!r} for {name}") from exc
        if not 0 <= level <= 10:
            raise SystemExit(f"Out-of-range historical q_level {level} for {name}")
        q_by_name[name] = str(level)
    missing = [name for name in names if name not in q_by_name]
    if missing:
        raise SystemExit(f"Missing historical q provenance for {missing[0]}")
    for row, name in zip(rows, names):
        row["q_level"] = q_by_name[name]


def caption_sha256(caption: str) -> str:
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def write_or_verify_metadata(
    output_dir: Path,
    output_tsv: Path,
    output_cache: Path | None,
    names: list[str],
    rows: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if output_cache is not None:
        output_cache.parent.mkdir(parents=True, exist_ok=True)
    expected = [
        {
            "row_index": str(i),
            "catalog_index": row["catalog_index"],
            "clip_id": row["id"],
            "npz_fname": name,
            "caption_sha256": caption_sha256(row["caption"]),
            "historical_q_level": row["q_level"],
        }
        for i, (name, row) in enumerate(zip(names, rows))
    ]

    if manifest_path.exists() or output_tsv.exists():
        if not manifest_path.exists() or not output_tsv.exists():
            raise SystemExit("Refusing partial metadata resume: manifest/TSV pair incomplete")
        with manifest_path.open(newline="") as handle:
            actual_manifest = list(csv.DictReader(handle, delimiter="\t"))
        with output_tsv.open(newline="") as handle:
            actual_tsv = list(csv.DictReader(handle, delimiter="\t"))
        expected_tsv = [
            {
                "id": row["id"],
                "caption": row["caption"],
                "q_level": row["q_level"],
            }
            for row in rows
        ]
        if actual_manifest != expected or actual_tsv != expected_tsv:
            raise SystemExit("Existing legacy metadata does not match requested catalog/cache")
        print(f"[Metadata] verified resumable {len(rows):,}-row manifest and TSV")
    else:
        manifest_tmp = manifest_path.with_suffix(".tsv.tmp")
        with manifest_tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(expected[0]),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(expected)
        os.replace(manifest_tmp, manifest_path)

        tsv_tmp = output_tsv.with_suffix(output_tsv.suffix + ".tmp")
        with tsv_tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "caption", "q_level"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {
                    "id": row["id"],
                    "caption": row["caption"],
                    "q_level": row["q_level"],
                }
                for row in rows
            )
        os.replace(tsv_tmp, output_tsv)
        print(f"[Metadata] wrote {len(rows):,}-row manifest and TSV")

    if output_cache is not None:
        expected_cache = "".join(f"{name}\n" for name in names)
        if output_cache.exists():
            if output_cache.read_text() != expected_cache:
                raise SystemExit(
                    f"Existing output cache does not match requested rows: {output_cache}"
                )
            print(f"[Metadata] verified cache list: {output_cache}")
        else:
            cache_tmp = output_cache.with_suffix(output_cache.suffix + ".tmp")
            cache_tmp.write_text(expected_cache)
            os.replace(cache_tmp, output_cache)
            print(f"[Metadata] wrote cache list: {output_cache}")


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
    return features.float().cpu().numpy(), attention_mask.bool().cpu().numpy()


@torch.inference_mode()
def encode_clap(model: laion_clap.CLAP_Module, captions: list[str]) -> np.ndarray:
    return model.get_text_embedding(captions, use_tensor=True).float().cpu().numpy()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp_path, path)


def preflight_source(source_dir: Path, names: list[str]) -> None:
    sample_positions = sorted({0, min(1, len(names) - 1), len(names) - 1})
    for position in sample_positions:
        path = source_dir / names[position]
        if not path.is_file():
            raise SystemExit(f"Missing source NPZ: {path}")
        with np.load(path) as data:
            if data["mean"].shape != (312, 20) or data["std"].shape != (312, 20):
                raise SystemExit(f"Bad source audio statistics: {path}")


def validate_outputs(
    source_dir: Path,
    output_dir: Path,
    names: list[str],
    rows: list[dict[str, str]],
) -> None:
    sample_positions = sorted({0, min(1, len(names) - 1), len(names) - 1})
    for position in sample_positions:
        name = names[position]
        with np.load(source_dir / name) as source, np.load(output_dir / name) as output:
            if not np.array_equal(source["mean"], output["mean"]):
                raise SystemExit(f"Audio mean changed: {name}")
            if not np.array_equal(source["std"], output["std"]):
                raise SystemExit(f"Audio std changed: {name}")
            if output["text_features"].shape != (77, 1024):
                raise SystemExit(f"Bad T5 feature shape: {name}")
            if output["text_features_c"].shape != (512,):
                raise SystemExit(f"Bad CLAP feature shape: {name}")
            if output["text_attention_mask"].shape != (77,):
                raise SystemExit(f"Bad attention-mask shape: {name}")
            if output["clip_id"].item() != rows[position]["id"]:
                raise SystemExit(f"Embedded clip provenance mismatch: {name}")
    print(f"[Validation] {len(sample_positions)} sampled outputs passed")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    for path in (
        args.catalog,
        args.cache,
        args.q_tsv,
        args.q_cache,
        args.source_dir,
        args.clap_ckpt,
    ):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    names = select_names(read_cache(args.cache), args.sample_size, args.sample_seed)
    rows = read_catalog_rows(args.catalog, names)
    attach_historical_q_levels(rows, names, args.q_tsv, args.q_cache)
    preflight_source(args.source_dir, names)
    write_or_verify_metadata(
        args.output_dir, args.output_tsv, args.output_cache, names, rows
    )

    pending = [i for i, name in enumerate(names) if not (args.output_dir / name).exists()]
    print(
        f"[Input] requested={len(names):,}, pending={len(pending):,}, "
        f"source={args.source_dir}, output={args.output_dir}"
    )
    if pending:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
        t5_model = T5EncoderModel.from_pretrained(T5_MODEL_NAME).eval().to(device)
        clap_model = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base"
        ).eval()
        clap_model.load_ckpt(str(args.clap_ckpt), verbose=False)
        clap_model = clap_model.to(device)

        for start in tqdm(
            range(0, len(pending), args.batch_size), desc="legacy matched NPZ"
        ):
            positions = pending[start : start + args.batch_size]
            captions = [rows[position]["caption"] for position in positions]
            text, masks = encode_t5(tokenizer, t5_model, captions, device)
            clap = encode_clap(clap_model, captions)
            for batch_index, position in enumerate(positions):
                name = names[position]
                row = rows[position]
                with np.load(args.source_dir / name) as source:
                    atomic_savez(
                        args.output_dir / name,
                        mean=source["mean"],
                        std=source["std"],
                        text_features=text[batch_index],
                        text_features_c=clap[batch_index],
                        text_attention_mask=masks[batch_index],
                        clip_id=np.asarray(row["id"]),
                        catalog_index=np.asarray(int(row["catalog_index"])),
                        caption_sha256=np.asarray(caption_sha256(row["caption"])),
                    )

    validate_outputs(args.source_dir, args.output_dir, names, rows)


if __name__ == "__main__":
    main()
