#!/usr/bin/env python3
"""Build an ID-bound stacked-caption text overlay without touching audio NPZs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import stat
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ENCODER_SOURCE = Path("/home/kojiek/research/meanaudio_training/caption10s_pipeline/reextract_text_inplace_caption10s.py")
ENCODER_SOURCE_SHA256 = "eb692393994a414b5578e6ab4e5c46c8aa7e66f2a09e39f2061bfe83768374dc"
OUTPUT_ROOT = Path("/home/kojiek/text_overlays")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temp, path)


def stored_caption_hashes(value: np.ndarray) -> list[str]:
    """Overlay files store caption_sha256 as a 0-d comma-joined string; accept 1-D too."""
    if value.ndim == 0:
        return str(value.item()).split(",")
    return [str(item) for item in value.tolist()]


def safe_output(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"symlink output rejected: {path}")
    resolved = path.resolve(strict=False)
    if OUTPUT_ROOT not in resolved.parents:
        raise ValueError(f"output outside {OUTPUT_ROOT}: {resolved}")
    path.mkdir(parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError(f"unsafe output directory: {path}")


def load_encoder():
    if sha256(ENCODER_SOURCE) != ENCODER_SOURCE_SHA256:
        raise RuntimeError("encoder source drift")
    spec = importlib.util.spec_from_file_location("bound_text_encoder", ENCODER_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bound text encoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-tsv", type=Path, required=True)
    parser.add_argument("--train-tsv", type=Path, required=True)
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--done-json", type=Path, required=True)
    parser.add_argument("--n-caps", type=int, default=3, choices=[3])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    safe_output(args.output_dir)
    if args.progress_json.parent != args.output_dir or args.done_json.parent != args.output_dir:
        raise ValueError("reports must be contained by output-dir")
    grouped: dict[str, list[str]] = defaultdict(list)
    with args.extraction_tsv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            grouped[row["id"]].append(row["caption"].strip())
    with args.train_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(rows) != 251599 or len(names) != len(rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("train/cache ID cardinality mismatch")
    bad = [row["id"] for row in rows if len(grouped.get(row["id"], [])) != args.n_caps]
    if bad:
        raise ValueError(f"stacked caption coverage mismatch: {len(bad)}")
    encoder = load_encoder()
    fingerprint = encoder.encoder_fingerprint()
    pending = []
    for index, (row, name) in enumerate(zip(rows, names)):
        target = args.output_dir / name
        if target.is_symlink():
            raise ValueError(f"symlink overlay rejected: {target}")
        cap_hashes = [encoder.sha_caption(text) for text in grouped[row["id"]]]
        if target.is_file():
            with np.load(target, allow_pickle=False) as data:
                valid = (
                    str(data["clip_id"].item()) == row["id"]
                    and stored_caption_hashes(data["caption_sha256"]) == cap_hashes
                    and str(data["text_encoder_fingerprint"].item()) == fingerprint
                    and data["text_features"].shape == (3, 77, 1024)
                    and data["text_features_c"].shape == (3, 512)
                    and data["text_attention_mask"].shape == (3, 77)
                )
            if valid:
                continue
        pending.append(index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = encoder.AutoTokenizer.from_pretrained(encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True)
    t5 = encoder.T5EncoderModel.from_pretrained(encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True).eval().to(device)
    clap = encoder.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap.load_ckpt(str(encoder.CLAP_CKPT), verbose=False)
    completed = len(rows) - len(pending)
    for offset in tqdm(range(0, len(pending), args.batch_size), desc="stacked-overlay"):
        indices = pending[offset:offset + args.batch_size]
        texts = [text for index in indices for text in grouped[rows[index]["id"]]]
        features, masks = encoder.encode_t5(tokenizer, t5, texts, device)
        pooled = encoder.encode_clap(clap, texts)
        for local, index in enumerate(indices):
            row, name = rows[index], names[index]
            sl = slice(local * 3, local * 3 + 3)
            encoder.atomic_savez(args.output_dir / name, {
                "clip_id": np.asarray(row["id"]),
                "text_features": features[sl].astype(np.float32),
                "text_features_c": pooled[sl].astype(np.float32),
                "text_attention_mask": masks[sl].astype(np.int64),
                "caption_sha256": np.asarray(",".join(encoder.sha_caption(x) for x in grouped[row["id"]])),
                "text_encoder_fingerprint": np.asarray(fingerprint),
            })
            completed += 1
        atomic_json(args.progress_json, {"completed": completed, "total": len(rows)})
    if completed != len(rows):
        raise RuntimeError(f"overlay incomplete: {completed}/{len(rows)}")
    atomic_json(args.done_json, {
        "status": "passed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "n_caps": 3,
        "extraction_tsv_sha256": sha256(args.extraction_tsv),
        "train_tsv_sha256": sha256(args.train_tsv),
        "cache_list_sha256": sha256(args.cache_list),
        "encoder_source_sha256": ENCODER_SOURCE_SHA256,
        "text_encoder_fingerprint": fingerprint,
        "shapes": {"text_features": [3, 77, 1024], "text_features_c": [3, 512], "text_attention_mask": [3, 77]}
    })


if __name__ == "__main__":
    main()
