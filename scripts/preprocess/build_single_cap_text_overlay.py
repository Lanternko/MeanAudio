#!/usr/bin/env python3
"""Encode a one-caption-per-clip text overlay, named to match an audio cache list.

The stacked builder (build_stacked_text_overlay_cache.py) is hard-wired to 3
captions and to the 251,599-row c2p0 corpus. The paired59k control needs a
single caption for an arbitrary row count, so this is that same procedure with
the stacking removed. It deliberately imports the SAME bound encoder module and
records the SAME fingerprint, because a captioner-only control is only honest if
both arms' features come out of one encoder.

Resumable: an existing file whose clip_id, caption hash, fingerprint and shapes
all check out is left alone, so a killed run restarts where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

csv.field_size_limit(10**9)

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
    parser.add_argument("--train-tsv", type=Path, required=True)
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()

    safe_output(args.output_dir)
    progress_json = args.output_dir / "PROGRESS.json"
    done_json = args.output_dir / "DONE.json"

    rows = list(csv.DictReader(args.train_tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(names) != len(rows):
        raise ValueError(f"cache list {len(names)} != train tsv {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("train tsv has duplicate ids")
    if len(set(names)) != len(names):
        raise ValueError("cache list has duplicate names")

    encoder = load_encoder()
    fingerprint = encoder.encoder_fingerprint()

    pending = []
    for index, (row, name) in enumerate(zip(rows, names)):
        target = args.output_dir / name
        if target.is_symlink():
            raise ValueError(f"symlink overlay rejected: {target}")
        if target.is_file():
            try:
                with np.load(target, allow_pickle=False) as data:
                    valid = (
                        str(data["clip_id"].item()) == row["id"]
                        and str(data["caption_sha256"].item()) == encoder.sha_caption(row["caption"])
                        and str(data["text_encoder_fingerprint"].item()) == fingerprint
                        and data["text_features"].shape == (77, 1024)
                        and data["text_features_c"].shape == (512,)
                        and data["text_attention_mask"].shape == (77,)
                    )
            except Exception:
                valid = False
            if valid:
                continue
        pending.append(index)
    print(f"rows={len(rows)} pending={len(pending)} fingerprint={fingerprint}")

    if pending:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = encoder.AutoTokenizer.from_pretrained(
            encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True)
        t5 = encoder.T5EncoderModel.from_pretrained(
            encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True).eval().to(device)
        clap = encoder.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
        clap.load_ckpt(str(encoder.CLAP_CKPT), verbose=False)

        completed = len(rows) - len(pending)
        for offset in tqdm(range(0, len(pending), args.batch_size), desc="single-cap-overlay"):
            indices = pending[offset:offset + args.batch_size]
            texts = [rows[i]["caption"] for i in indices]
            features, masks = encoder.encode_t5(tokenizer, t5, texts, device)
            pooled = encoder.encode_clap(clap, texts)
            for local, index in enumerate(indices):
                row, name = rows[index], names[index]
                encoder.atomic_savez(args.output_dir / name, {
                    "clip_id": np.asarray(row["id"]),
                    "text_features": features[local].astype(np.float32),
                    "text_features_c": pooled[local].astype(np.float32),
                    "text_attention_mask": masks[local].astype(np.int64),
                    "caption_sha256": np.asarray(encoder.sha_caption(row["caption"])),
                    "text_encoder_fingerprint": np.asarray(fingerprint),
                })
                completed += 1
            atomic_json(progress_json, {"completed": completed, "total": len(rows)})
        if completed != len(rows):
            raise RuntimeError(f"overlay incomplete: {completed}/{len(rows)}")

    atomic_json(done_json, {
        "status": "passed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "n_caps": 1,
        "train_tsv": str(args.train_tsv),
        "train_tsv_sha256": sha256(args.train_tsv),
        "cache_list_sha256": sha256(args.cache_list),
        "encoder_source_sha256": ENCODER_SOURCE_SHA256,
        "text_encoder_fingerprint": fingerprint,
        "shapes": {"text_features": [77, 1024], "text_features_c": [512], "text_attention_mask": [77]},
    })
    print(f"wrote {done_json}")


if __name__ == "__main__":
    main()
