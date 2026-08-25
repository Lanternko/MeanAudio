#!/usr/bin/env python3
"""Build a text-only NPZ overlay without mutating canonical audio latents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ENCODER_SOURCE = Path(
    "/home/kojiek/research/meanaudio_training/caption10s_pipeline/"
    "reextract_text_inplace_caption10s.py"
)
ENCODER_SOURCE_SHA256 = "eb692393994a414b5578e6ab4e5c46c8aa7e66f2a09e39f2061bfe83768374dc"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder_module():
    actual = sha_file(ENCODER_SOURCE)
    if actual != ENCODER_SOURCE_SHA256:
        raise RuntimeError(f"encoder implementation drift: {actual}")
    spec = importlib.util.spec_from_file_location("bound_text_encoder", ENCODER_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bound encoder implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tsv", type=Path, required=True)
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--audio-npz-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--done-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    encoder = load_encoder_module()
    with args.train_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(rows) != len(names):
        raise SystemExit(f"rows {len(rows)} != cache names {len(names)}")
    if args.limit:
        rows, names = rows[: args.limit], names[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    fingerprint = encoder.encoder_fingerprint()
    for index, (row, name) in enumerate(zip(rows, names)):
        overlay_path = args.output_dir / name
        if overlay_path.is_file():
            try:
                with np.load(overlay_path, allow_pickle=False) as overlay:
                    valid = (
                        str(overlay["clip_id"].item()) == row["id"]
                        and str(overlay["caption_sha256"].item()) == encoder.sha_caption(row["caption"])
                        and str(overlay["text_encoder_fingerprint"].item()) == fingerprint
                    )
                if valid:
                    continue
            except (KeyError, OSError, ValueError, EOFError):
                pass
        pending.append(index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = encoder.AutoTokenizer.from_pretrained(
        encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True
    )
    t5 = encoder.T5EncoderModel.from_pretrained(
        encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True
    ).eval().to(device)
    clap = encoder.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap.load_ckpt(str(encoder.CLAP_CKPT), verbose=False)

    completed = len(rows) - len(pending)
    for offset in tqdm(range(0, len(pending), args.batch_size), desc="text-overlay"):
        indices = pending[offset : offset + args.batch_size]
        texts = [rows[index]["caption"] for index in indices]
        features, masks = encoder.encode_t5(tokenizer, t5, texts, device)
        pooled = encoder.encode_clap(clap, texts)
        for item, index in enumerate(indices):
            row, name = rows[index], names[index]
            encoder.atomic_savez(args.output_dir / name, {
                "clip_id": np.asarray(row["id"]),
                "text_features": features[item].astype(np.float32),
                "text_features_c": pooled[item].astype(np.float32),
                "text_attention_mask": masks[item].astype(np.int64),
                "caption_sha256": np.asarray(encoder.sha_caption(row["caption"])),
                "text_encoder_fingerprint": np.asarray(fingerprint),
            })
            completed += 1
        atomic_json(args.progress_json, {
            "completed": completed,
            "total": len(rows),
            "pct": completed / len(rows),
            "train_tsv_sha256": sha_file(args.train_tsv),
            "cache_list_sha256": sha_file(args.cache_list),
            "encoder_source_sha256": ENCODER_SOURCE_SHA256,
            "text_encoder_fingerprint": fingerprint,
        })

    if completed != len(rows):
        raise SystemExit(f"overlay incomplete: {completed}/{len(rows)}")
    atomic_json(args.done_json, {
        "status": "passed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total": completed,
        "train_tsv": str(args.train_tsv),
        "train_tsv_sha256": sha_file(args.train_tsv),
        "cache_list": str(args.cache_list),
        "cache_list_sha256": sha_file(args.cache_list),
        "audio_npz_dir": str(args.audio_npz_dir),
        "output_dir": str(args.output_dir),
        "encoder_source": str(ENCODER_SOURCE),
        "encoder_source_sha256": ENCODER_SOURCE_SHA256,
        "text_encoder_fingerprint": fingerprint,
    })


if __name__ == "__main__":
    main()
