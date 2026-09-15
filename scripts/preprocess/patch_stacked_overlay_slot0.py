#!/usr/bin/env python3
"""Reuse true_random stacked overlay; re-encode only rows whose slot0 caption changed.

Hardlinks every *.npz from true_random into the output dir (same filesystem, 0 extra
bytes), then atomically replaces the files whose slot-0 caption no longer matches
the TSV. os.replace breaks the hardlink for those files only, so true_random is
never mutated. Train with cap_index_fixed=0.
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

ENCODER_SOURCE = Path(
    "/home/kojiek/research/meanaudio_training/caption10s_pipeline/"
    "reextract_text_inplace_caption10s.py"
)
ENCODER_SOURCE_SHA256 = "eb692393994a414b5578e6ab4e5c46c8aa7e66f2a09e39f2061bfe83768374dc"
TRUE_RANDOM = Path("/home/kojiek/text_overlays/true_random")
OUTPUT_ROOT = Path("/home/kojiek/text_overlays")
EXPECTED_FP = "27e88fac68d94a8a10e44d2db930a8f79db8ca0454ce996b82e448c48c40ab4c"


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
    if resolved == TRUE_RANDOM.resolve():
        raise ValueError("refusing to write into true_random")
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


def slot0_hash(path: Path) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            stored = str(data["caption_sha256"].item()).split(",")
            return stored[0] if stored else None
    except Exception:
        return None


def hardlink_base(output_dir: Path, names: list[str]) -> None:
    marker = output_dir / "HARDLINK.OK"
    if marker.is_file():
        return
    missing = 0
    for name in names:
        src = TRUE_RANDOM / name
        dst = output_dir / name
        if not src.is_file():
            missing += 1
            continue
        if dst.exists() or dst.is_symlink():
            continue
        os.link(src, dst)
    if missing:
        raise FileNotFoundError(f"{missing} source overlay files missing in {TRUE_RANDOM}")
    marker.write_text("ok\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tsv", type=Path, required=True)
    parser.add_argument("--cache-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()

    safe_output(args.output_dir)
    rows = list(csv.DictReader(args.train_tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(rows) != len(names):
        raise ValueError(f"tsv {len(rows)} != cache {len(names)}")

    encoder = load_encoder()
    fingerprint = encoder.encoder_fingerprint()
    if fingerprint != EXPECTED_FP:
        raise RuntimeError(f"encoder fingerprint drift: {fingerprint}")

    hardlink_base(args.output_dir, names)

    pending = []
    for index, (row, name) in enumerate(zip(rows, names)):
        target = args.output_dir / name
        if target.is_symlink():
            raise ValueError(f"symlink overlay rejected: {target}")
        expected = encoder.sha_caption(row["caption"])
        if slot0_hash(target) == expected:
            continue
        pending.append(index)
    print(f"rows={len(rows)} pending={len(pending)} fingerprint={fingerprint}")

    if pending:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = encoder.AutoTokenizer.from_pretrained(
            encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True
        )
        t5 = encoder.T5EncoderModel.from_pretrained(
            encoder.T5_MODEL, revision=encoder.T5_REVISION, local_files_only=True
        ).eval().to(device)
        clap = encoder.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
        clap.load_ckpt(str(encoder.CLAP_CKPT), verbose=False)

        for offset in tqdm(range(0, len(pending), args.batch_size), desc="patch-slot0"):
            indices = pending[offset : offset + args.batch_size]
            texts = [rows[i]["caption"] for i in indices]
            features, masks = encoder.encode_t5(tokenizer, t5, texts, device)
            pooled = encoder.encode_clap(clap, texts)
            for local, index in enumerate(indices):
                row, name = rows[index], names[index]
                target = args.output_dir / name
                source = TRUE_RANDOM / name
                with np.load(source, allow_pickle=False) as orig:
                    if str(orig["clip_id"].item()) != row["id"]:
                        raise ValueError(f"clip_id mismatch {name}")
                    text_features = np.array(orig["text_features"], copy=True)
                    text_features_c = np.array(orig["text_features_c"], copy=True)
                    text_attention_mask = np.array(orig["text_attention_mask"], copy=True)
                    hashes = str(orig["caption_sha256"].item()).split(",")
                if text_features.ndim != 3:
                    raise ValueError(f"{name} is not a stacked overlay: {text_features.shape}")
                text_features[0] = features[local].astype(np.float32)
                text_features_c[0] = pooled[local].astype(np.float32)
                text_attention_mask[0] = masks[local].astype(np.int64)
                hashes[0] = encoder.sha_caption(row["caption"])
                encoder.atomic_savez(
                    target,
                    {
                        "clip_id": np.asarray(row["id"]),
                        "text_features": text_features,
                        "text_features_c": text_features_c,
                        "text_attention_mask": text_attention_mask,
                        "caption_sha256": np.asarray(",".join(hashes)),
                        "text_encoder_fingerprint": np.asarray(fingerprint),
                    },
                )
                if os.stat(target).st_ino == os.stat(source).st_ino:
                    raise RuntimeError(f"atomic save did not break hardlink for {name}")

    patched_rows = 0
    for name in names:
        if os.stat(args.output_dir / name).st_ino != os.stat(TRUE_RANDOM / name).st_ino:
            patched_rows += 1

    atomic_json(
        args.output_dir / "DONE.json",
        {
            "status": "passed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(rows),
            "n_caps": 3,
            "patched_slot": 0,
            "patched_rows": patched_rows,
            "base_overlay": str(TRUE_RANDOM),
            "train_tsv": str(args.train_tsv),
            "train_tsv_sha256": sha256(args.train_tsv),
            "cache_list_sha256": sha256(args.cache_list),
            "encoder_source_sha256": ENCODER_SOURCE_SHA256,
            "text_encoder_fingerprint": fingerprint,
            "train_with": "cap_index_fixed=0",
        },
    )
    print(f"DONE patched={len(pending)}")


if __name__ == "__main__":
    main()
