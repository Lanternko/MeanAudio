#!/usr/bin/env python3
"""Stacked 3-caption text overlay; does not mutate canonical audio NPZs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ENCODER_SOURCE = Path(
    "/home/kojiek/research/meanaudio_training/caption10s_pipeline/"
    "reextract_text_inplace_caption10s.py"
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder_module():
    spec = importlib.util.spec_from_file_location("bound_text_encoder", ENCODER_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load encoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction-tsv", type=Path, required=True)
    ap.add_argument("--official-tsv", type=Path, required=True)
    ap.add_argument("--cache-list", type=Path, required=True)
    ap.add_argument("--audio-npz-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--progress-json", type=Path, required=True)
    ap.add_argument("--done-json", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-caps", type=int, default=3)
    args = ap.parse_args()

    encoder = load_encoder_module()
    grouped: dict[str, list[str]] = defaultdict(list)
    with args.extraction_tsv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            grouped[row["id"]].append(row["caption"].strip())
    with args.official_tsv.open(encoding="utf-8", newline="") as handle:
        official = list(csv.DictReader(handle, delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(official) != len(names):
        raise SystemExit(f"official {len(official)} != cache {len(names)}")
    missing = [row["id"] for row in official if len(grouped.get(row["id"], [])) != args.n_caps]
    if missing:
        raise SystemExit(f"incomplete stacked captions: {len(missing)} e.g. {missing[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = encoder.encoder_fingerprint()
    pending = []
    for index, (row, name) in enumerate(zip(official, names)):
        overlay_path = args.output_dir / name
        caps = grouped[row["id"]]
        cap_sha = ",".join(encoder.sha_caption(c) for c in caps)
        if overlay_path.is_file():
            try:
                with np.load(overlay_path, allow_pickle=False) as overlay:
                    valid = (
                        str(overlay["clip_id"].item()) == row["id"]
                        and overlay["text_features"].shape[0] == args.n_caps
                        and str(overlay["caption_sha256"].item()) == cap_sha
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

    completed = len(official) - len(pending)
    print(f"stacked overlay todo={len(pending)} done={completed} device={device}", flush=True)
    for offset in tqdm(range(0, len(pending), args.batch_size), desc="stacked-overlay"):
        indices = pending[offset : offset + args.batch_size]
        texts = []
        owners = []
        for index in indices:
            caps = grouped[official[index]["id"]]
            texts.extend(caps)
            owners.extend([index] * args.n_caps)
        features, masks = encoder.encode_t5(tokenizer, t5, texts, device)
        pooled = encoder.encode_clap(clap, texts)
        by_index: dict[int, list[int]] = defaultdict(list)
        for item, index in enumerate(owners):
            by_index[index].append(item)
        for index, items in by_index.items():
            row, name = official[index], names[index]
            cap_sha = ",".join(encoder.sha_caption(c) for c in grouped[row["id"]])
            encoder.atomic_savez(args.output_dir / name, {
                "clip_id": np.asarray(row["id"]),
                "text_features": np.stack([features[i] for i in items]).astype(np.float32),
                "text_features_c": np.stack([pooled[i] for i in items]).astype(np.float32),
                "text_attention_mask": np.stack([masks[i] for i in items]).astype(np.int64),
                "caption_sha256": np.asarray(cap_sha),
                "text_encoder_fingerprint": np.asarray(fingerprint),
            })
            completed += 1
        atomic_json(args.progress_json, {
            "completed": completed,
            "total": len(official),
            "pct": completed / len(official),
        })

    if completed != len(official):
        raise SystemExit(f"overlay incomplete: {completed}/{len(official)}")
    atomic_json(args.done_json, {
        "status": "passed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total": completed,
        "output_dir": str(args.output_dir),
        "n_caps": args.n_caps,
        "extraction_tsv_sha256": sha_file(args.extraction_tsv),
    })
    print("STACKED_OVERLAY_DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
