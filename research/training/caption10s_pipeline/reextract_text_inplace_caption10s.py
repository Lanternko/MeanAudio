#!/usr/bin/env python3
"""In-place text re-extract. Same-dir atomic write via file handle (exfat-safe)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import laion_clap
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

T5_MODEL = "google/flan-t5-large"
T5_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
MAX_TEXT_LEN = 128
SEQ_LEN = 77
CLAP_CKPT = Path("/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt")
CLAP_CKPT_SHA256 = "51c68f12f9d7ea25fdaaccf741ec7f81e93ee594455410f3bca4f47f88d8e006"


def encoder_contract() -> dict:
    """Feature semantics that must match before an NPZ can be resumed/skipped."""
    return {
        "schema_version": 1,
        "t5_model": T5_MODEL,
        "t5_revision": T5_REVISION,
        "tokenizer_max_length": MAX_TEXT_LEN,
        "stored_sequence_length": SEQ_LEN,
        "t5_algorithm": "bidirectional_encode_then_prefix_slice_v1",
        "clap_checkpoint": str(CLAP_CKPT),
        "clap_checkpoint_sha256": CLAP_CKPT_SHA256,
    }


def encoder_fingerprint() -> str:
    payload = json.dumps(encoder_contract(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha_caption(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def encode_t5(tokenizer, model, texts, device):
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TEXT_LEN
    ).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    B, L, D = out.shape
    if L < SEQ_LEN:
        out = torch.cat(
            [out, torch.zeros(B, SEQ_LEN - L, D, device=device, dtype=out.dtype)], dim=1
        )
    else:
        out = out[:, :SEQ_LEN, :]
    lengths = enc["attention_mask"].sum(dim=1).tolist()
    masks = []
    for L0 in lengths:
        m = np.zeros(SEQ_LEN, dtype=np.int64)
        m[: min(int(L0), SEQ_LEN)] = 1
        masks.append(m)
    return out.cpu().float().numpy(), np.stack(masks, axis=0)


def encode_clap(clap, texts):
    with torch.no_grad():
        emb = clap.get_text_embedding(texts, use_tensor=True)
    return emb.cpu().float().numpy()


def atomic_savez(path: Path, data: dict) -> None:
    """Same-dir temp + file handle so numpy does not append another .npz suffix."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix="." + path.stem + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", type=Path, required=True)
    ap.add_argument("--cache_list", type=Path, required=True)
    ap.add_argument("--npz_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--progress_json", type=Path, required=True)
    ap.add_argument("--done_json", type=Path)
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--scan_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with args.train_tsv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    names = [ln.strip() for ln in args.cache_list.open() if ln.strip()]
    if len(rows) != len(names):
        raise SystemExit(f"rows {len(rows)} != cache {len(names)}")
    if args.dry_run:
        rows, names = rows[: args.dry_run], names[: args.dry_run]
        print(f"DRY RUN n={len(rows)}", flush=True)
    if not args.npz_dir.is_dir():
        raise SystemExit(f"npz_dir missing: {args.npz_dir}")

    actual_clap_sha = sha_file(CLAP_CKPT)
    if actual_clap_sha != CLAP_CKPT_SHA256:
        raise SystemExit(
            f"CLAP checkpoint hash mismatch: expected {CLAP_CKPT_SHA256} "
            f"got {actual_clap_sha}"
        )
    feature_fingerprint = encoder_fingerprint()
    feature_contract = encoder_contract()
    print(
        f"feature_encoder_fingerprint={feature_fingerprint} "
        f"t5_revision={T5_REVISION} tokenizer_max={MAX_TEXT_LEN} stored_seq={SEQ_LEN}",
        flush=True,
    )

    todo, skipped = [], 0
    for i, (row, name) in enumerate(zip(rows, names)):
        path = args.npz_dir / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
                if "clip_id" not in data.files:
                    raise SystemExit(f"clip_id missing: {name}")
                stored_id = str(data["clip_id"].item())
                if stored_id != row["id"]:
                    raise SystemExit(
                        f"clip_id mismatch {name}: {stored_id} vs {row['id']}"
                    )
                caption_matches = (
                    "caption_sha256" in data.files
                    and str(data["caption_sha256"].item())
                    == sha_caption(row["caption"])
                )
                encoder_matches = (
                    "text_encoder_fingerprint" in data.files
                    and str(data["text_encoder_fingerprint"].item())
                    == feature_fingerprint
                )
        except (OSError, ValueError, EOFError) as exc:
            raise SystemExit(f"unreadable NPZ {name}: {exc}") from exc
        if not args.force and caption_matches and encoder_matches:
            skipped += 1
            continue
        todo.append(i)
    print(f"total={len(rows)} todo={len(todo)} skipped_sha_match={skipped}", flush=True)
    if args.scan_only:
        print("[SCAN ONLY] no NPZ files modified", flush=True)
        return

    train_tsv_sha256 = sha_file(args.train_tsv)
    cache_list_sha256 = sha_file(args.cache_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} ATOMIC_SAME_DIR_FILEHANDLE=1", flush=True)
    tokenizer = t5 = clap = None
    if todo:
        print("Loading T5...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            T5_MODEL, revision=T5_REVISION, local_files_only=True
        )
        t5 = (
            T5EncoderModel.from_pretrained(
                T5_MODEL, revision=T5_REVISION, local_files_only=True
            )
            .eval()
            .to(device)
        )
        print("Loading CLAP...", flush=True)
        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
        clap.load_ckpt(str(CLAP_CKPT), verbose=False)

    # smoke rewrite identity on first file
    if todo:
        path0 = args.npz_dir / names[todo[0]]
        d0 = dict(np.load(path0, allow_pickle=False))
        atomic_savez(path0, d0)
        assert path0.exists(), path0
        print(f"smoke rewrite ok: {path0}", flush=True)

    updated = skipped
    for bi in tqdm(range(0, len(todo), args.batch_size), desc="inplace-text"):
        batch_i = todo[bi : bi + args.batch_size]
        batch_rows = [rows[i] for i in batch_i]
        batch_names = [names[i] for i in batch_i]
        texts = [r["caption"] for r in batch_rows]
        t_feat, t_mask = encode_t5(tokenizer, t5, texts, device)
        t_c = encode_clap(clap, texts)
        for j, (row, name) in enumerate(zip(batch_rows, batch_names)):
            path = args.npz_dir / name
            data = dict(np.load(path, allow_pickle=False))
            if "clip_id" in data:
                stored = str(
                    data["clip_id"].item() if hasattr(data["clip_id"], "item") else data["clip_id"]
                )
                if stored != row["id"]:
                    raise SystemExit(f"clip_id mismatch {name}: {stored} vs {row['id']}")
            data["text_features"] = t_feat[j].astype(np.float32)
            data["text_features_c"] = t_c[j].astype(np.float32)
            data["text_attention_mask"] = t_mask[j].astype(np.int64)
            data["caption_sha256"] = np.asarray(sha_caption(row["caption"]))
            data["text_encoder_fingerprint"] = np.asarray(feature_fingerprint)
            atomic_savez(path, data)
            updated += 1
        atomic_json(
            args.progress_json,
            {
                "updated_or_skipped": updated,
                "total": len(rows),
                "todo_done": min(bi + args.batch_size, len(todo)),
                "todo_total": len(todo),
                "pct": updated / len(rows),
                "train_tsv_sha256": train_tsv_sha256,
                "cache_list_sha256": cache_list_sha256,
                "feature_encoder_fingerprint": feature_fingerprint,
            },
        )

    print(f"[DONE] inplace text reextract on {args.npz_dir} ({updated}/{len(rows)})", flush=True)
    if updated != len(rows):
        raise SystemExit(f"reextract completeness failure: {updated}/{len(rows)}")
    done_json = args.done_json or args.progress_json.with_suffix(
        args.progress_json.suffix + ".DONE"
    )
    atomic_json(
        done_json,
        {
            "status": "passed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "updated_or_verified": updated,
            "total": len(rows),
            "train_tsv": str(args.train_tsv),
            "train_tsv_sha256": train_tsv_sha256,
            "cache_list": str(args.cache_list),
            "cache_list_sha256": cache_list_sha256,
            "npz_dir": str(args.npz_dir),
            "feature_encoder_fingerprint": feature_fingerprint,
            "feature_encoder_contract": feature_contract,
        },
    )


if __name__ == "__main__":
    main()
