#!/usr/bin/env python3
"""Write NEW npz dir: copy audio latent fields from source, re-encode text from new TSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel
import laion_clap

T5_MODEL = "google/flan-t5-large"
MAX_TEXT_LEN = 128
SEQ_LEN = 77
CLAP_CKPT = Path(
    "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
)


def sha_caption(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_t5(tokenizer, model, texts, device):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LEN,
    ).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    B, L, D = out.shape
    if L < SEQ_LEN:
        out = torch.cat(
            [out, torch.zeros(B, SEQ_LEN - L, D, device=device, dtype=out.dtype)],
            dim=1,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", type=Path, required=True)
    ap.add_argument("--cache_list", type=Path, required=True)
    ap.add_argument("--source_npz_dir", type=Path, required=True)
    ap.add_argument("--dest_npz_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--progress_json", type=Path, required=True)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    with args.train_tsv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    names = [ln.strip() for ln in args.cache_list.open() if ln.strip()]
    if len(rows) != len(names):
        raise SystemExit(f"rows {len(rows)} != cache {len(names)}")
    if args.dry_run:
        rows, names = rows[: args.dry_run], names[: args.dry_run]

    args.dest_npz_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading T5...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    t5 = T5EncoderModel.from_pretrained(T5_MODEL).eval().to(device)
    print("Loading CLAP...", flush=True)
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap.load_ckpt(str(CLAP_CKPT), verbose=False)

    # skip already written
    todo_idx = [i for i, n in enumerate(names) if not (args.dest_npz_dir / n).exists()]
    print(f"total={len(names)} todo={len(todo_idx)}", flush=True)
    updated = len(names) - len(todo_idx)

    for bi in tqdm(range(0, len(todo_idx), args.batch_size), desc="reextract"):
        batch_i = todo_idx[bi : bi + args.batch_size]
        batch_rows = [rows[i] for i in batch_i]
        batch_names = [names[i] for i in batch_i]
        texts = [r["caption"] for r in batch_rows]
        t_feat, t_mask = encode_t5(tokenizer, t5, texts, device)
        t_c = encode_clap(clap, texts)

        for j, (row, name) in enumerate(zip(batch_rows, batch_names)):
            src = args.source_npz_dir / name
            dst = args.dest_npz_dir / name
            if not src.exists():
                raise SystemExit(f"missing source {src}")
            data = dict(np.load(src, allow_pickle=False))
            if "clip_id" in data:
                stored = str(
                    data["clip_id"].item()
                    if hasattr(data["clip_id"], "item")
                    else data["clip_id"]
                )
                if stored != row["id"]:
                    raise SystemExit(f"clip_id mismatch {name}: {stored} vs {row['id']}")
            data["text_features"] = t_feat[j].astype(np.float32)
            data["text_features_c"] = t_c[j].astype(np.float32)
            data["text_attention_mask"] = t_mask[j].astype(np.int64)
            data["caption_sha256"] = np.asarray(sha_caption(row["caption"]))
            tmp = dst.with_suffix(".npz.tmp")
            np.savez(tmp, **data)
            tmp.replace(dst)
            updated += 1

        args.progress_json.write_text(
            json.dumps({"updated": updated, "total": len(names), "pct": updated / len(names)})
            + "\n"
        )

    print(f"[DONE] wrote {updated}/{len(names)} -> {args.dest_npz_dir}")


if __name__ == "__main__":
    main()
