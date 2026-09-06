#!/usr/bin/env python3
"""Assemble the training inputs for the Music Flamingo full-coverage arms.

paired59k answered "which captioner writes better captions" on the 59,614 clips
MF and c2p0 shared. It could not answer "is MF viable at c2p0's scale", because
MF only covered 23.7% of the corpus. The full-coverage recaption closes that
gap: every one of c2p0's 251,599 clips now has a short_direct_v2 caption, so an
MF arm can be trained on the exact rows, in the exact order, against the exact
audio latents the c2p0 arms used.

Outputs, all under --out-dir:
  mf_fullcov_train.tsv   id + caption, 251,599 rows, in c2p0 cache-list order
  bindings.json          hashes, coverage counts and the audit result

Two things are easy to get silently wrong here and are both handled:

1. id normalization, which is asymmetric. c2p0 TSV ids carry a trailing slot
   suffix (00_1014400_segment_2_0); MF / recaption ids do not
   (00_1014400_segment_2). A raw set intersection returns 0 rows and raises
   nothing -- see memory reference_c2p0_id_slot_suffix.md. So exactly one
   trailing _<digits> is stripped, and only from the c2p0 side: MF ids end in
   the segment number, so normalizing them too collapses every segment of a
   track onto one key and silently drops 82% of the corpus.

2. which id ends up in the TSV. The emitted id is the c2p0 id *with* its slot
   suffix, because that is what the audio NPZs store in clip_id. Matching it
   lets the arms run with require_text_overlay=true, which paired59k could not
   do (its audio NPZs predate the clip_id field). That closes the gating gap
   recorded in memory project_c2p0_corpus_provenance_2026_08_26.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

csv.field_size_limit(10**9)

SLOT_SUFFIX = re.compile(r"_\d+$")
C2P0_TRAIN_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv")
C2P0_CACHE_LIST = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt")
C2P0_AUDIO_NPZ = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz")
RECAPTION_JSONL = Path("/home/kojiek/eval_output/mf_recaption_full_coverage/caption.jsonl")
OUT_DIR = Path("/home/kojiek/exps_nvme/mf_full_coverage/arm_inputs")


def base_id(clip_id: str) -> str:
    return SLOT_SUFFIX.sub("", clip_id)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def caption_of(row: dict) -> str:
    out = row.get("output")
    if isinstance(out, dict):
        return out.get("text", "") or ""
    if isinstance(out, str):
        return out
    return row.get("raw_text", "") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=RECAPTION_JSONL)
    ap.add_argument("--c2p0-tsv", type=Path, default=C2P0_TRAIN_TSV)
    ap.add_argument("--cache-list", type=Path, default=C2P0_CACHE_LIST)
    ap.add_argument("--audio-npz", type=Path, default=C2P0_AUDIO_NPZ)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--audit-n", type=int, default=300)
    ap.add_argument("--require-enforced", action="store_true",
                    help="reject clips whose caption never passed the 77-token window check")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # -- row order comes from c2p0, unchanged --------------------------------
    rows = list(csv.DictReader(args.c2p0_tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
    names = [line.strip() for line in args.cache_list.open() if line.strip()]
    if len(names) != len(rows):
        raise SystemExit(f"[FAIL] cache list {len(names)} != c2p0 tsv {len(rows)}")
    if len(set(names)) != len(names):
        raise SystemExit("[FAIL] duplicate names in cache list")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("[FAIL] duplicate ids in c2p0 tsv")

    # -- MF captions, keyed by normalized id ---------------------------------
    captions: dict[str, str] = {}
    enforced: dict[str, bool] = {}
    n_lines = n_ok = 0
    with args.jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            cap = caption_of(rec).strip()
            if not cap:
                continue
            n_ok += 1
            # The recaption id is already the MF-native form and must NOT be
            # normalized: MF ids end in the segment number (26_282626_segment_0),
            # so stripping a trailing _<digits> here collapses every segment of a
            # track onto one key. Only the c2p0 side carries a slot suffix.
            cid = str(rec["id"])
            captions[cid] = cap
            enforced[cid] = bool(rec.get("enforced_ok"))
    print(f"recaption jsonl: lines={n_lines} usable={n_ok} unique_ids={len(captions)}")

    missing = [r["id"] for r in rows if base_id(r["id"]) not in captions]
    if missing:
        raise SystemExit(
            f"[FAIL] {len(missing)} of {len(rows)} c2p0 clips have no MF caption "
            f"(first: {missing[:3]}); recaption is not complete"
        )

    not_enforced = [r["id"] for r in rows if not enforced[base_id(r["id"])]]
    print(f"coverage: {len(rows)}/{len(rows)}; enforced_ok "
          f"{len(rows) - len(not_enforced)} ({100 * (1 - len(not_enforced) / len(rows)):.2f}%)")
    if args.require_enforced and not_enforced:
        raise SystemExit(f"[FAIL] --require-enforced set but {len(not_enforced)} rows are best-effort")

    # -- emit the training TSV in cache-list order ---------------------------
    train_tsv = out / "mf_fullcov_train.tsv"
    tmp = train_tsv.with_suffix(".tsv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL,
                       lineterminator="\n")
        w.writerow(["id", "caption"])
        for r in rows:
            w.writerow([r["id"], captions[base_id(r["id"])]])
    tmp.replace(train_tsv)

    written = list(csv.DictReader(train_tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
    if len(written) != len(rows):
        raise SystemExit(f"[FAIL] wrote {len(written)} rows, expected {len(rows)}")

    # -- audit: the emitted id must equal the audio NPZ's clip_id ------------
    # This is the binding require_text_overlay=true will enforce for every row
    # at training time; check a sample now so a mismatch fails here, cheaply,
    # instead of 100k iterations in.
    import random

    import numpy as np

    bad = []
    for i in random.Random(0).sample(range(len(rows)), min(args.audit_n, len(rows))):
        data = np.load(args.audio_npz / names[i], allow_pickle=False)
        clip_id = str(data["clip_id"].item())
        if clip_id != written[i]["id"]:
            bad.append((i, clip_id, written[i]["id"]))
        if data["mean"].shape != (312, 20):
            bad.append((i, "mean shape", str(data["mean"].shape)))
    if bad:
        for b in bad[:5]:
            print(f"  [MISMATCH] {b}")
        raise SystemExit(f"[FAIL] clip_id audit failed on {len(bad)} of {args.audit_n}")
    print(f"clip_id audit: {args.audit_n}/{args.audit_n} ok")

    caption_lengths = [len(r["caption"]) for r in written]
    unique_caps = len({r["caption"] for r in written})
    bindings = {
        "status": "passed",
        "rows": len(written),
        "train_tsv": str(train_tsv),
        "train_tsv_sha256": sha_file(train_tsv),
        "cache_list": str(args.cache_list),
        "cache_list_sha256": sha_file(args.cache_list),
        "audio_npz_dir": str(args.audio_npz),
        "c2p0_tsv": str(args.c2p0_tsv),
        "c2p0_tsv_sha256": sha_file(args.c2p0_tsv),
        "recaption_jsonl": str(args.jsonl),
        "recaption_jsonl_sha256": sha_file(args.jsonl),
        "id_convention": "c2p0 id with slot suffix; equals audio npz clip_id, so "
                         "require_text_overlay=true is usable",
        "coverage": {"c2p0_rows": len(rows), "mf_captioned": len(rows), "missing": 0},
        "enforced_ok_rows": len(rows) - len(not_enforced),
        "best_effort_rows": len(not_enforced),
        "caption_stats": {
            "unique": unique_caps,
            "unique_rate": round(unique_caps / len(written), 6),
            "chars_min": min(caption_lengths),
            "chars_mean": round(sum(caption_lengths) / len(caption_lengths), 1),
            "chars_max": max(caption_lengths),
        },
        "clip_id_audit": {"sampled": args.audit_n, "mismatches": 0},
        "first_row_sha256": sha_text(written[0]["caption"]),
    }
    (out / "bindings.json").write_text(json.dumps(bindings, indent=1, sort_keys=True))
    print(json.dumps({k: bindings[k] for k in
                      ("rows", "enforced_ok_rows", "best_effort_rows", "caption_stats")}, indent=1))
    print(f"PREP OK -> {train_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
