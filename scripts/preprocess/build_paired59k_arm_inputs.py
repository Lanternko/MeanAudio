#!/usr/bin/env python3
"""Assemble the training inputs for the paired59k captioner-only control.

Both arms train on the SAME audio latents (the MF 100k extraction) and the same
59,614 rows in the same order; only the caption text differs. That is the whole
point of the control, so this script builds the bindings rather than letting the
loader fall back to sequential i.npz -- the fallback is what silently misaligned
the Phase 9 multi-cap runs.

Outputs, all under --out-dir:
  cache_train.txt            audio NPZ names ({mf_row}.npz), one per paired row
  mf_recaption_train.tsv     MF arm captions, rebuilt from the recaption jsonl
  qwen_text_overlay/         symlinks into text_overlays/true_random, renamed to
                             match cache_train.txt so text_npz_dir lines up
  bindings.json              hashes + audit results

The Qwen arm needs no encoding at all: slot 0 of the existing true_random
3-stack IS the c2p0 slot0 caption, so the arm reads the very same float32
features the c2p0 runs used. The MF arm still needs a fresh encode because its
captions did not exist until today.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

csv.field_size_limit(10**9)

SLOT_SUFFIX = re.compile(r"_(\d+)$")
QWEN_TRAIN_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv")
QWEN_CACHE_LIST = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt")
TRUE_RANDOM = Path("/home/kojiek/text_overlays/true_random")
PAIRED_DIR = Path("/home/kojiek/exps_nvme/paired59k_mf_qwen")


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
    ap.add_argument("--jsonl", type=Path,
                    default=Path("/home/kojiek/eval_output/mf_recaption_paired59k_v2/caption.jsonl"))
    ap.add_argument("--paired-mf-tsv", type=Path, default=PAIRED_DIR / "paired59k_mf_shortdirect_train.tsv")
    ap.add_argument("--paired-qwen-tsv", type=Path, default=PAIRED_DIR / "paired59k_qwen_slot0_train.tsv")
    ap.add_argument("--row-index", type=Path, default=PAIRED_DIR / "paired59k_mf_npz_row_index.txt")
    ap.add_argument("--out-dir", type=Path, default=PAIRED_DIR / "arm_inputs")
    ap.add_argument("--audit-n", type=int, default=200)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # -- paired row order (both arms share it) ------------------------------
    mf_rows = list(csv.DictReader(args.paired_mf_tsv.open(), delimiter="\t"))
    qwen_rows = list(csv.DictReader(args.paired_qwen_tsv.open(), delimiter="\t"))
    order = [r["id"] for r in mf_rows]
    n = len(order)
    if [r["id"] for r in qwen_rows] != order:
        raise SystemExit("[FAIL] MF and Qwen paired TSVs are not in the same row order")
    audio_rows = [int(line) for line in args.row_index.read_text().split()]
    if len(audio_rows) != n:
        raise SystemExit(f"[FAIL] row index has {len(audio_rows)} entries, TSVs have {n}")
    print(f"paired rows: {n}")

    # -- 1. audio cache list -------------------------------------------------
    cache_path = out / "cache_train.txt"
    cache_path.write_text("".join(f"{i}.npz\n" for i in audio_rows))
    print(f"wrote {cache_path}")

    # -- 2. MF recaption TSV -------------------------------------------------
    caps: dict[str, str] = {}
    for line in args.jsonl.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        caps[row["id"]] = caption_of(row)
    missing = [i for i in order if not caps.get(i, "").strip()]
    if missing:
        raise SystemExit(f"[FAIL] {len(missing)} paired clips have no recaption, e.g. {missing[:3]}")
    mf_tsv = out / "mf_recaption_train.tsv"
    with mf_tsv.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "caption"])
        for i in order:
            w.writerow([i, caps[i].replace("\t", " ").replace("\n", " ").strip()])
    print(f"wrote {mf_tsv}")

    # -- 3. Qwen text-overlay symlink farm -----------------------------------
    # true_random files are named by the c2p0 cache list, which is indexed by the
    # 251,599-row Qwen training TSV. Rename (by symlink) into the audio cache
    # list's namespace, because ExtractedAudio uses ONE name list for both dirs.
    qwen_tsv_rows = list(csv.DictReader(QWEN_TRAIN_TSV.open(), delimiter="\t"))
    qwen_names = [ln.strip() for ln in QWEN_CACHE_LIST.open() if ln.strip()]
    if len(qwen_tsv_rows) != len(qwen_names):
        raise SystemExit(f"[FAIL] qwen tsv {len(qwen_tsv_rows)} vs cache {len(qwen_names)}")
    base_to_name: dict[str, str] = {}
    for row, name in zip(qwen_tsv_rows, qwen_names):
        base = SLOT_SUFFIX.sub("", row["id"])
        base_to_name.setdefault(base, name)
    overlay_dir = out / "qwen_text_overlay"
    overlay_dir.mkdir(exist_ok=True)
    linked = 0
    for clip_id, audio_row in zip(order, audio_rows):
        src_name = base_to_name.get(clip_id)
        if src_name is None:
            raise SystemExit(f"[FAIL] no c2p0 overlay row for {clip_id}")
        src = TRUE_RANDOM / src_name
        if not src.is_file():
            raise SystemExit(f"[FAIL] overlay file missing: {src}")
        dst = overlay_dir / f"{audio_row}.npz"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        linked += 1
    print(f"linked {linked} overlay files into {overlay_dir}")

    # -- 4. audit: the overlay slot 0 really is this row's Qwen caption -------
    import random
    rng = random.Random(4242)
    sample = rng.sample(range(n), min(args.audit_n, n))
    bad = []
    for idx in sample:
        clip_id, audio_row = order[idx], audio_rows[idx]
        with np.load(overlay_dir / f"{audio_row}.npz", allow_pickle=False) as data:
            stored_id = str(data["clip_id"].item())
            hashes = str(data["caption_sha256"].item()).split(",")
        # overlay clip_id keeps the Qwen "_<slot>" suffix; the paired TSVs strip it
        if SLOT_SUFFIX.sub("", stored_id) != clip_id:
            bad.append(f"{clip_id}: overlay holds {stored_id}")
            continue
        if hashes[0] != sha_text(qwen_rows[idx]["caption"]):
            bad.append(f"{clip_id}: slot0 hash != paired Qwen caption")
    print(f"overlay audit: {len(sample) - len(bad)}/{len(sample)} ok")
    for b in bad[:5]:
        print(f"  [MISMATCH] {b}")
    if bad:
        raise SystemExit(f"[FAIL] overlay pairing audit failed ({len(bad)}/{len(sample)})")

    # -- 5. bindings ---------------------------------------------------------
    bindings = {
        "rows": n,
        "audio_npz_dir": "/home/kojiek/exps_nvme/mfshort100k_direct_noq_c2p0recipe_npz",
        "cache_train": str(cache_path),
        "cache_train_sha256": sha_file(cache_path),
        "mf_recaption_tsv": str(mf_tsv),
        "mf_recaption_tsv_sha256": sha_file(mf_tsv),
        "mf_recaption_jsonl_sha256": sha_file(args.jsonl),
        "qwen_paired_tsv": str(args.paired_qwen_tsv),
        "qwen_paired_tsv_sha256": sha_file(args.paired_qwen_tsv),
        "qwen_text_overlay_dir": str(overlay_dir),
        "qwen_overlay_source": str(TRUE_RANDOM),
        "qwen_overlay_cap_index_fixed": 0,
        "overlay_audit_sampled": len(sample),
        "overlay_audit_failures": len(bad),
        "note": ("Audio NPZs carry no clip_id and the overlay's clip_id keeps the Qwen "
                 "_<slot> suffix, so require_text_overlay cannot be enabled; this audit "
                 "is the substitute and must be re-run if any input changes."),
    }
    (out / "bindings.json").write_text(json.dumps(bindings, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out / 'bindings.json'}")
    print("\n=== PREP OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
