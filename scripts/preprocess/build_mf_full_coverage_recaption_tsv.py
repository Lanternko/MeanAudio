#!/usr/bin/env python3
"""Build the MF recaption target list covering the whole c2p0 clip set.

The paired59k control only recaptioned the 59,614 clips MF and c2p0 already
shared. That leaves MF at 59,614 clips against Qwen's 251,599, which is the
one asymmetry the control cannot absorb -- it decides coverage, not caption
quality. This emits every c2p0 clip so MF ends up on equal footing.

id normalization matters and is easy to get silently wrong: c2p0 TSV ids carry
a trailing slot suffix (00_1014400_segment_2_0) while the MF / paired59k ids do
not (26_282626_segment_0). A raw set intersection of the two returns 0 rows and
raises nothing. Strip exactly one trailing _<digits> -- two strips would eat the
segment number as well. See memory reference_c2p0_id_slot_suffix.md.
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


def base_id(clip_id: str) -> str:
    return SLOT_SUFFIX.sub("", clip_id)


def read_ids(tsv: Path, normalize: bool) -> list[str]:
    out, seen = [], set()
    with tsv.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cid = base_id(row["id"]) if normalize else row["id"]
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2p0-tsv", type=Path, required=True)
    ap.add_argument("--done-jsonl", type=Path, required=True,
                    help="captions already produced; counted, never re-emitted as new work")
    ap.add_argument("--wav-root", type=Path,
                    default=Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio"))
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--out-bindings", type=Path, required=True)
    args = ap.parse_args()

    targets = read_ids(args.c2p0_tsv, normalize=True)
    print(f"[c2p0]   {len(targets)} distinct clips")

    done = set()
    with args.done_jsonl.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(str(rec["id"]))
    print(f"[done]   {len(done)} already captioned")

    stray = done - set(targets)
    if stray:
        raise SystemExit(f"[FAIL] {len(stray)} captioned ids are not c2p0 clips, e.g. {sorted(stray)[:3]}")

    # One directory listing beats 251k stat calls on the HDD.
    have_wav = {p.stem for p in args.wav_root.iterdir() if p.suffix == ".wav"}
    missing = [c for c in targets if c not in have_wav]
    print(f"[wav]    {len(have_wav)} files under {args.wav_root}, {len(missing)} targets missing")
    if missing:
        raise SystemExit(f"[FAIL] no wav for e.g. {missing[:3]}")

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "caption"], delimiter="\t")
        w.writeheader()
        for cid in targets:
            w.writerow({"id": cid, "caption": ""})

    backlog = len(targets) - len(done)
    bindings = {
        "c2p0_tsv": str(args.c2p0_tsv),
        "c2p0_tsv_sha256": sha256(args.c2p0_tsv),
        "done_jsonl": str(args.done_jsonl),
        "done_jsonl_sha256": sha256(args.done_jsonl),
        "out_tsv": str(args.out_tsv),
        "out_tsv_sha256": sha256(args.out_tsv),
        "wav_root": str(args.wav_root),
        "targets": len(targets),
        "already_captioned": len(done),
        "backlog": backlog,
        "note": "caption column is empty on purpose: short_direct_v2 captions the audio "
                "directly and the loader never reads lpmc_caption. Run the captioner with "
                "--resume against an out_dir seeded with done_jsonl so the uniqueness set "
                "spans the whole corpus.",
    }
    args.out_bindings.write_text(json.dumps(bindings, indent=1, sort_keys=True) + "\n")

    print(f"[out]    {args.out_tsv}")
    print(f"[PREP OK] targets={len(targets)} done={len(done)} backlog={backlog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
