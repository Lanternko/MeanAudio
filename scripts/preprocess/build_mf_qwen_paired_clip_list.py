#!/usr/bin/env python3
"""Build the MF-vs-Qwen paired clip list for the captioner-isolating control.

The 036/037 arms compare a Music Flamingo corpus against the c2p0 Qwen corpus,
but they move three variables at once: captioner, corpus size (100,000 vs
251,599 rows) and audio selection (the two clip sets overlap by only 59.6%).
The 036 contract's own `not_controlled` field concedes the first two.

This script emits the intersection so a captioner-only control can be run:
identical audio, identical row count, identical recipe and budget, with only
the caption text differing.

Qwen row ids carry a trailing `_<caption_index>` suffix relative to the MF
segment ids (`00_1014400_segment_2_0` vs `26_282626_segment_0`), so the join
strips that suffix. Only caption index 0 (the c2p0 slot0 arm) is kept.

Emits a TSV per side with the shared clips in a single shared order, plus the
row indices into the MF training TSV so the already-extracted audio latents in
`mfshort100k_direct_noq_c2p0recipe_npz` can be reused for both sides.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

csv.field_size_limit(10**9)

MF_TSV = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/music_flamingo_slice10_100k_short_direct_train.tsv"
)
QWEN_TSV = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv"
)
SLOT_SUFFIX = re.compile(r"_(\d+)$")


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mf-tsv", type=Path, default=MF_TSV)
    ap.add_argument("--qwen-tsv", type=Path, default=QWEN_TSV)
    ap.add_argument("--slot", type=int, default=0, help="Qwen caption index to keep")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    mf_rows = read_tsv(args.mf_tsv)
    qwen_rows = read_tsv(args.qwen_tsv)
    print(f"[load] mf={len(mf_rows):,} qwen={len(qwen_rows):,}")

    # MF row order defines the npz index, so keep it as the canonical order.
    mf_index = {row["id"]: i for i, row in enumerate(mf_rows)}
    if len(mf_index) != len(mf_rows):
        raise SystemExit("[fatal] duplicate ids in the MF tsv; npz mapping would be ambiguous")

    qwen_by_clip: dict[str, str] = {}
    for row in qwen_rows:
        rid = row["id"]
        m = SLOT_SUFFIX.search(rid)
        if not m or int(m.group(1)) != args.slot:
            continue
        clip = rid[: m.start()]
        if clip in qwen_by_clip:
            raise SystemExit(f"[fatal] duplicate qwen clip at slot {args.slot}: {clip}")
        qwen_by_clip[clip] = row["caption"]
    print(f"[qwen] slot{args.slot} clips={len(qwen_by_clip):,}")

    shared = [row["id"] for row in mf_rows if row["id"] in qwen_by_clip]
    print(f"[join] shared clips={len(shared):,} ({len(shared) / len(mf_rows):.2%} of MF)")
    if not shared:
        raise SystemExit("[fatal] empty intersection")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mf_caption = {row["id"]: row["caption"] for row in mf_rows}

    # Shared clip order == MF tsv order, so npz row indices stay monotonic.
    npz_rows = [mf_index[cid] for cid in shared]

    for name, captions in (
        ("mf_shortdirect", mf_caption),
        (f"qwen_slot{args.slot}", qwen_by_clip),
    ):
        out = args.out_dir / f"paired59k_{name}_train.tsv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["id", "caption"])
            for cid in shared:
                w.writerow([cid, captions[cid]])
        print(f"[write] {out}  rows={len(shared):,}")

    idx_path = args.out_dir / "paired59k_mf_npz_row_index.txt"
    idx_path.write_text("\n".join(str(i) for i in npz_rows) + "\n")
    print(f"[write] {idx_path}")

    meta = {
        "mf_tsv": str(args.mf_tsv),
        "qwen_tsv": str(args.qwen_tsv),
        "qwen_slot": args.slot,
        "mf_rows": len(mf_rows),
        "qwen_slot_clips": len(qwen_by_clip),
        "shared_clips": len(shared),
        "order": "MF training tsv order (defines npz index)",
        "audio_latent_source": "/home/kojiek/exps_nvme/mfshort100k_direct_noq_c2p0recipe_npz",
        "audio_latent_note": (
            "Both arms share these audio latents; only text_features / "
            "text_features_c / text_attention_mask differ between arms."
        ),
    }
    meta_path = args.out_dir / "paired59k_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[write] {meta_path}")


if __name__ == "__main__":
    main()
