#!/usr/bin/env python3
"""Build 512-row TSVs + gt_cache for Qwen3-K vs LP-K vs shuffle-K S2 probes."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

OUT = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_pilot_n512")
FULL_TSV = Path("/home/kojiek/MeanAudio/data/phase8_caption2p0_k3_balanced_train.tsv")
CACHE = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt")
ASSIGN = OUT / "qwen3_vs_lp_k3.tsv"


def main() -> None:
    assign = {}
    with ASSIGN.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            assign[row["id"]] = row
    captions = {}
    order = []
    with (OUT / "slot0_c2p0.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") in assign:
                captions[rec["id"]] = rec["caption"]
                order.append(rec["id"])
    id_to_idx = {}
    with FULL_TSV.open() as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            id_to_idx[row["id"]] = i
    cache_lines = [ln.strip() for ln in CACHE.read_text().splitlines() if ln.strip()]
    if len(cache_lines) != len(id_to_idx):
        raise SystemExit(f"cache {len(cache_lines)} vs tsv {len(id_to_idx)}")

    rng = random.Random(14159265)
    qwen_qs = [int(assign[i]["qwen_k3"]) for i in order]
    shuf = qwen_qs[:]
    rng.shuffle(shuf)

    missing = [i for i in order if i not in id_to_idx]
    if missing:
        raise SystemExit(f"{len(missing)} ids not in full tsv e.g. {missing[:3]}")

    cache_out = []
    rows_qwen, rows_lp, rows_shuf = [], [], []
    for i, cid in enumerate(order):
        cap = captions[cid]
        idx = id_to_idx[cid]
        cache_out.append(cache_lines[idx])
        lp = assign[cid]["lp_k3"]
        rows_qwen.append({"id": cid, "caption": cap, "q_level": assign[cid]["qwen_k3"]})
        rows_lp.append({"id": cid, "caption": cap, "q_level": lp if lp not in (None, "") else "5"})
        rows_shuf.append({"id": cid, "caption": cap, "q_level": str(shuf[i])})

    (OUT / "probe_gt_cache.txt").write_text("\n".join(cache_out) + "\n")
    for name, rows in ("qwen3", rows_qwen), ("lp", rows_lp), ("shuffle", rows_shuf):
        path = OUT / f"probe_{name}_k3.tsv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "caption", "q_level"], delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print("wrote", path, len(rows))
    print("wrote", OUT / "probe_gt_cache.txt", len(cache_out))


if __name__ == "__main__":
    main()
