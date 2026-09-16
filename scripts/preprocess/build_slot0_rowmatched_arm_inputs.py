#!/usr/bin/env python3
"""Build the row-matched slot0 control corpus for the paired-seed comparison.

slot0nm (057) excludes 1 row, which shifts the sampler permutation relative to the
historic slot0 arm. The paired comparison needs both arms on the same id set, so this
builds slot0 -- original captions, byte-identical to the source -- restricted to exactly
the ids slot0nm trains on.

Fails closed on: the slot0nm manifest not being ready; the source TSV having drifted from
the sha the slot0nm build recorded; the kept ids not equalling the slot0nm ids in order;
any caption differing from the source; pandas (training loader) and csv (overlay reader)
disagreeing on any row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

csv.field_size_limit(10**9)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot0nm-inputs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    nm = json.loads((args.slot0nm_inputs / "manifest.json").read_text())
    if nm["status"] != "arm_inputs_ready":
        raise SystemExit("[FAIL] slot0nm manifest not ready")
    source = Path(nm["source_tsv"])
    cache_list = Path(nm["source_cache_list"])
    if sha256(source) != nm["source_tsv_sha256"]:
        raise SystemExit("[FAIL] source TSV drifted since the slot0nm build")
    if sha256(cache_list) != nm["source_cache_list_sha256"]:
        raise SystemExit("[FAIL] source cache list drifted since the slot0nm build")

    rows = read_rows(source)
    names = [l.strip() for l in cache_list.open() if l.strip()]
    if len(rows) != len(names):
        raise SystemExit(f"[FAIL] source {len(rows)} != cache {len(names)}")
    excluded = set(nm["excluded"])
    keep_idx = [i for i, r in enumerate(rows) if r["id"] not in excluded]
    nm_rows = read_rows(Path(nm["train_tsv"]))
    if [rows[i]["id"] for i in keep_idx] != [r["id"] for r in nm_rows]:
        raise SystemExit("[FAIL] kept ids != slot0nm ids (order included)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tsv = args.out_dir / "phase8_caption2p0_slot0_rowmatched_train.tsv"
    cache_out = args.out_dir / "cache_train.txt"
    with source.open(encoding="utf-8", newline="") as fh:
        header = fh.readline()
    tmp = train_tsv.with_name(f".{train_tsv.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as out:
        out.write(header)
        w = csv.DictWriter(out, fieldnames=list(rows[0].keys()), delimiter="\t",
                           lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        for i in keep_idx:
            w.writerow(rows[i])
    os.replace(tmp, train_tsv)
    tmp = cache_out.with_name(f".{cache_out.name}.tmp")
    tmp.write_text("".join(names[i] + "\n" for i in keep_idx))
    os.replace(tmp, cache_out)

    written = read_rows(train_tsv)
    if len(written) != len(keep_idx):
        raise SystemExit(f"[FAIL] wrote {len(written)} rows, expected {len(keep_idx)}")
    for i, w_row in zip(keep_idx, written):
        if w_row != rows[i]:
            raise SystemExit(f"[FAIL] row {rows[i]['id']} not byte-equal to the source row")
    df = pd.read_csv(train_tsv, sep="\t").to_dict("records")  # exactly as extracted_audio.py reads it
    if len(df) != len(written) or any(str(d["id"]) != w["id"] or str(d["caption"]) != w["caption"]
                                      for d, w in zip(df, written)):
        raise SystemExit("[FAIL] pandas/csv parity broken")

    manifest = {
        "status": "arm_inputs_ready",
        "arm": "c2p0_slot0_rowmatched",
        "purpose": "paired-seed control for slot0nm; original slot0 captions, slot0nm id set",
        "rows": len(written), "source_rows": len(rows), "excluded": sorted(excluded),
        "changed_vs_source": 0,
        "train_tsv": str(train_tsv), "train_tsv_sha256": sha256(train_tsv),
        "cache_list": str(cache_out), "cache_list_sha256": sha256(cache_out),
        "source_tsv": str(source), "source_tsv_sha256": nm["source_tsv_sha256"],
        "source_cache_list": str(cache_list), "source_cache_list_sha256": nm["source_cache_list_sha256"],
        "slot0nm_manifest": str(args.slot0nm_inputs / "manifest.json"),
        "slot0nm_train_tsv_sha256": nm["train_tsv_sha256"],
        "slot0nm_cache_list_sha256": nm["cache_list_sha256"],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "train_tsv_sha256", "cache_list_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
