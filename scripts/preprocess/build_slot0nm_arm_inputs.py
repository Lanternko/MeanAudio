#!/usr/bin/env python3
"""Build training inputs for c2p0 slot0nm (slot0clean + numbers/key/meter removed).

Fails closed on: rewrite not complete; any caption still matching MEASURE; id order not
equal to source ids minus excluded_rows.json; pandas (training loader) and csv (overlay
patcher) disagreeing on any caption. Writes the train TSV (byte copy), a cache list
filtered to the same rows, and a manifest with every sha.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewrite_slot0nm_no_measurements as R  # noqa: E402

csv.field_size_limit(10**9)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--cache-list", type=Path, required=True)
    ap.add_argument("--clean-manifest", type=Path, required=True, help="slot0clean arm manifest (provenance)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    summary = json.loads((args.rewrite_dir / "summary.json").read_text())
    if summary["status"] != "rewrite_complete_not_released":
        raise SystemExit("[FAIL] rewrite not complete")
    corpus_path = args.rewrite_dir / "phase8_caption2p0_slot0nm_train.tsv"
    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8", newline=""), delimiter="\t"))  # noqa: E731
    corpus, source = read(corpus_path), read(args.source)
    names = [l.strip() for l in args.cache_list.open() if l.strip()]
    if len(source) != len(names):
        raise SystemExit(f"[FAIL] source {len(source)} != cache {len(names)}")
    excluded = {r["id"] for r in json.loads((args.rewrite_dir / "excluded_rows.json").read_text())}
    clean_manifest = json.loads(args.clean_manifest.read_text())
    excluded |= set(clean_manifest["unresolved_excluded"])
    keep_idx = [i for i, r in enumerate(source) if r["id"] not in excluded]
    if [source[i]["id"] for i in keep_idx] != [r["id"] for r in corpus]:
        raise SystemExit("[FAIL] corpus ids != source ids minus excluded (order included)")
    bad = [r["id"] for r in corpus if R.MEASURE.search(r["caption"]) or not r["caption"].strip()]
    if bad:
        raise SystemExit(f"[FAIL] {len(bad)} captions still hold a measurement or are empty, e.g. {bad[:3]}")
    df = pd.read_csv(corpus_path, sep="\t").to_dict("records")
    if len(df) != len(corpus) or any(str(d["id"]) != c["id"] or str(d["caption"]) != c["caption"]
                                     for d, c in zip(df, corpus)):
        raise SystemExit("[FAIL] pandas/csv parity broken")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tsv = args.out_dir / "phase8_caption2p0_slot0nm_train.tsv"
    cache_out = args.out_dir / "cache_train.txt"
    tmp = train_tsv.with_name(f".{train_tsv.name}.tmp")
    shutil.copyfile(corpus_path, tmp)
    os.replace(tmp, train_tsv)
    tmp = cache_out.with_name(f".{cache_out.name}.tmp")
    tmp.write_text("".join(names[i] + "\n" for i in keep_idx))
    os.replace(tmp, cache_out)
    changed_vs_source = sum(1 for i, r in zip(keep_idx, corpus) if r["caption"] != source[i]["caption"])
    manifest = {
        "status": "arm_inputs_ready", "rows": len(corpus), "source_rows": len(source),
        "excluded": sorted(excluded), "changed_vs_source": changed_vs_source,
        "train_tsv": str(train_tsv), "train_tsv_sha256": sha256(train_tsv),
        "cache_list": str(cache_out), "cache_list_sha256": sha256(cache_out),
        "source_tsv": str(args.source), "source_tsv_sha256": sha256(args.source),
        "source_cache_list": str(args.cache_list), "source_cache_list_sha256": sha256(args.cache_list),
        "rewrite_summary": summary,
        "rewrite_script_sha256": sha256(Path(R.__file__)),
        "sentence_log_sha256": sha256(args.rewrite_dir / "sentence_log.jsonl"),
        "clean_manifest": str(args.clean_manifest), "clean_manifest_sha256": sha256(args.clean_manifest),
        "clean_spotcheck": clean_manifest["spotcheck"], "clean_operator_override": clean_manifest.get("operator_override"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "changed_vs_source", "train_tsv_sha256", "cache_list_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
