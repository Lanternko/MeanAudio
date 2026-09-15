#!/usr/bin/env python3
"""Build training inputs for the c2p0 slot0-clean (definition A) arm.

Runs only after the paired Luna spot check returned PASS. Fails closed on:
  - spot-check verdict other than PASS, or the corpus changed after it was sampled
  - regeneration loop not complete
  - corpus ids not equal to source ids (in order) minus unresolved.tsv
  - any row outside the regenerated set differing from the source
  - pandas (training loader) and csv (overlay patcher) parsing any caption differently
Writes the train TSV (byte copy of the candidate corpus), a cache list filtered to
the same rows (overlay/audio npz are named by cache-list name, not row index), and
a manifest with every sha. Unresolved rows are excluded from both files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
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
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--cache-list", type=Path, required=True)
    ap.add_argument("--regen-dir", type=Path, required=True)
    ap.add_argument("--spotcheck-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--operator-override", default=None,
                    help="operator decision text; required to build when the verdict is not PASS")
    args = ap.parse_args()

    report = json.loads((args.spotcheck_dir / "spotcheck_report.json").read_text())
    if report["verdict"] != "PASS" and not args.operator_override:
        raise SystemExit(f"[FAIL] spot check verdict {report['verdict']} != PASS (no operator override)")
    if report["verdict"] in ("INCOMPLETE",):
        raise SystemExit("[FAIL] spot check incomplete; override not allowed")
    sample = json.loads((args.spotcheck_dir / "sample.json").read_text())
    corpus_path = args.regen_dir / "slot0_regen_candidate_corpus.tsv"
    corpus_sha = sha256(corpus_path)
    if Path(sample["corpus"]).resolve() != corpus_path.resolve() or sample["corpus_sha256"] != corpus_sha:
        raise SystemExit("[FAIL] corpus is not the one the spot check sampled")
    if sample["source_sha256"] != sha256(args.source):
        raise SystemExit("[FAIL] source TSV changed since the spot check")
    summary = json.loads((args.regen_dir / "summary.json").read_text())
    if summary["status"] != "regen_complete_not_released":
        raise SystemExit("[FAIL] regeneration loop not complete")

    source = read_rows(args.source)
    names = [l.strip() for l in args.cache_list.open() if l.strip()]
    if len(source) != len(names):
        raise SystemExit(f"[FAIL] source {len(source)} != cache list {len(names)}")
    corpus = read_rows(corpus_path)
    accepted = json.loads((args.regen_dir / "accepted.json").read_text())
    unresolved = {r["id"] for r in read_rows(args.regen_dir / "unresolved.tsv")}

    keep_idx = [i for i, r in enumerate(source) if r["id"] not in unresolved]
    if [source[i]["id"] for i in keep_idx] != [r["id"] for r in corpus]:
        raise SystemExit("[FAIL] corpus ids != source ids minus unresolved (order included)")
    changed = 0
    for i, row in zip(keep_idx, corpus):
        if row["caption"] != source[i]["caption"]:
            if row["id"] not in accepted or accepted[row["id"]]["caption"] != row["caption"]:
                raise SystemExit(f"[FAIL] row {row['id']} changed without an accepted regeneration")
            changed += 1
        elif row["id"] in accepted:
            raise SystemExit(f"[FAIL] accepted regeneration for {row['id']} identical to source")
        if not row["caption"].strip():
            raise SystemExit(f"[FAIL] empty caption {row['id']}")
    if changed != len(accepted):
        raise SystemExit(f"[FAIL] changed rows {changed} != accepted {len(accepted)}")

    df = pd.read_csv(corpus_path, sep="\t").to_dict("records")  # exactly as extracted_audio.py reads it
    if len(df) != len(corpus):
        raise SystemExit(f"[FAIL] pandas rows {len(df)} != csv rows {len(corpus)}")
    bad = [c["id"] for d, c in zip(df, corpus) if str(d["id"]) != c["id"] or str(d["caption"]) != c["caption"]]
    if bad:
        raise SystemExit(f"[FAIL] pandas/csv caption parity broken for {len(bad)} rows, e.g. {bad[:3]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tsv = args.out_dir / "phase8_caption2p0_slot0clean_train.tsv"
    cache_out = args.out_dir / "cache_train.txt"
    tmp = train_tsv.with_name(f".{train_tsv.name}.tmp")
    shutil.copyfile(corpus_path, tmp)
    os.replace(tmp, train_tsv)
    tmp = cache_out.with_name(f".{cache_out.name}.tmp")
    tmp.write_text("".join(names[i] + "\n" for i in keep_idx))
    os.replace(tmp, cache_out)

    manifest = {
        "status": "arm_inputs_ready",
        "rows": len(corpus), "source_rows": len(source),
        "changed_rows": changed, "unresolved_excluded": sorted(unresolved),
        "train_tsv": str(train_tsv), "train_tsv_sha256": sha256(train_tsv),
        "cache_list": str(cache_out), "cache_list_sha256": sha256(cache_out),
        "source_tsv": str(args.source), "source_tsv_sha256": sample["source_sha256"],
        "source_cache_list": str(args.cache_list), "source_cache_list_sha256": sha256(args.cache_list),
        "candidate_corpus": str(corpus_path), "candidate_corpus_sha256": corpus_sha,
        "spotcheck_report_sha256": sha256(args.spotcheck_dir / "spotcheck_report.json"),
        "spotcheck": {"verdict": report["verdict"],
                      "base_flags": report["paired"]["base_flags"],
                      "residual_flags": report["paired"]["residual_flags"],
                      "n": report["paired"]["n"],
                      "regenerated_keep_rate": report["regenerated"]["keep_rate"]},
        "regen_summary": summary,
        "operator_override": ({"verdict": report["verdict"], "text": args.operator_override}
                              if report["verdict"] != "PASS" else None),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "changed_rows", "train_tsv_sha256", "cache_list_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
