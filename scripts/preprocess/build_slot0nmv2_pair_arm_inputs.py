#!/usr/bin/env python3
"""Build both arms of the slot0clean vs slot0nmv2 multi-seed quarter comparison.

slot0nmv2 is slot0clean with only the measurement phrases (BPM, meter, key/mode/chord
quality, Hz/dB, durations) removed; 3 rows lost every sentence and were excluded. The
paired comparison needs both arms on the same ids in the same order (one sampler
permutation per seed), so this writes:

  <nmv2-out>/   slot0nmv2 captions, slot0clean ids minus the 3 excluded rows
  <ctrl-out>/   slot0clean captions, byte-identical to slot0clean, same ids

Each dir gets the train TSV, the matching slice of the cache list, and a manifest.
Fails closed on: slot0clean manifest/TSV/cache drift; corpus ids != slot0clean ids minus
excluded (order included); any control row differing from slot0clean; any slot0nmv2
caption still matching MEASURE, or protected style tokens (decades, 808, 8-bit, ...) not
preserved row by row; any row outside the rewrite touching a caption; pandas (training
loader) and csv (overlay reader) disagreeing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

csv.field_size_limit(10**9)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewrite_slot0nmv2_measurements as R  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_arm(out_dir: Path, header: str, fields: list[str], rows: list[dict], names: list[str]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"phase8_caption2p0_{out_dir.parent.name}_train.tsv"
    cache = out_dir / "cache_train.txt"
    tmp = tsv.with_name(f".{tsv.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as out:
        out.write(header)
        w = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n",
                           quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, tsv)
    tmp = cache.with_name(f".{cache.name}.tmp")
    tmp.write_text("".join(n + "\n" for n in names))
    os.replace(tmp, cache)
    written = read_rows(tsv)
    if written != rows:
        raise SystemExit(f"[FAIL] {tsv} does not read back byte-equal")
    df = pd.read_csv(tsv, sep="\t").to_dict("records")  # exactly as extracted_audio.py reads it
    if len(df) != len(rows) or any(str(d["id"]) != r["id"] or str(d["caption"]) != r["caption"]
                                   for d, r in zip(df, rows)):
        raise SystemExit(f"[FAIL] pandas/csv parity broken in {tsv}")
    return tsv, cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-inputs", type=Path, default=Path.home() / "exps_nvme/slot0clean/arm_inputs")
    ap.add_argument("--rewrite-dir", type=Path, default=Path.home() / "exps_nvme/slot0nmv2/full")
    ap.add_argument("--nmv2-out", type=Path, default=Path.home() / "exps_nvme/slot0nmv2/arm_inputs")
    ap.add_argument("--ctrl-out", type=Path, default=Path.home() / "exps_nvme/slot0clean_nmv2matched/arm_inputs")
    args = ap.parse_args()

    cm = json.loads((args.clean_inputs / "manifest.json").read_text())
    if cm["status"] != "arm_inputs_ready":
        raise SystemExit("[FAIL] slot0clean manifest not ready")
    clean_tsv, clean_cache = Path(cm["train_tsv"]), Path(cm["cache_list"])
    if sha256(clean_tsv) != cm["train_tsv_sha256"] or sha256(clean_cache) != cm["cache_list_sha256"]:
        raise SystemExit("[FAIL] slot0clean TSV / cache list drifted since its manifest")
    corpus_path = args.rewrite_dir / "phase8_caption2p0_slot0nmv2_train.tsv"
    summary = json.loads((args.rewrite_dir / "summary.json").read_text())

    clean = read_rows(clean_tsv)
    names = [l.strip() for l in clean_cache.open() if l.strip()]
    if len(clean) != len(names):
        raise SystemExit(f"[FAIL] slot0clean {len(clean)} rows != cache {len(names)}")
    corpus = read_rows(corpus_path)
    excluded = {r["id"] for r in json.loads((args.rewrite_dir / "excluded_rows.json").read_text())}
    if len(excluded) != summary["excluded_all_sentences_dropped"]:
        raise SystemExit("[FAIL] excluded_rows.json disagrees with summary.json")
    keep = [i for i, r in enumerate(clean) if r["id"] not in excluded]
    if [clean[i]["id"] for i in keep] != [r["id"] for r in corpus]:
        raise SystemExit("[FAIL] slot0nmv2 ids != slot0clean ids minus excluded (order included)")

    # slot0nmv2 content gates
    left = [r["id"] for r in corpus if R.MEASURE.search(r["caption"])]
    if left:
        raise SystemExit(f"[FAIL] {len(left)} slot0nmv2 captions still hold a measurement, e.g. {left[:3]}")
    prot_bad, other_col_bad, changed = [], [], 0
    for i, r in zip(keep, corpus):
        src = clean[i]
        if any(r[k] != src[k] for k in src if k != "caption"):
            other_col_bad.append(r["id"])
        if r["caption"] != src["caption"]:
            changed += 1
            if Counter(R.protected(r["caption"])) != Counter(R.protected(src["caption"])):
                prot_bad.append(r["id"])
        elif R.MEASURE.search(src["caption"]):
            raise SystemExit(f"[FAIL] {r['id']} unchanged but source holds a measurement")
    if other_col_bad:
        raise SystemExit(f"[FAIL] {len(other_col_bad)} rows changed a non-caption column")
    if prot_bad:
        raise SystemExit(f"[FAIL] {len(prot_bad)} rows lost/altered protected style tokens, e.g. {prot_bad[:3]}")
    if changed != summary["rows_changed"]:
        raise SystemExit(f"[FAIL] changed rows {changed} != summary {summary['rows_changed']}")

    with clean_tsv.open(encoding="utf-8", newline="") as fh:
        header = fh.readline()
    fields = list(clean[0].keys())
    kept_names = [names[i] for i in keep]
    ctrl_rows = [clean[i] for i in keep]
    n_tsv, n_cache = write_arm(args.nmv2_out, header, fields, corpus, kept_names)
    c_tsv, c_cache = write_arm(args.ctrl_out, header, fields, ctrl_rows, kept_names)
    if sha256(n_cache) != sha256(c_cache):
        raise SystemExit("[FAIL] the two arms' cache lists differ")

    common = {
        "status": "arm_inputs_ready", "rows": len(corpus), "source_rows": len(clean),
        "excluded": sorted(excluded), "unresolved_excluded": [],
        "source_tsv": str(clean_tsv), "source_tsv_sha256": cm["train_tsv_sha256"],
        "source_cache_list": str(clean_cache), "source_cache_list_sha256": cm["cache_list_sha256"],
        "clean_manifest": str(args.clean_inputs / "manifest.json"),
        "clean_manifest_sha256": sha256(args.clean_inputs / "manifest.json"),
        "rewrite_script": R.__file__, "rewrite_script_sha256": sha256(Path(R.__file__)),
        "build_script": __file__, "build_script_sha256": sha256(Path(__file__)),
    }
    for arm, out, tsv, cache, n_changed, purpose in (
        ("c2p0_slot0nmv2", args.nmv2_out, n_tsv, n_cache, changed,
         "slot0clean with measurement phrases removed (Qwen3.6-27B, word-exact gates)"),
        ("c2p0_slot0clean_nmv2matched", args.ctrl_out, c_tsv, c_cache, 0,
         "paired control for slot0nmv2: slot0clean captions byte-identical, slot0nmv2 id set"),
    ):
        m = dict(common, arm=arm, purpose=purpose, changed_vs_source=n_changed,
                 train_tsv=str(tsv), train_tsv_sha256=sha256(tsv),
                 cache_list=str(cache), cache_list_sha256=sha256(cache))
        if arm == "c2p0_slot0nmv2":
            m["corpus_tsv"] = str(corpus_path)
            m["corpus_tsv_sha256"] = sha256(corpus_path)
            m["rewrite_summary"] = summary
        (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"arm": arm, "rows": m["rows"], "changed_vs_source": n_changed,
                          "train_tsv_sha256": m["train_tsv_sha256"], "cache_list_sha256": m["cache_list_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
