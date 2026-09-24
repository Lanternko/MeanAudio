#!/usr/bin/env python3
"""Build training TSV from new 10s captions, preserving q_level/order from official matched."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_tsv", type=Path, required=True)
    ap.add_argument("--caption_jsonl", type=Path, required=True)
    ap.add_argument("--out_tsv", type=Path, required=True)
    ap.add_argument("--out_manifest", type=Path, required=True)
    args = ap.parse_args()

    caps = {}
    duplicate_caps = []
    nulls = 0
    with args.caption_jsonl.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec.get("id")
            cap = rec.get("caption")
            if not cid:
                continue
            if not cap:
                nulls += 1
                continue
            if cid in caps:
                duplicate_caps.append(cid)
            caps[cid] = cap.strip()

    with args.official_tsv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
        fields = list(rows[0].keys()) if rows else ["id", "caption", "q_level"]

    official_ids = [r["id"] for r in rows]
    if len(official_ids) != len(set(official_ids)):
        raise SystemExit("official TSV contains duplicate ids")
    if duplicate_caps:
        raise SystemExit(f"caption JSONL contains duplicate ids: {len(duplicate_caps)}")
    extra = set(caps) - set(official_ids)
    if extra:
        raise SystemExit(f"extra captions for {len(extra)} ids e.g. {sorted(extra)[:3]}")

    missing = []
    out_rows = []
    for r in rows:
        cid = r["id"]
        if cid not in caps:
            missing.append(cid)
            continue
        nr = dict(r)
        nr["caption"] = caps[cid]
        out_rows.append(nr)

    if missing:
        raise SystemExit(f"missing captions for {len(missing)} ids e.g. {missing[:3]}")
    if nulls:
        print(f"[WARN] null captions skipped in jsonl: {nulls}")

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_tsv = args.out_tsv.with_name(f".{args.out_tsv.name}.tmp.{os.getpid()}")
    with tmp_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    os.replace(tmp_tsv, args.out_tsv)
    out_sha = sha256(args.out_tsv)

    man = {
        "status": "passed",
        "rows": len(out_rows),
        "nulls_in_jsonl": nulls,
        "out_tsv": str(args.out_tsv),
        "sha256": out_sha,
        "source_caption_sha256": sha256(args.caption_jsonl),
        "source_official_sha256": sha256(args.official_tsv),
        "source_official_tsv": str(args.official_tsv),
        "source_caption_jsonl": str(args.caption_jsonl),
        "window_sec": 10,
        "captioner": "Qwen2.5-Omni-3B first-10s-crop",
    }
    tmp_manifest = args.out_manifest.with_name(
        f".{args.out_manifest.name}.tmp.{os.getpid()}"
    )
    tmp_manifest.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, args.out_manifest)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
