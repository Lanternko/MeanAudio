#!/usr/bin/env python3
"""Fail-closed full-corpus gate for the caption10s multisent experiment.

This validator is intentionally independent from generation.  It runs even when
generation has a .done marker, records cryptographic provenance, and optionally
checks the derived TSV/manifest against both the corpus and official TSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_multisent_first_entity_line import classify, n_sents  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--official-tsv", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--defects-tsv", type=Path, required=True)
    ap.add_argument("--train-tsv", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--reextract-report", type=Path)
    ap.add_argument("--cache-list", type=Path)
    ap.add_argument("--expected-corpus-sha256")
    args = ap.parse_args()

    failures: list[str] = []
    defect_rows: list[tuple[str, list[str], str]] = []
    tag_counts: Counter[str] = Counter()
    rows: list[dict] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    parse_errors = 0

    with args.corpus.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            cid = row.get("id")
            cap = row.get("caption")
            if not isinstance(cid, str) or not cid:
                failures.append(f"line {lineno}: missing id")
                continue
            if cid in seen:
                duplicate_ids.append(cid)
            seen.add(cid)
            tags = classify(cap if isinstance(cap, str) else "")
            if isinstance(cap, str):
                if row.get("n_chars") != len(cap):
                    tags.append("n_chars_mismatch")
                if row.get("n_words") != len(cap.split()):
                    tags.append("n_words_mismatch")
                if row.get("n_sents") != n_sents(cap):
                    tags.append("n_sents_mismatch")
            if tags:
                unique_tags = list(dict.fromkeys(tags))
                defect_rows.append((cid, unique_tags, (cap or "")[:240]))
                tag_counts.update(unique_tags)
            rows.append(row)

    with args.official_tsv.open(encoding="utf-8", newline="") as f:
        official_rows = list(csv.DictReader(f, delimiter="\t"))
    official_ids = [r["id"] for r in official_rows]
    official_set = set(official_ids)
    have_ids = [r["id"] for r in rows]
    have_set = set(have_ids)
    missing = official_set - have_set
    extra = have_set - official_set

    corpus_sha = sha256(args.corpus)
    official_sha = sha256(args.official_tsv)
    if parse_errors:
        failures.append(f"json_parse_errors={parse_errors}")
    if duplicate_ids:
        failures.append(f"duplicate_ids={len(duplicate_ids)}")
    if len(official_ids) != len(official_set):
        failures.append("official TSV contains duplicate ids")
    if missing:
        failures.append(f"missing_ids={len(missing)}")
    if extra:
        failures.append(f"extra_ids={len(extra)}")
    if have_ids != official_ids:
        failures.append("corpus id order differs from official TSV")
    if defect_rows:
        failures.append(f"caption_defects={len(defect_rows)}")
    if args.expected_corpus_sha256 and corpus_sha != args.expected_corpus_sha256:
        failures.append(
            f"corpus sha mismatch: expected {args.expected_corpus_sha256} got {corpus_sha}"
        )

    train_contract = None
    if bool(args.train_tsv) != bool(args.manifest):
        failures.append("--train-tsv and --manifest must be provided together")
    elif args.train_tsv and args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        train_sha = sha256(args.train_tsv)
        tsv_mismatches = 0
        with args.train_tsv.open(encoding="utf-8", newline="") as f:
            train_rows = list(csv.DictReader(f, delimiter="\t"))
        if len(train_rows) != len(rows):
            failures.append(f"train TSV rows={len(train_rows)} corpus rows={len(rows)}")
        for index, (train, corpus, official) in enumerate(
            zip(train_rows, rows, official_rows), 1
        ):
            if train.get("id") != corpus["id"] or train.get("caption") != corpus["caption"]:
                tsv_mismatches += 1
            for key, value in official.items():
                if key != "caption" and train.get(key) != value:
                    tsv_mismatches += 1
                    break
        expected_manifest = {
            "sha256": train_sha,
            "source_caption_sha256": corpus_sha,
            "source_official_sha256": official_sha,
            "rows": len(rows),
        }
        for key, value in expected_manifest.items():
            if manifest.get(key) != value:
                failures.append(
                    f"manifest {key} mismatch: expected {value!r} got {manifest.get(key)!r}"
                )
        if tsv_mismatches:
            failures.append(f"train TSV contract mismatches={tsv_mismatches}")
        train_contract = {
            "path": str(args.train_tsv),
            "sha256": train_sha,
            "rows": len(train_rows),
            "mismatches": tsv_mismatches,
            "manifest": str(args.manifest),
        }

    reextract_contract = None
    if bool(args.reextract_report) != bool(args.cache_list):
        failures.append("--reextract-report and --cache-list must be provided together")
    elif args.reextract_report and args.cache_list:
        if train_contract is None:
            failures.append("reextract validation requires --train-tsv and --manifest")
        reextract = json.loads(args.reextract_report.read_text(encoding="utf-8"))
        cache_sha = sha256(args.cache_list)
        expected_reextract = {
            "status": "passed",
            "total": len(rows),
            "updated_or_verified": len(rows),
            "train_tsv_sha256": train_contract["sha256"] if train_contract else None,
            "cache_list_sha256": cache_sha,
        }
        for key, value in expected_reextract.items():
            if reextract.get(key) != value:
                failures.append(
                    f"reextract report {key} mismatch: expected {value!r} "
                    f"got {reextract.get(key)!r}"
                )
        fingerprint = reextract.get("feature_encoder_fingerprint")
        feature_contract = reextract.get("feature_encoder_contract")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            failures.append("reextract report missing valid feature encoder fingerprint")
        if not isinstance(feature_contract, dict):
            failures.append("reextract report missing feature encoder contract")
        elif isinstance(fingerprint, str):
            encoded_contract = json.dumps(
                feature_contract, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            computed_fingerprint = hashlib.sha256(encoded_contract).hexdigest()
            if fingerprint != computed_fingerprint:
                failures.append(
                    "reextract feature encoder fingerprint does not match its contract"
                )
        reextract_contract = {
            "path": str(args.reextract_report),
            "cache_list": str(args.cache_list),
            "cache_list_sha256": cache_sha,
            "feature_encoder_fingerprint": fingerprint,
        }

    words = [len(r["caption"].split()) for r in rows if isinstance(r.get("caption"), str)]
    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {"path": str(args.corpus), "sha256": corpus_sha, "rows": len(rows)},
        "official_tsv": {
            "path": str(args.official_tsv),
            "sha256": official_sha,
            "rows": len(official_rows),
        },
        "train_contract": train_contract,
        "reextract_contract": reextract_contract,
        "identity": {
            "unique_ids": len(have_set),
            "missing_ids": len(missing),
            "extra_ids": len(extra),
            "duplicate_ids": len(duplicate_ids),
            "order_matches": have_ids == official_ids,
        },
        "quality": {
            "defect_rows": len(defect_rows),
            "tag_counts": dict(sorted(tag_counts.items())),
            "mean_words": round(st.mean(words), 3) if words else 0,
            "median_words": st.median(words) if words else 0,
        },
        "failures": failures,
    }

    defects_tmp = args.defects_tsv.with_name(f".{args.defects_tsv.name}.tmp.{os.getpid()}")
    with defects_tmp.open("w", encoding="utf-8") as f:
        f.write("id\ttags\tcaption_prefix\n")
        for cid, tags, prefix in defect_rows:
            safe_prefix = prefix.replace("\t", " ").replace("\n", " ")
            f.write(f"{cid}\t{','.join(tags)}\t{safe_prefix}\n")
    os.replace(defects_tmp, args.defects_tsv)
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    if failures:
        print("[GATE FAIL] corpus/derived-artifact validation failed", file=sys.stderr)
        return 2
    print("[GATE PASS] strict multisent corpus contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
