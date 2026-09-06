#!/usr/bin/env python3
"""Acceptance gate for the paired59k Music Flamingo recaption (R1 + R2).

R1 (uniqueness) and R2 (77-token window) are the two reasons the recaption was
run at all: the original corpus was 73% unique and 79% truncated. This checks
the finished caption.jsonl against those targets plus the integrity properties
that would silently poison training (missing clips, duplicate ids, empty text).

Exit 0 = PASS (safe to build the overlay and launch), 1 = FAIL.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

csv.field_size_limit(10**9)

# Thresholds. Uniqueness is a soft gate: greedy decoding on homogeneous EDM
# clips collides no matter how many attempts are allowed, and 92% still beats
# the 73% corpus this run exists to replace. The window gate is hard: an
# over-window caption is exactly the R2 defect.
MIN_UNIQUE_PCT = 88.0
MAX_TOKENS = 77


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
    ap.add_argument("--tsv", type=Path,
                    default=Path("/home/kojiek/exps_nvme/paired59k_mf_qwen/paired59k_mf_shortdirect_train.tsv"))
    ap.add_argument("--expect-n", type=int, default=59614)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for lineno, line in enumerate(args.jsonl.open(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[FAIL] malformed JSON at line {lineno}: {exc}")
            return 1

    fails: list[str] = []
    warns: list[str] = []
    n = len(rows)
    print(f"rows: {n}")

    # -- completeness -------------------------------------------------------
    if n != args.expect_n:
        fails.append(f"row count {n} != expected {args.expect_n}")

    ids = [r.get("id") for r in rows]
    if len(set(ids)) != n:
        dup = [i for i, c in collections.Counter(ids).items() if c > 1]
        fails.append(f"{n - len(set(ids))} duplicate ids, e.g. {dup[:3]}")

    # every clip the training TSV expects must be captioned, in that id set
    want = {r["id"] for r in csv.DictReader(args.tsv.open(), delimiter="\t")}
    missing = want - set(ids)
    extra = set(ids) - want
    if missing:
        fails.append(f"{len(missing)} TSV clips have no caption, e.g. {sorted(missing)[:3]}")
    if extra:
        fails.append(f"{len(extra)} captions are not in the TSV, e.g. {sorted(extra)[:3]}")

    # -- generation health --------------------------------------------------
    errs = [r for r in rows if r.get("error")]
    if errs:
        fails.append(f"{len(errs)} rows carry an error field")
    not_ok = [r for r in rows if not r.get("ok")]
    if not_ok:
        fails.append(f"{len(not_ok)} rows have ok=False")

    caps = [caption_of(r) for r in rows]
    empty = [i for i, c in zip(ids, caps) if not c.strip()]
    if empty:
        fails.append(f"{len(empty)} empty captions, e.g. {empty[:3]}")

    # -- R2: 77-token window (hard) ----------------------------------------
    toks = [r.get("tokens", 0) for r in rows]
    over = [i for i, t in zip(ids, toks) if t > MAX_TOKENS]
    print(f"tokens: mean={statistics.mean(toks):.2f} p50={statistics.median(toks)} "
          f"max={max(toks)} over_{MAX_TOKENS}={len(over)}")
    if over:
        fails.append(f"R2: {len(over)} captions exceed {MAX_TOKENS} tokens, e.g. {over[:3]}")

    ends = sum(1 for c in caps if c.rstrip().endswith((".", "!", "?")))
    print(f"ends with sentence punctuation: {ends / n * 100:.2f}%")
    if ends / n < 0.98:
        warns.append(f"only {ends / n * 100:.2f}% end on sentence punctuation")

    # -- R1: uniqueness (soft) ---------------------------------------------
    uniq = len(set(caps))
    uniq_pct = uniq / n * 100
    counts = collections.Counter(caps)
    worst_text, worst_n = counts.most_common(1)[0]
    print(f"unique captions: {uniq} ({uniq_pct:.2f}%)")
    print(f"largest collision cluster: {worst_n} clips share one caption")
    if uniq_pct < MIN_UNIQUE_PCT:
        fails.append(f"R1: uniqueness {uniq_pct:.2f}% < {MIN_UNIQUE_PCT}%")
    elif uniq_pct < 100.0:
        warns.append(f"R1: {n - uniq} clips ({100 - uniq_pct:.2f}%) reuse a caption; "
                     f"largest cluster {worst_n}")

    attempts = [r.get("attempts", 0) for r in rows]
    att = collections.Counter(attempts)
    print(f"attempts: mean={statistics.mean(attempts):.3f} dist={sorted(att.items())}")

    # -- verdict ------------------------------------------------------------
    for w in warns:
        print(f"[WARN] {w}")
    for f in fails:
        print(f"[FAIL] {f}")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\n=== ACCEPTANCE {verdict} ===")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "verdict": verdict,
            "jsonl": str(args.jsonl),
            "rows": n,
            "expected_rows": args.expect_n,
            "unique_captions": uniq,
            "unique_pct": round(uniq_pct, 4),
            "largest_collision_cluster": worst_n,
            "tokens_mean": round(statistics.mean(toks), 4),
            "tokens_max": max(toks),
            "over_window": len(over),
            "ends_sentence_pct": round(ends / n * 100, 4),
            "attempts_dist": {str(k): v for k, v in sorted(att.items())},
            "warnings": warns,
            "failures": fails,
        }, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.report}")

    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
