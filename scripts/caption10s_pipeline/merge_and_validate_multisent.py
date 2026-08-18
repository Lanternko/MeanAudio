#!/usr/bin/env python
"""Atomically merge regenerated captions into the repaired multisent corpus, then
run the full acceptance gate. Any failed check aborts with a non-zero exit and
leaves the corpus untouched — nothing downstream (TSV / NPZ / training) may start.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_multisent_first_entity_line import (  # noqa: E402
    CJK_RE,
    MIN_WORDS,
    TURN_MARKERS,
    DEGENERATE_RE,
    classify,
    n_sents,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--regen", type=Path, required=True)
    ap.add_argument("--official_tsv", type=Path, required=True)
    ap.add_argument("--backup", type=Path, required=True)
    ap.add_argument("--backup_sha256", required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    failures: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    # --- backup integrity first: it is the only way back ---
    actual = sha256(args.backup)
    backup_ok = actual == args.backup_sha256
    if not backup_ok:
        failures.append(f"backup sha256 mismatch: expected {args.backup_sha256} got {actual}")
    mode = oct(args.backup.stat().st_mode & 0o777)
    if args.backup.stat().st_mode & 0o222:
        failures.append(f"backup is writable (mode {mode}) — must stay immutable")

    regen = {}
    with args.regen.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            regen[r["id"]] = r

    rows = []
    with args.corpus.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    merged = 0
    out_rows = []
    for r in rows:
        rr = regen.get(r["id"])
        if rr is not None:
            cap = rr["caption"]
            r = dict(r)
            r["caption"] = cap
            r["n_chars"] = len(cap)
            r["n_words"] = len(cap.split())
            r["n_sents"] = n_sents(cap)
            r["regen"] = rr["regen"]
            r["max_new_tokens"] = rr["max_new_tokens"]
            merged += 1
        out_rows.append(r)

    if merged != len(regen):
        failures.append(f"merged {merged} rows but regen file has {len(regen)}")

    # --- acceptance gate ---
    with args.official_tsv.open(encoding="utf-8", newline="") as f:
        want = {r["id"] for r in csv.DictReader(f, delimiter="\t")}

    ids = [r["id"] for r in out_rows]
    have = set(ids)
    if len(ids) != len(have):
        failures.append(f"duplicate ids: {len(ids)} rows vs {len(have)} unique")
    missing = want - have
    extra = have - want
    if missing:
        failures.append(f"missing ids: {len(missing)} e.g. {sorted(missing)[:3]}")
    if extra:
        failures.append(f"extra ids: {len(extra)} e.g. {sorted(extra)[:3]}")

    nulls = cjk = turn = short = degen = multiline = 0
    strict_tags: Counter[str] = Counter()
    words, sents = [], []
    for r in out_rows:
        c = r.get("caption")
        if not c or not c.strip():
            nulls += 1
            continue
        if CJK_RE.search(c):
            cjk += 1
        if any(m in c for m in TURN_MARKERS):
            turn += 1
        if DEGENERATE_RE.match(c):
            degen += 1
        if len(c.split()) < MIN_WORDS:
            short += 1
        if "\n" in c or "\r" in c:
            multiline += 1
        strict_tags.update(classify(c))
        words.append(len(c.split()))
        sents.append(n_sents(c))

    for name, val in [("null_captions", nulls), ("cjk_rows", cjk),
                      ("turn_marker_rows", turn), ("degenerate_rows", degen),
                      ("too_short_rows", short), ("multiline_rows", multiline)]:
        if val:
            failures.append(f"{name}={val} (must be 0)")
    if strict_tags:
        failures.append(
            "strict_caption_defects="
            + ",".join(f"{key}:{value}" for key, value in sorted(strict_tags.items()))
        )

    mean_w = st.mean(words) if words else 0.0
    if mean_w < 30:
        failures.append(f"mean_words too low: {mean_w:.1f}")

    report = {
        "checked_at": now,
        "corpus": str(args.corpus),
        "rows": len(out_rows),
        "expected_rows": len(want),
        "unique_ids": len(have),
        "regen_merged": merged,
        "backup": {"path": str(args.backup), "sha256": actual,
                   "matches_recorded": backup_ok, "mode": mode},
        "gate": {
            "null_captions": nulls, "cjk_rows": cjk, "turn_marker_rows": turn,
            "degenerate_rows": degen, "too_short_rows": short,
            "multiline_rows": multiline, "missing_ids": len(missing),
            "extra_ids": len(extra),
            "strict_tag_counts": dict(sorted(strict_tags.items())),
        },
        "stats": {
            "mean_words": round(mean_w, 2),
            "median_words": st.median(words) if words else 0,
            "mean_sents": round(st.mean(sents), 3) if sents else 0,
            "still_1sent_frac": round(sum(1 for s in sents if s == 1) / len(sents), 4) if sents else 0,
        },
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
    }

    if failures:
        print(json.dumps(report, indent=2))
        print("\n[FAIL] acceptance gate failed — corpus NOT modified, "
              "do NOT start TSV / NPZ / training")
        return 4

    tmp = args.corpus.with_suffix(args.corpus.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.corpus)
    report["final_sha256"] = sha256(args.corpus)

    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\n[PASS] all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
