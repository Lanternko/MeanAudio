#!/usr/bin/env python3
"""Pre-launch caption corpus audit gate.

The CLAUDE.md pre-experiment checklist already requires a caption diversity
check (stop training if the unique rate is below 90%), but `sanity_check_50.py`
only samples 50 rows and never looks at the T5 window. That combination let the
Music Flamingo short-direct 100k corpus through with a 73.17% unique rate and
78.85% of its captions truncated at 77 tokens (audited 2026-09-04, after the
036 arm had already been trained on it twice).

This script audits the whole corpus on the two axes that actually bit us:

  diversity  -- unique caption rate, rows sharing a caption with another clip,
                and opening n-gram concentration (greedy decoding collapses
                captioners onto a handful of sentence openings)
  truncation -- flan-t5-large token length against the 77-token window that
                `features_utils.py` enforces, measured BEFORE and AFTER
                truncation, because truncation itself destroys uniqueness

Exit code is 1 if any hard gate fails, so it can be wired into a pipeline.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(10**9)

T5_WINDOW = 77
T5_MODEL = "google/flan-t5-large"


def load_captions(tsv_path: Path, id_col: str, caption_col: str) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    caps: list[str] = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if caption_col not in (reader.fieldnames or []):
            sys.exit(f"[fatal] column {caption_col!r} not in {reader.fieldnames}")
        for row in reader:
            ids.append(row.get(id_col, ""))
            caps.append(row.get(caption_col) or "")
    return ids, caps


def token_lengths(caps: list[str], batch: int = 2000) -> tuple[list[int], list[str]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(T5_MODEL)
    lengths: list[int] = []
    truncated_text: list[str] = []
    for i in range(0, len(caps), batch):
        chunk = caps[i : i + batch]
        ids = tok(chunk, add_special_tokens=True)["input_ids"]
        lengths.extend(len(x) for x in ids)
        truncated_text.extend(
            tok.decode(x[:T5_WINDOW], skip_special_tokens=True) for x in ids
        )
    return lengths, truncated_text


def pct(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--caption-col", default="caption")
    ap.add_argument("--ngram", type=int, default=5, help="opening n-gram width")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="diversity only; skips the flan-t5-large load",
    )
    # hard gates
    ap.add_argument("--min-unique-rate", type=float, default=0.90)
    ap.add_argument("--max-shared-caption-rate", type=float, default=0.05)
    ap.add_argument("--max-opening-ngram-share", type=float, default=0.10)
    ap.add_argument("--max-truncation-rate", type=float, default=0.50)
    args = ap.parse_args()

    ids, caps = load_captions(args.tsv, args.id_col, args.caption_col)
    n = len(caps)
    if n == 0:
        sys.exit("[fatal] empty corpus")

    counts = collections.Counter(caps)
    unique_rate = pct(len(counts), n)
    shared_rows = sum(v for v in counts.values() if v > 1)
    shared_rate = pct(shared_rows, n)
    worst_caption, worst_reuse = counts.most_common(1)[0]

    openings = collections.Counter(
        " ".join(c.split()[: args.ngram]).lower() for c in caps
    )
    top_opening, top_opening_count = openings.most_common(1)[0]
    top_opening_share = pct(top_opening_count, n)

    words = sorted(len(c.split()) for c in caps)

    report: dict = {
        "tsv": str(args.tsv),
        "rows": n,
        "unique_ids": len(set(ids)),
        "diversity": {
            "unique_caption_rate": round(unique_rate, 6),
            "rows_sharing_a_caption": shared_rows,
            "shared_caption_rate": round(shared_rate, 6),
            "worst_caption_reuse_count": worst_reuse,
            "top_opening_ngram": top_opening,
            "top_opening_ngram_share": round(top_opening_share, 6),
            "words_p50": words[n // 2],
            "words_p90": words[9 * n // 10],
        },
    }

    print(f"=== caption corpus audit: {args.tsv}")
    print(f"  rows={n:,}  unique_ids={len(set(ids)):,}")
    print("  -- diversity")
    print(f"     unique caption rate      : {unique_rate:.2%}")
    print(f"     rows sharing a caption   : {shared_rows:,} ({shared_rate:.2%})")
    print(f"     worst caption reused     : {worst_reuse:,} times")
    print(f'     top opening {args.ngram}-gram      : {top_opening_share:.2%}  "{top_opening}"')
    print(f"     words p50/p90            : {words[n // 2]} / {words[9 * n // 10]}")

    failures: list[str] = []
    if unique_rate < args.min_unique_rate:
        failures.append(
            f"unique caption rate {unique_rate:.2%} < {args.min_unique_rate:.2%}"
        )
    if shared_rate > args.max_shared_caption_rate:
        failures.append(
            f"shared caption rate {shared_rate:.2%} > {args.max_shared_caption_rate:.2%}"
        )
    if top_opening_share > args.max_opening_ngram_share:
        failures.append(
            f"top opening {args.ngram}-gram share {top_opening_share:.2%} > "
            f"{args.max_opening_ngram_share:.2%}"
        )

    if not args.skip_tokenizer:
        lengths, seen = token_lengths(caps)
        sl = sorted(lengths)
        trunc_rate = pct(sum(1 for x in lengths if x > T5_WINDOW), n)
        kept = sum(min(x, T5_WINDOW) / x for x in lengths) / n
        seen_counts = collections.Counter(seen)
        seen_unique_rate = pct(len(seen_counts), n)

        report["truncation"] = {
            "t5_window": T5_WINDOW,
            "tokens_p50": sl[n // 2],
            "tokens_p90": sl[9 * n // 10],
            "tokens_max": sl[-1],
            "truncation_rate": round(trunc_rate, 6),
            "mean_token_kept": round(kept, 6),
            "unique_rate_after_truncation": round(seen_unique_rate, 6),
            "worst_visible_caption_reuse": seen_counts.most_common(1)[0][1],
        }

        print(f"  -- truncation (flan-t5-large, window {T5_WINDOW})")
        print(f"     tokens p50/p90/max       : {sl[n // 2]} / {sl[9 * n // 10]} / {sl[-1]}")
        print(f"     truncated                : {trunc_rate:.2%}")
        print(f"     mean token kept          : {kept:.3f}")
        print(f"     unique rate AFTER trunc  : {seen_unique_rate:.2%}  <- what the model sees")
        print(f"     worst visible reuse      : {seen_counts.most_common(1)[0][1]:,} times")

        if trunc_rate > args.max_truncation_rate:
            failures.append(
                f"truncation rate {trunc_rate:.2%} > {args.max_truncation_rate:.2%}"
            )
        if seen_unique_rate < args.min_unique_rate:
            failures.append(
                f"post-truncation unique rate {seen_unique_rate:.2%} < "
                f"{args.min_unique_rate:.2%}"
            )

    report["gate"] = {"passed": not failures, "failures": failures}

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n[json] {args.json_out}")

    if failures:
        print("\n[GATE FAILED] do not train on this corpus:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)
    print("\n[GATE PASSED]")


if __name__ == "__main__":
    main()
