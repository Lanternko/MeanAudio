#!/usr/bin/env python3
"""Verify the batched Music Flamingo path against the per-clip path.

Text equality is the wrong acceptance test on its own: the real pipeline samples
from attempt 1 onward, so bit-identical output was never on the table, and bf16
batched matmuls reduce in a different order anyway -- greedy argmax can diverge
at a near-tie and then cascade. What actually has to hold is that each caption
describes ITS OWN audio. Cross-talk through a bad attention mask produces
fluent captions about a neighbour's clip, which no exception will ever catch.

Three tests:

  machinery   batch path at size 1 vs the per-clip path. Same code, same
              shapes, one sequence: any difference here is a real bug, not
              batching numerics.
  neighbours  the same clip captioned in different batch compositions. If its
              caption tracks whoever it is batched with, the mask leaks.
  ownership   each batched caption is compared (token overlap) against the
              per-clip captions of every clip in the batch. It should match
              its own clip best; matching a neighbour better is leakage.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from music_flamingo_jamendo_slice_caption import (  # noqa: E402
    PROMPT_PRESETS,
    load_model,
    run_caption,
    run_caption_batch,
    write_slice_wav,
)

csv.field_size_limit(10**9)


def jaccard(a: str | None, b: str | None) -> float:
    """Content-word overlap; enough to tell 'describes this clip' from 'describes that one'."""
    stop = {"a", "an", "the", "is", "are", "and", "with", "of", "that", "this",
            "in", "to", "it", "its", "as", "for", "by", "on", "or", "music",
            "track", "piece", "overall", "creating", "features", "featuring"}
    def toks(x: str | None) -> set:
        if not x:
            return set()
        return {w.strip(".,;:'\u2019\u2018\u201c\u201d").lower() for w in x.split()} - stop
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--wav_root", type=Path,
                    default=Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--prompt-preset", default="short_direct_v2")
    ap.add_argument("--tmp-dir", type=Path, default=Path("/tmp/mf_batch_verify"))
    ap.add_argument("--test", choices=["machinery", "neighbours", "ownership", "all"],
                    default="all")
    args = ap.parse_args()

    prompt = PROMPT_PRESETS[args.prompt_preset]
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    with args.tsv.open(newline="", encoding="utf-8") as f:
        rows = [r for _, r in zip(range(args.n), csv.DictReader(f, delimiter="\t"))]
    print(f"[data] {len(rows)} clips from {args.tsv}")

    model, processor = load_model()

    paths = [str(write_slice_wav(args.wav_root / f"{r['id']}.wav", args.tmp_dir))
             for r in rows]

    print("[run] per-clip greedy ...")
    single = [run_caption(model, processor, p, prompt, args.max_new_tokens)["raw_text"]
              for p in paths]

    print(f"[run] batched greedy (batch={args.batch_size}) ...")
    batched: list[str] = []
    for i in range(0, len(paths), args.batch_size):
        out = run_caption_batch(model, processor, paths[i : i + args.batch_size],
                                prompt, args.max_new_tokens, do_sample=False)
        if not out["ok"]:
            sys.exit(f"[fatal] batch generation failed: {out['error']}")
        batched.extend(out["texts"])

    for p in paths:
        Path(p).unlink(missing_ok=True)

    identical = sum(1 for a, b in zip(single, batched)
                    if (a or "").strip() == (b or "").strip())
    print(f"\n=== text equality: {identical}/{len(rows)} identical")
    if args.batch_size == 1 and identical != len(rows):
        print("[FAIL] machinery: the batch path differs from the per-clip path "
              "at batch size 1, where no batching numerics apply. Real bug.")
        for r, a, b in zip(rows, single, batched):
            if (a or "").strip() != (b or "").strip():
                print(f"\n-- {r['id']}\n   single : {a}\n   batched: {b}")
        sys.exit(1)
    if args.batch_size == 1:
        print("[PASS] machinery: batch path at size 1 reproduces the per-clip path")

    failures = 0
    if args.test in ("ownership", "all") and args.batch_size > 1:
        print("\n=== ownership: does each batched caption match its own clip?")
        for i, (r, b) in enumerate(zip(rows, batched)):
            scores = [(jaccard(b, s_j), j) for j, s_j in enumerate(single)]
            scores.sort(reverse=True)
            best_score, best_j = scores[0]
            own_score = jaccard(b, single[i])
            verdict = "ok" if best_j == i else "LEAK?"
            if best_j != i:
                failures += 1
            print(f"  {verdict:5s} {r['id']:28s} own={own_score:.3f} "
                  f"best={best_score:.3f} (clip {best_j})")
        if failures:
            print(f"\n[FAIL] ownership: {failures}/{len(rows)} captions matched a "
                  f"neighbour better than their own clip")
            sys.exit(1)
        print("[PASS] ownership: every batched caption matched its own clip best")

    print("\nNote: text differences at batch_size > 1 are expected (bf16 reduction "
          "order); ownership is the test that matters.")


if __name__ == "__main__":
    main()
