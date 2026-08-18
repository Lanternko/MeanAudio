#!/usr/bin/env python3
"""Review gate for caption-10s pilot SUMMARY.json. Exit 0 only if all checks pass."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out_verdict", type=Path, required=True)
    ap.add_argument("--min_n", type=int, default=500)
    ap.add_argument("--min_delta", type=float, default=0.01)
    ap.add_argument("--max_null_rate", type=float, default=0.02)
    ap.add_argument("--min_full_dur", type=float, default=28.0)
    ap.add_argument("--max_full_dur", type=float, default=32.0)
    ap.add_argument("--min_caption_len", type=float, default=20.0)
    args = ap.parse_args()

    s = json.loads(args.summary.read_text())
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    n = int(s.get("n_compared") or s.get("n") or 0)
    add("n_compared", n >= args.min_n, f"n={n} min={args.min_n}")

    null_rate = float(s.get("null_rate", 1.0))
    add("null_rate", null_rate <= args.max_null_rate, f"null_rate={null_rate:.4f}")

    mean_dur = float(s.get("mean_full_dur", 0.0))
    add(
        "mean_full_dur",
        args.min_full_dur <= mean_dur <= args.max_full_dur,
        f"mean_full_dur={mean_dur:.3f}",
    )

    # clap structure from compare_caption10s_pilot
    clap = s.get("clap") or {}
    if "old_vs_10s" in clap:
        old10 = float(clap["old_vs_10s"]["mean"])
        new10 = float(clap["new10s_vs_10s"]["mean"])
        old30 = float(clap["old_vs_30s"]["mean"])
    else:
        # legacy pilot format
        cs = s.get("clap_stats") or {}
        old10 = float(cs["old_vs_10s_audio"]["mean"])
        new10 = float(cs["cap10s_vs_10s_audio"]["mean"])
        old30 = float(cs["old_vs_30s_audio"]["mean"])

    add("bug_old_prefers_30s", old30 > old10, f"old30={old30:.4f} old10={old10:.4f}")
    delta = new10 - old10
    add(
        "new_improves_on_10s",
        delta >= args.min_delta,
        f"delta={delta:.4f} min={args.min_delta}",
    )

    mean_len = float(s.get("mean_new_caption_len") or s.get("mean_new_caption_len", 0) or 0)
    if mean_len == 0 and "mean_new_caption_len" not in s:
        # optional
        add("caption_len", True, "skipped (not in summary)")
    else:
        add("caption_len", mean_len >= args.min_caption_len, f"mean_len={mean_len:.1f}")

    failed = [c for c in checks if not c["ok"]]
    status = "passed" if not failed else "failed"
    verdict = {
        "schema_version": 1,
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(args.summary),
        "checks": checks,
        "failed": [c["name"] for c in failed],
        "metrics": {
            "n": n,
            "null_rate": null_rate,
            "mean_full_dur": mean_dur,
            "old10": old10,
            "new10": new10,
            "old30": old30,
            "delta": delta,
        },
    }
    args.out_verdict.parent.mkdir(parents=True, exist_ok=True)
    args.out_verdict.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    if status != "passed":
        print(f"[GATE FAIL] {failed}", file=sys.stderr)
        sys.exit(2)
    print("[GATE PASS]")
    sys.exit(0)


if __name__ == "__main__":
    main()
