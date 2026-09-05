#!/usr/bin/env python3
"""Prune PostHocEMA snapshots in an exp dir down to quartile anchors.

Policy (memory reference_ema_ckpts_prune_policy.md): the files in ema_ckpts/
hold EMA weights only -- they cannot resume training, and once the run has
synthesized its *_ema_final.pth they are no longer needed to reproduce the
published number. Keep quartile anchors so a later post-hoc EMA sweep still
has something to interpolate between.

Refuses to touch a directory whose ema_final is missing: that run either has
not finished or still needs every snapshot to synthesize.

Snapshot names are "<sigma_index>.<iteration>.pt". Anchors are chosen on the
iteration axis and kept for every sigma index.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SNAPSHOT_RE = re.compile(r"^(\d+)\.(\d+)\.pt$")


def snapshot_iterations(ema_dir: Path) -> dict[int, list[Path]]:
    by_iter: dict[int, list[Path]] = {}
    for p in ema_dir.iterdir():
        m = SNAPSHOT_RE.match(p.name)
        if m:
            by_iter.setdefault(int(m.group(2)), []).append(p)
    return by_iter


def quartile_anchors(iterations: list[int], keep: int) -> list[int]:
    """The final snapshot plus evenly spaced earlier ones, on the iteration axis."""
    if not iterations:
        return []
    ordered = sorted(iterations)
    lo, hi = ordered[0], ordered[-1]
    targets = [lo + (hi - lo) * k / keep for k in range(1, keep + 1)]
    anchors = {min(ordered, key=lambda it: abs(it - t)) for t in targets}
    anchors.add(hi)
    return sorted(anchors)


def prune(exp_dir: Path, keep: int, dry_run: bool) -> tuple[int, int]:
    ema_dir = exp_dir / "ema_ckpts"
    if not ema_dir.is_dir():
        return 0, 0
    if not list(exp_dir.glob("*_ema_final.pth")):
        print(f"  [SKIP] {exp_dir.name}: no *_ema_final.pth, snapshots still needed")
        return 0, 0

    by_iter = snapshot_iterations(ema_dir)
    if not by_iter:
        return 0, 0
    # Already at (or below) the anchor budget -- re-running must not erode it
    # further, otherwise repeated calls walk the dir down to a single snapshot.
    if len(by_iter) <= keep + 1:
        print(f"  [SKIP] {exp_dir.name}: {len(by_iter)} snapshots <= keep+1, already pruned")
        return 0, 0
    anchors = quartile_anchors(list(by_iter), keep)
    freed = 0
    removed = 0
    for it in sorted(by_iter):
        if it in anchors:
            continue
        for p in by_iter[it]:
            freed += p.stat().st_size
            removed += 1
            if not dry_run:
                p.unlink()
    kept = sum(len(by_iter[it]) for it in anchors)
    verb = "would remove" if dry_run else "removed"
    print(
        f"  {exp_dir.name}: {verb} {removed} snapshots "
        f"({freed / 2**30:.1f}G), kept {kept} at {anchors}"
    )
    return removed, freed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="+", type=Path)
    ap.add_argument("--keep", type=int, default=4, help="anchors on the iteration axis")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_removed = 0
    total_freed = 0
    for d in args.exp_dirs:
        if not d.is_dir():
            print(f"  [SKIP] {d}: not a directory")
            continue
        r, f = prune(d, args.keep, args.dry_run)
        total_removed += r
        total_freed += f
    print(f"total: {total_removed} snapshots, {total_freed / 2**30:.1f}G")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
