#!/usr/bin/env python3
"""Strict completeness check for 053 before reading summary.json.

summarize_tscore_beta_probe.py (hash-bound to the running 053) accepts >=30 val
points and takes the last 4 as final_val without checking they are the planned
iterations; metrics are only checked for file existence. This post-hoc check
does not change the preregistered rule; it decides whether the summary is
admissible at all. Exit 0 = admissible, 1 = not.

Checks per arm x seed:
  - val its are exactly {499, 999, ..., 19999} (40 points); duplicate its from
    resumed appends must carry identical fm_mse, otherwise flagged
  - every val point has finite fm_mse and the five fm_mse_t* grid values
  - ema_final exists
  - mc500 / val100 metrics.txt have all five keys, finite, in sane ranges,
    and "Test clips" equals the TSV record count
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

ARMS = ["base", "lam0p2", "lam1p0", "lam1p0shuf"]
SEEDS = ["14159265", "27182818"]
ITERS = 20000
VAL_ITS = list(range(499, ITERS, 500))
GRID = [f"fm_mse_t{t}" for t in ("0.1", "0.3", "0.5", "0.7", "0.9")]
METRIC_RANGES = {"clap_score": (-0.1, 0.6), "aes_CE": (1, 10), "aes_CU": (1, 10),
                 "aes_PC": (1, 10), "aes_PQ": (1, 10)}
VAL_RE = re.compile(r"-val - it\s+(\d+):\s*(.*)")
KV_RE = re.compile(r"([A-Za-z_0-9.]+):\s*(-?[0-9.]+(?:e-?\d+)?|nan|inf|-inf)", re.I)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def exp_name(arm, seed):
    return f"tscore_beta_probe_{arm}_seed{seed}_s1_{ITERS}"


def tsv_records(path: Path) -> int:
    csv.field_size_limit(10**9)
    with path.open(newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=str(Path.home() / "logs/tscore_beta_probe_20260913"))
    ap.add_argument("--art", default=str(Path.home() / "nvme_experiment_artifacts/meanaudio/tscore_beta_probe_20260913"))
    ap.add_argument("--exps-dir", default=str(Path.home() / "exps_nvme"))
    ap.add_argument("--out")
    args = ap.parse_args()
    art = Path(args.art)
    want = {"mc500": tsv_records(art / "inputs/musiccaps_head500.tsv"),
            "val100": tsv_records(art / "inputs/val_scored.tsv")}
    problems, per_run = [], {}
    for arm in ARMS:
        for seed in SEEDS:
            exp = exp_name(arm, seed)
            p = []
            log = Path(args.state_dir) / f"train_{exp}.log"
            if not log.is_file():
                problems.append(f"{exp}: no train log")
                per_run[exp] = {"admissible": False}
                continue
            seen: dict[int, list[dict]] = {}
            for raw in log.read_text(errors="replace").splitlines():
                m = VAL_RE.search(ANSI_RE.sub("", raw))
                if m:
                    seen.setdefault(int(m.group(1)), []).append(
                        {k: float(v) for k, v in KV_RE.findall(m.group(2))})
            its = sorted(seen)
            if its != VAL_ITS:
                missing = sorted(set(VAL_ITS) - set(its))
                extra = sorted(set(its) - set(VAL_ITS))
                p.append(f"val its mismatch: {len(its)} points, missing {missing[:5]}{'...' if len(missing) > 5 else ''}, extra {extra[:5]}")
            for it, recs in seen.items():
                if len({r.get("fm_mse") for r in recs}) > 1:
                    p.append(f"it {it}: {len(recs)} differing duplicate val records (resume append)")
                last = recs[-1]
                for k in ["fm_mse"] + GRID:
                    if k not in last or not math.isfinite(last[k]):
                        p.append(f"it {it}: {k} missing or non-finite")
            if not (Path(args.exps_dir) / exp / f"{exp}_ema_final.pth").is_file():
                p.append("ema_final missing")
            for split, n in want.items():
                mpath = art / "metrics" / f"{exp}_{split}" / "metrics.txt"
                if not mpath.is_file():
                    p.append(f"{split}: metrics.txt missing")
                    continue
                kv = {}
                for line in mpath.read_text().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        kv[k.strip()] = v.strip()
                if kv.get("Test clips") != str(n):
                    p.append(f"{split}: Test clips {kv.get('Test clips')} != {n}")
                for k, (lo, hi) in METRIC_RANGES.items():
                    try:
                        v = float(kv[k])
                    except (KeyError, ValueError):
                        p.append(f"{split}: {k} missing/unparseable")
                        continue
                    if not math.isfinite(v) or not lo <= v <= hi:
                        p.append(f"{split}: {k}={v} non-finite or out of range")
            per_run[exp] = {"admissible": not p, "problems": p}
            problems.extend(f"{exp}: {x}" for x in p)
    report = {"admissible": not problems, "problems": problems, "runs": per_run,
              "note": "admissibility of summary.json only; says nothing about effect size or causality"}
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text if problems else json.dumps({"admissible": True}))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
