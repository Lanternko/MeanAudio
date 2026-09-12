#!/usr/bin/env python3
"""Summarise the score-aware Beta timestep probe (stage A) and apply the preregistered rule.

Primary metric: held-out conditional velocity MSE on the fixed t grid {0.1,...,0.9}
(`fm_mse`, logged every 500 iters on 100 val rows from disjoint tracks). Per arm and seed:
  min_val, argmin_it, final_val = mean of the last 4 evals, overfit_gap = final_val - min_val.
Noise floor = |final_val(base, seed A) - final_val(base, seed B)|; effects need >= 2x floor
and the same sign in both seeds (reference_training_seed_pq_noise_floor).

  R1 base overfits        : base argmin_it <= 15000 in both seeds and mean overfit_gap > 2x its seed |delta|
  R2 lambda regularises   : final_val(arm) < final_val(base) in both seeds, mean paired gap > 2x floor
  R3 score carries signal : final_val(lam1p0) < final_val(lam1p0shuf) in both seeds, mean gap > 2x floor

MusicCaps-500 / val-100 CLAP and AES deltas are reported, never gating (n too small).
Exits non-zero if any expected curve or metrics file is missing.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ARMS = ["base", "lam0p2", "lam1p0", "lam1p0shuf"]
SEEDS = ["14159265", "27182818"]
ITERS = 20000
SPLITS = ["mc500", "val100"]
METRIC_KEYS = ["clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"]
VAL_RE = re.compile(r"-val - it\s+(\d+):\s*(.*)")
TRAIN_RE = re.compile(r"-train - it\s+(\d+):\s*(.*)")
KV_RE = re.compile(r"([A-Za-z_0-9.]+):\s*(-?[0-9.]+(?:e-?\d+)?)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def exp_name(arm: str, seed: str) -> str:
    return f"tscore_beta_probe_{arm}_seed{seed}_s1_{ITERS}"


def parse_log(path: Path, pattern: re.Pattern) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for raw in path.read_text(errors="replace").splitlines():
        m = pattern.search(ANSI_RE.sub("", raw))
        if m:
            rows[int(m.group(1))] = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
    return rows


def parse_metrics(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    missing = []
    for arm in ARMS:
        for seed in SEEDS:
            exp = exp_name(arm, seed)
            log = Path(args.state_dir) / f"train_{exp}.log"
            if not log.is_file():
                missing.append(str(log))
                continue
            val = {it: r["fm_mse"] for it, r in parse_log(log, VAL_RE).items() if "fm_mse" in r}
            train = parse_log(log, TRAIN_RE)
            if len(val) < 30:
                missing.append(f"{exp}: only {len(val)} val points")
                continue
            its = sorted(val)
            min_it = min(its, key=lambda i: val[i])
            final = sum(val[i] for i in its[-4:]) / 4
            per_t = {}
            last_val = parse_log(log, VAL_RE)[its[-1]]
            for k, v in last_val.items():
                if k.startswith("fm_mse_t"):
                    per_t[k] = v
            last_train = train[max(train)] if train else {}
            metrics = {}
            for split in SPLITS:
                mpath = Path(args.metrics_dir) / f"{exp}_{split}" / "metrics.txt"
                if not mpath.is_file():
                    missing.append(str(mpath))
                    continue
                m = parse_metrics(mpath)
                metrics[split] = {k: m.get(k) for k in METRIC_KEYS}
            runs[arm][seed] = {
                "curve": [[i, val[i]] for i in its],
                "min_val": val[min_it], "argmin_it": min_it, "final_val": final,
                "overfit_gap": final - val[min_it], "final_val_per_t": per_t,
                "final_train_loss": last_train.get("loss"), "final_t_mean": last_train.get("t_mean"),
                "metrics": metrics,
            }
    if missing:
        raise SystemExit("[FAIL] incomplete:\n  " + "\n  ".join(missing))

    fv = lambda arm, seed: runs[arm][seed]["final_val"]  # noqa: E731
    floor = abs(fv("base", SEEDS[0]) - fv("base", SEEDS[1]))
    gaps = [runs["base"][s]["overfit_gap"] for s in SEEDS]
    r1 = (all(runs["base"][s]["argmin_it"] <= 15000 for s in SEEDS)
          and sum(gaps) / 2 > 2 * abs(gaps[0] - gaps[1]))

    def paired(better: str, worse: str) -> dict:
        diffs = [fv(worse, s) - fv(better, s) for s in SEEDS]  # > 0 means `better` has lower val MSE
        mean = sum(diffs) / 2
        return {"diffs": diffs, "mean": mean, "ratio_to_floor": (mean / floor) if floor > 0 else None,
                "pass": all(d > 0 for d in diffs) and mean > 2 * floor}

    r2 = {arm: paired(arm, "base") for arm in ("lam0p2", "lam1p0")}
    r3 = paired("lam1p0", "lam1p0shuf")

    if not (r2["lam0p2"]["pass"] or r2["lam1p0"]["pass"]):
        decision = "stop: no regularisation effect at 2x seed floor; stage B not launched"
    elif r3["pass"]:
        decision = "go: regularisation reproduces and score information beats the permuted control; run stage B B0-B3"
    else:
        decision = ("go-with-reframe: regularisation reproduces but permuted S is as good; the effect is a "
                    "t-distribution shift, not score awareness. Stage B B1 vs B3 expected null")

    secondary = {}
    for split in SPLITS:
        secondary[split] = {}
        for arm in ARMS[1:]:
            secondary[split][arm] = {
                k: [runs[arm][s]["metrics"][split][k] - runs["base"][s]["metrics"][split][k] for s in SEEDS]
                for k in METRIC_KEYS
            }

    summary = {
        "experiment": "tscore_beta_probe_20260913",
        "primary_metric": "held-out conditional velocity MSE, t grid {0.1,0.3,0.5,0.7,0.9}, final = mean of last 4 evals",
        "noise_floor_final_val": floor,
        "R1_base_overfits": {"pass": r1, "overfit_gaps": gaps,
                             "argmin_its": [runs["base"][s]["argmin_it"] for s in SEEDS]},
        "R2_regularises": r2,
        "R3_score_information": r3,
        "decision": decision,
        "secondary_deltas_vs_base_per_seed": secondary,
        "runs": runs,
        "score_manifest": json.loads((Path(args.inputs_dir) / "score_manifest.json").read_text()),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    brief = {k: summary[k] for k in ("noise_floor_final_val", "R1_base_overfits", "R3_score_information", "decision")}
    brief["R2"] = {a: {"mean": v["mean"], "pass": v["pass"]} for a, v in r2.items()}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
