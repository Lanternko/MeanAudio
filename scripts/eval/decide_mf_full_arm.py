#!/usr/bin/env python3
"""Pick which MF corpus earns the full training budget.

Two quarter arms now differ only in caption text over the same audio latents,
rows, order, recipe and budget:

  mf_fullcov  full-coverage short_direct_v2, 0.8916 caption unique rate
  mf_dedup    same corpus with 27,264 duplicate captions regenerated

Lanternko 2026-09-07: "誰 clap+aes 好就跑誰，不用問我". This replaces the
absolute CFG0 >= 0.1900 gate that 039 pre-registered and self-aborted on --
that gate asked "is MF viable at all", which the quarter numbers have now
answered; the open question is which of the two corpora to spend ~19h on.

Scoring, so the rule is not re-argued after the numbers land:

  * both eval cells count -- CFG0 canonical and CFG 3.0 + fidelity negative --
    because the two cells disagreed for MF vs Qwen and there is no reason to
    assume they agree here. 5 metrics x 2 cells = 10 comparisons.
  * a comparison counts only if the margin clears 2x the measured
    training-seed noise floor FOR THAT CELL. The floors are protocol-specific
    (CFG3+neg inflates AES seed noise 2-3x while shrinking CLAP's), so they
    are not interchangeable -- see memory reference_training_seed_pq_noise_floor.
  * winner = more counted wins; tie broken by total effect size in floor units.
  * an exact tie goes to mf_dedup, because it carries no audit deviation. That
    is a stated preference, not a measurement, and is reported as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HOME = Path.home()
METRICS = ["clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"]

# |delta| between two training seeds of the same configuration, measured on
# c2p0_slot0 full (14159265 vs 27182818), MusicCaps 5521 / MF25 / gen seed 42.
FLOOR = {
    "cfg0":     {"clap_score": 0.0042, "aes_CE": 0.1343, "aes_CU": 0.0520,
                 "aes_PC": 0.0554, "aes_PQ": 0.0523},
    "cfg3_neg": {"clap_score": 0.0003, "aes_CE": 0.2960, "aes_CU": 0.1053,
                 "aes_PC": 0.1884, "aes_PQ": 0.1416},
}
FACTOR = 2.0  # cross-checkpoint threshold; 1.0-1.5x is treated as null
# The tie-break sums margins in floor units. CFG3+neg CLAP has a floor of
# 0.0003, so without a cap a single CLAP gap scores tens of units and decides
# the tie-break alone -- that is an artefact of CLAP's seed stability, not of
# CLAP mattering ten times more. Cap each metric's contribution.
EFFECT_CAP = 10.0

ARMS = {
    "mf_fullcov": "mf_fullcov_noq_quarter",
    "mf_dedup": "mf_dedup_noq_quarter",
}


def read_cfg0(prefix: str) -> dict[str, float]:
    p = HOME / "cfg0_eval_runtime/reports" / f"{prefix}_musiccaps_mf25_cfg0_noq_REPORT.json"
    if not p.exists():
        raise SystemExit(f"[FAIL] missing CFG0 report: {p}")
    d = json.loads(p.read_text())
    if d.get("status") != "passed":
        raise SystemExit(f"[FAIL] CFG0 report status={d.get('status')!r}: {p}")
    return {k: float(d["metrics"][k]) for k in METRICS}


def read_cfg3(prefix: str) -> dict[str, float]:
    name = f"{prefix}_mc_mf25_cfg3_neg"
    p = HOME / "eval_output_nvme" / name / name / "metrics.txt"
    if not p.exists():
        raise SystemExit(f"[FAIL] missing CFG3+neg metrics: {p}")
    out: dict[str, float] = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    missing = [k for k in METRICS if k not in out]
    if missing:
        raise SystemExit(f"[FAIL] {p} missing {missing}")
    return {k: out[k] for k in METRICS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-winner", choices=sorted(ARMS),
                    help="exit 5 unless this arm wins")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    vals = {arm: {"cfg0": read_cfg0(pre), "cfg3_neg": read_cfg3(pre)}
            for arm, pre in ARMS.items()}

    score = {a: 0 for a in ARMS}
    effect = {a: 0.0 for a in ARMS}
    rows = []
    for cell in ("cfg0", "cfg3_neg"):
        for m in METRICS:
            a, b = vals["mf_fullcov"][cell][m], vals["mf_dedup"][cell][m]
            delta = b - a                      # positive => dedup better
            thr = FLOOR[cell][m] * FACTOR
            if abs(delta) >= thr:
                win = "mf_dedup" if delta > 0 else "mf_fullcov"
                score[win] += 1
                effect[win] += min(abs(delta) / FLOOR[cell][m], EFFECT_CAP)
            else:
                win = "tie"
            rows.append((cell, m, a, b, delta, thr, win))

    print(f"{'cell':<9} {'metric':<11} {'fullcov':>9} {'dedup':>9} "
          f"{'delta':>9} {'2x floor':>9}  winner")
    for cell, m, a, b, d, thr, win in rows:
        print(f"{cell:<9} {m:<11} {a:>9.4f} {b:>9.4f} {d:>+9.4f} {thr:>9.4f}  {win}")

    ties = sum(1 for r in rows if r[6] == "tie")
    print(f"\ncounted wins: mf_fullcov={score['mf_fullcov']} "
          f"mf_dedup={score['mf_dedup']} tie={ties}")
    print(f"effect (floor units): mf_fullcov={effect['mf_fullcov']:.2f} "
          f"mf_dedup={effect['mf_dedup']:.2f}")

    if score["mf_dedup"] != score["mf_fullcov"]:
        winner = max(score, key=score.get)
        basis = "counted wins"
    elif abs(effect["mf_dedup"] - effect["mf_fullcov"]) > 1e-9:
        winner = max(effect, key=effect.get)
        basis = "tie on wins, decided on total effect size"
    else:
        winner = "mf_dedup"
        basis = "exact tie; preference for the corpus with no audit deviation"

    print(f"\nWINNER: {winner}  ({basis})")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "winner": winner, "basis": basis, "counted_wins": score,
            "effect_floor_units": effect, "effect_cap": EFFECT_CAP, "ties": ties,
            "values": vals, "floor": FLOOR, "factor": FACTOR,
            "rows": [{"cell": c, "metric": m, "mf_fullcov": a, "mf_dedup": b,
                      "delta_dedup_minus_fullcov": d, "threshold": t, "winner": w}
                     for c, m, a, b, d, t, w in rows],
        }, indent=1, sort_keys=True))

    if args.assert_winner and args.assert_winner != winner:
        print(f"[FAIL] {args.assert_winner} did not win; {winner} takes the full budget")
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
