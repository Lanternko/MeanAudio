#!/usr/bin/env python3
"""3-seed paired readout for the slot0nmv2 vs slot0clean quarter ablation (066-071).

Follows the preregistered decision_rule in
docs/experiments/caption2p0_slot0nmv2_nmv2pair_quarter_s*_contract.json:
  - report the 3 per-seed deltas (slot0nmv2 - slot0clean) per cell and metric
  - mean + t-based 95% CI (df=2)
  - CFG0 CLAP non-inferior iff CI lower bound > -0.0084
  - a gain is claimable only if the CI excludes 0 AND all 3 deltas share a sign
  - LUFS / silent_n are reported beside AES+CLAP (063/065, slot0nm silence mode)
"""
import json, os, statistics, sys

EVAL = os.path.expanduser("~/eval_output_nvme")
SEEDS = ["14159265", "27182818", "16180339"]
CELLS = ["cfg0", "cfg3_neg"]
ARMS = {"clean": "slot0clean", "nmv2": "slot0nmv2"}
DELTA_METRICS = ["clap_score", "aes_PQ", "aes_CU", "aes_CE", "aes_PC"]
LEVEL_METRICS = ["level_lufs_mean", "level_silent_n", "level_crest_mean"]
T975_DF2 = 4.302652729911275
NI_BOUND = -0.0084
SEED_FLOOR = {"clap_score": 0.0042, "aes_CE": 0.1343, "aes_CU": 0.0520,
              "aes_PC": 0.0554, "aes_PQ": 0.0523}


def load(arm, seed, cell):
    exp = f"phase8_qwen_caption2p0_{ARMS[arm]}_nmv2pair_noq_quarter_s{seed}"
    d = f"{EVAL}/{exp}_mc_mf25_{cell}"
    rep = f"{d}/{exp}_mc_mf25_{cell}_REPORT.json"
    met = f"{d}/{exp}_mc_mf25_{cell}/metrics.json"
    if not (os.path.exists(rep) and os.path.exists(met)):
        return None
    m = json.load(open(met))
    assert m["n_present"] == m["n_rows"] == 5521, f"incomplete eval: {met}"
    return m["metrics"]


def ci(deltas):
    mean = statistics.mean(deltas)
    if len(deltas) < 2:
        return mean, None, None
    half = T975_DF2 * statistics.stdev(deltas) / len(deltas) ** 0.5
    return mean, mean - half, mean + half


def main():
    out = {"document_kind": "nmv2pair_three_seed_readout", "cells": {}}
    missing = []
    for cell in CELLS:
        rows, cell_out = {}, {"per_seed": {}, "delta": {}, "levels": {}}
        for seed in SEEDS:
            got = {a: load(a, seed, cell) for a in ARMS}
            if any(v is None for v in got.values()):
                missing += [f"{a}/s{seed}/{cell}" for a, v in got.items() if v is None]
                continue
            rows[seed] = got
            cell_out["per_seed"][seed] = {
                a: {k: got[a][k] for k in DELTA_METRICS + LEVEL_METRICS} for a in ARMS}
        for k in DELTA_METRICS:
            d = [rows[s]["nmv2"][k] - rows[s]["clean"][k] for s in rows]
            if not d:
                continue
            mean, lo, hi = ci(d)
            e = {"per_seed": {s: rows[s]["nmv2"][k] - rows[s]["clean"][k] for s in rows},
                 "mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n_seeds": len(d),
                 "same_sign": len({x > 0 for x in d}) == 1,
                 "seed_floor": SEED_FLOOR.get(k)}
            if lo is not None:
                e["claimable_gain"] = bool(e["same_sign"] and lo > 0)
                e["claimable_loss"] = bool(e["same_sign"] and hi < 0)
                if cell == "cfg0" and k == "clap_score":
                    e["non_inferior"] = bool(lo > NI_BOUND)
                    e["ni_bound"] = NI_BOUND
            cell_out["delta"][k] = e
        for k in LEVEL_METRICS:
            for a in ARMS:
                v = [rows[s][a][k] for s in rows]
                if v:
                    cell_out["levels"].setdefault(k, {})[a] = {
                        "per_seed": {s: rows[s][a][k] for s in rows},
                        "mean": statistics.mean(v)}
        out["cells"][cell] = cell_out
    out["missing"] = missing
    out["complete"] = not missing

    for cell in CELLS:
        c = out["cells"].get(cell)
        if not c or not c["delta"]:
            print(f"== {cell}: no complete pairs yet"); continue
        n = c["delta"]["clap_score"]["n_seeds"]
        print(f"\n== {cell}  (n={n} seed pairs)   delta = slot0nmv2 - slot0clean")
        print(f"{'metric':12}" + "".join(f"{'s'+s:>13}" for s in SEEDS) +
              f"{'mean':>11}{'ci95_lo':>11}{'ci95_hi':>11}{'floor':>9}  verdict")
        for k in DELTA_METRICS:
            e = c["delta"][k]
            cells = "".join(f"{e['per_seed'][s]:>+13.4f}" if s in e["per_seed"] else f"{'-':>13}"
                            for s in SEEDS)
            lo = f"{e['ci95_lo']:>+11.4f}" if e["ci95_lo"] is not None else f"{'-':>11}"
            hi = f"{e['ci95_hi']:>+11.4f}" if e["ci95_hi"] is not None else f"{'-':>11}"
            fl = f"{e['seed_floor']:>9.4f}" if e["seed_floor"] else f"{'-':>9}"
            if e.get("claimable_gain"):
                v = "GAIN (CI excludes 0, signs agree)"
            elif e.get("claimable_loss"):
                v = "LOSS (CI excludes 0, signs agree)"
            else:
                v = "inconclusive" + ("" if e["same_sign"] else " (signs disagree)")
            if k == "clap_score" and "non_inferior" in e:
                v += f" | non-inferiority({NI_BOUND}): {'PASS' if e['non_inferior'] else 'FAIL'}"
            print(f"{k:12}{cells}{e['mean']:>+11.4f}{lo}{hi}{fl}  {v}")
        print(f"{'':12}levels (mean over seeds):", end=" ")
        for k, d in c["levels"].items():
            print(f"{k}: clean {d['clean']['mean']:.2f} / nmv2 {d['nmv2']['mean']:.2f}", end="   ")
        print()
    if missing:
        print("\nMISSING (still running or not evaluated):")
        for m in missing:
            print("  -", m)

    dst = os.path.expanduser(
        "~/MeanAudio/docs/experiments/results/phase8/nmv2pair_three_seed_readout.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")
    return 0 if out["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
