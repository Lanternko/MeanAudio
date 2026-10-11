#!/usr/bin/env python3
"""Per-prompt bootstrap 95% CIs for TTM table B (lvl30 block).

Reads the per-clip scores written by p2 115 / 118.  MeanAudio cells are the
per-prompt mean over the three training seeds (114/116/117), so the CI covers
prompt sampling only, not training-seed variance.  Also writes paired
differences against MusicGen-medium on the shared prompts.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

R115 = Path.home() / "eval_output_nvme/ttm_metrics_115_20261009"
R118 = Path.home() / "eval_output_nvme/ttm_metrics_seeds_118_20261011"
BLOCK = "lvl30"
KEYS = ["clap", "PQ", "CE", "CU", "PC", "meva"]
N_BOOT, SEED = 10000, 42


def load(root, name):
    out = {}
    d = root / name / BLOCK
    (pc,) = list((d / "scores").glob("*/per_clip.tsv"))
    with open(pc, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["id"]] = {k: float(r[k]) for k in KEYS if k != "meva"}
    with open(d / "meva" / "meva_per_clip.tsv", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        col = [c for c in rd.fieldnames if c != "id"][0]
        for r in rd:
            if r["id"] in out:
                out[r["id"]]["meva"] = float(r[col])
    return out


def ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(out_path):
    rng = np.random.default_rng(SEED)
    sysd = {n: load(R115, n) for n in ["musicgen_medium", "musicldm", "musicldm_fid8", "musiccaps_real"]}
    for c in ["cfg0", "cfg3_neg"]:
        seeds = [load(R115, f"meanaudio_{c}"), load(R118, f"meanaudio_s27182818_{c}"), load(R118, f"meanaudio_s16180339_{c}")]
        ids = set(seeds[0]) & set(seeds[1]) & set(seeds[2])
        sysd[f"meanaudio_{c}_3seed"] = {
            i: {k: float(np.mean([s[i][k] for s in seeds])) for k in KEYS if all(k in s[i] for s in seeds)} for i in ids
        }
    res = {"block": BLOCK, "n_boot": N_BOOT, "seed": SEED, "systems": {}, "paired_vs_musicgen_medium": {}}
    for n, d in sysd.items():
        res["systems"][n] = {"n": len(d)}
        for k in KEYS:
            v = [r[k] for r in d.values() if k in r and np.isfinite(r[k])]
            res["systems"][n][k] = {"n": len(v), "mean_ci95": ci(v, rng)}
    ref = sysd["musicgen_medium"]
    for n, d in sysd.items():
        if n == "musicgen_medium":
            continue
        ids = sorted(set(d) & set(ref))
        res["paired_vs_musicgen_medium"][n] = {"n": len(ids)}
        for k in KEYS:
            v = [d[i][k] - ref[i][k] for i in ids if k in d[i] and k in ref[i]]
            res["paired_vs_musicgen_medium"][n][k] = {"n": len(v), "diff_ci95": ci(v, rng)}
    Path(out_path).write_text(json.dumps(res, indent=1))
    for n, s in res["systems"].items():
        print(n, s["n"], {k: [round(x, 4) for x in s[k]["mean_ci95"]] for k in KEYS})
    for n, s in res["paired_vs_musicgen_medium"].items():
        print("vs MG", n, s["n"], {k: [round(x, 4) for x in s[k]["diff_ci95"]] for k in KEYS})


if __name__ == "__main__":
    main(sys.argv[1])
