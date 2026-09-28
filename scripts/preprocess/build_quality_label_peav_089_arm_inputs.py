#!/usr/bin/env python
"""089: quality-label prefix arm with the tiers defined by PE-AV instead of AES PQ.

Identical to 081 (scripts/preprocess/build_quality_label_081_arm_inputs.py, imported and
sha-pinned; source rows, prefixes, tier fraction, overlay format, verify are all its own)
except for the ranking score:

  081  AES PQ of the 10 s training window
  089  PE-AV cos(caption, 10 s training window)   (score_peav_corpus_089.py, ~/venvs/peav)

Ranked set = the rows 081 ranked (same silence exclusion) that PE-AV scored ok.
bottom 20% -> "Low quality recording. <caption>", top 20% -> "High quality recording. <caption>".

Overlays: a prefixed row whose 081 tier equals its 089 tier has a byte-identical caption to
081, so its overlay is 081's (HDD /mnt/HDD/kojiek/quality_label_081/overlay_new, read-only
dependency). Every other prefixed row gets a new overlay on NVMe (<root>/overlay_new).
The symlink farm (<root>/overlay_farm) points prefixed rows at those, everything else at
text_overlays/slot0clean.

Stages:
  window  assert score_peav_corpus_089.load_window == 075 load_window on 32 stems (array equal)
  build   tiers, train TSV, overlays, farm, manifest (incl. PE-AV<->PQ Spearman, 081x089 tier
          crosstab, per-tier mean PQ -> the "PQ contrast" the analysis needs)
  verify  081's verify() (manifest shas, caption = prefix + source, pandas parse, overlay
          clip_id / caption sha on a sample, re-encode determinism)

--limit N: smoke only, non-production --root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/preprocess"))

Q81_SOURCE = ROOT / "scripts/preprocess/build_quality_label_081_arm_inputs.py"
Q81_SOURCE_SHA256 = "710d1c5cf3a3134b7c7be9a6c937d570aa1bab5d28f4411729e8706af9ac0af0"
SCORER_SOURCE = ROOT / "scripts/preprocess/score_peav_corpus_089.py"
Q81_ROOT = Path("/home/kojiek/exps_nvme/quality_label_081")
Q81_OVERLAY_NEW = Path("/mnt/HDD/kojiek/quality_label_081/overlay_new")
DEFAULT_ROOT = Path("/home/kojiek/exps_nvme/quality_label_089")
OVERLAY_BYTES = 330_000          # one (1,77,1024) f32 overlay npz incl. CLAP + mask
NVME_MARGIN = 25_000_000_000     # left for the S1/S2 checkpoints after the overlay write


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_q81():
    got = sha256(Q81_SOURCE)
    assert got == Q81_SOURCE_SHA256, f"081 builder drift: {got}"
    import build_quality_label_081_arm_inputs as q81
    return q81


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_peav_corpus_089", SCORER_SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def window_gate(q81, rows):
    ref = q81.load_window_fn()
    mine = load_scorer().load_window
    stems = [q81.stem_of(r["id"]) for r in random.Random(20260929).sample(rows, min(32, len(rows)))]
    n = 0
    for s in stems:
        a, b = ref(s), mine(s)
        assert (a is None) == (b is None), s
        if a is not None:
            assert a.dtype == b.dtype and np.array_equal(a, b), f"[FAIL] window mismatch {s}"
            n += 1
    print(f"[window] 089 scorer window == 075 load_window on {n} stems", flush=True)
    return {"n": n, "array_equal": True}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def tiers(q81, rows, peav, tier81):
    ranked = []
    for i, r in enumerate(rows):
        s = peav.get(q81.stem_of(r["id"]))
        if tier81[i] != "unranked" and s and s["status"] == "ok":
            ranked.append((float(s["peav_cos"]), i))
    ranked.sort()
    k = int(round(q81.TIER_FRAC * len(ranked)))
    out = ["unranked"] * len(rows)
    for j, (_, i) in enumerate(ranked):
        out[i] = "low" if j < k else ("high" if j >= len(ranked) - k else "mid")
    cut = {"peav_low_max": ranked[k - 1][0], "peav_high_min": ranked[len(ranked) - k][0]} if k else {}
    return out, len(ranked), cut


def build(q81, root: Path, man, header, rows, names, window, limit):
    peav = {r["stem"]: r for r in csv.DictReader(open(root / "peav_scores.tsv"), delimiter="\t")}
    t81 = list(csv.DictReader(open(Q81_ROOT / "tiers.tsv", encoding="utf-8"), delimiter="\t"))
    assert [r["id"] for r in t81[:len(rows)]] == [r["id"] for r in rows], "081 tiers row order differs"
    tier81 = [r["tier"] for r in t81[:len(rows)]]
    pq = [float(r["PQ"]) if r["PQ"] else float("nan") for r in t81[:len(rows)]]
    tier, n_ranked, cut = tiers(q81, rows, peav, tier81)
    prefix = {"low": q81.LOW, "high": q81.HIGH}
    out_rows = []
    for r, t in zip(rows, tier):
        r2 = dict(r)
        if t in prefix:
            r2["caption"] = prefix[t] + r["caption"]
        out_rows.append(r2)
    arm = root / "arm_inputs"
    arm.mkdir(parents=True, exist_ok=True)
    tsv = arm / "train.tsv"
    with open(tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    (arm / "cache_train.txt").write_text("\n".join(names) + "\n")
    with open(root / "tiers.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["row", "id", "name", "tier", "peav_cos", "tier081", "PQ"])
        for i, (r, t) in enumerate(zip(rows, tier)):
            s = peav.get(q81.stem_of(r["id"])) or {}
            w.writerow([i, r["id"], names[i], t, s.get("peav_cos", ""), tier81[i], t81[i]["PQ"]])

    # overlays: reuse 081's where the prefixed caption is byte-identical, else encode to NVMe
    new_dir = root / "overlay_new"
    reuse = [i for i, t in enumerate(tier) if t in prefix and tier81[i] == t]
    for i in reuse[:64] + random.Random(1).sample(reuse, min(256, len(reuse))):
        assert (Q81_OVERLAY_NEW / q81.overlay_rel(names[i])).exists(), f"081 overlay missing for row {i}"
    todo = [i for i, t in enumerate(tier) if t in prefix and tier81[i] != t
            and not (new_dir / q81.overlay_rel(names[i])).exists()]
    need = len(todo) * OVERLAY_BYTES + (0 if limit else NVME_MARGIN)
    free = shutil.disk_usage(root).free
    print(f"[build] prefixed {sum(t in prefix for t in tier)}: reuse 081 {len(reuse)}, "
          f"new {len(todo)} to write (~{len(todo) * OVERLAY_BYTES / 1e9:.1f} GB); NVMe free {free / 1e9:.1f} GB",
          flush=True)
    if free < need:
        print(f"[FAIL] NVMe free {free / 1e9:.1f} GB < {need / 1e9:.1f} GB (overlays + checkpoint margin)", flush=True)
        sys.exit(3)
    enc, tok, t5, clap, dev = q81.load_encoder()
    fp = enc.encoder_fingerprint()
    ref_fp = str(np.load(q81.OVERLAY_DIR / names[0])["text_encoder_fingerprint"].item())
    assert fp == ref_fp, f"[FAIL] encoder fingerprint {fp} != slot0clean overlay {ref_fp}"
    if reuse:
        z = np.load(Q81_OVERLAY_NEW / q81.overlay_rel(names[reuse[0]]))
        assert str(z["text_encoder_fingerprint"].item()) == fp, "[FAIL] 081 overlay fingerprint differs"
    t0 = time.time()
    for o in range(0, len(todo), 48):
        idx = todo[o:o + 48]
        texts = [out_rows[i]["caption"] for i in idx]
        feats, masks = enc.encode_t5(tok, t5, texts, dev)
        pooled = enc.encode_clap(clap, texts)
        for k, i in enumerate(idx):
            dst = new_dir / q81.overlay_rel(names[i])
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp.npz")
            np.savez(tmp, clip_id=np.asarray(rows[i]["id"]),
                     text_features=feats[k][None].astype(np.float32),
                     text_features_c=pooled[k][None].astype(np.float32),
                     text_attention_mask=masks[k][None].astype(np.int64),
                     caption_sha256=np.asarray(enc.sha_caption(texts[k])),
                     text_encoder_fingerprint=np.asarray(fp))
            os.replace(tmp, dst)
        if (o // 48) % 100 == 0:
            done = o + len(idx)
            print(f"[build] overlays {done}/{len(todo)}, {done / (time.time() - t0):.1f}/s", flush=True)

    farm = root / "overlay_farm"
    farm.mkdir(exist_ok=True)
    for i, n in enumerate(names):
        p = farm / n
        if tier[i] in prefix:
            target = (Q81_OVERLAY_NEW if tier81[i] == tier[i] else new_dir) / q81.overlay_rel(n)
        else:
            target = q81.OVERLAY_DIR / n
        if p.is_symlink():
            if os.readlink(p) == str(target):
                continue
            p.unlink()
        p.symlink_to(target)

    # interpretation stats: how PQ-like are the PE-AV tiers?
    rk = [i for i, t in enumerate(tier) if t != "unranked" and np.isfinite(pq[i])]
    sp = spearman(np.array([float(peav[q81.stem_of(rows[i]["id"])]["peav_cos"]) for i in rk]),
                  np.array([pq[i] for i in rk])) if len(rk) > 2 else float("nan")
    cross = {a: {b: sum(1 for i in range(len(rows)) if tier[i] == a and tier81[i] == b)
                 for b in ("low", "mid", "high", "unranked")} for a in ("low", "mid", "high", "unranked")}

    def tier_mean(vals, sel, tt):
        v = [vals[i] for i, t in enumerate(tt) if t == sel and np.isfinite(vals[i])]
        return float(np.mean(v)) if v else float("nan")

    pq_by = {t: tier_mean(pq, t, tier) for t in ("low", "mid", "high")}
    pq_by81 = {t: tier_mean(pq, t, tier81) for t in ("low", "mid", "high")}
    pv = [float(peav[q81.stem_of(r["id"])]["peav_cos"]) if (peav.get(q81.stem_of(r["id"])) or {}).get("status") == "ok"
          else float("nan") for r in rows]
    peav_by = {t: tier_mean(pv, t, tier) for t in ("low", "mid", "high")}
    peav_by81 = {t: tier_mean(pv, t, tier81) for t in ("low", "mid", "high")}
    lens = {key: len(tok(pfx.strip()).input_ids) - 1 for key, pfx in prefix.items()}

    def mention(sel):
        idx = [i for i, t in enumerate(tier) if t == sel]
        return {"n": len(idx), "quality_word_rate": (sum(bool(q81.QUALITY_WORDS.search(rows[i]["caption"]))
                                                         for i in idx) / max(1, len(idx)))}

    stats = {
        "rows": len(rows), "ranked": n_ranked,
        "tier_counts": {t: tier.count(t) for t in ("low", "mid", "high", "unranked")},
        "tier_cut": cut, "prefix_tokens": lens,
        "spearman_peav_vs_pq": sp,
        "crosstab_089_by_081": cross,
        "pq_by_089_tier": pq_by, "pq_by_081_tier": pq_by81,
        "pq_contrast_high_minus_low": {"089": pq_by["high"] - pq_by["low"], "081": pq_by81["high"] - pq_by81["low"]},
        "peav_by_089_tier": peav_by, "peav_by_081_tier": peav_by81,
        "overlays": {"reused_081": len(reuse), "new_nvme": sum(1 for i, t in enumerate(tier)
                                                                 if t in prefix and tier81[i] != t)},
        "source_caption_quality_word_rate": {t: mention(t) for t in ("low", "mid", "high")},
    }
    manifest = {
        "status": "arm_inputs_ready" if not limit else "smoke_only", "experiment": "089_quality_label_peav",
        "limit": limit, "rows": len(rows), "prefixes": {"low": q81.LOW, "high": q81.HIGH},
        "tier_frac": q81.TIER_FRAC,
        "label_metric": "PE-AV (facebook/pe-av-large) cos(plain caption, 10 s training window @48k) "
                        "over the rows 081 ranked",
        "train_tsv": str(tsv), "train_tsv_sha256": sha256(tsv),
        "cache_list": str(arm / "cache_train.txt"), "cache_list_sha256": sha256(arm / "cache_train.txt"),
        "npz_dir": str(q81.NPZ_DIR), "text_npz_dir": str(farm), "overlay_new_dir": str(new_dir),
        "overlay_reuse_dir_081": str(Q81_OVERLAY_NEW),
        "peav_scores_tsv": str(root / "peav_scores.tsv"), "peav_scores_tsv_sha256": sha256(root / "peav_scores.tsv"),
        "peav_gate": json.load(open(root / "peav_gate.json")), "window_gate": window,
        "tiers_tsv_sha256": sha256(root / "tiers.tsv"), "tiers081_tsv_sha256": sha256(Q81_ROOT / "tiers.tsv"),
        "source_manifest": str(q81.SRC_MANIFEST), "source_train_tsv_sha256": man["train_tsv_sha256"],
        "source_cache_list_sha256": man["cache_list_sha256"], "text_encoder_fingerprint": fp,
        "build_script_sha256": sha256(Path(__file__)), "scorer_sha256": sha256(SCORER_SOURCE),
        "q81_builder_sha256": Q81_SOURCE_SHA256, "stats": stats,
    }
    json.dump(manifest, open(arm / "manifest.json", "w"), indent=1)
    print("[build] done", json.dumps(stats, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["window", "build", "verify", "all"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify_sample", type=int, default=4000)
    ap.add_argument("--reencode", type=int, default=16)
    a = ap.parse_args()
    if a.limit:
        assert a.root != DEFAULT_ROOT, "--limit must use a non-production root"
    a.root.mkdir(parents=True, exist_ok=True)
    q81 = load_q81()
    man, header, rows, names = q81.source(a.limit)
    wfile = a.root / "window_gate.json"
    if a.stage in ("window", "all") and not wfile.exists():
        wfile.write_text(json.dumps(window_gate(q81, rows)) + "\n")
    if a.stage in ("build", "all"):
        assert (a.root / "peav_scores.tsv").exists(), "run score_peav_corpus_089.py first"
        if not (a.root / "arm_inputs/manifest.json").exists() or a.stage == "build":
            build(q81, a.root, man, header, rows, names, json.load(open(wfile)), a.limit)
    if a.stage in ("verify", "all"):
        q81.verify(a.root, rows, names, a.verify_sample, a.reencode)


if __name__ == "__main__":
    main()
