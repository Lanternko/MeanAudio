#!/usr/bin/env python
"""081: build the quality-label prefix arm inputs (QA-MDT-style real quality labels).

Source: the slot0clean_nmv2matched corpus (251,596 rows), the exact inputs of the
066/070/071 nmv2pair control. Same audio NPZ, same cache list, same row order.
Only the caption of rows in the bottom / top quality tier changes:

  bottom 20% by PQ   "Low quality recording. <caption>"
  top 20% by PQ      "High quality recording. <caption>"
  middle 60%         <caption>                (byte-identical to the control)

The label is the corpus's own Audiobox Aesthetics PQ, scored on the exact audio
window the model trains on: the 30 s wav peak-normalised to 0.95, first 160,000
samples (10 s @ 16 kHz), i.e. load_window() of the 075 builder (imported, sha
pinned). Rows whose window is silent (peak < 1e-6 or RMS < -45 dBFS) or failed to
score keep their caption and are not ranked.

Stages (each idempotent / resumable):
  score   AES over the needed wavs in os.scandir order (the wav dir is one flat
          exFAT directory: random opens cost ~2 s each, scandir-order opens ~8 ms).
          Appends to labels.partial.tsv; `labels.tsv` is written when complete.
          Gate first: in-memory AES == path-based eval_metrics.score_aes (|d| <= 5e-3).
  build   tiers, train TSV, new (1, 77, 1024) overlays for the prefixed rows on HDD
          (sharded <root_hdd>/overlay_new/<int(stem)//1000>/<name>, tmp+rename), and
          a flat symlink farm on NVMe (prefixed -> new overlay, else slot0clean).
  verify  manifest shas, caption = prefix + source, pandas == csv parse, overlay
          clip_id / caption_sha256 binding on a sample, re-encode determinism.

--limit N restricts every stage to the first N source rows (smoke only; writes
under --root/--root_hdd given explicitly, never the production dirs).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/preprocess"))

SRC_INPUTS = Path("/home/kojiek/exps_nvme/slot0clean_nmv2matched/arm_inputs")
SRC_TSV = SRC_INPUTS / "phase8_caption2p0_slot0clean_nmv2matched_train.tsv"
SRC_CACHE = SRC_INPUTS / "cache_train.txt"
SRC_MANIFEST = SRC_INPUTS / "manifest.json"
NPZ_DIR = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz")
OVERLAY_DIR = Path("/home/kojiek/text_overlays/slot0clean")
WAV_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio")
ENCODER_SOURCE = Path("/home/kojiek/research/meanaudio_training/caption10s_pipeline/reextract_text_inplace_caption10s.py")
ENCODER_SOURCE_SHA256 = "eb692393994a414b5578e6ab4e5c46c8aa7e66f2a09e39f2061bfe83768374dc"
WINDOW_SOURCE = ROOT / "scripts/preprocess/build_defect_negsample_arm_inputs.py"
WINDOW_SOURCE_SHA256 = "a63b49270bf395e42e4000366220bf34ee379770ea19d4e5e18b8476eaaedc37"
DEFAULT_ROOT = Path("/home/kojiek/exps_nvme/quality_label_081")
DEFAULT_ROOT_HDD = Path("/mnt/HDD/kojiek/quality_label_081")

LOW, HIGH = "Low quality recording. ", "High quality recording. "
TIER_FRAC = 0.20
SILENT_RMS_DBFS = -45.0
AES_KEYS = ("CE", "CU", "PC", "PQ")
LABEL_FIELDS = ["stem", "status", "rms_dbfs", *AES_KEYS]
QUALITY_WORDS = re.compile(r"\b(low[- ]quality|poor(?:ly)?|lo-?fi|noisy|muffled|distort\w*|amateur|"
                           r"high[- ]quality|hi-?fi|pristine|polished|professional\w*)\b", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_window_fn():
    got = sha256(WINDOW_SOURCE)
    assert got == WINDOW_SOURCE_SHA256, f"075 builder drift: {got}"
    import build_defect_negsample_arm_inputs as b075
    assert b075.WAV_DIR == WAV_DIR and b075.NUM_SAMPLES == 160_000 and b075.SR == 16_000
    return b075.load_window


def source(limit: int | None):
    man = json.load(open(SRC_MANIFEST))
    assert sha256(SRC_TSV) == man["train_tsv_sha256"], "source tsv drift"
    assert sha256(SRC_CACHE) == man["cache_list_sha256"], "source cache drift"
    header = open(SRC_TSV, encoding="utf-8").readline().rstrip("\n").split("\t")
    rows = list(csv.DictReader(open(SRC_TSV, encoding="utf-8", newline=""), delimiter="\t"))
    names = [l.strip() for l in open(SRC_CACHE) if l.strip()]
    assert len(rows) == len(names) == man["rows"]
    if limit:
        rows, names = rows[:limit], names[:limit]
    return man, header, rows, names


def stem_of(row_id: str) -> str:
    return row_id.rsplit("_", 1)[0]


# ------------------------------------------------------------------ score
def aes_predictor():
    import audiobox_aesthetics.infer as aes_infer
    from eval_metrics import _read_wav_sf
    aes_infer.read_wav = _read_wav_sf
    return aes_infer.AesPredictor(checkpoint_pth=None, batch_size=32)


def aes_tensors(pred, wins):
    import torch
    out = pred.forward([{"path": torch.from_numpy(w.astype(np.float32))[None], "sample_rate": 16_000}
                        for w in wins])
    return [{k: float(r[k]) for k in AES_KEYS} for r in out]


def score_gate(pred, load_window, stems, tmp: Path):
    """The in-memory tensor path must give the same scores as eval_metrics.score_aes on a wav file."""
    import soundfile as sf
    from eval_metrics import score_aes
    tmp.mkdir(parents=True, exist_ok=True)
    wins, paths = [], []
    for s in stems:
        w = load_window(s)
        if w is None:
            continue
        p = tmp / f"{s}.wav"
        sf.write(p, w.astype(np.float32), 16_000, subtype="FLOAT")
        wins.append(w); paths.append(str(p))
    mem = aes_tensors(pred, wins)
    ref, failed = score_aes(paths, batch_size=32, progress=False)
    assert not failed, failed
    d = max(abs(m[k] - ref[p][k]) for m, p in zip(mem, paths) for k in AES_KEYS)
    for p in paths:
        os.unlink(p)
    gate = {"n": len(paths), "max_abs_diff": d}
    print("[score] AES path gate:", gate, flush=True)
    assert len(paths) >= 16 and d <= 5e-3, f"[FAIL] AES path gate {gate}"
    return gate


def score(root: Path, rows, limit):
    load_window = load_window_fn()
    need = {stem_of(r["id"]) for r in rows}
    final = root / "labels.tsv"
    if final.exists():
        done = {r["stem"] for r in csv.DictReader(open(final), delimiter="\t")}
        assert need <= done, f"labels.tsv misses {len(need - done)} stems"
        print(f"[score] labels.tsv complete ({len(done)} stems)", flush=True)
        return json.load(open(root / "score_gate.json"))
    part = root / "labels.partial.tsv"
    done = set()
    if part.exists():
        done = {r["stem"] for r in csv.DictReader(open(part), delimiter="\t")}
    todo = need - done
    print(f"[score] need {len(need)} stems, {len(done)} done, {len(todo)} to go", flush=True)
    pred = aes_predictor()
    gate_file = root / "score_gate.json"
    if not gate_file.exists():
        gate = score_gate(pred, load_window, sorted(need)[:32], root / "_gate_tmp")
        gate_file.write_text(json.dumps(gate) + "\n")
    gate = json.load(open(gate_file))

    q: queue.Queue = queue.Queue(maxsize=256)

    def reader():
        left = set(todo)
        with os.scandir(WAV_DIR) as it:
            for e in it:
                if not left:
                    break
                if not e.name.endswith(".wav"):
                    continue
                s = e.name[:-4]
                if s in left:
                    left.discard(s)
                    try:
                        q.put((s, load_window(s), None))
                    except Exception as ex:  # unreadable file: record, keep going
                        q.put((s, None, repr(ex)))
        for s in sorted(left):  # not in the directory at all
            q.put((s, None, "missing"))
        q.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    new = not part.exists()
    f = open(part, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=LABEL_FIELDS, delimiter="\t", lineterminator="\n")
    if new:
        w.writeheader()
    buf, n, t0 = [], 0, time.time()

    def flush():
        nonlocal n
        if buf:
            res = aes_tensors(pred, [x[1] for x in buf])
            for (s, _, rms), r in zip(buf, res):
                w.writerow({"stem": s, "status": "ok", "rms_dbfs": f"{rms:.3f}",
                            **{k: f"{r[k]:.6f}" for k in AES_KEYS}})
            n += len(buf)
            buf.clear()
            f.flush()

    while True:
        item = q.get()
        if item is None:
            break
        s, win, err = item
        if win is None:
            w.writerow({"stem": s, "status": err or "silent_peak", "rms_dbfs": "", **{k: "" for k in AES_KEYS}})
            continue
        rms = 20 * np.log10(np.sqrt(np.mean(win**2)) + 1e-12)
        buf.append((s, win, rms))
        if len(buf) == 32:
            flush()
            if n % 3200 == 0:
                rate = n / (time.time() - t0)
                print(f"[score] {n}/{len(todo)} scored, {rate:.1f}/s, eta {(len(todo) - n) / rate / 60:.1f} min",
                      flush=True)
    flush()
    f.close()
    got = {r["stem"] for r in csv.DictReader(open(part), delimiter="\t")}
    assert need <= got, f"score incomplete: {len(need - got)} stems missing"
    if not limit:
        os.replace(part, final)
    else:
        (root / "labels.tsv").write_text(part.read_text())
    return gate


# ------------------------------------------------------------------ build
def tiers(rows, labels):
    """row index -> 'low' / 'high' / 'mid' / 'unranked'. Rank by PQ, ties broken by row index."""
    ranked = []
    for i, r in enumerate(rows):
        lab = labels.get(stem_of(r["id"]))
        if lab and lab["status"] == "ok" and float(lab["rms_dbfs"]) >= SILENT_RMS_DBFS:
            ranked.append((float(lab["PQ"]), i))
    ranked.sort()
    k = int(round(TIER_FRAC * len(ranked)))
    out = ["unranked"] * len(rows)
    for j, (_, i) in enumerate(ranked):
        out[i] = "low" if j < k else ("high" if j >= len(ranked) - k else "mid")
    cut = {"pq_low_max": ranked[k - 1][0], "pq_high_min": ranked[len(ranked) - k][0]} if k else {}
    return out, len(ranked), cut


def overlay_rel(name: str) -> str:
    return f"{int(Path(name).stem) // 1000:04d}/{name}"


def load_encoder():
    import torch
    spec = importlib.util.spec_from_file_location("bound_text_encoder", ENCODER_SOURCE)
    assert sha256(ENCODER_SOURCE) == ENCODER_SOURCE_SHA256, "encoder source drift"
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    dev = torch.device("cuda")
    tok = enc.AutoTokenizer.from_pretrained(enc.T5_MODEL, revision=enc.T5_REVISION, local_files_only=True)
    t5 = enc.T5EncoderModel.from_pretrained(enc.T5_MODEL, revision=enc.T5_REVISION,
                                            local_files_only=True).eval().to(dev)
    clap = enc.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap.load_ckpt(str(enc.CLAP_CKPT), verbose=False)
    return enc, tok, t5, clap, dev


def build(root: Path, root_hdd: Path, man, header, rows, names, gate, limit):
    labels = {r["stem"]: r for r in csv.DictReader(open(root / "labels.tsv"), delimiter="\t")}
    tier, n_ranked, cut = tiers(rows, labels)
    prefix = {"low": LOW, "high": HIGH}
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
        w.writerow(["row", "id", "name", "tier", "PQ"])
        for i, (r, t) in enumerate(zip(rows, tier)):
            lab = labels.get(stem_of(r["id"])) or {}
            w.writerow([i, r["id"], names[i], t, lab.get("PQ", "")])

    enc, tok, t5, clap, dev = load_encoder()
    fp = enc.encoder_fingerprint()
    ref_fp = str(np.load(OVERLAY_DIR / names[0])["text_encoder_fingerprint"].item())
    assert fp == ref_fp, f"[FAIL] encoder fingerprint {fp} != slot0clean overlay {ref_fp}"

    new_dir = root_hdd / "overlay_new"
    todo = [i for i, t in enumerate(tier) if t in prefix and not (new_dir / overlay_rel(names[i])).exists()]
    n_new = sum(t in prefix for t in tier)
    print(f"[build] {n_new} prefixed rows, {len(todo)} overlays to write", flush=True)
    t0 = time.time()
    for o in range(0, len(todo), 48):
        idx = todo[o:o + 48]
        texts = [out_rows[i]["caption"] for i in idx]
        feats, masks = enc.encode_t5(tok, t5, texts, dev)
        pooled = enc.encode_clap(clap, texts)
        for k, i in enumerate(idx):
            dst = new_dir / overlay_rel(names[i])
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
        target = (new_dir / overlay_rel(n)) if tier[i] in prefix else (OVERLAY_DIR / n)
        if p.is_symlink():
            if os.readlink(p) == str(target):
                continue
            p.unlink()
        p.symlink_to(target)

    # the prefix must survive T5's 77-token window; report how often it pushes caption text out
    lens = {}
    for key, pfx in prefix.items():
        lens[key] = len(tok(pfx.strip()).input_ids) - 1   # without </s>
    rng = random.Random(20260926)
    sample = rng.sample([i for i, t in enumerate(tier) if t in prefix], min(5000, n_new))
    trunc_src = sum(len(tok(rows[i]["caption"]).input_ids) > 77 for i in sample)
    trunc_new = sum(len(tok(out_rows[i]["caption"]).input_ids) > 77 for i in sample)

    def mention(sel):
        idx = [i for i, t in enumerate(tier) if t == sel]
        return {"n": len(idx), "quality_word_rate": (sum(bool(QUALITY_WORDS.search(rows[i]["caption"]))
                                                         for i in idx) / max(1, len(idx)))}

    pq_by_tier = {}
    for sel in ("low", "mid", "high"):
        v = [float(labels[stem_of(rows[i]["id"])]["PQ"]) for i, t in enumerate(tier) if t == sel]
        pq_by_tier[sel] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
    stats = {
        "rows": len(rows), "ranked": n_ranked, "unranked": tier.count("unranked"),
        "tier_counts": {t: tier.count(t) for t in ("low", "mid", "high", "unranked")},
        "tier_cut": cut, "pq_by_tier": pq_by_tier,
        "prefix_tokens": lens,
        "truncated_over_77_tokens_sample": {"n": len(sample), "source": trunc_src, "prefixed": trunc_new},
        "source_caption_quality_word_rate": {t: mention(t) for t in ("low", "mid", "high")},
    }
    manifest = {
        "status": "arm_inputs_ready" if not limit else "smoke_only", "experiment": "081_quality_label_prefix",
        "limit": limit, "rows": len(rows), "prefixes": {"low": LOW, "high": HIGH}, "tier_frac": TIER_FRAC,
        "silent_rms_dbfs": SILENT_RMS_DBFS, "label_metric": "audiobox_aesthetics PQ on the 10 s training window",
        "train_tsv": str(tsv), "train_tsv_sha256": sha256(tsv),
        "cache_list": str(arm / "cache_train.txt"), "cache_list_sha256": sha256(arm / "cache_train.txt"),
        "npz_dir": str(NPZ_DIR), "text_npz_dir": str(farm), "overlay_new_dir": str(new_dir),
        "labels_tsv": str(root / "labels.tsv"), "labels_tsv_sha256": sha256(root / "labels.tsv"),
        "tiers_tsv_sha256": sha256(root / "tiers.tsv"),
        "source_manifest": str(SRC_MANIFEST), "source_train_tsv_sha256": man["train_tsv_sha256"],
        "source_cache_list_sha256": man["cache_list_sha256"],
        "aes_gate": gate, "text_encoder_fingerprint": fp,
        "build_script_sha256": sha256(Path(__file__)), "window_source_sha256": WINDOW_SOURCE_SHA256,
        "stats": stats,
    }
    json.dump(manifest, open(arm / "manifest.json", "w"), indent=1)
    print("[build] done", json.dumps(stats, indent=1), flush=True)


# ------------------------------------------------------------------ verify
def verify(root: Path, rows, names, n_sample: int, reencode: int):
    import pandas as pd
    arm = root / "arm_inputs"
    man = json.load(open(arm / "manifest.json"))
    tsv, cache = Path(man["train_tsv"]), Path(man["cache_list"])
    assert sha256(tsv) == man["train_tsv_sha256"] and sha256(cache) == man["cache_list_sha256"]
    new = list(csv.DictReader(open(tsv, encoding="utf-8", newline=""), delimiter="\t"))
    assert [l.strip() for l in open(cache) if l.strip()] == names
    assert len(new) == len(rows)
    tier = [r["tier"] for r in csv.DictReader(open(root / "tiers.tsv"), delimiter="\t")]
    pfx = {"low": LOW, "high": HIGH}
    for r0, r1, t in zip(rows, new, tier):
        assert r1["id"] == r0["id"] and r1["q_level"] == r0["q_level"]
        assert r1["caption"] == pfx.get(t, "") + r0["caption"], r0["id"]
    df = pd.read_csv(tsv, sep="\t")
    assert len(df) == len(new) and list(df["id"].astype(str)) == [r["id"] for r in new]
    assert [str(c) for c in df["caption"]] == [r["caption"] for r in new], "pandas/csv caption parse differ"
    farm = Path(man["text_npz_dir"])
    rng = random.Random(20260926)
    idx = sorted(set(rng.sample(range(len(new)), min(n_sample, len(new)))
                     + [i for i, t in enumerate(tier) if t in pfx][:64]))
    for i in idx:
        z = np.load(farm / names[i])
        assert str(z["clip_id"].item()) == new[i]["id"], (i, "clip_id")
        stored = str(z["caption_sha256"].item()).split(",")
        assert stored[0] == hashlib.sha256(new[i]["caption"].encode("utf-8")).hexdigest(), (i, "caption sha")
        assert z["text_features"].shape[-2:] == (77, 1024) and z["text_features"].dtype == np.float32
        if tier[i] in pfx:
            assert z["text_features"].shape == (1, 77, 1024), (i, z["text_features"].shape)
    re_out = {}
    if reencode:
        enc, tok, t5, clap, dev = load_encoder()
        pick = [i for i in idx if tier[i] in pfx][:reencode]
        feats, _ = enc.encode_t5(tok, t5, [new[i]["caption"] for i in pick], dev)
        diffs = [float(np.abs(np.load(farm / names[i])["text_features"][0] - feats[k]).max())
                 for k, i in enumerate(pick)]
        re_out = {"n": len(pick), "max_abs_diff": max(diffs)}
        assert max(diffs) < 1e-3, re_out
    # training-read cost of the HDD overlays, cold and random order
    t0 = time.time()
    sel = [i for i in rng.sample(range(len(new)), min(200, len(new))) if tier[i] in pfx]
    for i in sel:
        np.load(farm / names[i])["text_features"]
    read = {"n": len(sel), "ms_per_file": 1000 * (time.time() - t0) / max(1, len(sel))}
    out = {"verified_rows": len(new), "binding_sample": len(idx), "reencode": re_out, "hdd_random_read": read}
    (arm / "verify.json").write_text(json.dumps(out, indent=1) + "\n")
    print("[verify] OK", out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "build", "verify", "all"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--root_hdd", type=Path, default=DEFAULT_ROOT_HDD)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify_sample", type=int, default=4000)
    ap.add_argument("--reencode", type=int, default=16)
    a = ap.parse_args()
    if a.limit:
        assert a.root != DEFAULT_ROOT and a.root_hdd != DEFAULT_ROOT_HDD, "--limit must use non-production roots"
    a.root.mkdir(parents=True, exist_ok=True)
    man, header, rows, names = source(a.limit)
    if a.stage in ("score", "all"):
        gate = score(a.root, rows, a.limit)
    else:
        gate = json.load(open(a.root / "score_gate.json"))
    if a.stage in ("build", "all"):
        if not (a.root / "arm_inputs/manifest.json").exists() or a.stage == "build":
            build(a.root, a.root_hdd, man, header, rows, names, gate, a.limit)
    if a.stage in ("verify", "all"):
        verify(a.root, rows, names, a.verify_sample, a.reencode)


if __name__ == "__main__":
    main()
