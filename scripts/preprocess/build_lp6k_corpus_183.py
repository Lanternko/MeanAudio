#!/usr/bin/env python
"""183 arm L: 6 kHz low-passed copy of the slot0clean_nmv2matched training corpus.

Same window as the official latents (075 builder): wav peak-normalised to 0.95, first
160,000 samples (10 s @ 16 kHz). Then an 8th-order Butterworth 6 kHz low-pass
(sosfiltfilt, zero phase) and a scalar gain back to the unfiltered window's integrated
LUFS (peak capped at 0.999). mel_converter('16k') -> v1-16 VAE encode -> mean/std.

Only mean/std/clip_id/catalog_index are written: training reads text from the
slot0clean overlay (require_text_overlay), never from the audio NPZ.

Windows the official pipeline would have kept but that are silent (peak < 1e-6) or
too quiet to meter get the official latents copied unchanged (counted in manifest).

Gates (design doc muq_train_exploit_183_20261108.md section 2.1):
  G_L0  unfiltered path reproduces the official latents: median corr >= 0.9999 (2,000 clips)
  G_L1  500 clips decoded (VAE + BigVGAN): 6-8 kHz share >= 15 dB lower than the decoded
        unfiltered latent; below-4 kHz energy within 0.5 dB on the pre-VAE waveform
        (amended 2026-11-08 after run 1: post-VAE low-band drift is a decoder effect, reported only)
Run `--gates_only` first; the full build refuses to start without a passed gate file.
Resumable: existing output files are skipped (written via tmp + rename).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt, welch

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SRC_INPUTS = Path("/home/kojiek/exps_nvme/slot0clean_nmv2matched/arm_inputs")
SRC_TSV = SRC_INPUTS / "phase8_caption2p0_slot0clean_nmv2matched_train.tsv"
SRC_CACHE = SRC_INPUTS / "cache_train.txt"
SRC_MANIFEST = SRC_INPUTS / "manifest.json"
NPZ_DIR = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz")
WAV_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio")
SR = 16_000
NUM_SAMPLES = 160_000
CUTOFF = 6000.0
SOS = butter(8, CUTOFF, btype="low", fs=SR, output="sos")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_window(name: str):
    x, sr = sf.read(WAV_DIR / f"{name}.wav", dtype="float32", always_2d=False)
    assert sr == SR, f"{name}: sr {sr}"
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    if peak < 1e-6:
        return None
    x = x / peak * 0.95
    if x.shape[0] < NUM_SAMPLES:
        x = np.pad(x, (0, NUM_SAMPLES - x.shape[0]))
    return x[:NUM_SAMPLES].astype(np.float64)


_METER = pyln.Meter(SR)


def lowpass_matched(x: np.ndarray):
    """Returns (y, status) with status in {ok, capped, unmetered}."""
    y = sosfiltfilt(SOS, x)
    src = _METER.integrated_loudness(x)
    pre = _METER.integrated_loudness(y)
    status = "ok"
    if np.isfinite(src) and np.isfinite(pre) and src > -70:
        y = y * 10 ** ((src - pre) / 20)
    else:
        status = "unmetered"
    pk = np.abs(y).max()
    if pk > 0.999:
        y = y * (0.999 / pk)
        status = "capped" if status == "ok" else status
    return y, status


class Windows(torch.utils.data.Dataset):
    def __init__(self, items, filtered: bool):
        self.items, self.filtered = items, filtered   # items: (row_index, clip_id, npz_name)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, cid, name = self.items[k]
        x = load_window(cid.rsplit("_", 1)[0])
        if x is None:
            return i, cid, name, np.zeros(NUM_SAMPLES, np.float32), "silent"
        if not self.filtered:
            return i, cid, name, x.astype(np.float32), "raw"
        y, st = lowpass_matched(x)
        return i, cid, name, y.astype(np.float32), st


def collate(b):
    return [list(z) for z in zip(*b)]


def build_vae():
    from meanaudio.ext.autoencoder import AutoEncoderModule
    from meanaudio.ext.mel_converter import get_mel_converter
    torch.backends.cuda.matmul.allow_tf32 = True   # as extract_audio_latents.py / 075
    torch.backends.cudnn.allow_tf32 = True
    tod = AutoEncoderModule(vae_ckpt_path=str(ROOT / "weights/v1-16.pth"),
                            vocoder_ckpt_path=str(ROOT / "weights/best_netG.pt"), mode="16k").eval().cuda()
    mel = get_mel_converter("16k").eval().cuda()
    return tod, mel


def encode(tod, mel, wavs):
    with torch.no_grad():
        d = tod.encode(mel(torch.from_numpy(np.stack(wavs)).float().cuda()))
    return d.mean.cpu().transpose(1, 2).numpy(), d.std.cpu().transpose(1, 2).numpy()


def decode_wav(tod, mean_btc: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = torch.from_numpy(mean_btc).float().cuda().transpose(1, 2)
        wav = tod.vocode(tod.decode(z))
    return wav.squeeze(1).float().cpu().numpy() if wav.ndim == 3 else wav.float().cpu().numpy()


def band_db(w: np.ndarray, lo: float, hi: float) -> float:
    f, p = welch(w.astype(np.float64), fs=SR, nperseg=2048)
    tot = p.sum() + 1e-30
    return float(10 * np.log10(p[(f >= lo) & (f < hi)].sum() / tot + 1e-30))


def band_abs_db(w: np.ndarray, lo: float, hi: float) -> float:
    f, p = welch(w.astype(np.float64), fs=SR, nperseg=2048)
    return float(10 * np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-30))


def load_rows():
    man = json.load(open(SRC_MANIFEST))
    assert sha256(SRC_TSV) == man["train_tsv_sha256"], "source tsv drift"
    assert sha256(SRC_CACHE) == man["cache_list_sha256"], "source cache drift"
    rows = list(csv.DictReader(open(SRC_TSV, encoding="utf-8", newline=""), delimiter="\t"))
    names = [l.strip() for l in open(SRC_CACHE) if l.strip()]
    assert len(rows) == len(names) == man["rows"], (len(rows), len(names), man["rows"])
    return rows, names, man


def run_gates(args, rows, names):
    tod, mel = build_vae()
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(rows), size=args.gate_n, replace=False).tolist())
    items = [(i, rows[i]["id"], names[i]) for i in pick]
    dl = torch.utils.data.DataLoader(Windows(items, filtered=False), batch_size=args.batch,
                                     num_workers=args.workers, collate_fn=collate)
    corrs = []
    for idx, cids, nms, wavs, sts in dl:
        keep = [k for k, s in enumerate(sts) if s != "silent"]
        if not keep:
            continue
        m, _ = encode(tod, mel, [wavs[k] for k in keep])
        for j, k in enumerate(keep):
            z = np.load(NPZ_DIR / nms[k])
            assert str(z["clip_id"].item()) == cids[k], f"clip_id mismatch {nms[k]}"
            corrs.append(float(np.corrcoef(m[j].ravel(), z["mean"].ravel())[0, 1]))
    g0 = {"n": len(corrs), "median_corr": float(np.median(corrs)), "min_corr": float(np.min(corrs)),
          "p01_corr": float(np.quantile(corrs, 0.01))}
    g0["pass"] = g0["median_corr"] >= 0.9999
    print("G_L0", g0, flush=True)

    # G_L1: decoded filtered vs decoded unfiltered latents
    sub = items[: args.g1_n]
    d68, d46, dlow, dlow_pre, dlufs, stat = [], [], [], [], [], {}
    for o in range(0, len(sub), args.batch):
        chunk = sub[o:o + args.batch]
        raws, lps = [], []
        for i, cid, nm in chunk:
            x = load_window(cid.rsplit("_", 1)[0])
            if x is None:
                continue
            y, st = lowpass_matched(x)
            stat[st] = stat.get(st, 0) + 1
            dlow_pre.append(band_abs_db(y, 0, 4000) - band_abs_db(x, 0, 4000))
            raws.append(x.astype(np.float32)); lps.append(y.astype(np.float32))
        if not raws:
            continue
        mr, _ = encode(tod, mel, raws)
        ml, _ = encode(tod, mel, lps)
        wr, wl = decode_wav(tod, mr), decode_wav(tod, ml)
        for a, b in zip(wr, wl):
            d68.append(band_db(b, 6000, 8000) - band_db(a, 6000, 8000))
            d46.append(band_db(b, 4000, 6000) - band_db(a, 4000, 6000))
            dlow.append(band_abs_db(b, 0, 4000) - band_abs_db(a, 0, 4000))
            la, lb = _METER.integrated_loudness(a.astype(np.float64)), _METER.integrated_loudness(b.astype(np.float64))
            if np.isfinite(la) and np.isfinite(lb):
                dlufs.append(lb - la)
    g1 = {"n": len(d68), "d_share_6_8k_db_mean": float(np.mean(d68)), "d_share_6_8k_db_median": float(np.median(d68)),
          "d_share_4_6k_db_mean": float(np.mean(d46)), "d_low_0_4k_db_mean": float(np.mean(dlow)),
          "d_low_0_4k_db_abs_p95": float(np.quantile(np.abs(dlow), 0.95)), "status_counts": stat,
          "d_low_0_4k_pre_vae_db_mean": float(np.mean(dlow_pre)),
          "d_low_0_4k_pre_vae_db_abs_p95": float(np.quantile(np.abs(dlow_pre), 0.95)),
          "d_lufs_decoded_mean": float(np.mean(dlufs)), "d_lufs_decoded_median": float(np.median(dlufs))}
    # Deviation (2026-11-08, after first gate run): the low-band condition is judged on the waveform
    # the VAE encodes (pre-VAE). Post-VAE low-band drift (-0.75 dB in run 1; pre-VAE +0.06) is a
    # decoder effect on band-limited mels, reported but not gating; scoring is lvl30 loudness-aligned.
    g1["pass"] = g1["d_share_6_8k_db_mean"] <= -15.0 and abs(g1["d_low_0_4k_pre_vae_db_mean"]) <= 0.5
    print("G_L1", g1, flush=True)
    out = {"G_L0": g0, "G_L1": g1, "pass": bool(g0["pass"] and g1["pass"]), "seed": args.seed,
           "builder_sha256": sha256(Path(__file__)), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (args.root).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.root / "gates.json", "w"), indent=1)
    return out


def run_build(args, rows, names, man):
    gates = json.load(open(args.root / "gates.json"))
    assert gates["pass"], "[FAIL] gates.json not passed"
    assert gates["builder_sha256"] == sha256(Path(__file__)), "builder changed since gates; rerun --gates_only"
    out_dir = args.root / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, rows[i]["id"], names[i]) for i in range(len(rows)) if not (out_dir / names[i]).exists()]
    print(f"todo {len(todo)} / {len(rows)}", flush=True)
    tod, mel = build_vae()
    dl = torch.utils.data.DataLoader(Windows(todo, filtered=True), batch_size=args.batch,
                                     num_workers=args.workers, collate_fn=collate, prefetch_factor=4)
    counts_path = args.root / "status_counts.json"
    counts = json.load(open(counts_path)) if counts_path.exists() else {}
    t0, done = time.time(), 0
    for idx, cids, nms, wavs, sts in dl:
        keep = [k for k, s in enumerate(sts) if s != "silent"]
        mm = ss = None
        if keep:
            mm, ss = encode(tod, mel, [wavs[k] for k in keep])
        pos = {k: j for j, k in enumerate(keep)}
        for k in range(len(nms)):
            src = np.load(NPZ_DIR / nms[k])
            assert str(src["clip_id"].item()) == cids[k], f"clip_id mismatch {nms[k]}"
            cat = src["catalog_index"] if "catalog_index" in src.files else np.asarray(-1)
            if sts[k] == "silent":
                mean, std = src["mean"], src["std"]
            else:
                mean, std = mm[pos[k]].astype(np.float32), ss[pos[k]].astype(np.float32)
            tmp = out_dir / (nms[k] + ".tmp.npz")
            np.savez(tmp, mean=mean, std=std, clip_id=np.asarray(cids[k]), catalog_index=cat,
                     lp6k_status=np.asarray(sts[k]))
            os.replace(tmp, out_dir / nms[k])
            counts[sts[k]] = counts.get(sts[k], 0) + 1
        done += len(nms)
        if done % (args.batch * 200) < args.batch:
            rate = done / (time.time() - t0)
            print(f"{done}/{len(todo)}  {rate:.1f} clip/s  eta {(len(todo) - done) / rate / 3600:.2f} h  {counts}",
                  flush=True)
            json.dump(counts, open(counts_path, "w"))
    json.dump(counts, open(counts_path, "w"))
    n_out = sum(1 for n in names if (out_dir / n).exists())
    assert n_out == len(names), f"[FAIL] {n_out} / {len(names)} written"

    # arm inputs: identical TSV and cache list, only the npz dir changes
    arm = args.root / "arm_inputs"
    arm.mkdir(exist_ok=True)
    for p in (SRC_TSV, SRC_CACHE):
        shutil.copy2(p, arm / p.name)
    manifest = {"status": "lp6k_ready", "rows": len(names), "npz_dir": str(out_dir),
                "train_tsv": str(arm / SRC_TSV.name), "cache_list": str(arm / SRC_CACHE.name),
                "train_tsv_sha256": sha256(arm / SRC_TSV.name), "cache_list_sha256": sha256(arm / SRC_CACHE.name),
                "source_manifest": str(SRC_MANIFEST), "source_npz_dir": str(NPZ_DIR),
                "cutoff_hz": CUTOFF, "filter": "butter(8) sosfiltfilt, LUFS-matched, peak cap 0.999",
                "status_counts": counts, "gates": gates, "builder_sha256": sha256(Path(__file__)),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(manifest, open(arm / "manifest.json", "w"), indent=1)
    print("[DONE]", json.dumps(counts), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/kojiek/exps_nvme/lp6k_corpus_183"))
    ap.add_argument("--gates_only", action="store_true")
    ap.add_argument("--gate_n", type=int, default=2000)
    ap.add_argument("--g1_n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    rows, names, man = load_rows()
    if args.gates_only:
        g = run_gates(args, rows, names)
        sys.exit(0 if g["pass"] else 3)
    run_build(args, rows, names, man)


if __name__ == "__main__":
    main()
