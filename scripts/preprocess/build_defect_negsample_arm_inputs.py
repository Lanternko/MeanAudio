#!/usr/bin/env python
"""075 D2: build the defect negative-sample arm inputs.

Adds N programmatically degraded copies of existing training clips to the
slot0clean_nmv2matched corpus (251,596 rows, the 066 control's exact inputs).
Two arms share the SAME extra audio and differ only in the extra rows' captions:

  defectlab    "<defect sentence> <original caption>"   (degradation is named)
  defectunlab  "<original caption>"                     (degradation is unnamed)

Clean rows are untouched in both arms (same NPZ, same slot0clean overlay slot 0).

Audio path replicates training/extract_audio_latents.py exactly: the whole 30 s
wav is peak-normalised to 0.95, the first 160,000 samples (10 s @ 16 kHz) are
cut, mel_converter('16k') -> v1-16 VAE encode -> mean/std (312, 20). Each
degraded clip is renormalised to its clean window's integrated LUFS (peak
guarded at 0.999), so the taught direction is timbral, not level (063/065).

Gate: re-encoding 32 clean windows through this path must correlate >= 0.9999
(median) with the official NPZ means, or nothing is written.

Outputs under --root (NVMe):
  latents/dgr_XXXXX.npz          mean, std, clip_id, degradation, severity
  overlay_new_{lab,unlab}/...    (1, 77, 1024) stacked overlays for the extra rows
  npz_farm/                      symlinks: official NPZ + latents/
  overlay_farm_{lab,unlab}/      symlinks: slot0clean overlay + overlay_new_*/
  arm_{lab,unlab}/train.tsv, cache_train.txt, manifest.json
  degradations.tsv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt
from tqdm import tqdm

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SRC_INPUTS = Path("/home/kojiek/exps_nvme/slot0clean_nmv2matched/arm_inputs")
SRC_TSV = SRC_INPUTS / "phase8_caption2p0_slot0clean_nmv2matched_train.tsv"
SRC_CACHE = SRC_INPUTS / "cache_train.txt"
SRC_MANIFEST = SRC_INPUTS / "manifest.json"
NPZ_DIR = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz")
OVERLAY_DIR = Path("/home/kojiek/text_overlays/slot0clean")
WAV_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio")
ENCODER_SOURCE = Path("/home/kojiek/research/meanaudio_training/caption10s_pipeline/reextract_text_inplace_caption10s.py")
ENCODER_SOURCE_SHA256 = "eb692393994a414b5578e6ab4e5c46c8aa7e66f2a09e39f2061bfe83768374dc"
SR = 16_000
NUM_SAMPLES = 160_000

TYPES = ["noise", "clip", "lowpass", "bitcrush", "crackle"]
# Deliberately NOT the D1 probe strings (those stay held out for the manipulation check).
TEMPLATES = {
    "noise": [
        "A noisy recording buried in hiss and static.",
        "Poor quality audio with heavy background noise.",
        "The music is drowned in broadband hiss.",
        "Low fidelity recording with constant white noise.",
    ],
    "clip": [
        "A badly clipped recording with harsh distortion.",
        "Overdriven audio with crunchy digital clipping.",
        "The recording is distorted and clipping throughout.",
        "Low quality audio with harsh clipping artifacts.",
    ],
    "lowpass": [
        "A muffled recording that lacks high frequencies.",
        "Dull, muffled audio as if heard through a wall.",
        "Low fidelity recording with the treble cut off.",
        "The sound is muffled and dark with no clarity.",
    ],
    "bitcrush": [
        "A lo-fi, bit-crushed recording with grainy digital artifacts.",
        "Low quality audio with harsh quantization noise.",
        "Grainy low-bit digital audio with poor fidelity.",
        "The recording sounds crushed and gritty, like a cheap digital file.",
    ],
    "crackle": [
        "A recording full of clicks and crackles.",
        "Poor quality audio with constant popping and crackling.",
        "The music is covered in crackle and pops like a damaged record.",
        "Low fidelity recording with loud clicks and pops.",
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_window(name: str) -> np.ndarray:
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


def degrade(kind: str, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    peak = np.abs(x).max()
    if kind == "noise":        # white noise at SNR U[0, 15] dB
        snr = rng.uniform(0, 15)
        n = rng.standard_normal(x.shape)
        return x + n * np.sqrt(np.mean(x**2) / (np.mean(n**2) * 10 ** (snr / 10))), snr
    if kind == "clip":         # hard clip at the original peak after U[9, 21] dB drive
        drive = rng.uniform(9, 21)
        return np.clip(x * 10 ** (drive / 20), -peak, peak), drive
    if kind == "lowpass":      # 8th-order Butterworth, cutoff U[1, 3] kHz
        fc = rng.uniform(1000, 3000)
        return sosfilt(butter(8, fc / (SR / 2), btype="low", output="sos"), x), fc
    if kind == "bitcrush":     # 4-6 bit quantisation + sample-and-hold to SR/U{2..4}
        bits = int(rng.integers(4, 7))
        hold = int(rng.integers(2, 5))
        step = 2 * peak / (2**bits)
        y = np.round(x / step) * step
        y = np.repeat(y[::hold], hold)[: x.shape[0]]
        return y, bits + hold / 10
    if kind == "crackle":      # sparse bipolar impulses, 20-80 /s, 0.3-0.9 x peak
        rate = rng.uniform(20, 80)
        k = rng.poisson(rate * x.shape[0] / SR)
        pos = rng.integers(0, x.shape[0] - 8, size=k)
        amp = rng.uniform(0.3, 0.9, size=k) * peak * rng.choice([-1, 1], size=k)
        y = x.copy()
        kern = np.array([1.0, -0.6, 0.3, -0.1])
        for p, a in zip(pos, amp):
            y[p:p + 4] += a * kern
        return y, rate
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/kojiek/exps_nvme/defect_negsample"))
    ap.add_argument("--n", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    root = args.root
    for d in ["latents", "overlay_new_lab", "overlay_new_unlab", "npz_farm",
              "overlay_farm_lab", "overlay_farm_unlab", "arm_lab", "arm_unlab"]:
        (root / d).mkdir(parents=True, exist_ok=True)

    src_manifest = json.load(open(SRC_MANIFEST))
    assert sha256(SRC_TSV) == src_manifest["train_tsv_sha256"], "source tsv drift"
    assert sha256(SRC_CACHE) == src_manifest["cache_list_sha256"], "source cache drift"
    header = open(SRC_TSV, encoding="utf-8").readline().rstrip("\n").split("\t")
    rows = list(csv.DictReader(open(SRC_TSV, encoding="utf-8", newline=""), delimiter="\t"))
    names = [l.strip() for l in open(SRC_CACHE) if l.strip()]
    assert len(rows) == len(names) == src_manifest["rows"]

    # ---- selection: segment_0 excluded (fade-in silence, reference_training_corpus_silence_rate)
    rng = np.random.default_rng(args.seed)
    pool = [i for i, r in enumerate(rows) if "_segment_0_" not in r["id"]]
    pick = sorted(rng.choice(len(pool), size=args.n + 500, replace=False))  # spare for silent rejects
    pick = [pool[i] for i in pick]
    kinds = rng.permutation(np.repeat(np.arange(len(TYPES)), (args.n + 500) // len(TYPES) + 1))

    from meanaudio.ext.autoencoder import AutoEncoderModule
    from meanaudio.ext.mel_converter import get_mel_converter
    torch.backends.cuda.matmul.allow_tf32 = True   # as extract_audio_latents.py
    torch.backends.cudnn.allow_tf32 = True
    tod = AutoEncoderModule(vae_ckpt_path=str(ROOT / "weights/v1-16.pth"),
                            vocoder_ckpt_path=str(ROOT / "weights/best_netG.pt"), mode="16k").eval().cuda()
    mel = get_mel_converter("16k").eval().cuda()

    def encode(batch: list[np.ndarray]):
        with torch.no_grad():
            d = tod.encode(mel(torch.from_numpy(np.stack(batch)).float().cuda()))
        return d.mean.cpu().transpose(1, 2).numpy(), d.std.cpu().transpose(1, 2).numpy()

    # ---- gate: this path reproduces the official latents on clean windows
    gate_idx = [i for i in pick[:64] if "_segment_0_" not in rows[i]["id"]][:32]
    wins = [load_window(rows[i]["id"].rsplit("_", 1)[0]) for i in gate_idx]
    m, _ = encode([w for w in wins])
    corrs = [float(np.corrcoef(m[k].ravel(), np.load(NPZ_DIR / names[i])["mean"].ravel())[0, 1])
             for k, i in enumerate(gate_idx)]
    gate = {"n": len(corrs), "median_corr": float(np.median(corrs)), "min_corr": float(min(corrs))}
    print("encode gate:", gate)
    assert gate["median_corr"] >= 0.9999, f"[FAIL] encode path does not reproduce official latents {gate}"

    # ---- degrade + encode
    meter = pyln.Meter(SR)
    deg_rows = []   # (new_id, src_index, kind, severity, template, src_lufs, pre_lufs)
    buf, meta = [], []
    tmpl_rng = np.random.default_rng(args.seed + 1)

    def flush():
        if not buf:
            return
        mm, ss = encode(buf)
        for k, (nid, extra) in enumerate(meta):
            np.savez(root / "latents" / f"{extra['file']}", mean=mm[k].astype(np.float32),
                     std=ss[k].astype(np.float32), clip_id=np.asarray(nid),
                     degradation=np.asarray(extra["kind"]), severity=np.asarray(extra["severity"]))
        buf.clear(); meta.clear()

    for j, (i, kind_i) in enumerate(tqdm(list(zip(pick, kinds)), desc="degrade+encode")):
        if len(deg_rows) >= args.n:
            break
        r = rows[i]
        x = load_window(r["id"].rsplit("_", 1)[0])
        if x is None:
            continue
        src_lufs = meter.integrated_loudness(x)
        if not np.isfinite(src_lufs) or src_lufs < -45:
            continue
        kind = TYPES[int(kind_i)]
        y, sev = degrade(kind, x, rng)
        pre = meter.integrated_loudness(y)
        y = y * 10 ** ((src_lufs - pre) / 20)
        pk = np.abs(y).max()
        if pk > 0.999:
            y = y * (0.999 / pk)
        nid = f"{r['id']}__dgr_{kind}"
        fname = f"dgr_{len(deg_rows):05d}.npz"
        tmpl = TEMPLATES[kind][int(tmpl_rng.integers(len(TEMPLATES[kind])))]
        deg_rows.append(dict(new_id=nid, src_index=i, src_id=r["id"], kind=kind, severity=f"{sev:.4f}",
                             template=tmpl, src_lufs=f"{src_lufs:.3f}", pre_match_lufs=f"{pre:.3f}",
                             file=fname))
        buf.append(y.astype(np.float32)); meta.append((nid, {"file": fname, "kind": kind, "severity": sev}))
        if len(buf) >= args.batch:
            flush()
    flush()
    assert len(deg_rows) == args.n, f"only {len(deg_rows)} degraded rows"
    with open(root / "degradations.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(deg_rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(deg_rows)

    # ---- captions / TSVs / cache lists
    arms = {}
    for arm in ["lab", "unlab"]:
        extra = []
        for d in deg_rows:
            src = dict(rows[d["src_index"]])
            src["id"] = d["new_id"]
            if arm == "lab":
                src["caption"] = f"{d['template']} {src['caption']}"
            extra.append(src)
        arm_rows = rows + extra
        arm_names = names + [d["file"] for d in deg_rows]
        tsv = root / f"arm_{arm}" / "train.tsv"
        with open(tsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, delimiter="\t", lineterminator="\n")
            w.writeheader(); w.writerows(arm_rows)
        (root / f"arm_{arm}" / "cache_train.txt").write_text("\n".join(arm_names) + "\n")
        arms[arm] = (arm_rows, arm_names, extra)

    # ---- overlays for the extra rows
    spec = importlib.util.spec_from_file_location("bound_text_encoder", ENCODER_SOURCE)
    assert sha256(ENCODER_SOURCE) == ENCODER_SOURCE_SHA256, "encoder source drift"
    enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
    fp = enc.encoder_fingerprint()
    ref_fp = str(np.load(OVERLAY_DIR / names[0])["text_encoder_fingerprint"].item())
    assert fp == ref_fp, f"[FAIL] encoder fingerprint {fp} != slot0clean overlay {ref_fp}"

    # unlab: slot 0 of the source row's slot0clean overlay, re-keyed to the new id
    for d, row in zip(deg_rows, arms["unlab"][2]):
        z = np.load(OVERLAY_DIR / names[d["src_index"]])
        assert str(z["clip_id"].item()) == d["src_id"]
        stored = str(z["caption_sha256"].item()).split(",")
        assert stored[0] == enc.sha_caption(row["caption"]), f"slot0 mismatch {d['src_id']}"
        np.savez(root / "overlay_new_unlab" / d["file"], clip_id=np.asarray(d["new_id"]),
                 text_features=z["text_features"][:1], text_features_c=z["text_features_c"][:1],
                 text_attention_mask=z["text_attention_mask"][:1],
                 caption_sha256=np.asarray(stored[0]), text_encoder_fingerprint=np.asarray(fp))

    # lab: encode the new captions with the bound encoder
    dev = torch.device("cuda")
    tok = enc.AutoTokenizer.from_pretrained(enc.T5_MODEL, revision=enc.T5_REVISION, local_files_only=True)
    t5 = enc.T5EncoderModel.from_pretrained(enc.T5_MODEL, revision=enc.T5_REVISION,
                                            local_files_only=True).eval().to(dev)
    clap = enc.laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap.load_ckpt(str(enc.CLAP_CKPT), verbose=False)
    lab_extra = arms["lab"][2]
    for o in tqdm(range(0, len(deg_rows), 48), desc="lab overlay"):
        texts = [r["caption"] for r in lab_extra[o:o + 48]]
        feats, masks = enc.encode_t5(tok, t5, texts, dev)
        pooled = enc.encode_clap(clap, texts)
        for k, d in enumerate(deg_rows[o:o + 48]):
            np.savez(root / "overlay_new_lab" / d["file"], clip_id=np.asarray(d["new_id"]),
                     text_features=feats[k][None].astype(np.float32),
                     text_features_c=pooled[k][None].astype(np.float32),
                     text_attention_mask=masks[k][None].astype(np.int64),
                     caption_sha256=np.asarray(enc.sha_caption(texts[k])),
                     text_encoder_fingerprint=np.asarray(fp))
    # the prefix must survive T5's 77-token window, or the label never reaches the model
    tok_len = [len(tok(d["template"]).input_ids) for d in deg_rows[:2000]]
    assert max(tok_len) < 40, f"template too long: {max(tok_len)} tokens"

    # ---- symlink farms (the loader reads npz_dir/<name> and text_npz_dir/<name>)
    def farm(dst: Path, base: Path, new: Path):
        for n in names:
            p = dst / n
            if not p.is_symlink():
                p.symlink_to(base / n)
        for d in deg_rows:
            p = dst / d["file"]
            if not p.is_symlink():
                p.symlink_to(new / d["file"])
    farm(root / "npz_farm", NPZ_DIR, root / "latents")
    farm(root / "overlay_farm_lab", OVERLAY_DIR, root / "overlay_new_lab")
    farm(root / "overlay_farm_unlab", OVERLAY_DIR, root / "overlay_new_unlab")

    # ---- manifests
    for arm in ["lab", "unlab"]:
        d = root / f"arm_{arm}"
        json.dump({
            "status": "arm_inputs_ready", "experiment": "075_d2_defect_negsample", "arm": f"defect{arm}",
            "rows": len(arms[arm][0]), "clean_rows": len(rows), "extra_rows": len(deg_rows),
            "train_tsv": str(d / "train.tsv"), "train_tsv_sha256": sha256(d / "train.tsv"),
            "cache_list": str(d / "cache_train.txt"), "cache_list_sha256": sha256(d / "cache_train.txt"),
            "npz_dir": str(root / "npz_farm"), "text_npz_dir": str(root / f"overlay_farm_{arm}"),
            "source_manifest": str(SRC_MANIFEST), "source_train_tsv_sha256": src_manifest["train_tsv_sha256"],
            "degradations_tsv_sha256": sha256(root / "degradations.tsv"),
            "encode_gate": gate, "text_encoder_fingerprint": fp, "seed": args.seed,
            "build_script_sha256": sha256(Path(__file__)),
            "kind_counts": {k: sum(1 for r in deg_rows if r["kind"] == k) for k in TYPES},
        }, open(d / "manifest.json", "w"), indent=1)
    print("done", {k: sum(1 for r in deg_rows if r["kind"] == k) for k in TYPES})


if __name__ == "__main__":
    main()
