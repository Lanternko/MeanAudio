#!/usr/bin/env python
"""089: PE-AV caption<->audio cosine for every ranked row of the slot0clean_nmv2matched corpus.

Runs in ~/venvs/peav (transformers PeAudioVideoModel; that venv has no pyloudnorm, so the
075 builder cannot be imported here). The training window is re-implemented in
`load_window` below -- a line-for-line copy of build_defect_negsample_arm_inputs.load_window
(30 s wav, mono mean, peak-normalise to 0.95, first 160,000 samples @ 16 kHz). The 089
builder (dac venv) imports this module and asserts array equality against the 075 function
before it trusts the scores (`window_gate`).

Score per row = cos(normalize(audio_embeds), normalize(text_audio_embeds)) of facebook/pe-av-large,
audio = the training window resampled 16 k -> 48 k (librosa, as research/eval/peav_eval.py),
text  = the row's exact training caption (the plain slot0clean caption, no prefix).

Rows scored = the rows 081 ranked (tier != unranked in quality_label_081/tiers.tsv), so the
silence exclusion is identical to 081. Reads wavs in os.scandir order (flat exFAT dir:
random opens ~2 s, scandir order ~8 ms). Appends to <root>/peav_scores.partial.tsv and
renames to peav_scores.tsv when complete; rerunning resumes.

Gate (written to <root>/peav_gate.json before the full run): the same 16 rows scored in a
batch of 8 and one at a time must agree to |d| <= 2e-3 (CLAP taught us batch padding can
move scores; the audio windows are all the same length, only the text is padded).

Usage: ~/venvs/peav/bin/python score_peav_corpus_089.py --root DIR [--limit N] [--batch_size 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

csv.field_size_limit(10**9)
WAV_DIR = Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio")
SR, NUM_SAMPLES, TARGET_SR = 16_000, 160_000, 48_000
SRC_TSV = Path("/home/kojiek/exps_nvme/slot0clean_nmv2matched/arm_inputs/"
               "phase8_caption2p0_slot0clean_nmv2matched_train.tsv")
TIERS_081 = Path("/home/kojiek/exps_nvme/quality_label_081/tiers.tsv")
MODEL_ID = "facebook/pe-av-large"
FIELDS = ["stem", "status", "peav_cos"]


def load_window(name: str):
    """Copy of build_defect_negsample_arm_inputs.load_window (equality asserted by the 089 builder)."""
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


def to_48k(win: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.resample(win.astype(np.float32), orig_sr=SR, target_sr=TARGET_SR)


def ranked_rows(limit: int | None):
    """stem -> caption for the rows 081 ranked, in source order."""
    tier = {r["id"]: r["tier"] for r in csv.DictReader(open(TIERS_081, encoding="utf-8"), delimiter="\t")}
    out = {}
    for i, r in enumerate(csv.DictReader(open(SRC_TSV, encoding="utf-8", newline=""), delimiter="\t")):
        if limit and i >= limit:
            break
        if tier[r["id"]] != "unranked":
            out[r["id"].rsplit("_", 1)[0]] = r["caption"]
    return out


class Scorer:
    def __init__(self):
        import torch
        from transformers import PeAudioVideoModel, PeAudioVideoProcessor
        self.torch = torch
        self.dev = torch.device("cuda")
        self.model = PeAudioVideoModel.from_pretrained(MODEL_ID).to(self.dev).eval()
        self.proc = PeAudioVideoProcessor.from_pretrained(MODEL_ID)

    def __call__(self, wavs, caps):
        torch = self.torch
        inp = self.proc(audio=wavs, sampling_rate=TARGET_SR, text=caps, return_tensors="pt", padding=True)
        inp = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp)
        a = torch.nn.functional.normalize(out.audio_embeds.float(), dim=-1)
        t = torch.nn.functional.normalize(out.text_audio_embeds.float(), dim=-1)
        return (a * t).sum(-1).cpu().numpy().tolist()


def gate(scorer, caps, root: Path, bs: int):
    stems = sorted(caps)[:16]
    wavs = [to_48k(load_window(s)) for s in stems]
    batched = []
    for o in range(0, len(stems), bs):
        batched += scorer(wavs[o:o + bs], [caps[s] for s in stems[o:o + bs]])
    single = [scorer([w], [caps[s]])[0] for w, s in zip(wavs, stems)]
    d = float(max(abs(a - b) for a, b in zip(batched, single)))
    g = {"n": len(stems), "batch_size": bs, "max_abs_diff_batch_vs_single": d,
         "mean": float(np.mean(single)), "model": MODEL_ID}
    print("[peav] batch gate:", g, flush=True)
    assert d <= 2e-3, f"[FAIL] PE-AV batch gate {g}"
    (root / "peav_gate.json").write_text(json.dumps(g, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    caps = ranked_rows(a.limit)
    final, part = a.root / "peav_scores.tsv", a.root / "peav_scores.partial.tsv"
    if final.exists():
        done = {r["stem"] for r in csv.DictReader(open(final), delimiter="\t")}
        assert set(caps) <= done, f"peav_scores.tsv misses {len(set(caps) - done)} stems"
        print(f"[peav] peav_scores.tsv complete ({len(done)} stems)", flush=True)
        return
    done = {r["stem"] for r in csv.DictReader(open(part), delimiter="\t")} if part.exists() else set()
    todo = set(caps) - done
    print(f"[peav] need {len(caps)} stems, {len(done)} done, {len(todo)} to go", flush=True)
    scorer = Scorer()
    if not (a.root / "peav_gate.json").exists():
        gate(scorer, caps, a.root, a.batch_size)

    q: queue.Queue = queue.Queue(maxsize=128)

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
                        w = load_window(s)
                        q.put((s, None if w is None else to_48k(w), None if w is not None else "silent_peak"))
                    except Exception as ex:
                        q.put((s, None, repr(ex)))
        for s in sorted(left):
            q.put((s, None, "missing"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    new = not part.exists()
    f = open(part, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    if new:
        w.writeheader()
    buf, n, t0 = [], 0, time.time()

    def flush():
        nonlocal n
        if buf:
            res = scorer([x[1] for x in buf], [caps[x[0]] for x in buf])
            for (s, _), v in zip(buf, res):
                w.writerow({"stem": s, "status": "ok", "peav_cos": f"{v:.6f}"})
            n += len(buf)
            buf.clear()
            f.flush()

    while True:
        item = q.get()
        if item is None:
            break
        s, wav, err = item
        if wav is None:
            w.writerow({"stem": s, "status": err, "peav_cos": ""})
            continue
        buf.append((s, wav))
        if len(buf) == a.batch_size:
            flush()
            if n % 4000 == 0:
                rate = n / (time.time() - t0)
                print(f"[peav] {n}/{len(todo)} scored, {rate:.1f}/s, eta {(len(todo) - n) / rate / 3600:.2f} h",
                      flush=True)
    flush()
    f.close()
    got = {r["stem"] for r in csv.DictReader(open(part), delimiter="\t")}
    assert set(caps) <= got, f"score incomplete: {len(set(caps) - got)} stems missing"
    os.replace(part, final)
    print(f"[peav] done: {final}", flush=True)


if __name__ == "__main__":
    main()
