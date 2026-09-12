#!/usr/bin/env python3
"""PE-AV cosine(caption, audio) for training rows -- the alignment score S of the Beta timestep probe.

Deliberately not LAION-CLAP: CLAP is the primary eval metric, so it must not steer training
(professor 2026-03-27 leakage rule). Audio = first 10 s of the segment mp3 at 48 kHz, the
window the c2p0 captioner saw. Resumable: ids already in --out are skipped.

Run inside ~/venvs/peav.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import PeAudioVideoModel, PeAudioVideoProcessor

csv.field_size_limit(10**9)
SR = 48000
WINDOW_SEC = 10.0
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    return AUDIO_ROOT / "_".join(parts[: seg_idx - 1]) / parts[seg_idx - 1] / f"segment_{parts[seg_idx + 1]}.mp3"


def load_crop(clip_id: str) -> np.ndarray:
    wav, _ = librosa.load(str(id_to_audio_path(clip_id)), sr=SR, mono=True, duration=WINDOW_SEC)
    n = int(SR * WINDOW_SEC)
    wav = np.asarray(wav, dtype=np.float32)
    return np.pad(wav, (0, n - len(wav))) if len(wav) < n else wav[:n]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="facebook/pe-av-large")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    rows = []
    for tsv in args.tsv:
        with open(tsv, newline="", encoding="utf-8") as f:
            rows += [(r["id"], r["caption"]) for r in csv.DictReader(f, delimiter="\t")]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open(encoding="utf-8") if line.strip()}
    todo = [r for r in rows if r[0] not in done]
    print(f"[peav_score] rows={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    device = torch.device("cuda")
    model = PeAudioVideoModel.from_pretrained(args.model_id).to(device).eval()
    proc = PeAudioVideoProcessor.from_pretrained(args.model_id)
    with out.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            inp = proc(audio=[load_crop(cid) for cid, _ in batch], sampling_rate=SR,
                       text=[cap for _, cap in batch], return_tensors="pt", padding=True)
            res = model(**{k: v.to(device) for k, v in inp.items()})
            a = torch.nn.functional.normalize(res.audio_embeds.float(), dim=-1)
            t = torch.nn.functional.normalize(res.text_audio_embeds.float(), dim=-1)
            cos = (a * t).sum(-1).cpu().tolist()
            for (cid, _), c in zip(batch, cos):
                if not np.isfinite(c):
                    raise SystemExit(f"[FAIL] non-finite score for {cid}")
                fh.write(json.dumps({"id": cid, "peav_cos": c}) + "\n")
            fh.flush()
            if (i // args.batch_size) % 25 == 0:
                print(f"[peav_score] {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
    print("[peav_score] done", flush=True)


if __name__ == "__main__":
    main()
