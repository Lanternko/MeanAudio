"""
PE-AV evaluation for MeanAudio generations.

Computes per-pair cosine similarity (analogous to CLAP score) and
retrieval metrics (R@1/5/10) using facebook/pe-av-large.

Usage:
    python peav_eval.py \
        --gen_dir /path/to/audio \
        --tsv     /path/to/musiccaps_test.tsv \
        --out     /path/to/peav_metrics.json \
        [--batch_size 8] [--max_samples N]

Requires transformers>=5.6.0, torch (cu128 nightly for Blackwell).
Run inside ~/venvs/peav.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from transformers import PeAudioVideoModel, PeAudioVideoProcessor

TARGET_SR = 48000


def load_and_resample(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


def read_tsv(tsv_path: Path):
    rows = []
    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for r in reader:
            if len(r) < 2:
                continue
            rows.append((r[0], r[1]))
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="facebook/pe-av-large")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--ext", default="flac")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    rows = read_tsv(Path(args.tsv))

    pairs = []
    for id_, cap in rows:
        p = gen_dir / f"{id_}.{args.ext}"
        if p.exists():
            pairs.append((id_, cap, p))
    print(f"[peav_eval] TSV rows: {len(rows)}, audio found: {len(pairs)}")
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
        print(f"[peav_eval] capped to {len(pairs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[peav_eval] device: {device}")
    print(f"[peav_eval] loading model: {args.model_id}")
    model = PeAudioVideoModel.from_pretrained(args.model_id).to(device).eval()
    proc = PeAudioVideoProcessor.from_pretrained(args.model_id)

    audio_embs, text_embs = [], []
    ids_out = []

    bs = args.batch_size
    t0 = time.time()
    for i in tqdm(range(0, len(pairs), bs), desc="peav"):
        batch = pairs[i : i + bs]
        wavs = [load_and_resample(p) for _, _, p in batch]
        caps = [c for _, c, _ in batch]
        ids = [i_ for i_, _, _ in batch]

        inp = proc(
            audio=wavs,
            sampling_rate=TARGET_SR,
            text=caps,
            return_tensors="pt",
            padding=True,
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp)

        a = torch.nn.functional.normalize(out.audio_embeds, dim=-1).cpu()
        t = torch.nn.functional.normalize(out.text_audio_embeds, dim=-1).cpu()
        audio_embs.append(a)
        text_embs.append(t)
        ids_out.extend(ids)

    audio_embs = torch.cat(audio_embs, dim=0)  # (N, D)
    text_embs = torch.cat(text_embs, dim=0)
    N = audio_embs.shape[0]
    dt = time.time() - t0
    print(f"[peav_eval] embedded {N} pairs in {dt/60:.1f} min ({dt/N:.2f} s/pair)")

    # Per-pair cosine similarity (PE-AV score, analogous to CLAP score)
    per_pair_sim = (audio_embs * text_embs).sum(-1)  # (N,)
    peav_score = per_pair_sim.mean().item()
    peav_std = per_pair_sim.std().item()

    # Full retrieval matrix for R@K (text-to-audio)
    # sim[i, j] = sim(text_i, audio_j); diagonal = ground-truth pair
    sim_mat = text_embs @ audio_embs.T  # (N, N)
    # Rank of ground-truth audio for each text query
    gt_sim = sim_mat.diag()  # (N,)
    # Count how many distractors score >= gt for each query
    ranks = (sim_mat > gt_sim.unsqueeze(1)).sum(dim=1) + 1
    # Avg metrics
    r1 = (ranks <= 1).float().mean().item() * 100
    r5 = (ranks <= 5).float().mean().item() * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    median_rank = float(ranks.median().item())
    mean_rank = float(ranks.float().mean().item())

    # Also report audio-to-text (reverse direction)
    sim_mat_at = audio_embs @ text_embs.T
    gt_sim_at = sim_mat_at.diag()
    ranks_at = (sim_mat_at > gt_sim_at.unsqueeze(1)).sum(dim=1) + 1
    a2t_r1 = (ranks_at <= 1).float().mean().item() * 100
    a2t_r5 = (ranks_at <= 5).float().mean().item() * 100
    a2t_r10 = (ranks_at <= 10).float().mean().item() * 100

    results = {
        "model_id": args.model_id,
        "gen_dir": str(gen_dir),
        "tsv": str(args.tsv),
        "n_pairs": N,
        "elapsed_minutes": round(dt / 60, 2),
        "peav_score_mean": round(peav_score, 6),
        "peav_score_std": round(peav_std, 6),
        "t2a_R@1": round(r1, 3),
        "t2a_R@5": round(r5, 3),
        "t2a_R@10": round(r10, 3),
        "t2a_median_rank": median_rank,
        "t2a_mean_rank": mean_rank,
        "a2t_R@1": round(a2t_r1, 3),
        "a2t_R@5": round(a2t_r5, 3),
        "a2t_R@10": round(a2t_r10, 3),
    }
    print(json.dumps(results, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[peav_eval] saved → {args.out}")


if __name__ == "__main__":
    main()
