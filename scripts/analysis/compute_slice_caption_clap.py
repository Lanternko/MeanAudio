#!/usr/bin/env python3
"""Compute CLAP audio-text scores for slice-level caption review TSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_CLAP_CKPT = Path("/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt")
DEFAULT_TSV = Path(
    "/home/kojiek/eval_output/music_flamingo_slice10_10k/lpmc_review/"
    "sample20_tracks_seed20260522_slice10_flamingo_lpmc.tsv"
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def encode_audio(model, paths: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for i in tqdm(range(0, len(paths), batch_size), desc="audio-emb"):
        batch = paths[i : i + batch_size]
        with torch.no_grad():
            emb = model.get_audio_embedding_from_filelist(batch, use_tensor=True)
            chunks.append(normalize(emb).detach().cpu())
    return torch.cat(chunks, dim=0)


def encode_text(model, captions: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for i in tqdm(range(0, len(captions), batch_size), desc="text-emb"):
        batch = captions[i : i + batch_size]
        with torch.no_grad():
            emb = model.get_text_embedding(batch, use_tensor=True)
            chunks.append(normalize(emb).detach().cpu())
    return torch.cat(chunks, dim=0)


def source_metrics(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> dict:
    sim = (audio_emb @ text_emb.T).numpy()
    n = sim.shape[0]
    diag = np.diag(sim)
    off_diag = sim[~np.eye(n, dtype=bool)] if n > 1 else np.array([], dtype=np.float32)
    ranks = np.array([1 + int((sim[i] > sim[i, i]).sum()) for i in range(n)])
    return {
        "n": int(n),
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_median": float(np.median(diag)),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
        "shuffled_mean": float(off_diag.mean()) if len(off_diag) else None,
        "diag_minus_shuffled": float(diag.mean() - off_diag.mean()) if len(off_diag) else None,
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "R@1": float((ranks <= 1).mean()),
        "R@5": float((ranks <= 5).mean()),
        "R@10": float((ranks <= 10).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--clap-ckpt", type=Path, default=DEFAULT_CLAP_CKPT)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rows = read_rows(args.review_tsv)
    if not rows:
        raise SystemExit(f"No rows found: {args.review_tsv}")

    out_dir = args.out_dir or args.review_tsv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = [row["review_audio_path"] or row["source_audio_path"] for row in rows]
    flamingo_caps = [row["music_flamingo_slice10_caption"] for row in rows]
    lpmc_caps = [row["lpmc_caption"] for row in rows]

    import laion_clap

    print("Loading CLAP...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(args.clap_ckpt))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"device={device}")

    audio_emb = encode_audio(model, audio_paths, args.batch_size)
    flamingo_emb = encode_text(model, flamingo_caps, args.batch_size)
    lpmc_emb = encode_text(model, lpmc_caps, args.batch_size)

    flamingo_matrix = (audio_emb @ flamingo_emb.T).numpy()
    lpmc_matrix = (audio_emb @ lpmc_emb.T).numpy()
    flamingo_diag = np.diag(flamingo_matrix)
    lpmc_diag = np.diag(lpmc_matrix)

    detail_path = out_dir / f"{args.review_tsv.stem}_clap.tsv"
    with detail_path.open("w", newline="") as f:
        fieldnames = [
            "id",
            "review_audio_path",
            "clap_audio_flamingo",
            "clap_audio_lpmc",
            "delta_flamingo_minus_lpmc",
            "flamingo_rank",
            "lpmc_rank",
            "music_flamingo_slice10_caption",
            "lpmc_caption",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for i, row in enumerate(rows):
            flamingo_rank = 1 + int((flamingo_matrix[i] > flamingo_matrix[i, i]).sum())
            lpmc_rank = 1 + int((lpmc_matrix[i] > lpmc_matrix[i, i]).sum())
            writer.writerow(
                {
                    "id": row["id"],
                    "review_audio_path": audio_paths[i],
                    "clap_audio_flamingo": f"{flamingo_diag[i]:.6f}",
                    "clap_audio_lpmc": f"{lpmc_diag[i]:.6f}",
                    "delta_flamingo_minus_lpmc": f"{(flamingo_diag[i] - lpmc_diag[i]):.6f}",
                    "flamingo_rank": flamingo_rank,
                    "lpmc_rank": lpmc_rank,
                    "music_flamingo_slice10_caption": row["music_flamingo_slice10_caption"],
                    "lpmc_caption": row["lpmc_caption"],
                }
            )

    summary = {
        "review_tsv": str(args.review_tsv),
        "clap_ckpt": str(args.clap_ckpt),
        "definition": "Cosine similarity between normalized LAION-CLAP audio and text embeddings.",
        "audio": "review_audio_path: exported first-10s slice used by MeanAudio training",
        "compared_texts": {
            "music_flamingo": "music_flamingo_slice10_caption for the same slice id",
            "lpmc": "LP-MC caption for the same segment id",
            "shuffled_baseline": "same audio compared with other rows' captions in this 20-pair set",
        },
        "music_flamingo": source_metrics(audio_emb, flamingo_emb),
        "lpmc": source_metrics(audio_emb, lpmc_emb),
        "paired_delta_flamingo_minus_lpmc": {
            "mean": float((flamingo_diag - lpmc_diag).mean()),
            "median": float(np.median(flamingo_diag - lpmc_diag)),
            "flamingo_higher_count": int((flamingo_diag > lpmc_diag).sum()),
            "lpmc_higher_count": int((lpmc_diag > flamingo_diag).sum()),
            "tie_count": int((flamingo_diag == lpmc_diag).sum()),
        },
    }

    summary_path = out_dir / f"{args.review_tsv.stem}_clap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    print(
        "music_flamingo "
        f"diag={summary['music_flamingo']['diag_mean']:.4f} "
        f"gap={summary['music_flamingo']['diag_minus_shuffled']:.4f} "
        f"R@1={summary['music_flamingo']['R@1']:.2%}"
    )
    print(
        "lpmc            "
        f"diag={summary['lpmc']['diag_mean']:.4f} "
        f"gap={summary['lpmc']['diag_minus_shuffled']:.4f} "
        f"R@1={summary['lpmc']['R@1']:.2%}"
    )
    print(
        "delta           "
        f"mean={summary['paired_delta_flamingo_minus_lpmc']['mean']:.4f} "
        f"flamingo_higher={summary['paired_delta_flamingo_minus_lpmc']['flamingo_higher_count']}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
