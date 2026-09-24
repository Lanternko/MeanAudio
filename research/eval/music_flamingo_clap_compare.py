"""
Compare CLAP alignment for Music Flamingo full-track captions against
Qwen and LP-MC captions on the same Jamendo full-track audio set.

Music Flamingo produces one caption per full track. Existing Qwen/LP-MC
captions are segment-level, so this script encodes all segment captions
for a track with CLAP, averages the normalized text embeddings, and then
compares that track-level text embedding to the full-track audio embedding.

Outputs:
  - clap_compare.json: summary metrics
  - clap_compare_per_track.jsonl: per-track diagonal scores
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


CLAP_CKPT = Path.home() / "MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
DEFAULT_FLAMINGO = Path.home() / "eval_output/music_flamingo_1k/caption.jsonl"
DEFAULT_LPMC = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv")
DEFAULT_QWEN = Path("/mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_captions.jsonl")


def track_id_from_segment_id(segment_id: str) -> str:
    if "_segment_" not in segment_id:
        return segment_id
    return segment_id.split("_segment_", 1)[0]


def load_flamingo(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            text = (rec.get("output") or {}).get("text") or rec.get("raw_text")
            if not text:
                continue
            audio_path = rec.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                continue
            rows.append(
                {
                    "track_id": rec["track_id"],
                    "audio_path": audio_path,
                    "duration_sec": rec.get("duration_sec"),
                    "caption": text,
                }
            )
    return rows


def load_lpmc_by_track(path: Path, wanted: set[str]) -> dict[str, list[str]]:
    by_track: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("id", "")
            caption = (row.get("caption") or "").strip()
            tid = track_id_from_segment_id(sid)
            if tid not in wanted or not caption:
                continue
            seg = 0
            if "_segment_" in sid:
                try:
                    seg = int(sid.rsplit("_segment_", 1)[1])
                except ValueError:
                    seg = 0
            by_track[tid].append((seg, caption))
    return {tid: [cap for _, cap in sorted(items)] for tid, items in by_track.items()}


def load_qwen_by_track(path: Path, wanted: set[str], mode: str) -> dict[str, list[str]]:
    by_track: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open() as f:
        for line in tqdm(f, desc="load-qwen", unit="line"):
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec.get("id", ""))
            tid = track_id_from_segment_id(sid)
            if tid not in wanted:
                continue

            captions: list[str]
            if "captions" in rec and isinstance(rec["captions"], list):
                vals = [str(c).strip() for c in rec["captions"] if str(c).strip()]
                captions = vals[:1] if mode == "slot0" else vals
            else:
                cap = str(rec.get("caption", "")).strip()
                captions = [cap] if cap else []

            for cap in captions:
                by_track[tid].append((sid, cap))

    return {tid: [cap for _, cap in sorted(items)] for tid, items in by_track.items()}


def encode_audio(model, paths: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for i in tqdm(range(0, len(paths), batch_size), desc="audio-emb"):
        batch = paths[i : i + batch_size]
        with torch.no_grad():
            emb = model.get_audio_embedding_from_filelist(batch, use_tensor=True)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        chunks.append(emb.detach().cpu())
    return torch.cat(chunks, dim=0)


def encode_text_list(model, captions: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for i in tqdm(range(0, len(captions), batch_size), desc="text-emb"):
        batch = captions[i : i + batch_size]
        with torch.no_grad():
            emb = model.get_text_embedding(batch, use_tensor=True)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        chunks.append(emb.detach().cpu())
    return torch.cat(chunks, dim=0)


def encode_track_mean_text(
    model,
    track_ids: list[str],
    captions_by_track: dict[str, list[str]],
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    flat_caps: list[str] = []
    flat_track_ids: list[str] = []
    counts = {}

    for tid in track_ids:
        caps = captions_by_track.get(tid, [])
        counts[tid] = len(caps)
        for cap in caps:
            flat_track_ids.append(tid)
            flat_caps.append(cap)

    if not flat_caps:
        raise RuntimeError("No captions available for track-mean text embedding")

    flat_emb = encode_text_list(model, flat_caps, batch_size=batch_size)
    dim = flat_emb.shape[-1]
    sums = {tid: torch.zeros(dim) for tid in track_ids}

    for tid, emb in zip(flat_track_ids, flat_emb):
        sums[tid] += emb

    out = []
    for tid in track_ids:
        if counts[tid] == 0:
            out.append(torch.full((dim,), float("nan")))
        else:
            v = torch.nn.functional.normalize(sums[tid], dim=0)
            out.append(v)
    return torch.stack(out, dim=0), counts


def retrieval_metrics(
    audio_emb: torch.Tensor,
    text_emb: torch.Tensor,
    base_valid: torch.Tensor | None = None,
) -> dict:
    valid = ~(torch.isnan(text_emb).any(dim=1))
    if base_valid is not None:
        valid = valid & base_valid
    a = audio_emb[valid]
    t = text_emb[valid]
    sim = (a @ t.T).numpy()
    n = sim.shape[0]
    diag = np.diag(sim)
    shuffled_mean = float((sim.sum() - diag.sum()) / (n * (n - 1))) if n > 1 else float("nan")
    ranks = np.array([1 + int((sim[i] > sim[i, i]).sum()) for i in range(n)])
    return {
        "n": int(n),
        "missing_text": int((~valid).sum().item()),
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_p05": float(np.percentile(diag, 5)),
        "diag_p10": float(np.percentile(diag, 10)),
        "diag_p50": float(np.percentile(diag, 50)),
        "diag_p90": float(np.percentile(diag, 90)),
        "diag_p95": float(np.percentile(diag, 95)),
        "frac_below_0.10": float((diag < 0.10).mean()),
        "frac_below_0.20": float((diag < 0.20).mean()),
        "shuffled_mean": shuffled_mean,
        "diag_minus_shuffled": float(diag.mean() - shuffled_mean),
        "median_rank": float(np.median(ranks)),
        "R@1": float((ranks <= 1).mean()),
        "R@5": float((ranks <= 5).mean()),
        "R@10": float((ranks <= 10).mean()),
        "random_R@1": float(1 / n),
        "random_R@10": float(min(10, n) / n),
    }


def diag_scores(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> list[float | None]:
    scores = []
    for a, t in zip(audio_emb, text_emb):
        if torch.isnan(t).any():
            scores.append(None)
        else:
            scores.append(float((a * t).sum().item()))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flamingo", type=Path, default=DEFAULT_FLAMINGO)
    parser.add_argument("--lpmc", type=Path, default=DEFAULT_LPMC)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_FLAMINGO.parent)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--qwen_mode", choices=["slot0", "mean5"], default="slot0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"clap_compare_qwen_{args.qwen_mode}.json"
    per_track_path = args.out_dir / f"clap_compare_qwen_{args.qwen_mode}_per_track.jsonl"

    flamingo_rows = load_flamingo(args.flamingo)
    if not flamingo_rows:
        raise RuntimeError(f"No usable Flamingo rows from {args.flamingo}")

    track_ids = [r["track_id"] for r in flamingo_rows]
    wanted = set(track_ids)
    audio_paths = [r["audio_path"] for r in flamingo_rows]
    flamingo_caps = [r["caption"] for r in flamingo_rows]

    print(f"Loaded Flamingo rows: {len(flamingo_rows)}")
    print("Loading LP-MC segment captions...")
    lpmc_by_track = load_lpmc_by_track(args.lpmc, wanted)
    print(f"  tracks covered: {len(lpmc_by_track)}/{len(track_ids)}")
    print(f"Loading Qwen segment captions ({args.qwen_mode})...")
    qwen_by_track = load_qwen_by_track(args.qwen, wanted, args.qwen_mode)
    print(f"  tracks covered: {len(qwen_by_track)}/{len(track_ids)}")

    import laion_clap

    print("Loading CLAP...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(CLAP_CKPT))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  device={device}")

    audio_emb = encode_audio(model, audio_paths, args.batch_size)

    print("Encoding Music Flamingo text...")
    flamingo_emb = encode_text_list(model, flamingo_caps, args.batch_size)

    print("Encoding LP-MC track-mean text...")
    lpmc_emb, lpmc_counts = encode_track_mean_text(model, track_ids, lpmc_by_track, args.batch_size)

    print(f"Encoding Qwen {args.qwen_mode} track-mean text...")
    qwen_emb, qwen_counts = encode_track_mean_text(model, track_ids, qwen_by_track, args.batch_size)

    embeddings = {
        "music_flamingo": flamingo_emb,
        "lpmc_track_mean": lpmc_emb,
        f"qwen_{args.qwen_mode}_track_mean": qwen_emb,
    }
    metrics = {name: retrieval_metrics(audio_emb, emb) for name, emb in embeddings.items()}
    common_valid = torch.ones(len(track_ids), dtype=torch.bool)
    for emb in embeddings.values():
        common_valid &= ~(torch.isnan(emb).any(dim=1))
    common_metrics = {
        name: retrieval_metrics(audio_emb, emb, base_valid=common_valid)
        for name, emb in embeddings.items()
    }
    scores = {name: diag_scores(audio_emb, emb) for name, emb in embeddings.items()}

    per_track_rows = []
    with per_track_path.open("w") as f:
        for i, row in enumerate(flamingo_rows):
            out = {
                "track_id": row["track_id"],
                "audio_path": row["audio_path"],
                "duration_sec": row["duration_sec"],
                "lpmc_segment_count": lpmc_counts.get(row["track_id"], 0),
                "qwen_caption_count": qwen_counts.get(row["track_id"], 0),
            }
            for name, vals in scores.items():
                out[f"{name}_clap_sim"] = vals[i]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            per_track_rows.append(out)

    summary = {
        "meta": {
            "flamingo_caption_jsonl": str(args.flamingo),
            "lpmc_tsv": str(args.lpmc),
            "qwen_jsonl": str(args.qwen),
            "qwen_mode": args.qwen_mode,
            "clap_ckpt": str(CLAP_CKPT),
            "tracks": len(track_ids),
            "comparison_note": (
                "Flamingo has one full-track caption. Qwen/LP-MC are segment-level; "
                "their normalized CLAP text embeddings are averaged per track."
            ),
        },
        "metrics_all_available": metrics,
        "metrics_qwen_common": common_metrics,
        "coverage": {
            "lpmc_tracks": len(lpmc_by_track),
            "qwen_tracks": len(qwen_by_track),
            "qwen_common_tracks": int(common_valid.sum().item()),
            "lpmc_segment_count_mean": float(np.mean(list(lpmc_counts.values()))),
            "qwen_caption_count_mean": float(np.mean(list(qwen_counts.values()))),
        },
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== CLAP comparison: all available ===")
    for name, m in metrics.items():
        print(
            f"{name:<24} n={m['n']:4d} diag={m['diag_mean']:.4f} "
            f"gap={m['diag_minus_shuffled']:.4f} R@1={100*m['R@1']:.2f}% "
            f"R@10={100*m['R@10']:.2f}% p10={m['diag_p10']:.4f}"
        )
    print("\n=== CLAP comparison: Qwen common subset ===")
    for name, m in common_metrics.items():
        print(
            f"{name:<24} n={m['n']:4d} diag={m['diag_mean']:.4f} "
            f"gap={m['diag_minus_shuffled']:.4f} R@1={100*m['R@1']:.2f}% "
            f"R@10={100*m['R@10']:.2f}% p10={m['diag_p10']:.4f}"
        )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-track: {per_track_path}")


if __name__ == "__main__":
    main()
