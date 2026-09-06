#!/usr/bin/env python3
"""Generic paired per-prompt CLAP bootstrap for historical Q closure."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def records(tsv: Path) -> list[tuple[str, str]]:
    with tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty TSV: {tsv}")
    return [(str(row["id"]), str(row["caption"])) for row in rows]


def paired_scores(
    rows: list[tuple[str, str]],
    baseline_dir: Path,
    treatment_dir: Path,
    checkpoint: Path,
    *,
    batch_size: int,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    baseline: list[Path] = []
    treatment: list[Path] = []
    captions: list[str] = []
    ids: list[str] = []
    for clip_id, caption in rows:
        base = baseline_dir / f"{clip_id}.flac"
        treat = treatment_dir / f"{clip_id}.flac"
        if not base.is_file() or not treat.is_file():
            raise ValueError(f"missing paired audio for {clip_id}: {base}, {treat}")
        ids.append(clip_id)
        captions.append(caption)
        baseline.append(base)
        treatment.append(treat)
    if not ids:
        raise ValueError("no complete paired audio rows")
    import torch
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(checkpoint), verbose=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    base_scores: list[np.ndarray] = []
    treatment_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            text = model.get_text_embedding(captions[start:stop], use_tensor=True)
            base_audio = model.get_audio_embedding_from_filelist(
                [str(path) for path in baseline[start:stop]], use_tensor=True
            )
            treat_audio = model.get_audio_embedding_from_filelist(
                [str(path) for path in treatment[start:stop]], use_tensor=True
            )
            base_scores.append(
                torch.nn.functional.cosine_similarity(base_audio, text, dim=-1)
                .float().cpu().numpy()
            )
            treatment_scores.append(
                torch.nn.functional.cosine_similarity(treat_audio, text, dim=-1)
                .float().cpu().numpy()
            )
            print(f"paired CLAP {stop}/{len(ids)}", flush=True)
    base_score = np.concatenate(base_scores)
    treatment_score = np.concatenate(treatment_scores)
    return treatment_score - base_score, ids, base_score, treatment_score


def bootstrap(diff: np.ndarray, *, seed: int, replicates: int) -> dict[str, Any]:
    if diff.ndim != 1 or diff.size < 2:
        raise ValueError("paired bootstrap needs at least two differences")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = np.mean(rng.choice(diff, size=diff.size, replace=True))
    return {
        "n": int(diff.size),
        "mean_delta_treatment_minus_baseline": float(np.mean(diff)),
        "median_delta": float(np.median(diff)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "interpretation": "descriptive paired effect; no positive threshold is an audit gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--clap-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14159265)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 100:
        raise SystemExit("--replicates must be >=100")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    diff, ids, baseline_scores, treatment_scores = paired_scores(
        records(args.tsv), args.baseline_dir, args.treatment_dir,
        args.clap_checkpoint, batch_size=args.batch_size,
    )
    result = bootstrap(diff, seed=args.seed, replicates=args.replicates)
    result.update(
        {
            "tsv": str(args.tsv),
            "baseline_dir": str(args.baseline_dir),
            "treatment_dir": str(args.treatment_dir),
            "paired_id_sha256": __import__("hashlib").sha256("\n".join(ids).encode()).hexdigest(),
            "baseline_mean": float(np.mean(baseline_scores)),
            "treatment_mean": float(np.mean(treatment_scores)),
        }
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.json_out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
