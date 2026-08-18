#!/usr/bin/env python3
"""Paired per-prompt CLAP bootstrap: Fixed-Q9 vs matched-NoQ on MusicCaps.

Uses the existing internal 89.98 CLAP checkpoint for continuity with prior
Phase-8 MusicCaps metrics.  ATTM official 90.14 comparison is intentionally
out of scope here and remains blocked until the exact official 100-prompt
CSV is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

DEFAULT_CLAP = Path(
    "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
)
EXPECTED_ROWS = 5521


def bootstrap_ci(diff: np.ndarray, *, seed: int, samples: int = 20_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 200
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        idx = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        means[start:stop] = diff[idx].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--fixedq-dir", type=Path, required=True)
    parser.add_argument("--noq-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--clap-ckpt", type=Path, default=DEFAULT_CLAP)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda|cpu; default auto")
    args = parser.parse_args()

    if "90.14" in args.clap_ckpt.name:
        raise SystemExit(
            "[FAIL] this paired experiment intentionally keeps the internal "
            "89.98 CLAP metric; 90.14 ATTM official is a separate blocked evaluator"
        )

    import laion_clap

    records = list(csv.DictReader(args.tsv.open(), delimiter="\t"))
    if len(records) != EXPECTED_ROWS:
        raise SystemExit(
            f"[FAIL] MusicCaps rows={len(records)}, expected={EXPECTED_ROWS}"
        )

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(args.clap_ckpt))
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            ids = [record["id"] for record in batch]
            captions = [record["caption"] for record in batch]
            paths = {
                "fixedq": [args.fixedq_dir / f"{clip_id}.flac" for clip_id in ids],
                "noq": [args.noq_dir / f"{clip_id}.flac" for clip_id in ids],
            }
            missing = [str(path) for arm in paths.values() for path in arm if not path.is_file()]
            if missing:
                raise SystemExit(f"[FAIL] missing audio files: {missing[:10]}")
            text = model.get_text_embedding(captions, use_tensor=True)
            values: dict[str, np.ndarray] = {}
            for name, arm_paths in paths.items():
                audio = model.get_audio_embedding_from_filelist(
                    [str(path) for path in arm_paths], use_tensor=True
                )
                values[name] = (
                    torch.nn.functional.cosine_similarity(audio, text, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            rows.extend(
                {
                    "id": clip_id,
                    "fixedq": float(values["fixedq"][offset]),
                    "noq": float(values["noq"][offset]),
                }
                for offset, clip_id in enumerate(ids)
            )
            print(
                f"paired CLAP {min(start + len(batch), EXPECTED_ROWS)}/{EXPECTED_ROWS}",
                flush=True,
            )

    args.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "fixedq", "noq"])
        writer.writeheader()
        writer.writerows(rows)

    fixed = np.asarray([row["fixedq"] for row in rows], dtype=np.float64)
    noq = np.asarray([row["noq"] for row in rows], dtype=np.float64)
    diff = fixed - noq
    ci = bootstrap_ci(diff, seed=args.seed)
    means = {"fixedq": float(fixed.mean()), "noq": float(noq.mean())}
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "seed": args.seed,
        "bootstrap_samples": 20000,
        "clap_checkpoint": str(args.clap_ckpt),
        "clap_metric_family": "internal_89.98_continuity",
        "attm_official_90_14": "blocked_until_exact_100_prompt_csv",
        "means": means,
        "fixedq_minus_noq": float(diff.mean()),
        "fixedq_minus_noq_ci95": ci,
        "fixedq_win_rate_vs_noq": float((diff > 0).mean()),
        "fixedq_benefit_supported": ci[0] > 0,
        "restored_clap_0p19": means["fixedq"] >= 0.19,
        "primary_checkpoint_iteration": 700000,
        "no_checkpoint_cherrypick": True,
    }
    payload["primary_objective_met"] = payload["fixedq_benefit_supported"]
    payload["fallback_restoration_met"] = payload["restored_clap_0p19"]
    payload["program_goal_met"] = bool(
        payload["fixedq_benefit_supported"] or payload["restored_clap_0p19"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
