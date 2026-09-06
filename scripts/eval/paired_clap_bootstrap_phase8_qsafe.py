#!/usr/bin/env python3
"""Paired CLAP bootstrap for Q-safe Real-Q vs controls on MusicCaps."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


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
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--real-dir", type=Path, required=True)
    p.add_argument("--shuffled-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scores-csv", type=Path, required=True)
    p.add_argument("--clap-ckpt", type=Path, default=Path("/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"))
    p.add_argument("--seed", type=int, default=20260721)
    args = p.parse_args()

    import laion_clap

    records = list(csv.DictReader(args.tsv.open(), delimiter="\t"))
    if len(records) != 5521:
        raise SystemExit(f"[FAIL] MusicCaps rows={len(records)}, expected=5521")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(args.clap_ckpt))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            clip_id, caption = record["id"], record["caption"]
            paths = {
                "baseline": args.baseline_dir / f"{clip_id}.flac",
                "real": args.real_dir / f"{clip_id}.flac",
                "shuffled": args.shuffled_dir / f"{clip_id}.flac",
            }
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise SystemExit(f"[FAIL] missing {clip_id}: {missing}")
            text = model.get_text_embedding([caption], use_tensor=True)
            values = {}
            for name, path in paths.items():
                audio = model.get_audio_embedding_from_filelist([str(path)], use_tensor=True)
                values[name] = float(torch.nn.functional.cosine_similarity(audio, text, dim=-1).item())
            rows.append({"id": clip_id, **values})
            if index % 250 == 0:
                print(f"paired CLAP {index}/5521", flush=True)

    args.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "baseline", "real", "shuffled"])
        writer.writeheader(); writer.writerows(rows)

    arrays = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in ("baseline", "real", "shuffled")}
    real_shuffled = arrays["real"] - arrays["shuffled"]
    real_baseline = arrays["real"] - arrays["baseline"]
    rs_ci = bootstrap_ci(real_shuffled, seed=args.seed)
    rb_ci = bootstrap_ci(real_baseline, seed=args.seed + 1)
    means = {key: float(value.mean()) for key, value in arrays.items()}
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n": len(rows), "seed": args.seed, "bootstrap_samples": 20000,
        "means": means,
        "real_minus_shuffled": float(real_shuffled.mean()),
        "real_minus_shuffled_ci95": rs_ci,
        "real_minus_baseline": float(real_baseline.mean()),
        "real_minus_baseline_ci95": rb_ci,
        "real_win_rate_vs_shuffled": float((real_shuffled > 0).mean()),
        "real_win_rate_vs_baseline": float((real_baseline > 0).mean()),
        "q_information_supported": rs_ci[0] > 0,
        "net_q_gain_supported": rs_ci[0] > 0 and rb_ci[0] > 0,
        "restored_clap_0p19": means["real"] >= 0.19,
    }
    payload["primary_objective_met"] = payload["net_q_gain_supported"] or payload["restored_clap_0p19"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
