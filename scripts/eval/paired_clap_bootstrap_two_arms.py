#!/usr/bin/env python3
"""Generic paired per-prompt CLAP bootstrap between two generated audio sets.

Reports mean CLAP per arm, the paired difference, a bootstrap 95% CI on that
difference, and the per-clip win rate.  Unlike an unpaired comparison of two
``metrics.txt`` numbers, this says whether an observed delta is separable from
per-prompt noise.

Uses the internal 89.98 CLAP checkpoint so the numbers stay comparable with the
existing Phase-8 MusicCaps metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

DEFAULT_CLAP = Path(
    "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--arm-a-name", required=True, help="label for the positive arm")
    parser.add_argument("--arm-a-dir", type=Path, required=True)
    parser.add_argument("--arm-b-name", required=True, help="label for the reference arm")
    parser.add_argument("--arm-b-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--clap-ckpt", type=Path, default=DEFAULT_CLAP)
    parser.add_argument("--expected-rows", type=int, default=5521)
    parser.add_argument("--gate-delta", type=float, default=None,
                        help="preregistered a-minus-b threshold to report against")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda|cpu; default auto")
    args = parser.parse_args()

    if args.arm_a_name == args.arm_b_name:
        raise SystemExit("[FAIL] arm names must differ")

    import laion_clap

    records = list(csv.DictReader(args.tsv.open(encoding="utf-8"), delimiter="\t"))
    if len(records) != args.expected_rows:
        raise SystemExit(
            f"[FAIL] tsv rows={len(records)}, expected={args.expected_rows}"
        )

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(args.clap_ckpt))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    arms = {"a": args.arm_a_dir, "b": args.arm_b_dir}
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            ids = [record["id"] for record in batch]
            captions = [record["caption"] for record in batch]
            paths = {
                arm: [directory / f"{clip_id}.flac" for clip_id in ids]
                for arm, directory in arms.items()
            }
            missing = [
                str(path) for arm in paths.values() for path in arm if not path.is_file()
            ]
            if missing:
                raise SystemExit(f"[FAIL] missing audio files: {missing[:10]}")
            text = model.get_text_embedding(captions, use_tensor=True)
            values: dict[str, np.ndarray] = {}
            for arm, arm_paths in paths.items():
                audio = model.get_audio_embedding_from_filelist(
                    [str(path) for path in arm_paths], use_tensor=True
                )
                values[arm] = (
                    torch.nn.functional.cosine_similarity(audio, text, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            rows.extend(
                {
                    "id": clip_id,
                    args.arm_a_name: float(values["a"][offset]),
                    args.arm_b_name: float(values["b"][offset]),
                }
                for offset, clip_id in enumerate(ids)
            )
            print(
                f"paired CLAP {min(start + len(batch), len(records))}/{len(records)}",
                flush=True,
            )

    args.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", args.arm_a_name, args.arm_b_name]
        )
        writer.writeheader()
        writer.writerows(rows)

    a = np.asarray([row[args.arm_a_name] for row in rows], dtype=np.float64)
    b = np.asarray([row[args.arm_b_name] for row in rows], dtype=np.float64)
    diff = a - b
    ci = bootstrap_ci(diff, seed=args.seed)
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "seed": args.seed,
        "bootstrap_samples": 20000,
        "clap_checkpoint": str(args.clap_ckpt),
        "clap_metric_family": "internal_89.98_continuity",
        "arms": {
            args.arm_a_name: str(args.arm_a_dir),
            args.arm_b_name: str(args.arm_b_dir),
        },
        "means": {args.arm_a_name: float(a.mean()), args.arm_b_name: float(b.mean())},
        "contrast": f"{args.arm_a_name} minus {args.arm_b_name}",
        "delta": float(diff.mean()),
        "delta_ci95": ci,
        "delta_separable_from_zero": bool(ci[0] > 0 or ci[1] < 0),
        "win_rate": float((diff > 0).mean()),
    }
    if args.gate_delta is not None:
        payload["gate_delta"] = args.gate_delta
        payload["point_estimate_meets_gate"] = bool(diff.mean() >= args.gate_delta)
        # The honest question for a small preregistered threshold: does the CI
        # actually exclude the threshold, or is the gate decision noise-limited?
        payload["ci_excludes_gate_delta"] = bool(
            ci[0] > args.gate_delta or ci[1] < args.gate_delta
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
