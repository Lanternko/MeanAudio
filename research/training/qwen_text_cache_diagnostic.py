#!/usr/bin/env python3
"""Inspect caption length, padding, and cached text-feature diversity.

MeanAudio stores the full 77-token T5 hidden-state sequence but no attention
mask. This script quantifies how much of each training source is padding and
how much the padded/full sequence representation is dominated by shared tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


CACHE_LIST = Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt")

CASES = {
    "lpmc_phase7_captions": {
        "tsv": Path("/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv"),
        "npz": None,
        "cache": None,
    },
    "phase8v4_generated_cache": {
        "tsv": Path("/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase8_v4_train.tsv"),
        "npz": Path("/home/kojiek/research/meanaudio_training/npz_phase8v4"),
        "cache": CACHE_LIST,
    },
    "qwen_slot0_cache": {
        "tsv": Path("/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv"),
        "npz": Path("/home/kojiek/exps_nvme/npz_qwen_slot0"),
        "cache": CACHE_LIST,
    },
    "qwen_boilerplate_cache": {
        "tsv": Path("/home/kojiek/eval_tsvs_p100/qwen_slot0_boilerplate_train.tsv"),
        "npz": Path("/home/kojiek/exps_nvme/npz_qwen_slot0_boilerplate"),
        "cache": CACHE_LIST,
    },
    "expH_rewrite_cache": {
        "tsv": Path("/home/kojiek/eval_tsvs_p100/expH_rewrite_train.tsv"),
        "npz": Path("/home/kojiek/exps_nvme/npz_expH_rewrite"),
        "cache": CACHE_LIST,
    },
}


def load_captions(tsv_path: Path, n: int) -> list[str]:
    captions: list[str] = []
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            captions.append(row["caption"])
            if len(captions) >= n:
                break
    return captions


def load_cache(cache_path: Path | None, n: int) -> list[str]:
    if cache_path is None:
        return [f"{i}.npz" for i in range(n)]
    with cache_path.open("r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    return names[:n]


def offdiag_cos_mean(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return float("nan")
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9
    x = x / norms
    cos = x @ x.T
    mask = ~np.eye(x.shape[0], dtype=bool)
    return float(cos[mask].mean())


def token_stats(tokenizer, captions: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = []
    lengths = []
    raw_lengths = []
    for start in range(0, len(captions), batch_size):
        batch = captions[start : start + batch_size]
        toks = tokenizer(
            batch,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        masks.append(toks["attention_mask"].astype(bool))
        lengths.append(toks["attention_mask"].sum(axis=1))
        raw = tokenizer(batch, padding=False, truncation=False)
        raw_lengths.extend(len(ids) for ids in raw["input_ids"])
    return np.concatenate(masks), np.concatenate(lengths), np.asarray(raw_lengths)


def npz_stats(npz_dir: Path, names: list[str], masks: np.ndarray, max_npz: int) -> dict[str, float]:
    full_means = []
    real_means = []
    pad_means = []
    clap = []
    real_norm_sum = 0.0
    pad_norm_sum = 0.0
    real_energy_sum = 0.0
    pad_energy_sum = 0.0
    real_count = 0
    pad_count = 0
    loaded = 0
    missing = 0

    for i, name in enumerate(names[:max_npz]):
        path = npz_dir / name
        if not path.exists():
            alt = npz_dir / f"{i}.npz"
            path = alt if alt.exists() else path
        if not path.exists():
            missing += 1
            continue

        data = np.load(path)
        text = data["text_features"].astype(np.float32)
        text_c = data["text_features_c"].astype(np.float32)
        mask = masks[i]
        real = text[mask]
        pad = text[~mask]

        full_means.append(text.mean(axis=0))
        real_means.append(real.mean(axis=0))
        if len(pad):
            pad_means.append(pad.mean(axis=0))
        clap.append(text_c)

        real_norms = np.linalg.norm(real, axis=1)
        pad_norms = np.linalg.norm(pad, axis=1) if len(pad) else np.asarray([], dtype=np.float32)
        real_norm_sum += float(real_norms.sum())
        pad_norm_sum += float(pad_norms.sum())
        real_energy_sum += float((real_norms**2).sum())
        pad_energy_sum += float((pad_norms**2).sum())
        real_count += len(real)
        pad_count += len(pad)
        loaded += 1

    out = {
        "npz_loaded": loaded,
        "npz_missing": missing,
        "t5_pad_token_fraction": pad_count / max(real_count + pad_count, 1),
        "t5_real_token_norm_mean": real_norm_sum / max(real_count, 1),
        "t5_pad_token_norm_mean": pad_norm_sum / max(pad_count, 1),
        "t5_pad_vs_real_norm_ratio": (pad_norm_sum / max(pad_count, 1)) / max(real_norm_sum / max(real_count, 1), 1.0e-9),
        "t5_pad_norm_fraction": pad_norm_sum / max(real_norm_sum + pad_norm_sum, 1.0e-9),
        "t5_pad_energy_fraction": pad_energy_sum / max(real_energy_sum + pad_energy_sum, 1.0e-9),
        "t5_full_mean_offdiag_cos": offdiag_cos_mean(np.stack(full_means)) if full_means else float("nan"),
        "t5_real_mean_offdiag_cos": offdiag_cos_mean(np.stack(real_means)) if real_means else float("nan"),
        "clap_offdiag_cos": offdiag_cos_mean(np.stack(clap)) if clap else float("nan"),
    }
    if pad_means:
        out["t5_pad_mean_offdiag_cos"] = offdiag_cos_mean(np.stack(pad_means))
    else:
        out["t5_pad_mean_offdiag_cos"] = float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--max-npz", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="/home/kojiek/research/meanaudio_training/qwen_text_cache_diagnostic.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    results: dict[str, object] = {"n": args.n, "max_npz": args.max_npz, "cases": {}}

    for name, cfg in CASES.items():
        tsv = cfg["tsv"]
        if not tsv.exists():
            print(f"SKIP missing TSV: {name} {tsv}", flush=True)
            continue

        captions = load_captions(tsv, args.n)
        masks, lengths, raw_lengths = token_stats(tokenizer, captions, args.batch_size)
        case = {
            "tsv": str(tsv),
            "caption_count": len(captions),
            "token_len_mean": float(lengths.mean()),
            "token_len_p10": float(np.percentile(lengths, 10)),
            "token_len_p50": float(np.percentile(lengths, 50)),
            "token_len_p90": float(np.percentile(lengths, 90)),
            "pad_fraction_mean": float(1.0 - lengths.mean() / 77.0),
            "raw_len_gt_77_fraction": float((raw_lengths > 77).mean()),
        }

        npz_dir = cfg["npz"]
        if npz_dir is not None and npz_dir.exists():
            names = load_cache(cfg["cache"], len(captions))
            case.update(npz_stats(npz_dir, names, masks, args.max_npz))
        results["cases"][name] = case

        print(
            f"{name:<28} "
            f"len={case['token_len_mean']:.1f} "
            f"pad={case['pad_fraction_mean']:.3f} "
            f">77={case['raw_len_gt_77_fraction']:.3f} "
            f"fullT5cos={case.get('t5_full_mean_offdiag_cos', float('nan')):.3f} "
            f"realT5cos={case.get('t5_real_mean_offdiag_cos', float('nan')):.3f} "
            f"CLAPcos={case.get('clap_offdiag_cos', float('nan')):.3f}",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
