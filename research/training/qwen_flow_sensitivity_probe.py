#!/usr/bin/env python3
"""Probe whether trained MeanAudio checkpoints still react to text.

The CLAP collapse shows up at generation time, but a much cheaper check is:
hold the latent/timestep fixed, change only the cached text features, and
measure how much the model's predicted flow changes. A caption-conditioned
model should have a measurable matched-vs-shuffled / matched-vs-empty delta.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO = Path("/home/kojiek/MeanAudio")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from meanaudio.model.networks import get_mean_audio  # noqa: E402


CHECKPOINTS = {
    "healthy_phase8_s2": "/home/kojiek/exps_nvme/phase8_stage2_200000/phase8_stage2_200000_ema_final.pth",
    "phase8v4_generated_caption_s2": "/home/kojiek/exps_nvme/phase8_v4_stage2_200000/phase8_v4_stage2_200000_ema_final.pth",
    "p8_qwen_s1": "/home/kojiek/exps_nvme/p8_qwen_stage1_400000/p8_qwen_stage1_400000_ema_final.pth",
    "p8_qwen_s2": "/home/kojiek/exps_nvme/p8_qwen_stage2_200000/p8_qwen_stage2_200000_ema_final.pth",
    "expb_qwen_slot0_s1": "/home/kojiek/exps_nvme/p_qwen_slot0_stage1_400000/p_qwen_slot0_stage1_400000_ema_final.pth",
    "expb_qwen_slot0_s2": "/home/kojiek/exps_nvme/p_qwen_slot0_stage2_200000/p_qwen_slot0_stage2_200000_ema_final.pth",
    "expg_lpmc_s1_qwen_s2": "/home/kojiek/exps_nvme/p_expg_lpmcs1_qwens2_stage2_200000/p_expg_lpmcs1_qwens2_stage2_200000_ema_final.pth",
}

TEXT_SETS = {
    "phase8v4_cache": "/home/kojiek/research/meanaudio_training/npz_phase8v4",
    "qwen_slot0_cache": "/home/kojiek/exps_nvme/npz_qwen_slot0",
    "qwen_boilerplate_cache": "/home/kojiek/exps_nvme/npz_qwen_slot0_boilerplate",
    "expH_rewrite_cache": "/home/kojiek/exps_nvme/npz_expH_rewrite",
}

DEFAULT_FILES = ("33.npz", "34.npz", "35.npz", "36.npz")


def tensor_norm(x: torch.Tensor) -> float:
    return x.float().pow(2).sum().sqrt().item()


def rel_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return tensor_norm(a - b) / max(tensor_norm(a), 1.0e-12)


def load_npz_features(npz_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    data = np.load(npz_path)
    text_f = torch.from_numpy(data["text_features"]).unsqueeze(0).to(device=device, dtype=dtype)
    text_fc = torch.from_numpy(data["text_features_c"]).unsqueeze(0).to(device=device, dtype=dtype)
    text_mask = None
    if "text_attention_mask" in data.files:
        text_mask = torch.from_numpy(data["text_attention_mask"]).unsqueeze(0).to(device=device, dtype=torch.bool)
    return text_f, text_fc, text_mask


def usable_files(npz_dir: Path, requested: Iterable[str]) -> list[str]:
    found = [name for name in requested if (npz_dir / name).exists()]
    if len(found) >= 2:
        return found
    found = sorted(p.name for p in npz_dir.glob("*.npz"))[:4]
    if len(found) < 2:
        raise FileNotFoundError(f"Need at least two npz files in {npz_dir}")
    return found


def summarize_conditions(net, text_a, text_ac, mask_a, text_b, text_bc, mask_b) -> dict[str, float]:
    cond_a = net.preprocess_conditions(text_a, text_ac, mask_a)
    cond_b = net.preprocess_conditions(text_b, text_bc, mask_b)
    empty = net.get_empty_conditions(1)
    return {
        "text_proj_norm_a": tensor_norm(cond_a.text_f) / cond_a.text_f.shape[1],
        "text_proj_norm_b": tensor_norm(cond_b.text_f) / cond_b.text_f.shape[1],
        "text_proj_rel_delta_ab": rel_delta(cond_a.text_f, cond_b.text_f),
        "global_cond_norm_a": tensor_norm(cond_a.text_f_c),
        "global_cond_norm_b": tensor_norm(cond_b.text_f_c),
        "global_cond_rel_delta_ab": rel_delta(cond_a.text_f_c, cond_b.text_f_c),
        "text_proj_rel_delta_empty": rel_delta(cond_a.text_f, empty.text_f),
        "global_cond_rel_delta_empty": rel_delta(cond_a.text_f_c, empty.text_f_c),
    }


def run_pair(
    net,
    text_a: torch.Tensor,
    text_ac: torch.Tensor,
    mask_a: torch.Tensor | None,
    text_b: torch.Tensor,
    text_bc: torch.Tensor,
    mask_b: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    q_level: int,
) -> dict[str, float]:
    cond_a = net.preprocess_conditions(text_a, text_ac, mask_a)
    cond_b = net.preprocess_conditions(text_b, text_bc, mask_b)
    empty = net.get_empty_conditions(1)

    generator = torch.Generator(device=device.type).manual_seed(seed)
    latent = torch.randn(
        1,
        net.latent_seq_len,
        net.latent_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    t = torch.full((1,), 0.55, device=device, dtype=dtype)
    r = torch.full((1,), 0.15, device=device, dtype=dtype)
    q = torch.full((1,), q_level, dtype=torch.long, device=device)

    flow_a = net.predict_flow(latent, t, r, cond_a, q=q)
    flow_b = net.predict_flow(latent, t, r, cond_b, q=q)
    flow_empty = net.predict_flow(latent, t, r, empty, q=q)

    return {
        "flow_norm": tensor_norm(flow_a),
        "flow_text_rel_delta_ab": rel_delta(flow_a, flow_b),
        "flow_text_abs_delta_ab": tensor_norm(flow_a - flow_b),
        "flow_empty_rel_delta": rel_delta(flow_a, flow_empty),
        "flow_empty_abs_delta": tensor_norm(flow_a - flow_empty),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/kojiek/research/meanaudio_training/qwen_flow_sensitivity_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--q-level", type=int, default=10)
    parser.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    results: dict[str, object] = {
        "device": str(device),
        "dtype": str(dtype),
        "q_level": args.q_level,
        "checkpoints": {},
    }

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"SKIP missing checkpoint: {ckpt_name} {ckpt_path}", flush=True)
            continue

        print(f"\n=== {ckpt_name} ===", flush=True)
        net = get_mean_audio("meanaudio_s", use_rope=False, text_c_dim=512).to(device=device, dtype=dtype).eval()
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        net.load_weights(state)

        ckpt_results: dict[str, object] = {"path": ckpt_path, "text_sets": {}}

        with torch.no_grad():
            for text_set_name, text_set_dir in TEXT_SETS.items():
                npz_dir = Path(text_set_dir)
                if not npz_dir.exists():
                    print(f"  SKIP missing text set: {text_set_name}", flush=True)
                    continue
                files = usable_files(npz_dir, args.files)
                text_a, text_ac, mask_a = load_npz_features(npz_dir / files[0], device, dtype)
                text_b, text_bc, mask_b = load_npz_features(npz_dir / files[1], device, dtype)

                cond_stats = summarize_conditions(net, text_a, text_ac, mask_a, text_b, text_bc, mask_b)
                flow_stats = run_pair(
                    net,
                    text_a,
                    text_ac,
                    mask_a,
                    text_b,
                    text_bc,
                    mask_b,
                    device=device,
                    dtype=dtype,
                    seed=1234,
                    q_level=args.q_level,
                )
                merged = {
                    "files": files[:2],
                    **cond_stats,
                    **flow_stats,
                }
                ckpt_results["text_sets"][text_set_name] = merged
                print(
                    "  "
                    f"{text_set_name:<24} "
                    f"flow_ab={merged['flow_text_rel_delta_ab']:.5f} "
                    f"flow_empty={merged['flow_empty_rel_delta']:.5f} "
                    f"cond={merged['global_cond_norm_a']:.3f} "
                    f"text/tok={merged['text_proj_norm_a']:.3f}",
                    flush=True,
                )

        results["checkpoints"][ckpt_name] = ckpt_results
        del net, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
