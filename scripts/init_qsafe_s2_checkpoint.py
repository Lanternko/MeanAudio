#!/usr/bin/env python3
"""Create a NoQ->Q continuation checkpoint with an exactly neutral Q start.

Rows q=0..9 are copied from the trained null row q=10 in the online model and
both PostHocEMA tracks.  At creation time, inference with any q therefore has
exactly the same network weights as the source NoQ checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_null_rows(weight: torch.Tensor) -> dict[str, object]:
    if tuple(weight.shape)[0] != 11:
        raise ValueError(f"q_embed must have 11 rows, got {tuple(weight.shape)}")
    before = (weight[:10].float() - weight[10].float()).norm(dim=1).tolist()
    weight[:10].copy_(weight[10].unsqueeze(0).expand_as(weight[:10]))
    exact = bool(torch.equal(weight[:10], weight[10].unsqueeze(0).expand_as(weight[:10])))
    if not exact:
        raise RuntimeError("q=0..9 are not exactly equal to q=10 after initialization")
    return {
        "distance_to_q10_before": before,
        "exactly_equal_after": exact,
        "shape": list(weight.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-it", type=int, default=600_000)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        raise SystemExit("[FAIL] output/manifest already exists; Q-safe init is fresh-only")
    source_hash = sha256(args.source)
    if args.source_sha256 and source_hash != args.source_sha256:
        raise SystemExit("[FAIL] source checkpoint hash mismatch")

    state = torch.load(args.source, map_location="cpu", weights_only=False)
    if state.get("it") != args.expected_it:
        raise SystemExit(f"[FAIL] source iteration={state.get('it')}, expected={args.expected_it}")
    if not all(key in state for key in ("weights", "ema")):
        raise SystemExit("[FAIL] source is not a resumable MeanAudio checkpoint")

    audited: dict[str, object] = {}
    audited["weights.q_embed.weight"] = copy_null_rows(state["weights"]["q_embed.weight"])
    ema_keys = sorted(key for key in state["ema"] if key.endswith("q_embed.weight"))
    if len(ema_keys) != 2:
        raise SystemExit(f"[FAIL] expected two EMA q_embed tracks, got {ema_keys}")
    for key in ema_keys:
        audited[f"ema.{key}"] = copy_null_rows(state["ema"][key])

    # A new optimizer/scheduler is intentional: this is a bounded low-LR
    # continuation, not a replay of the completed baseline schedule.
    state["optimizer"] = None
    state["scheduler"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, args.output)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": source_hash,
        "source_iteration": state["it"],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "initialization": "copy_q10_exactly_to_q0_through_q9",
        "optimizer_reset": True,
        "scheduler_reset": True,
        "audited_tensors": audited,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_manifest, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
