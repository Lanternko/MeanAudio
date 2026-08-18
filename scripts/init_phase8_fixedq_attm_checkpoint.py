#!/usr/bin/env python3
"""Create a matched-reset continuation checkpoint for Fixed-Q / matched-NoQ FT.

Modes:
  fixedq9  Copy source q10 exactly into q0..q9 for online weights and both
           PostHocEMA tracks (function-preserving high-Q prior init).
  noq      Leave q_embed unchanged; still strip optimizer/scheduler so both
           arms share an identical matched optimizer reset.

Both modes require source iteration 600000 and a resumable MeanAudio S2
checkpoint with weights + ema.  Outputs are fresh-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_q_embed(weight: torch.Tensor) -> None:
    if tuple(weight.shape)[0] != 11:
        raise ValueError(f"q_embed must have 11 rows, got {tuple(weight.shape)}")


def copy_null_rows(weight: torch.Tensor) -> dict[str, Any]:
    """Copy q10 into q0..q9 in-place; return audit stats."""
    _require_q_embed(weight)
    before = (weight[:10].float() - weight[10].float()).norm(dim=1).tolist()
    weight[:10].copy_(weight[10].unsqueeze(0).expand_as(weight[:10]))
    exact = bool(torch.equal(weight[:10], weight[10].unsqueeze(0).expand_as(weight[:10])))
    if not exact:
        raise RuntimeError("q=0..9 are not exactly equal to q=10 after initialization")
    return {
        "distance_to_q10_before": before,
        "exactly_equal_after": exact,
        "shape": list(weight.shape),
        "mode": "copy_q10_to_q0_through_q9",
    }


def audit_noq_rows(weight: torch.Tensor) -> dict[str, Any]:
    """Record q-row state without mutation (NoQ arm)."""
    _require_q_embed(weight)
    distance = (weight[:10].float() - weight[10].float()).norm(dim=1).tolist()
    exact_to_null = bool(
        torch.equal(weight[:10], weight[10].unsqueeze(0).expand_as(weight[:10]))
    )
    return {
        "distance_to_q10": distance,
        "exactly_equal_to_q10": exact_to_null,
        "shape": list(weight.shape),
        "mode": "preserve_source_q_embed",
        "mutated": False,
    }


def process_q_embed(weight: torch.Tensor, mode: str) -> dict[str, Any]:
    if mode == "fixedq9":
        return copy_null_rows(weight)
    if mode == "noq":
        return audit_noq_rows(weight)
    raise ValueError(f"unknown mode {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["noq", "fixedq9"],
        required=True,
        help="noq: preserve q_embed; fixedq9: copy q10→q0..q9",
    )
    parser.add_argument("--expected-it", type=int, default=600_000)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        raise SystemExit(
            "[FAIL] output/manifest already exists; init is fresh-only"
        )
    source_hash = sha256(args.source)
    if args.source_sha256 and source_hash != args.source_sha256:
        raise SystemExit("[FAIL] source checkpoint hash mismatch")

    state = torch.load(args.source, map_location="cpu", weights_only=False)
    if state.get("it") != args.expected_it:
        raise SystemExit(
            f"[FAIL] source iteration={state.get('it')}, expected={args.expected_it}"
        )
    if not all(key in state for key in ("weights", "ema")):
        raise SystemExit("[FAIL] source is not a resumable MeanAudio checkpoint")
    if "q_embed.weight" not in state["weights"]:
        raise SystemExit("[FAIL] source weights lack q_embed.weight")

    audited: dict[str, Any] = {}
    audited["weights.q_embed.weight"] = process_q_embed(
        state["weights"]["q_embed.weight"], args.mode
    )

    ema_keys = sorted(key for key in state["ema"] if key.endswith("q_embed.weight"))
    if len(ema_keys) != 2:
        raise SystemExit(f"[FAIL] expected two EMA q_embed tracks, got {ema_keys}")
    for key in ema_keys:
        audited[f"ema.{key}"] = process_q_embed(state["ema"][key], args.mode)

    if args.mode == "fixedq9":
        for name, info in audited.items():
            if info.get("exactly_equal_after") is not True:
                raise SystemExit(f"[FAIL] fixedq9 equality failed for {name}")

    # Matched low-LR continuation: both arms reset optimizer/scheduler.
    state["optimizer"] = None
    state["scheduler"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, args.output)

    init_label = {
        "fixedq9": "copy_q10_exactly_to_q0_through_q9",
        "noq": "preserve_q_embed_matched_optimizer_reset",
    }[args.mode]

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "source": str(args.source),
        "source_sha256": source_hash,
        "source_iteration": int(state["it"]),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "initialization": init_label,
        "optimizer_reset": True,
        "scheduler_reset": True,
        "audited_tensors": audited,
        "contracts": {
            "q_none_maps_to": 10,
            "meanflow_unconditional_q": 10,
            "fixedq9_conditional_rows_use": 9 if args.mode == "fixedq9" else None,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_manifest, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
