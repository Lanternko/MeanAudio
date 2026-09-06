#!/usr/bin/env python3
"""Create and audit the eval-only neutral q9 copy.

Only row 9 of ``q_embed.weight`` is changed: q10 is copied to q9 in online
weights, both PostHocEMA tracks, and the synthesized EMA file.  Optimizer,
scheduler, iteration, and every non-target tensor are preserved exactly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(value: Any) -> str:
    if not torch.is_tensor(value):
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def nested_get(root: Mapping[str, Any], path: str) -> Any:
    if path.startswith("weights.") and isinstance(root.get("weights"), Mapping):
        flat_key = path[len("weights."):]
        if flat_key in root["weights"]:
            return root["weights"][flat_key]
    if path.startswith("ema.") and isinstance(root.get("ema"), Mapping):
        flat_key = path[len("ema."):]
        if flat_key in root["ema"]:
            return root["ema"][flat_key]
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def nested_set(root: Mapping[str, Any], path: str, value: Any) -> None:
    if path.startswith("weights.") and isinstance(root.get("weights"), Mapping):
        root["weights"][path[len("weights."):]] = value
        return
    if path.startswith("ema.") and isinstance(root.get("ema"), Mapping):
        root["ema"][path[len("ema."):]] = value
        return
    parts = path.split(".")
    current: Any = root
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def q_paths(state: Mapping[str, Any]) -> list[str]:
    paths = ["weights.q_embed.weight"]
    ema = state.get("ema")
    if not isinstance(ema, Mapping):
        raise ValueError("checkpoint lacks EMA state")
    paths.extend("ema." + key for key in sorted(ema) if key.endswith("q_embed.weight"))
    if len(paths) != 3:
        raise ValueError(f"expected online + two EMA q tensors, got {paths}")
    return paths


def copy_q10_to_q9(weight: torch.Tensor) -> dict[str, Any]:
    if tuple(weight.shape) != (11, 448):
        raise ValueError(f"unexpected q_embed shape: {tuple(weight.shape)}")
    before = tensor_digest(weight[9])
    q10 = weight[10].clone()
    weight[9].copy_(q10)
    if not torch.equal(weight[9], q10):
        raise RuntimeError("q9 does not exactly equal q10 after copy")
    return {
        "shape": list(weight.shape),
        "q9_sha256_before": before,
        "q9_sha256_after": tensor_digest(weight[9]),
        "q10_sha256": tensor_digest(weight[10]),
        "q9_equals_q10": True,
    }


def _flatten_tensors(value: Any, prefix: str = "") -> dict[str, str]:
    if torch.is_tensor(value):
        return {prefix: tensor_digest(value)}
    if isinstance(value, Mapping):
        output: dict[str, str] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_tensors(child, child_prefix))
        return output
    return {}


def invariant_report(before: Any, after: Any, target_paths: set[str]) -> dict[str, Any]:
    before_flat = _flatten_tensors(before)
    after_flat = _flatten_tensors(after)
    if set(before_flat) != set(after_flat):
        raise RuntimeError("tensor key set changed")
    changed = [path for path in before_flat if before_flat[path] != after_flat[path]]
    if set(changed) != target_paths:
        raise RuntimeError(f"unexpected tensor changes: {changed}; expected {sorted(target_paths)}")
    return {"tensor_count": len(before_flat), "changed_tensor_paths": sorted(changed)}


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def process_checkpoint(source: Path, output: Path, expected_it: int) -> dict[str, Any]:
    state = torch.load(source, map_location="cpu", weights_only=False)
    if state.get("it") != expected_it:
        raise ValueError(f"checkpoint it={state.get('it')}, expected={expected_it}")
    before = copy.deepcopy(state)
    audited: dict[str, Any] = {}
    paths = q_paths(state)
    for path in paths:
        weight = nested_get(state, path)
        audited[path] = copy_q10_to_q9(weight)
    invariant = invariant_report(before, state, set(paths))
    optimizer_scheduler_untouched = (
        _flatten_tensors(before.get("optimizer")) == _flatten_tensors(state.get("optimizer"))
        and _flatten_tensors(before.get("scheduler")) == _flatten_tensors(state.get("scheduler"))
    )
    if not optimizer_scheduler_untouched:
        raise RuntimeError("optimizer/scheduler changed in eval-only q-copy")
    atomic_torch_save(state, output)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "iteration": expected_it,
        "optimizer_scheduler_untouched": optimizer_scheduler_untouched,
        "audited": audited,
        "invariant": invariant,
    }


def process_ema(source: Path, output: Path) -> dict[str, Any]:
    state = torch.load(source, map_location="cpu", weights_only=False)
    before = copy.deepcopy(state)
    if "q_embed.weight" not in state:
        raise ValueError("EMA state lacks q_embed.weight")
    audited = {"q_embed.weight": copy_q10_to_q9(state["q_embed.weight"])}
    invariant = invariant_report(before, state, {"q_embed.weight"})
    atomic_torch_save(state, output)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "audited": audited,
        "invariant": invariant,
    }


def verify_existing(source_checkpoint: Path, source_ema: Path, output_checkpoint: Path, output_ema: Path, expected_it: int) -> dict[str, Any]:
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    output = torch.load(output_checkpoint, map_location="cpu", weights_only=False)
    if source.get("it") != expected_it or output.get("it") != expected_it:
        raise ValueError("neutral q-copy iteration drift")
    paths = set(q_paths(source))
    invariant_report(source, output, paths)
    for path in paths:
        if not torch.equal(nested_get(output, path)[9], nested_get(output, path)[10]):
            raise RuntimeError(f"existing checkpoint q9 != q10: {path}")
    source_ema_state = torch.load(source_ema, map_location="cpu", weights_only=False)
    output_ema_state = torch.load(output_ema, map_location="cpu", weights_only=False)
    invariant_report(source_ema_state, output_ema_state, {"q_embed.weight"})
    if not torch.equal(output_ema_state["q_embed.weight"][9], output_ema_state["q_embed.weight"][10]):
        raise RuntimeError("existing EMA q9 != q10")
    return {"status": "passed", "output_checkpoint_sha256": sha256_file(output_checkpoint), "output_ema_sha256": sha256_file(output_ema)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-ema", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-ema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-it", type=int, default=600_000)
    parser.add_argument("--source-checkpoint-sha256")
    parser.add_argument("--source-ema-sha256")
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_existing:
        if not all(path.is_file() for path in (args.output_checkpoint, args.output_ema, args.manifest)):
            raise SystemExit("[FAIL] neutral q-copy verification requires all existing outputs")
        print(json.dumps(verify_existing(args.source_checkpoint, args.source_ema, args.output_checkpoint, args.output_ema, args.expected_it), indent=2, sort_keys=True))
        return 0
    if any(path.exists() for path in (args.output_checkpoint, args.output_ema, args.manifest)):
        raise SystemExit("[FAIL] neutral q-copy is fresh-only; output exists")
    ckpt_hash = sha256_file(args.source_checkpoint)
    ema_hash = sha256_file(args.source_ema)
    if args.source_checkpoint_sha256 and ckpt_hash != args.source_checkpoint_sha256:
        raise SystemExit("[FAIL] source checkpoint hash mismatch")
    if args.source_ema_sha256 and ema_hash != args.source_ema_sha256:
        raise SystemExit("[FAIL] source EMA hash mismatch")
    checkpoint = process_checkpoint(args.source_checkpoint, args.output_checkpoint, args.expected_it)
    ema = process_ema(args.source_ema, args.output_ema)
    payload = {
        "schema_version": 1,
        "kind": "phase8_qwen_neutral_q9_eval_copy",
        "copy_rule": "q10_to_q9_only",
        "eval_only": True,
        "optimizer_reset": False,
        "scheduler_reset": False,
        "checkpoint": checkpoint,
        "ema_final": ema,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
