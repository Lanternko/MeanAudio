#!/usr/bin/env python3
"""Prepare a 600k -> 620k continuation checkpoint with fresh opt/scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value: Any, path: Path) -> None:
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


def tensor_fingerprint(value: Any, prefix: str = "") -> dict[str, str]:
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        return {prefix: hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, child in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            result.update(tensor_fingerprint(child, label))
        return result
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-it", type=int, default=600_000)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        raise SystemExit("[FAIL] fresh continuation initializer refuses existing output")
    source_hash = sha256_file(args.source)
    if args.source_sha256 and source_hash != args.source_sha256:
        raise SystemExit("[FAIL] source checkpoint hash mismatch")
    state = torch.load(args.source, map_location="cpu", weights_only=False)
    if state.get("it") != args.expected_it:
        raise SystemExit(f"[FAIL] source it={state.get('it')}, expected={args.expected_it}")
    for key in ("weights", "ema"):
        if key not in state or state[key] is None:
            raise SystemExit(f"[FAIL] source checkpoint lacks {key}")
    weights_before = tensor_fingerprint(state["weights"])
    ema_before = tensor_fingerprint(state["ema"])
    state["optimizer"] = None
    state["scheduler"] = None
    atomic_save(state, args.output)
    payload = {
        "schema_version": 1,
        "kind": "phase8_qwen_matched_20k_fresh_continuation",
        "source": str(args.source),
        "source_sha256": source_hash,
        "source_iteration": args.expected_it,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "weights_preserved": weights_before == tensor_fingerprint(state["weights"]),
        "ema_preserved": ema_before == tensor_fingerprint(state["ema"]),
        "optimizer_reset": state["optimizer"] is None,
        "scheduler_reset": state["scheduler"] is None,
        "final_iteration": 620_000,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
