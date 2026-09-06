#!/usr/bin/env python3
"""CFG-3 single-negative content ablation on the registered MusicCaps-1024 subset."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/kojiek/MeanAudio")
CONTRACT = ROOT / "docs/experiments/single_negprompt_cfg3_ablation_20260831_contract.json"
OUT = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/single_negprompt_cfg3_ablation")
AUDIO_ROOT = OUT / "_audio"
SUBSET = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation/musiccaps_subset1024.tsv")
BASE = ROOT / "scripts/eval/negprompt_ablation_matrix.py"
EXPECTED = 1024
CFG = "3.0"
EXP_ID = "phase8_qwen_caption10s_multisent_noq_full_stage2_200000"
NEGATIVES = {
    "none": None,
    "low_quality": "low quality",
    "noisy": "noisy",
    "distorted": "distorted",
    "muffled": "muffled",
    "poor_fidelity": "poor fidelity",
    "hiss": "hiss",
    "lo_fi": "lo-fi",
    "amateur": "amateur",
    "genre": "genre",
    "fidelity_short": "low quality, noisy",
}
ORDER = tuple(NEGATIVES)
METRICS = ("clap", "CE", "CU", "PC", "PQ")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_base():
    spec = importlib.util.spec_from_file_location("negprompt_matrix_base", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("[FAIL] cannot load scoring implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPS = ROOT / "exps"
    module.OUT = OUT
    module.AUDIO_ROOT = AUDIO_ROOT
    module.SUBSET_TSV = SUBSET
    module.SUBSET_N = EXPECTED
    module.NEGATIVES = NEGATIVES
    module.ARMS = {"c2p0_slot0": (EXP_ID, ["--no_q"])}
    return module


def rows() -> list[dict[str, str]]:
    with SUBSET.open(encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle, delimiter="\t"))
    ids = [row["id"] for row in data]
    if len(data) != EXPECTED or len(set(ids)) != EXPECTED:
        raise SystemExit(f"[FAIL] subset identity rows={len(data)} unique={len(set(ids))}")
    return data


def label(key: str) -> str:
    return f"c2p0_slot0__cfg3.0__single_{key}"


def report(key: str) -> Path:
    return OUT / f"{label(key)}.json"


def audio_dir(key: str) -> Path:
    return AUDIO_ROOT / label(key)


def valid_report(path: Path, key: str, expected_ids: set[str]) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        per = value["per_clip"]
        full = value["aggregates"]["full"]
        return (
            value.get("label") == label(key)
            and value.get("exp_id") == EXP_ID
            and float(value.get("cfg_strength")) == 3.0
            and value.get("negative_key") == key
            and value.get("negative_prompt") == NEGATIVES[key]
            and value.get("protocol_id") == "musiccaps1024_mf25_cfg3_single_negative_seed42_nomask_fp32"
            and int(full.get("n")) == EXPECTED
            and set(per) == expected_ids
            and all(math.isfinite(float(full[name])) for name in METRICS)
            and all(all(math.isfinite(float(item[name])) for name in METRICS) for item in per.values())
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def clear_partial(directory: Path) -> None:
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise SystemExit(f"[FAIL] unsafe transient directory: {directory}")
    children = list(directory.iterdir())
    if any(item.is_symlink() or not item.is_file() or item.suffix != ".flac" for item in children):
        raise SystemExit(f"[FAIL] unexpected transient artifact: {directory}")
    for item in children:
        item.unlink()


def paired_delta(per: dict, baseline: dict) -> dict:
    shared = sorted(set(per) & set(baseline))
    result = {"n_paired": len(shared)}
    for metric in METRICS:
        delta = np.array([per[item][metric] - baseline[item][metric] for item in shared])
        result[metric] = {
            "mean_delta": float(delta.mean()),
            "sd": float(delta.std(ddof=1)),
            "frac_improved": float((delta > 0).mean()),
        }
    return result


def validate_all() -> None:
    expected_ids = {row["id"] for row in rows()}
    bad = [key for key in ORDER if not valid_report(report(key), key, expected_ids)]
    if bad:
        raise SystemExit(f"[FAIL] incomplete/invalid reports: {','.join(bad)}")
    print(json.dumps({"status": "passed", "reports": len(ORDER), "rows_each": EXPECTED}))


def main() -> int:
    if "--validate-only" in sys.argv[1:]:
        validate_all()
        return 0
    base = load_base()
    data = rows()
    expected_ids = {row["id"] for row in data}
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    for key in ORDER:
        path = report(key)
        if valid_report(path, key, expected_ids):
            print(f"[skip] {key}")
            continue
        directory = audio_dir(key)
        clear_partial(directory)
        base.generate(EXP_ID, ["--no_q"], CFG, key, directory)
        actual = {item.stem for item in directory.glob("*.flac")}
        if actual != expected_ids:
            raise SystemExit(f"[FAIL] generated identity mismatch for {key}: {len(actual)}/{EXPECTED}")
        signal = base.signal_stats(directory)
        per = base.score(data, directory)
        aggregates, lofi_ids = base.aggregate(per, data)
        baseline = per if key == "none" else json.loads(report("none").read_text())["per_clip"]
        payload = {
            "schema_version": 1,
            "protocol_id": "musiccaps1024_mf25_cfg3_single_negative_seed42_nomask_fp32",
            "label": label(key), "arm": "c2p0_slot0", "exp_id": EXP_ID,
            "cfg_strength": 3.0, "negative_key": key,
            "negative_prompt": NEGATIVES[key],
            "subset": {"tsv": str(SUBSET), "n": EXPECTED, "seed": 20260830},
            "signal_stats": signal, "aggregates": aggregates,
            "lofi_ids": lofi_ids, "paired_delta_vs_cfg3_none": paired_delta(per, baseline),
            "per_clip": per,
        }
        atomic_json(path, payload)
        if not valid_report(path, key, expected_ids):
            raise SystemExit(f"[FAIL] post-write validation failed: {path}")
        shutil.rmtree(directory)
        atomic_json(OUT / "resume_progress.json", {
            "document_kind": "single_negprompt_cfg3_resume_v1",
            "completed": [name for name in ORDER if report(name).is_file()],
        })
    validate_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
