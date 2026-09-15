#!/usr/bin/env python3
"""Five-arm fidelity8 vocal-negative ablation on MusicCaps-5521 with fixed caption strata."""

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
CONTRACT = ROOT / "docs/experiments/vocal_negative_cfg3_20260908_contract.json"
OUT = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/vocal_negative_20260908")
AUDIO_ROOT = OUT / "_audio"
SUBSET = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/vocal_negative_20260908/musiccaps5521.tsv")
BASE = ROOT / "scripts/eval/negprompt_ablation_matrix.py"
EXPECTED = 5521
CFG = "3"
EXP_ID = "phase8_qwen_caption10s_multisent_noq_full_stage2_200000"
NEGATIVES = {'fidelity8': 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi', 'fidelity8_vocals': 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi, vocals', 'fidelity8_singing': 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi, singing', 'fidelity8_choir': 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi, choir', 'fidelity8_all': 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi, vocals, singing, choir'}
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
    return f"c2p0_slot0__cfg3.0__{key}"


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
            and value.get("provenance") == provenance()
            and float(value.get("cfg_strength")) == 3.0
            and value.get("negative_key") == key
            and value.get("negative_prompt") == NEGATIVES[key]
            and value.get("protocol_id") == "musiccaps5521_mf25_cfg3_fidelity8_vocal_ablation_seed42_nomask_fp32"
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
    summary = json.loads((OUT / "summary.json").read_text())
    if summary.get("provenance") != provenance() or set(summary.get("comparisons", {})) != set(ORDER[1:]):
        raise ValueError("summary provenance/completeness mismatch")
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
        preflight()
        notify("cell_" + key + "_preflight_pass", "start", "Preflight passed; generating " + key)
        import subprocess
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        argv = json.loads(CONTRACT.read_text())["commands_generation"][key]
        with (OUT / (key + "_generation.log")).open("a") as log:
            subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = {item.stem for item in directory.glob("*.flac")}
        if actual != expected_ids:
            raise SystemExit(f"[FAIL] generated identity mismatch for {key}: {len(actual)}/{EXPECTED}")
        validate_audio(directory, expected_ids)
        per_signal = base.per_clip_signal(directory)
        signal = base.signal_stats(per_signal, sample=EXPECTED)
        per = base.score(data, directory)
        aggregates, lofi_ids = base.aggregate(per, data)
        baseline = per if key == "fidelity8" else json.loads(report("fidelity8").read_text())["per_clip"]
        payload = {
            "schema_version": 1,
            "protocol_id": "musiccaps5521_mf25_cfg3_fidelity8_vocal_ablation_seed42_nomask_fp32",
            "label": label(key), "arm": "c2p0_slot0", "exp_id": EXP_ID,
            "cfg_strength": 3.0, "negative_key": key,
            "negative_prompt": NEGATIVES[key],
            "subset": {"tsv": str(SUBSET), "n": EXPECTED, "selection": "all_rows_original_order"},
            "signal_stats": signal, "per_clip_signal": per_signal,
            "provenance": provenance(), "aggregates": aggregates,
            "lofi_ids": lofi_ids, "paired_delta_vs_cfg3_fidelity8": paired_delta(per, baseline),
            "per_clip": per,
            "generated_audio_sha256": {p.stem: digest(p) for p in sorted(directory.glob("*.flac"))},
        }
        atomic_json(path, payload)
        if not valid_report(path, key, expected_ids):
            raise SystemExit(f"[FAIL] post-write validation failed: {path}")
        notify("cell_" + key + "_pass", "start", "Valid report persisted: " + str(path))
        clear_partial(directory)
        directory.rmdir()
        atomic_json(OUT / "resume_progress.json", {
            "document_kind": "single_negprompt_cfg3_resume_v1",
            "completed": [name for name in ORDER if report(name).is_file()],
        })
    write_summary()
    validate_all()
    notify("decision_report", "start", "Analysis complete; no automatic promotion. " + str(OUT / "summary.json") + " " + json.dumps(json.loads((OUT / "summary.json").read_text())["decision"]))
    return 0


def provenance():
    spec = json.loads(CONTRACT.read_text())
    return {"contract_sha256": digest(CONTRACT), "inputs": spec["inputs"]}


def preflight():
    import subprocess
    subprocess.run(json.loads(CONTRACT.read_text())["commands"]["preflight"], check=True)


def notify(event, status, summary):
    sys.path.insert(0, str(ROOT / "scripts/experiment_harness"))
    from notification_receipts import deliver_required
    spec = json.loads(CONTRACT.read_text())
    deliver_required(contract_path=CONTRACT, launcher_path=Path(os.environ["GPU_QUEUE_JOB_SCRIPT"]),
        event=event, status=status, summary=summary,
        idempotency_key=f"{spec['experiment_id']}:{spec['run_id']}:{event}",
        notifier=ROOT / "scripts/notify_experiment_webhook.py",
        python=Path("/home/kojiek/venvs/dac/bin/python"),
        root=Path(spec["notification_receipts"]["root"]))


def validate_audio(directory, ids):
    import soundfile as sf
    if {p.stem for p in directory.glob("*.flac")} != ids:
        raise ValueError("audio IDs mismatch")
    for p in directory.glob("*.flac"):
        wav, sr = sf.read(p, always_2d=True)
        if sr != 16000 or wav.shape[1] != 1 or not len(wav) or not np.isfinite(wav).all():
            raise ValueError(f"invalid audio: {p}")


def write_summary():
    groups = json.loads((OUT / "subsets.json").read_text())["groups"]
    baseline = json.loads(report("fidelity8").read_text())["per_clip"]
    result = {"provenance": provenance(), "comparisons": {}, "decision": {}}
    for key in ORDER[1:]:
        per = json.loads(report(key).read_text())["per_clip"]
        result["comparisons"][key] = {}
        for group, ids in groups.items():
            metrics = {}
            for metric in METRICS:
                d = np.array([per[i][metric] - baseline[i][metric] for i in ids])
                rng = np.random.default_rng(20260908)
                boot = np.concatenate([d[rng.integers(0, len(d), (100, len(d)))].mean(axis=1) for _ in range(100)])
                metrics[metric] = {"mean_delta": float(d.mean()),
                    "ci95": np.quantile(boot, [.025, .975]).tolist(),
                    "ci98_75": np.quantile(boot, [.00625, .99375]).tolist()}
            result["comparisons"][key][group] = {"n":len(ids), "metrics":metrics}
        m = result["comparisons"][key]["caption_no_vocal"]["metrics"]
        result["decision"][key] = "metric_improvement" if m["PQ"]["ci98_75"][0] > 0 and m["clap"]["ci98_75"][0] > -.01 else "not_demonstrated"
    result["interpretation"] = "Caption-defined subgroup; not verified vocal absence. PQ primary, CLAP noninferiority margin 0.01; four comparisons Bonferroni 98.75% bootstrap intervals. Other scores exploratory. No claim of audible vocal removal or automatic promotion."
    atomic_json(OUT / "summary.json", result)

if __name__ == "__main__":
    raise SystemExit(main())
