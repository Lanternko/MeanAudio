#!/usr/bin/env python3
"""Paired per-caption instrument-conflict ablation with fixed assignments and blind audio retention."""

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
CONTRACT = ROOT / "docs/experiments/instrument_conflict_cfg3_20260908_contract.json"
OUT = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/instrument_conflict_20260908")
AUDIO_ROOT = OUT / "_audio"
SUBSET = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/instrument_conflict_20260908/fidelity8.tsv")
BASE = ROOT / "scripts/eval/negprompt_ablation_matrix.py"
EXPECTED = 3207
CFG = "3"
EXP_ID = "phase8_qwen_caption10s_multisent_noq_full_stage2_200000"
NEGATIVES = {'fidelity8': 'fidelity8', 'fidelity8_conflict': 'fidelity8 + per-caption mentioned instrument', 'fidelity8_unmentioned': 'fidelity8 + per-caption unmentioned instrument'}
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
    with tmp.open("rb") as handle: os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


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
            and value.get("protocol_id") == "musiccaps3207_mf25_cfg3_fidelity8_instrument_conflict_seed42_nomask_fp32"
            and int(full.get("n")) == EXPECTED
            and set(per) == expected_ids
            and set(value["generated_audio_sha256"]) == expected_ids
            and all(len(h) == 64 for h in value["generated_audio_sha256"].values())
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
    if summary.get("provenance") != provenance() or set(summary.get("comparisons", {})) != {"conflict_vs_baseline", "unmentioned_vs_baseline", "conflict_vs_unmentioned"}:
        raise ValueError("summary provenance/completeness mismatch")
    for key in ORDER:
        if summary["reports_sha256"][key] != digest(report(key)): raise ValueError("summary report binding mismatch")
    assignments = json.loads((OUT / "assignments.json").read_text())
    for item in assignments["listening_sample"]:
        for slot, key in item["slots"].items():
            if digest(OUT / "listening" / item["id"] / (slot + ".flac")) != json.loads(report(key).read_text())["generated_audio_sha256"][item["id"]]:
                raise ValueError("listening artifact mismatch")
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
            notify("cell_" + key + "_pass", "start", "Valid report persisted: " + str(path))
            retain_and_cleanup(key)
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
            "protocol_id": "musiccaps3207_mf25_cfg3_fidelity8_instrument_conflict_seed42_nomask_fp32",
            "label": label(key), "arm": "c2p0_slot0", "exp_id": EXP_ID,
            "cfg_strength": 3.0, "negative_key": key,
            "negative_prompt": NEGATIVES[key],
            "subset": {"tsv": str(SUBSET), "n": EXPECTED, "selection": "explicit_instrument_subset_original_order"},
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
        retain_and_cleanup(key)
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
    rc = subprocess.run(json.loads(CONTRACT.read_text())["commands"]["preflight"]).returncode
    if rc: raise SystemExit(rc)


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


def interval(values, seed=20260908):
    d = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.concatenate([d[rng.integers(0, len(d), (100, len(d)))].mean(axis=1) for _ in range(100)])
    return {"n": len(d), "mean_delta": float(d.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def retain_and_cleanup(key):
    spec = json.loads((OUT / "assignments.json").read_text())
    registered = json.loads(report(key).read_text())
    directory = audio_dir(key)
    for item in spec["listening_sample"]:
        slot = next(k for k,v in item["slots"].items() if v == key)
        destination = OUT / "listening" / item["id"] / (slot + ".flac")
        if destination.exists():
            if digest(destination) != registered["generated_audio_sha256"][item["id"]]:
                raise ValueError("retained listening audio hash mismatch")
        else:
            source = directory / (item["id"] + ".flac")
            if digest(source) != registered["generated_audio_sha256"][item["id"]]:
                raise ValueError("source listening audio hash mismatch")
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if digest(destination) != digest(source): raise ValueError("copy verification failed")
    clear_partial(directory)
    if directory.exists(): directory.rmdir()


def write_summary():
    assignments = json.loads((OUT / "assignments.json").read_text())
    data = assignments["assignments"]
    per = {key: json.loads(report(key).read_text())["per_clip"] for key in ORDER}
    groups = {"all": [a["id"] for a in data]}
    groups.update({t: [a["id"] for a in data if a["target"] == t] for t in sorted({a["target"] for a in data})})
    contrasts = {"conflict_vs_baseline": ("fidelity8_conflict", "fidelity8"),
                 "unmentioned_vs_baseline": ("fidelity8_unmentioned", "fidelity8"),
                 "conflict_vs_unmentioned": ("fidelity8_conflict", "fidelity8_unmentioned")}
    result = {"provenance": provenance(), "comparisons": {}, "decision": {},
              "control_frequency_exact_match": assignments["control_frequency_exact_match"],
              "target_counts": assignments["target_counts"], "unmentioned_counts": assignments["unmentioned_counts"]}
    for name, (left, right) in contrasts.items():
        result["comparisons"][name] = {}
        for group, ids in groups.items():
            result["comparisons"][name][group] = {metric: interval([per[left][i][metric]-per[right][i][metric] for i in ids]) for metric in METRICS}
        result["comparisons"][name]["macro_instrument_mean"] = {metric: float(np.mean([result["comparisons"][name][t][metric]["mean_delta"] for t in groups if t != "all"])) for metric in METRICS}
    primary = result["comparisons"]["conflict_vs_unmentioned"]["all"]["clap"]
    result["decision"] = {"semantic_interference_proxy": "supported" if primary["ci95"][1] < 0 else "not_demonstrated",
        "audible_instrument_suppression": "pending_blind_listening", "promotion": "none"}
    result["interpretation"] = "One preregistered automated contrast: original-caption CLAP conflict minus unmentioned (upper95<0). Other metrics/groups exploratory. Not proof of instrument removal. Unmentioned does not mean acoustically absent. Negative-word frequency mismatch limits specificity attribution; report per-target and macro averages."
    result["reports_sha256"] = {key: digest(report(key)) for key in ORDER}
    atomic_json(OUT / "summary.json", result)
    # Blind randomized slots; assignment key stays separate from this rating sheet.
    if (OUT / "listening" / "ratings.csv").exists(): return
    with (OUT / "listening" / "ratings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "target", "slot", "audio", "presence_0_absent_1_uncertain_2_present", "prominence_0_to_4", "quality_1_to_5"])
        writer.writeheader()
        for item in assignments["listening_sample"]:
            for slot in "ABC": writer.writerow({"id": item["id"], "target": item["target"], "slot": slot, "audio": str(OUT / "listening" / item["id"] / (slot + ".flac"))})


if __name__ == "__main__":
    raise SystemExit(main())
