#!/usr/bin/env python3
"""Fail-closed 50k/100k paired caption-dose queue after the Qwen 20k probe."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from analysis.phase8_qwen_dose_provenance import validate_nested_cache_provenance


REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "docs/experiments/phase8_qwen_dose_contract.json"
PARENT_CONTRACT = REPO / "docs/experiments/phase8_qwen_official_matched_contract.json"
PARENT_ROOT = Path("/home/kojiek/exps_nvme/phase8_qwen_official_matched")
RUN_ROOT = Path("/home/kojiek/exps_nvme/phase8_qwen_official_matched_dose")
STATE_DIR = Path("/home/kojiek/logs/phase8_qwen_dose_monitor")
EXECUTION_MANIFEST = STATE_DIR / "execution_manifest.json"
ATTEMPTS = STATE_DIR / "attempts.json"
LOCK = STATE_DIR / "queue.lock"
MUSICCAPS = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
EVALUATOR = Path("/home/kojiek/research/meanaudio_eval/phase4_eval.py")
CLAP = REPO / "weights/music_speech_audioset_epoch_15_esc_89.98.pt"
TORCHRUN = Path("/home/kojiek/venvs/dac/bin/torchrun")

ARMS = {
    "control": {
        "tsv": Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train.tsv"),
        "npz": Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz"),
        "cache": Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt"),
        "cache_manifest": Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz/FULL_GATE_PASSED.json"),
    },
    "qwen": {
        "tsv": Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched.tsv"),
        "npz": Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"),
        "cache": Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt"),
        "cache_manifest": Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_manifest.json"),
    },
}
MILESTONES = (("50k", 620_000, 650_000), ("100k", 650_000, 700_000))
IMPLEMENTATION_RELATIVE_PATHS = (
    "docs/experiments/phase8_qwen_dose_contract.json",
    "docs/experiments/phase8_qwen_dose_luna_prompt.md",
    "docs/experiments/phase8_qwen_dose_sol_approval_prompt.md",
    "docs/experiments/phase8_qwen_dose_sol_incident_prompt.md",
    "scripts/phase8_qwen_dose_queue.py",
    "scripts/analysis/phase8_qwen_dose_audit.py",
    "scripts/analysis/phase8_qwen_dose_paired_report.py",
    "scripts/analysis/phase8_qwen_dose_provenance.py",
    "scripts/analysis/phase8_q_clap_bootstrap.py",
    "scripts/phase8_qwen_dose_monitor.py",
    "scripts/phase8_qwen_dose_approval.schema.json",
    "scripts/phase8_qwen_sol_verdict.schema.json",
    "scripts/phase8_qwen_parent_completion_gate.py",
    "scripts/phase8_qwen_dose_luna_loop.sh",
    "scripts/schedule_phase8_qwen_dose_after_20k.sh",
    "scripts/tests/selftest_phase8_qwen_dose.py",
)


def py() -> str:
    return sys.executable


def command(*parts: object) -> list[str]:
    return [py(), *[str(part) for part in parts]]


def display(args: Sequence[str]) -> str:
    return shlex.join([str(value) for value in args])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hashes() -> dict[str, str]:
    return {relative: sha256_file(REPO / relative) for relative in IMPLEMENTATION_RELATIVE_PATHS}


def tsv_id_sha256(path: Path) -> tuple[int, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        ids = [str(row["id"]) for row in csv.DictReader(handle, delimiter="\t")]
    return len(ids), hashlib.sha256("\n".join(ids).encode()).hexdigest()


def validate_data_provenance() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"arms": {}}
    for arm, paths in ARMS.items():
        declared = contract["arms"][arm]
        expected_paths = {
            "tsv": paths["tsv"], "npz_dir": paths["npz"],
            "cache_list": paths["cache"], "cache_manifest": paths["cache_manifest"],
        }
        drift = {
            key: (declared.get(key), str(value))
            for key, value in expected_paths.items()
            if declared.get(key) != str(value)
        }
        if drift:
            raise RuntimeError(f"{arm} declared data path drift: {drift}")
        current_hashes = {
            "tsv_sha256": sha256_file(paths["tsv"]),
            "cache_list_sha256": sha256_file(paths["cache"]),
            "cache_manifest_sha256": sha256_file(paths["cache_manifest"]),
        }
        hash_drift = {
            key: (declared.get(key), value)
            for key, value in current_hashes.items()
            if declared.get(key) != value
        }
        rows, id_hash = tsv_id_sha256(paths["tsv"])
        if hash_drift or declared.get("tsv_id_sha256") != id_hash or rows != 251_599:
            raise RuntimeError(
                f"{arm} data provenance drift: hashes={hash_drift}, rows={rows}, id_sha256={id_hash}"
            )
        manifest = json.loads(paths["cache_manifest"].read_text(encoding="utf-8"))
        if manifest.get("status") != "passed":
            raise RuntimeError(f"{arm} cache manifest/gate is not passed")
        nested = validate_nested_cache_provenance(arm, paths["npz"], manifest)
        if arm == "qwen":
            semantic = {
                "tsv": str(paths["tsv"]),
                "tsv_sha256": current_hashes["tsv_sha256"],
                "cache_list": str(paths["cache"]),
                "cache_list_sha256": current_hashes["cache_list_sha256"],
                "output_dir": str(paths["npz"]),
                "completed_rows": 251_599,
                "planned_rows": 251_599,
            }
            semantic_drift = {
                key: (manifest.get(key), value)
                for key, value in semantic.items()
                if manifest.get(key) != value
            }
            if semantic_drift:
                raise RuntimeError(f"qwen NPZ manifest semantic drift: {semantic_drift}")
        result["arms"][arm] = {
            **current_hashes, **nested, "rows": rows, "tsv_id_sha256": id_hash,
        }
    evaluation = contract["evaluation"]
    if evaluation.get("tsv") != str(MUSICCAPS) or evaluation.get("tsv_sha256") != sha256_file(MUSICCAPS):
        raise RuntimeError("MusicCaps evaluation TSV provenance drift")
    eval_rows, eval_id_hash = tsv_id_sha256(MUSICCAPS)
    if eval_rows != 5521 or evaluation.get("tsv_id_sha256") != eval_id_hash:
        raise RuntimeError("MusicCaps evaluation ID order drift")
    result["evaluation"] = {"rows": eval_rows, "tsv_id_sha256": eval_id_hash}
    return result


def json_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"
    except Exception:
        return False


def checkpoint_iteration(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    value = state.get("it")
    if not isinstance(value, int):
        raise RuntimeError(f"checkpoint has no integer iteration: {path}")
    return value


def exp_id(label: str, arm: str) -> str:
    return f"phase8_qwen_dose_{arm}_{label}"


def milestone_root(label: str) -> Path:
    return RUN_ROOT / label


def run_dir(label: str, arm: str) -> Path:
    return milestone_root(label) / arm


def checkpoint_path(label: str, arm: str) -> Path:
    eid = exp_id(label, arm)
    return run_dir(label, arm) / f"{eid}_ckpt_last.pth"


def ema_path(label: str, arm: str) -> Path:
    eid = exp_id(label, arm)
    return run_dir(label, arm) / f"{eid}_ema_final.pth"


def metrics_path(label: str, arm: str) -> Path:
    eid = exp_id(label, arm)
    return run_dir(label, arm) / "musiccaps_metrics" / eid / "metrics.txt"


def final_audit_path(label: str, arm: str) -> Path:
    return run_dir(label, arm) / "final_audit.json"


def paired_report_path(label: str) -> Path:
    return milestone_root(label) / "paired_final_report.json"


def source_checkpoint(label: str, arm: str) -> Path:
    if label == "50k":
        parent_id = f"phase8_qwen_official_matched_{arm}_20k"
        return PARENT_ROOT / arm / f"{parent_id}_ckpt_last.pth"
    return checkpoint_path("50k", arm)


def source_audit(label: str, arm: str) -> Path:
    if label == "50k":
        return PARENT_ROOT / arm / "final_audit.json"
    return final_audit_path("50k", arm)


def active_duplicates() -> list[str]:
    patterns = (
        "torchrun.*phase8_qwen_(official_matched|dose)",
        "train.py.*phase8_qwen_(official_matched|dose)",
    )
    own = str(os.getpid())
    lines: list[str] = []
    for pattern in patterns:
        result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, check=False)
        lines.extend(
            line for line in result.stdout.splitlines()
            if own not in line and "pgrep -af" not in line
        )
    return sorted(set(lines))


def verify_authorization(path: Path | None, run_mode: str) -> None:
    if path is None or not path.is_file():
        raise RuntimeError("execution requires Codex/Sol dose-chain authorization")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "approved",
        "codex_reviewed": True,
        "contract_sha256": sha256_file(CONTRACT),
        "run_mode": run_mode,
        "implementation_sha256": implementation_hashes(),
    }
    drift = {key: (payload.get(key), value) for key, value in required.items() if payload.get(key) != value}
    sol_path = Path(str(payload.get("sol_verdict_path", "")))
    if drift or not sol_path.is_file() or payload.get("sol_verdict_sha256") != sha256_file(sol_path):
        raise RuntimeError(f"invalid dose-chain authorization: drift={drift}")
    verdict = json.loads(sol_path.read_text(encoding="utf-8"))
    expected_verdict = "approve_predeclared_dose_chain" if run_mode == "fresh" else "resume_identical_contract"
    if verdict.get("verdict") != expected_verdict:
        raise RuntimeError(f"Sol verdict does not authorize {run_mode}: expected {expected_verdict}")
    if run_mode == "resume" and payload.get("same_prefix_and_contract") is not True:
        raise RuntimeError("resume authorization must bind the identical prefix and contract")


def preflight(strict: bool) -> dict[str, Any]:
    required_paths = [CONTRACT, PARENT_CONTRACT, MUSICCAPS, EVALUATOR, CLAP, TORCHRUN]
    for arm in ARMS.values():
        required_paths.extend((arm["tsv"], arm["npz"], arm["cache"], arm["cache_manifest"]))
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError("missing fixed input(s): " + ", ".join(missing))
    data_provenance = validate_data_provenance()
    predecessor = [
        PARENT_ROOT / "paired_final_report.json",
        PARENT_ROOT / "control/final_audit.json",
        PARENT_ROOT / "qwen/final_audit.json",
    ]
    failed = [str(path) for path in predecessor if not json_passed(path)]
    if failed:
        raise RuntimeError("20k predecessor gate is not passed: " + ", ".join(failed))
    parent_execution = STATE_DIR.parent / "phase8_qwen_official_matched_monitor/execution_manifest.json"
    try:
        parent_manifest = json.loads(parent_execution.read_text(encoding="utf-8"))
        parent_final = parent_manifest.get("steps", {}).get("paired_final_report", {})
        if parent_final.get("status") != "passed" or parent_final.get("exit_code") != 0:
            raise RuntimeError("parent final execution step is not passed")
    except Exception as exc:
        raise RuntimeError(f"20k parent execution completion gate failed: {exc!r}") from exc
    source_iterations = {
        arm: checkpoint_iteration(source_checkpoint("50k", arm)) for arm in ARMS
    }
    if set(source_iterations.values()) != {620_000}:
        raise RuntimeError(f"20k source iteration drift: {source_iterations}")
    source_checkpoint_sha256: dict[str, str] = {}
    for arm in ARMS:
        path = source_checkpoint("50k", arm)
        state = torch.load(path, map_location="cpu", weights_only=False)
        missing_state = sorted({"weights", "ema", "optimizer", "scheduler"} - set(state))
        del state
        if missing_state:
            raise RuntimeError(f"20k source checkpoint missing state for {arm}: {missing_state}")
        source_checkpoint_sha256[arm] = sha256_file(path)
    parent_contract = json.loads(PARENT_CONTRACT.read_text(encoding="utf-8"))
    runtime_drift = [
        relative for relative, expected in parent_contract["runtime_code_sha256"].items()
        if not (REPO / relative).is_file() or sha256_file(REPO / relative) != expected
    ]
    if runtime_drift:
        raise RuntimeError("scientific runtime code drift: " + ", ".join(runtime_drift))
    duplicates = active_duplicates()
    if duplicates:
        raise RuntimeError("duplicate/active experiment process: " + " | ".join(duplicates))
    root_free = shutil.disk_usage(Path("/")).free
    hdd_free = shutil.disk_usage(Path("/mnt/HDD")).free
    if root_free < 50 * 1024**3 or hdd_free < 50 * 1024**3:
        raise RuntimeError(f"free-space floor violated: root={root_free}, hdd={hdd_free}")
    if strict and shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi unavailable")
    return {
        "status": "passed",
        "contract_sha256": sha256_file(CONTRACT),
        "parent_contract_sha256": sha256_file(PARENT_CONTRACT),
        "source_iterations": source_iterations,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "implementation_sha256": implementation_hashes(),
        "data_provenance": data_provenance,
        "root_free_bytes": root_free,
        "hdd_free_bytes": hdd_free,
        "duplicate_processes": [],
        "metric_thresholds_are_not_gates": True,
    }


def train_args(label: str, arm: str, target: int) -> list[str]:
    cfg = ARMS[arm]
    eid = exp_id(label, arm)
    out = run_dir(label, arm)
    source = source_checkpoint(label, arm)
    return [
        str(TORCHRUN), "--standalone", "--nproc_per_node=1", str(REPO / "train.py"),
        "data=meanaudio", "model=meanaudio_s", f"exp_id={eid}",
        f"num_iterations={target}", "lr_schedule=step", "lr_schedule_steps=[999999,999999]",
        "batch_size=8", "learning_rate=1e-5", "linear_warmup_steps=1000", "seed=14159265",
        "num_workers=4", "save_weights_interval=10000", "save_checkpoint_interval=10000",
        "val_interval=999999", "eval_interval=999999", "save_eval_interval=999999",
        "+accumulation_steps=1", "+use_rope=False", "+use_q_conditioning=false",
        "+use_text_attention_mask=false", "+use_wandb=false", "++multi_cap=false",
        f"hydra.run.dir={out}", f"++data.AudioCaps_npz.tsv={cfg['tsv']}",
        f"++data.AudioCaps_npz.npz_dir={cfg['npz']}",
        f"++data.AudioCaps_npz.gt_cache={cfg['cache']}",
        f"++data.AudioCaps_val_npz.tsv={cfg['tsv']}",
        f"++data.AudioCaps_val_npz.npz_dir={cfg['npz']}",
        f"++data.AudioCaps_val_npz.gt_cache={cfg['cache']}",
        f"checkpoint={source}",
    ]


def eval_args(label: str, arm: str) -> list[str]:
    return command(
        REPO / "eval.py", "--variant", "meanaudio_s", "--model_path", ema_path(label, arm),
        "--output", run_dir(label, arm) / "musiccaps_eval/audio", "--tsv", MUSICCAPS,
        "--use_meanflow", "--num_steps", "1", "--encoder_name", "t5_clap", "--text_c_dim", "512",
        "--cfg_strength", "0.5", "--no_q", "--full_precision", "--no_text_attention_mask",
    )


def metric_args(label: str, arm: str) -> list[str]:
    eid = exp_id(label, arm)
    return command(
        EVALUATOR, "--gen_dir", run_dir(label, arm) / "musiccaps_eval/audio", "--tsv", MUSICCAPS,
        "--exp_name", eid, "--out_dir", run_dir(label, arm) / "musiccaps_metrics", "--num_samples", "5521",
    )


def audit_args(label: str, arm: str, source_it: int, target: int) -> list[str]:
    eid = exp_id(label, arm)
    return command(
        REPO / "scripts/analysis/phase8_qwen_dose_audit.py",
        "--arm", arm, "--exp-id", eid, "--run-dir", run_dir(label, arm),
        "--source-checkpoint", source_checkpoint(label, arm), "--source-iteration", source_it,
        "--expected-iteration", target, "--train-step", f"{label}_{arm}_train",
        "--eval-step", f"{label}_{arm}_eval", "--metrics-step", f"{label}_{arm}_metrics",
        "--metrics", metrics_path(label, arm), "--audio-tsv", MUSICCAPS,
        "--audio-dir", run_dir(label, arm) / "musiccaps_eval/audio", "--contract", CONTRACT,
        "--execution-manifest", EXECUTION_MANIFEST, "--source-audit", source_audit(label, arm),
        "--cache-manifest", ARMS[arm]["cache_manifest"], "--json-out", final_audit_path(label, arm),
    )


def paired_steps(label: str, target: int) -> list[tuple[str, list[str]]]:
    root = milestone_root(label)
    return [
        (
            f"{label}_paired_clap",
            command(
                REPO / "scripts/analysis/phase8_q_clap_bootstrap.py", "--tsv", MUSICCAPS,
                "--baseline-dir", run_dir(label, "control") / "musiccaps_eval/audio",
                "--treatment-dir", run_dir(label, "qwen") / "musiccaps_eval/audio",
                "--clap-checkpoint", CLAP, "--json-out", root / "paired_per_prompt_clap.json",
            ),
        ),
        (
            f"{label}_paired_report",
            command(
                REPO / "scripts/analysis/phase8_qwen_dose_paired_report.py",
                "--control-metrics", metrics_path(label, "control"),
                "--qwen-metrics", metrics_path(label, "qwen"),
                "--control-audit", final_audit_path(label, "control"),
                "--qwen-audit", final_audit_path(label, "qwen"),
                "--contract", CONTRACT, "--paired-bootstrap", root / "paired_per_prompt_clap.json",
                "--iteration", target, "--json-out", paired_report_path(label),
            ),
        ),
    ]


def build_queue(run_mode: str) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []
    for label, source_it, target in MILESTONES:
        if run_mode == "resume" and json_passed(paired_report_path(label)):
            continue
        for arm in ARMS:
            audit_path = final_audit_path(label, arm)
            if run_mode == "resume" and json_passed(audit_path):
                continue
            ckpt = checkpoint_path(label, arm)
            ema = ema_path(label, arm)
            if ckpt.exists():
                if checkpoint_iteration(ckpt) != target or not ema.is_file():
                    raise RuntimeError(f"unsafe partial final artifact for {label}/{arm}")
            else:
                steps.append((f"{label}_{arm}_train", train_args(label, arm, target)))
            metric_file = metrics_path(label, arm)
            audio_dir = run_dir(label, arm) / "musiccaps_eval/audio"
            if not metric_file.is_file():
                if run_mode == "resume" and audio_dir.exists():
                    raise RuntimeError(f"unsafe partial eval for {label}/{arm}; archive before audited resume")
                steps.append((f"{label}_{arm}_eval", eval_args(label, arm)))
                steps.append((f"{label}_{arm}_metrics", metric_args(label, arm)))
            steps.append((f"{label}_{arm}_audit", audit_args(label, arm, source_it, target)))
        steps.extend(paired_steps(label, target))
    return steps


def load_manifest(run_mode: str, preflight_report: Mapping[str, Any]) -> dict[str, Any]:
    if run_mode == "fresh":
        return {
            "schema_version": 1,
            "contract_sha256": sha256_file(CONTRACT),
            "implementation_sha256": implementation_hashes(),
            "preflight": dict(preflight_report),
            "steps": {},
        }
    if not EXECUTION_MANIFEST.is_file():
        raise RuntimeError("resume requires prior execution manifest")
    payload = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("contract_sha256") != sha256_file(CONTRACT):
        raise RuntimeError("execution manifest contract drift")
    if payload.get("implementation_sha256") != implementation_hashes():
        raise RuntimeError("execution manifest implementation drift")
    return payload


def write_manifest(payload: Mapping[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = EXECUTION_MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, EXECUTION_MANIFEST)


def verify_train_source(name: str, args: Sequence[str]) -> None:
    if not name.endswith("_train"):
        return
    checkpoint_arg = next((value for value in args if value.startswith("checkpoint=")), None)
    if checkpoint_arg is None:
        raise RuntimeError(f"train step has no source checkpoint: {name}")
    source = Path(checkpoint_arg.split("=", 1)[1])
    label = name.split("_", 1)[0]
    expected = 620_000 if label == "50k" else 650_000
    if checkpoint_iteration(source) != expected:
        raise RuntimeError(f"source gate failed for {name}: expected it={expected}")
    arm = "control" if "_control_" in name else "qwen"
    if not json_passed(source_audit(label, arm)):
        raise RuntimeError(f"source audit gate failed for {name}")
    if label == "100k" and not json_passed(paired_report_path("50k")):
        raise RuntimeError("100k launch blocked: 50k paired report not passed")


def execute(
    steps: Sequence[tuple[str, list[str]]], run_mode: str, preflight_report: Mapping[str, Any]
) -> None:
    manifest = load_manifest(run_mode, preflight_report)
    for name, args in steps:
        verify_train_source(name, args)
        print(f"[RUN] {name}: {display(args)}", flush=True)
        manifest["steps"][name] = {
            "command": list(args), "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        write_manifest(manifest)
        result = subprocess.run(args, cwd=REPO, check=False)
        step = manifest["steps"][name]
        step.update({
            "exit_code": result.returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if result.returncode == 0 else "failed",
        })
        write_manifest(manifest)
        if result.returncode != 0:
            raise RuntimeError(f"queue stopped at {name} with exit={result.returncode}")


def record_attempt(run_mode: str) -> None:
    payload = {"fresh": 0, "resume": 0}
    if ATTEMPTS.is_file():
        payload.update(json.loads(ATTEMPTS.read_text(encoding="utf-8")))
    if run_mode == "resume" and int(payload.get("resume", 0)) >= 2:
        raise RuntimeError("dose-chain resume limit reached")
    payload[run_mode] = int(payload.get(run_mode, 0)) + 1
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = ATTEMPTS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, ATTEMPTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--run-mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.plan_only:
        steps = build_queue(args.run_mode)
        print(json.dumps({"status": "plan_only", "contract_sha256": sha256_file(CONTRACT)}, indent=2))
    else:
        if args.execute:
            verify_authorization(args.authorization, args.run_mode)
        report = preflight(strict=args.execute)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run_mode == "fresh" and (RUN_ROOT.exists() or EXECUTION_MANIFEST.exists()):
            raise RuntimeError("fresh mode refuses existing dose outputs")
        steps = build_queue(args.run_mode)
    print("\nEXACT QUEUE")
    for index, (name, command_args) in enumerate(steps, start=1):
        print(f"{index:02d}. {name}: {display(command_args)}")
    if args.execute:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        lock_handle = LOCK.open("w")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("dose queue lock is held") from exc
        record_attempt(args.run_mode)
        execute(steps, args.run_mode, report)
    else:
        print("\nNO COMMAND STARTED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
