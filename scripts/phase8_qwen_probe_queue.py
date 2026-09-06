#!/usr/bin/env python3
"""Fail-closed queue for the official-Qwen matched 20k probe.

The default action is only to print the exact queue.  Execution requires both
``--execute`` and an explicit ``--run-mode fresh|resume``.
"""

from __future__ import annotations

import argparse
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


REPO = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/HDD/kojiek/phase4_jamendo_data")
BASE_NPZ = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz")
BASE_TSV = DATA / "phase8_legacy_catalog_train.tsv"
BASE_CACHE = DATA / "npz_cache_train.txt"
QWEN_JSON = Path(
    "/home/kojiek/reference-repos/ICME26-ATTM-GC-FluxAudio/data/captions/jamendo_qwen.json"
)
QWEN_TSV = DATA / "phase8_qwen_official_matched.tsv"
QWEN_CACHE = DATA / "phase8_qwen_official_matched_npz_cache_train.txt"
QWEN_METADATA_MANIFEST = DATA / "phase8_qwen_official_matched_manifest.json"
QWEN_NPZ = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz")
QWEN_NPZ_MANIFEST = DATA / "phase8_qwen_official_matched_npz_manifest.json"
QWEN_PROBE_NPZ = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz_probe_512")
QWEN_PROBE_MANIFEST = DATA / "phase8_qwen_official_matched_probe_512_manifest.json"
CLAP = Path("/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt")
SOURCE_CKPT = Path(
    "/home/kojiek/exps_nvme/phase8_catalog_matched_noq_stage2_200000/phase8_catalog_matched_noq_stage2_200000_ckpt_last.pth"
)
SOURCE_EMA = Path(
    "/home/kojiek/exps_nvme/phase8_catalog_matched_noq_stage2_200000/phase8_catalog_matched_noq_stage2_200000_ema_final.pth"
)
MUSICCAPS = DATA / "musiccaps_test.tsv"
EVALUATOR = Path("/home/kojiek/research/meanaudio_eval/phase4_eval.py")
RUN_ROOT = Path("/home/kojiek/exps_nvme/phase8_qwen_official_matched")
CONTROL_ID = "phase8_qwen_official_matched_control_20k"
QWEN_ID = "phase8_qwen_official_matched_qwen_20k"
CONTRACT = REPO / "docs/experiments/phase8_qwen_official_matched_contract.json"
TORCHRUN = Path("/home/kojiek/venvs/dac/bin/torchrun")
LOCK_PATH = Path("/home/kojiek/logs/phase8_qwen_official_matched_queue.lock")
ATTEMPT_STATE = Path("/home/kojiek/logs/phase8_qwen_official_matched_monitor/attempts.json")
EXECUTION_MANIFEST = Path("/home/kojiek/logs/phase8_qwen_official_matched_monitor/execution_manifest.json")


def py() -> str:
    return sys.executable


def command(*parts: object) -> list[str]:
    return [py(), *[str(part) for part in parts]]


def run_display(args: Sequence[str]) -> str:
    return shlex.join([str(arg) for arg in args])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_inputs() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))["inputs"]


def active_duplicates() -> list[str]:
    pattern = "torchrun.*phase8_qwen_official_matched_(control|qwen)_20k"
    result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, check=False)
    own = str(os.getpid())
    return [line for line in result.stdout.splitlines() if own not in line and "pgrep -af" not in line]


def reject_duplicates(lines: Sequence[str]) -> None:
    if lines:
        raise RuntimeError("duplicate/active probe process detected: " + " | ".join(lines))


def require_passed(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("status") != "passed":
        raise RuntimeError(f"downstream gate blocked by {label}: {payload.get('status')!r}")


def json_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"
    except Exception:
        return False


def execution_step_passed(name: str) -> bool:
    """Return true only for a durably recorded successful prior queue step."""
    if not EXECUTION_MANIFEST.is_file():
        return False
    try:
        manifest = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
        step = manifest.get("steps", {}).get(name, {})
        return step.get("status") == "passed" and step.get("exit_code") == 0
    except Exception:
        return False


def metrics_file_path(run_dir: Path, exp_id: str) -> Path:
    """Mirror phase4_eval.py's ``out_dir / exp_name / metrics.txt`` layout."""
    return run_dir / "musiccaps_metrics" / exp_id / "metrics.txt"


def checkpoint_iteration(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    value = state.get("it")
    if not isinstance(value, int):
        raise RuntimeError(f"checkpoint has no integer iteration: {path}")
    return value


def record_execute_attempt(run_mode: str) -> None:
    payload = {"fresh": 0, "resume": 0}
    if ATTEMPT_STATE.is_file():
        payload.update(json.loads(ATTEMPT_STATE.read_text(encoding="utf-8")))
    if run_mode == "resume" and int(payload.get("resume", 0)) >= 2:
        raise RuntimeError("resume limit reached (maximum two audited resume attempts)")
    payload[run_mode] = int(payload.get(run_mode, 0)) + 1
    ATTEMPT_STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = ATTEMPT_STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, ATTEMPT_STATE)


def verify_resume_authorization(path: Path | None) -> None:
    if path is None or not path.is_file():
        raise RuntimeError("resume requires --resume-authorization from Codex/Sol review")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "approved",
        "codex_reviewed": True,
        "same_prefix_and_contract": True,
        "contract_sha256": sha256_file(CONTRACT),
    }
    drift = {key: (payload.get(key), value) for key, value in required.items() if payload.get(key) != value}
    sol_path = Path(str(payload.get("sol_verdict_path", "")))
    if drift or not sol_path.is_file() or payload.get("sol_verdict_sha256") != sha256_file(sol_path):
        raise RuntimeError(f"invalid resume authorization: drift={drift}")
    verdict = json.loads(sol_path.read_text(encoding="utf-8"))
    if verdict.get("verdict") != "resume_identical_contract":
        raise RuntimeError("Sol verdict does not authorize identical-contract resume")


def write_execution_manifest(payload: Mapping[str, Any]) -> None:
    EXECUTION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temp = EXECUTION_MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, EXECUTION_MANIFEST)


def preflight(*, strict_environment: bool, run_mode: str) -> dict[str, Any]:
    required = [
        BASE_NPZ,
        BASE_TSV,
        BASE_CACHE,
        QWEN_JSON,
        SOURCE_CKPT,
        SOURCE_EMA,
        MUSICCAPS,
        EVALUATOR,
        CONTRACT,
        REPO / "set_training_stage.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing fixed input(s): " + ", ".join(missing))
    state = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    if state.get("it") != 600_000:
        raise RuntimeError(f"source checkpoint it={state.get('it')}, expected 600000")
    declared = contract_inputs()
    contract_payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    hash_pairs = {
        BASE_TSV: declared["base_tsv_sha256"],
        BASE_CACHE: declared["base_cache_list_sha256"],
        QWEN_JSON: declared["official_qwen_json_sha256"],
        SOURCE_CKPT: declared["source_checkpoint_sha256"],
        SOURCE_EMA: declared["source_ema_sha256"],
        MUSICCAPS: declared["musiccaps_tsv_sha256"],
        EVALUATOR: declared["evaluator_sha256"],
    }
    drift = [str(path) for path, expected in hash_pairs.items() if sha256_file(path) != expected]
    if drift:
        raise RuntimeError("immutable input SHA-256 drift: " + ", ".join(drift))
    code_drift = [
        relative for relative, expected in contract_payload["runtime_code_sha256"].items()
        if not (REPO / relative).is_file() or sha256_file(REPO / relative) != expected
    ]
    if strict_environment and code_drift:
        raise RuntimeError("runtime code SHA-256 drift: " + ", ".join(code_drift))
    duplicates = active_duplicates()
    reject_duplicates(duplicates)
    root_free = shutil.disk_usage(Path("/")).free
    hdd_free = shutil.disk_usage(Path("/mnt/HDD")).free
    if strict_environment:
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi is unavailable; refusing training launch")
        if not Path("/home/kojiek/venvs/dac/bin/python").is_file():
            raise RuntimeError("required DAC venv is unavailable")
        if not TORCHRUN.is_file():
            raise RuntimeError(f"required torchrun is unavailable: {TORCHRUN}")
        mean_flow = (REPO / "meanaudio/model/mean_flow.py").read_text(encoding="utf-8")
        stage2_markers = (
            "lambda z_f, r_f, t_f: model_partial(latent=z_f, r=r_f, t=t_f)",
            "(z, r, t)",
            "(v_hat, torch.zeros_like(r), torch.ones_like(t))",
        )
        if not all(marker in mean_flow for marker in stage2_markers):
            raise RuntimeError("live mean_flow.py is not verified Stage 2")
    if root_free < 50 * 1024**3 or hdd_free < 50 * 1024**3:
        raise RuntimeError(f"free-space floor already violated: root={root_free}, hdd={hdd_free}")
    if run_mode not in {"fresh", "resume"}:
        raise RuntimeError(f"invalid run mode: {run_mode}")
    return {
        "status": "passed",
        "run_mode": run_mode,
        "source_iteration": int(state["it"]),
        "root_free_bytes": int(root_free),
        "hdd_free_bytes": int(hdd_free),
        "expected_rows": 251_599,
        "gpu_environment_checked": strict_environment,
        "duplicate_processes": [],
        "no_threshold_retrain": True,
    }


def train_args(exp_id: str, run_dir: Path, tsv: Path, npz_dir: Path, init: Path | None) -> list[str]:
    args: list[str] = [
        str(TORCHRUN),
        "--standalone",
        "--nproc_per_node=1",
        str(REPO / "train.py"),
        "data=meanaudio",
        "model=meanaudio_s",
        f"exp_id={exp_id}",
        "num_iterations=620000",
        "lr_schedule=step",
        "lr_schedule_steps=[999999,999999]",
        "batch_size=8",
        "learning_rate=1e-5",
        "linear_warmup_steps=1000",
        "seed=14159265",
        "num_workers=4",
        "save_weights_interval=10000",
        "save_checkpoint_interval=10000",
        "val_interval=999999",
        "eval_interval=999999",
        "save_eval_interval=999999",
        "+accumulation_steps=1",
        "+use_rope=False",
        "+use_q_conditioning=false",
        "+use_text_attention_mask=false",
        "+use_wandb=false",
        "++multi_cap=false",
        f"hydra.run.dir={run_dir}",
        f"++data.AudioCaps_npz.tsv={tsv}",
        f"++data.AudioCaps_npz.npz_dir={npz_dir}",
        f"++data.AudioCaps_npz.gt_cache={BASE_CACHE if tsv == BASE_TSV else QWEN_CACHE}",
        f"++data.AudioCaps_val_npz.tsv={tsv}",
        f"++data.AudioCaps_val_npz.npz_dir={npz_dir}",
        f"++data.AudioCaps_val_npz.gt_cache={BASE_CACHE if tsv == BASE_TSV else QWEN_CACHE}",
    ]
    if init is not None:
        args.append(f"checkpoint={init}")
    return args


def eval_args(exp_id: str, ema: Path, output: Path, *, no_q: bool = True) -> list[str]:
    args = command(
        REPO / "eval.py",
        "--variant", "meanaudio_s",
        "--model_path", ema,
        "--output", output / "audio",
        "--tsv", MUSICCAPS,
        "--use_meanflow",
        "--num_steps", "1",
        "--encoder_name", "t5_clap",
        "--text_c_dim", "512",
        "--cfg_strength", "0.5",
        "--full_precision",
        "--no_text_attention_mask",
    )
    if no_q:
        args.insert(args.index("--full_precision"), "--no_q")
    else:
        args.insert(args.index("--full_precision"), "--quality_level")
        args.insert(args.index("--full_precision"), "9")
    return args


def evaluator_args(exp_id: str, audio_dir: Path, out_dir: Path) -> list[str]:
    return command(
        EVALUATOR,
        "--gen_dir", audio_dir,
        "--tsv", MUSICCAPS,
        "--exp_name", exp_id,
        "--out_dir", out_dir,
        "--num_samples", "5521",
    )


def build_queue(run_mode: str) -> list[tuple[str, list[str]]]:
    declared = contract_inputs()
    probe_progress = QWEN_PROBE_NPZ / ".phase8_qwen_npz_progress.json"
    full_progress = QWEN_NPZ / ".phase8_qwen_npz_progress.json"
    probe_resume = ["--resume"] if run_mode == "resume" and probe_progress.is_file() else []
    full_resume = ["--resume"] if run_mode == "resume" and full_progress.is_file() else []
    metadata_exists = all(path.is_file() for path in (QWEN_TSV, QWEN_CACHE, QWEN_METADATA_MANIFEST))
    if run_mode == "resume" and metadata_exists:
        metadata = command(
            REPO / "scripts/preprocess/phase8_qwen_official_mapper.py",
            "--local-tsv", BASE_TSV, "--cache-list", BASE_CACHE, "--official-json", QWEN_JSON,
            "--out-tsv", QWEN_TSV, "--out-cache", QWEN_CACHE, "--manifest", QWEN_METADATA_MANIFEST,
            "--verify-existing",
        )
    else:
        metadata = command(
            REPO / "scripts/preprocess/phase8_qwen_official_mapper.py",
            "--local-tsv", BASE_TSV, "--cache-list", BASE_CACHE, "--official-json", QWEN_JSON,
            "--out-tsv", QWEN_TSV, "--out-cache", QWEN_CACHE, "--manifest", QWEN_METADATA_MANIFEST,
            "--write",
        )
    probe_build = command(
        REPO / "scripts/preprocess/phase8_qwen_full_npz.py",
        "--base-npz", BASE_NPZ, "--tsv", QWEN_TSV, "--cache-list", QWEN_CACHE,
        "--mapper-manifest", QWEN_METADATA_MANIFEST, "--output-dir", QWEN_PROBE_NPZ,
        "--output-manifest", QWEN_PROBE_MANIFEST, "--limit", "512", *probe_resume,
    )
    full_build = command(
        REPO / "scripts/preprocess/phase8_qwen_full_npz.py",
        "--base-npz", BASE_NPZ, "--tsv", QWEN_TSV, "--cache-list", QWEN_CACHE,
        "--mapper-manifest", QWEN_METADATA_MANIFEST, "--output-dir", QWEN_NPZ,
        "--output-manifest", QWEN_NPZ_MANIFEST, *full_resume,
    )
    probe_audit = command(
        REPO / "scripts/analysis/phase8_qwen_cache_audit.py",
        "--tsv", QWEN_TSV, "--cache-list", QWEN_CACHE, "--npz-dir", QWEN_PROBE_NPZ,
        "--reference-npz-dir", BASE_NPZ, "--limit", "512", "--clap-checkpoint", CLAP,
        "--json-out", DATA / "phase8_qwen_official_matched_probe_audit.json",
    )
    control_audit = command(
        REPO / "scripts/analysis/phase8_qwen_cache_audit.py",
        "--tsv", BASE_TSV, "--cache-list", BASE_CACHE, "--npz-dir", BASE_NPZ,
        "--reference-npz-dir", BASE_NPZ, "--clap-checkpoint", CLAP,
        "--json-out", DATA / "phase8_qwen_official_matched_control_cache_audit.json",
    )
    qwen_audit = command(
        REPO / "scripts/analysis/phase8_qwen_cache_audit.py",
        "--tsv", QWEN_TSV, "--cache-list", QWEN_CACHE, "--npz-dir", QWEN_NPZ,
        "--reference-npz-dir", BASE_NPZ, "--clap-checkpoint", CLAP,
        "--json-out", DATA / "phase8_qwen_official_matched_qwen_cache_audit.json",
    )
    q9_dir = RUN_ROOT / "q9_neutral_sanity"
    q9_outputs = (q9_dir / "q9_neutral_ckpt.pth", q9_dir / "q9_neutral_ema.pth", q9_dir / "q9_neutral_manifest.json")
    q9_copy_mode = ["--verify-existing"] if run_mode == "resume" and all(path.is_file() for path in q9_outputs) else []
    q9_copy = command(
        REPO / "scripts/phase8_qwen_neutral_qcopy.py",
        "--source-checkpoint", SOURCE_CKPT, "--source-ema", SOURCE_EMA,
        "--output-checkpoint", q9_outputs[0], "--output-ema", q9_outputs[1],
        "--manifest", q9_outputs[2],
        "--source-checkpoint-sha256", declared["source_checkpoint_sha256"],
        "--source-ema-sha256", declared["source_ema_sha256"],
        *q9_copy_mode,
    )
    q9_eval = eval_args("q9_neutral_sanity", q9_dir / "q9_neutral_ema.pth", q9_dir / "eval", no_q=False)
    q9_metrics = evaluator_args("q9_neutral_sanity", q9_dir / "eval" / "audio", q9_dir / "eval" / "metrics")

    steps: list[tuple[str, list[str]]] = [
        ("preflight", []),
        ("qwen_metadata", metadata),
        ("q9_neutral_copy", q9_copy),
    ]
    q9_metrics_file = q9_dir / "eval/metrics/q9_neutral_sanity/metrics.txt"
    if not (run_mode == "resume" and q9_metrics_file.is_file()):
        if run_mode == "resume" and (q9_dir / "eval/audio").exists():
            raise RuntimeError("unsafe partial q9 eval; archive it before audited resume")
        steps.extend((("q9_neutral_eval", q9_eval), ("q9_neutral_metrics", q9_metrics)))

    historical_q = Path("/home/kojiek/logs/phase8_q_closure/historical_q9_vs_noq.json")
    if not historical_q.is_file():
        steps.append(
            (
                "historical_q_closure",
                command(
                    REPO / "scripts/analysis/phase8_q_clap_bootstrap.py",
                    "--tsv", MUSICCAPS,
                    "--baseline-dir", REPO / "eval_output/phase8_stage2_200000_no_q_musiccaps_qsweep_baseline/audio",
                    "--treatment-dir", REPO / "eval_output/phase8_stage2_200000_q9_musiccaps_qsweep_control/audio",
                    "--clap-checkpoint", CLAP,
                    "--json-out", historical_q,
                ),
            )
        )
    setup_steps = (
        ("cache_probe_512", probe_build, QWEN_PROBE_MANIFEST),
        ("probe_audit", probe_audit, DATA / "phase8_qwen_official_matched_probe_audit.json"),
        ("cache_full", full_build, QWEN_NPZ_MANIFEST),
        ("control_cache_audit", control_audit, DATA / "phase8_qwen_official_matched_control_cache_audit.json"),
        ("qwen_cache_audit", qwen_audit, DATA / "phase8_qwen_official_matched_qwen_cache_audit.json"),
    )
    for name, args, passed_artifact in setup_steps:
        # A post-training orchestration repair must not spend hours re-reading
        # immutable caches that this same execution manifest already audited.
        # Both the durable step record and its canonical JSON artifact must pass.
        if run_mode == "resume" and execution_step_passed(name) and json_passed(passed_artifact):
            continue
        steps.append((name, args))
    for arm, exp_id, tsv, npz_dir in (
        ("control", CONTROL_ID, BASE_TSV, BASE_NPZ),
        ("qwen", QWEN_ID, QWEN_TSV, QWEN_NPZ),
    ):
        run_dir = RUN_ROOT / arm
        init = run_dir / "source_reset_init.pth"
        init_manifest = run_dir / "source_reset_init_manifest.json"
        ckpt_last = run_dir / f"{exp_id}_ckpt_last.pth"
        ema_final = run_dir / f"{exp_id}_ema_final.pth"
        final_audit = run_dir / "final_audit.json"
        if run_mode == "resume" and json_passed(final_audit):
            continue
        if run_mode == "resume" and ckpt_last.is_file():
            current_it = checkpoint_iteration(ckpt_last)
            if current_it != 620_000:
                raise RuntimeError(
                    f"unsafe mid-training resume for {arm} at it={current_it}; "
                    "restart this arm from the immutable source after Sol/Codex approval"
                )
            if not ema_final.is_file():
                raise RuntimeError(f"final checkpoint exists but EMA is missing for {arm}")
        init_pair = (init.is_file(), init_manifest.is_file())
        if init_pair[0] != init_pair[1]:
            raise RuntimeError(f"partial reset initializer for {arm}")
        init_needed = not ckpt_last.exists() and not init.exists()
        if init_needed:
            steps.append(
                (
                    f"{arm}_init",
                    command(
                        REPO / "scripts/phase8_qwen_init_checkpoint.py",
                        "--source", SOURCE_CKPT, "--output", init, "--manifest", init_manifest,
                        "--source-sha256", declared["source_checkpoint_sha256"],
                    ),
                )
            )
        if not ckpt_last.exists():
            steps.append((f"{arm}_train", train_args(exp_id, run_dir, tsv, npz_dir, init)))
        eval_dir = run_dir / "musiccaps_eval"
        metrics_dir = run_dir / "musiccaps_metrics"
        metrics_file = metrics_file_path(run_dir, exp_id)
        if not metrics_file.is_file():
            if run_mode == "resume" and (eval_dir / "audio").exists():
                raise RuntimeError(f"unsafe partial eval for {arm}; archive it before audited resume")
            steps.append((f"{arm}_eval", eval_args(exp_id, ema_final, eval_dir)))
            steps.append((f"{arm}_metrics", evaluator_args(exp_id, eval_dir / "audio", metrics_dir)))
        steps.append(
            (
                f"{arm}_audit",
                command(
                    REPO / "scripts/analysis/phase8_qwen_experiment_audit.py",
                    "--arm", arm, "--exp-id", exp_id, "--run-dir", run_dir,
                    "--metrics", metrics_file,
                    "--audio-tsv", MUSICCAPS, "--audio-dir", eval_dir / "audio",
                    "--contract", CONTRACT,
                    "--execution-manifest", EXECUTION_MANIFEST,
                    "--init-manifest", init_manifest,
                    "--cache-manifest", QWEN_NPZ_MANIFEST if arm == "qwen" else BASE_NPZ / "FULL_GATE_PASSED.json",
                    "--json-out", final_audit,
                ),
            )
        )
    steps.append(
        (
            "paired_per_prompt_clap",
            command(
                REPO / "scripts/analysis/phase8_q_clap_bootstrap.py",
                "--tsv", MUSICCAPS,
                "--baseline-dir", RUN_ROOT / "control/musiccaps_eval/audio",
                "--treatment-dir", RUN_ROOT / "qwen/musiccaps_eval/audio",
                "--clap-checkpoint", CLAP,
                "--json-out", RUN_ROOT / "paired_per_prompt_clap.json",
            ),
        )
    )
    steps.append(
        (
            "paired_final_report",
            command(
                REPO / "scripts/analysis/phase8_qwen_paired_report.py",
                "--control-metrics", metrics_file_path(RUN_ROOT / "control", CONTROL_ID),
                "--qwen-metrics", metrics_file_path(RUN_ROOT / "qwen", QWEN_ID),
                "--control-audit", RUN_ROOT / "control/final_audit.json",
                "--qwen-audit", RUN_ROOT / "qwen/final_audit.json",
                "--contract", CONTRACT,
                "--paired-bootstrap", RUN_ROOT / "paired_per_prompt_clap.json",
                "--json-out", RUN_ROOT / "paired_final_report.json",
            ),
        )
    )
    return steps


def fresh_conflicts() -> list[Path]:
    return [
        QWEN_TSV, QWEN_CACHE, QWEN_METADATA_MANIFEST, QWEN_NPZ,
        QWEN_NPZ_MANIFEST, QWEN_PROBE_NPZ, QWEN_PROBE_MANIFEST,
        RUN_ROOT,
    ]


def load_execution_manifest(run_mode: str) -> dict[str, Any]:
    contract_sha256 = sha256_file(CONTRACT)
    if run_mode == "fresh":
        if EXECUTION_MANIFEST.exists():
            raise RuntimeError(
                f"fresh run refuses existing execution manifest: {EXECUTION_MANIFEST}"
            )
        return {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "steps": {},
        }
    if not EXECUTION_MANIFEST.is_file():
        raise RuntimeError(
            "resume requires the prior execution manifest so launch provenance is preserved"
        )
    try:
        manifest = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid prior execution manifest: {exc!r}") from exc
    if manifest.get("schema_version") != 1:
        raise RuntimeError("prior execution manifest schema drift")
    if manifest.get("contract_sha256") != contract_sha256:
        raise RuntimeError("prior execution manifest contract hash drift")
    if not isinstance(manifest.get("steps"), dict):
        raise RuntimeError("prior execution manifest has invalid steps")
    return manifest


def execute(steps: Sequence[tuple[str, list[str]]], run_mode: str) -> None:
    # Resume must retain passed train commands from the previous process.  Final
    # audits consume this same manifest as immutable launch provenance.
    manifest = load_execution_manifest(run_mode)
    for name, args in steps:
        if name == "preflight":
            continue
        print(f"[RUN] {name}: {run_display(args)}", flush=True)
        manifest["steps"][name] = {
            "command": list(args),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        write_execution_manifest(manifest)
        result = subprocess.run(args, cwd=REPO, check=False)
        manifest["steps"][name]["exit_code"] = result.returncode
        manifest["steps"][name]["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["steps"][name]["status"] = "passed" if result.returncode == 0 else "failed"
        write_execution_manifest(manifest)
        if result.returncode != 0:
            raise RuntimeError(f"queue stopped at {name} with exit={result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--run-mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--resume-authorization", type=Path)
    args = parser.parse_args()
    if args.run_mode == "resume":
        verify_resume_authorization(args.resume_authorization)
    lock_handle = None
    if args.execute:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = LOCK_PATH.open("w")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"queue lock is already held: {LOCK_PATH}") from exc
    report = preflight(strict_environment=args.execute, run_mode=args.run_mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.run_mode == "fresh":
        conflicts = fresh_conflicts()
        if any(path.exists() for path in conflicts):
            raise SystemExit("[FAIL] fresh mode has existing output; choose --run-mode resume")
    steps = build_queue(args.run_mode)
    print("\nEXACT QUEUE")
    for index, (name, command_args) in enumerate(steps, start=1):
        if name == "preflight":
            print(f"{index:02d}. {name}: preflight (already checked)")
        else:
            print(f"{index:02d}. {name}: {run_display(command_args)}")
    if args.execute:
        record_execute_attempt(args.run_mode)
        execute(steps, args.run_mode)
    else:
        print("\nDRY-RUN: no command was started.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
