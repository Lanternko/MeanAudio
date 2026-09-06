#!/usr/bin/env python3
"""Completion audit for the guarded Phase-8 legacy reproduction.

The guarded launcher intentionally keeps training and evaluation simple.  This
auditor is the stronger, independent completion proof: it ties the final model
and metrics back to the validated cache, checks both training stages, verifies
the exact legacy evaluation arguments, and inspects every generated MusicCaps
file.  A metrics.txt file by itself is not considered completion.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import yaml


ROOT = Path("/home/kojiek/MeanAudio")
DATA = Path("/mnt/HDD/kojiek/phase4_jamendo_data")
NPZ_DIR = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz")
STATE_DIR = Path("/home/kojiek/logs/phase8_legacy_repro_guard")

PREFIX = "phase8_legacy_repro"
S1_EXP = f"{PREFIX}_stage1_400000"
S2_EXP = f"{PREFIX}_stage2_200000"
S1_DIR = ROOT / "exps" / S1_EXP
S2_DIR = ROOT / "exps" / S2_EXP
S1_CKPT = S1_DIR / f"{S1_EXP}_ckpt_last.pth"
S2_CKPT = S2_DIR / f"{S2_EXP}_ckpt_last.pth"
S1_EMA = S1_DIR / f"{S1_EXP}_ema_final.pth"
S2_EMA = S2_DIR / f"{S2_EXP}_ema_final.pth"

VALIDATION = NPZ_DIR / "FULL_VALIDATION.json"
GATE = NPZ_DIR / "FULL_GATE_PASSED.json"
TRAIN_TSV = DATA / "phase8_legacy_catalog_train.tsv"
CACHE = DATA / "npz_cache_train.txt"
MUSICCAPS_TSV = DATA / "musiccaps_test.tsv"
EVAL_DIR = ROOT / "eval_output" / f"{S2_EXP}_musiccaps" / "audio"
METRICS = ROOT / "eval_output" / "metrics" / f"{S2_EXP}_musiccaps" / "metrics.txt"
EVAL_LOG = Path("/home/kojiek/logs") / f"{S2_EXP}_musiccaps_eval.log"
STATE_FILE = STATE_DIR / "state.json"
ALERT_FILE = STATE_DIR / "ALERT.json"
REPORT = STATE_DIR / "FINAL_AUDIT.json"

EXPECTED_TRAIN_ROWS = 251_599
EXPECTED_MUSICCAPS_ROWS = 5_521
# Comparison target is the historical --quality_level 9 measurement (2026-04-17:
# 0.1907), not the historical --no_q 0.1851.  The repro is effectively Q-trained,
# and under the fixed runners q=10 exists only as the CFG-unconditional marker,
# so the legacy --no_q semantics (q=10 as S1's universal default) cannot be
# reproduced; --no_q on this checkpoint generates unconditionally (CLAP 0.0134,
# 2026-07-18 incident).  The in-support q=9 condition is the faithful comparison.
HISTORICAL_CLAP = 0.1907
MAX_CLAP_DELTA_WITHOUT_REVIEW = 0.03
REQUIRED_METRICS = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Write an in-progress report and return success while future artifacts are absent.",
    )
    parser.add_argument(
        "--skip-large-loads",
        action="store_true",
        help="Skip multi-GB checkpoint deserialization (allowed only with --progress).",
    )
    parser.add_argument("--report", type=Path, default=REPORT)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def add_issue(issues: list[dict[str, str]], code: str, detail: str) -> None:
    issues.append({"code": code, "detail": detail})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_config(exp_dir: Path) -> Path | None:
    paths = sorted(exp_dir.glob("train-*-hydra/config.yaml"))
    return paths[-1] if paths else None


def audit_cache(evidence: dict[str, Any], issues: list[dict[str, str]]) -> None:
    validation = read_json(VALIDATION)
    gate = read_json(GATE)
    item: dict[str, Any] = {
        "validation": str(VALIDATION),
        "gate": str(GATE),
        "validation_status": validation.get("status"),
        "expected_rows": validation.get("expected_rows"),
        "deep_sample_size": validation.get("deep_sample_size"),
        "decoded_cache_clap": gate.get("decoded_cache_clap"),
        "decoded_samples": gate.get("decoded_samples"),
    }
    if validation.get("status") != "passed":
        add_issue(issues, "cache_validation", "structural validation did not pass")
    if validation.get("expected_rows") != EXPECTED_TRAIN_ROWS:
        add_issue(issues, "cache_rows", f"expected {EXPECTED_TRAIN_ROWS} validated rows")
    if int(validation.get("deep_sample_size", 0) or 0) < 4096:
        add_issue(issues, "cache_deep_sample", "deep validation covered fewer than 4,096 rows")

    expected_hashes = validation.get("sha256", {})
    validated_paths = validation.get("paths", {})
    current_hashes: dict[str, str] = {}
    for key in ("cache", "catalog", "manifest", "output_tsv"):
        value = validated_paths.get(key) if isinstance(validated_paths, dict) else None
        path = Path(value) if isinstance(value, str) else None
        if path is None or not path.is_file():
            add_issue(issues, "cache_artifact", f"missing validated {key} artifact")
            continue
        actual = sha256_file(path)
        current_hashes[key] = actual
        expected = expected_hashes.get(key) if isinstance(expected_hashes, dict) else None
        if actual != expected:
            add_issue(issues, "cache_hash", f"validated {key} hash changed")
    item["current_sha256"] = current_hashes

    validation_hash = sha256_file(VALIDATION) if VALIDATION.is_file() else None
    item["validation_report_sha256"] = validation_hash
    if gate.get("status") != "passed":
        add_issue(issues, "semantic_gate", "semantic cache gate did not pass")
    if gate.get("decoded_samples") != 512:
        add_issue(issues, "semantic_samples", "semantic gate did not use 512 decoded samples")
    score = gate.get("decoded_cache_clap")
    minimum = gate.get("minimum_clap")
    if not isinstance(score, (int, float)) or not isinstance(minimum, (int, float)) or score < minimum:
        add_issue(issues, "semantic_score", "decoded cache CLAP is missing or below its gate")
    if gate.get("validation_report_sha256") != validation_hash:
        add_issue(issues, "semantic_binding", "semantic gate is not tied to current validation report")
    evidence["cache_and_gates"] = item


def audit_config(
    label: str,
    exp_dir: Path,
    stage: int,
    evidence: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    path = latest_config(exp_dir)
    if path is None:
        add_issue(issues, f"{label}_config_missing", f"no Hydra config under {exp_dir}")
        evidence[label] = {"config": None}
        return
    try:
        cfg = yaml.safe_load(path.read_text())
    except Exception as exc:
        add_issue(issues, f"{label}_config_parse", str(exc))
        evidence[label] = {"config": str(path)}
        return

    train = cfg.get("data", {}).get("AudioCaps_npz", {})
    expected = {
        "model": "fluxaudio_s" if stage == 1 else "meanaudio_s",
        "num_iterations": 400_000 if stage == 1 else 600_000,
        "use_q_conditioning": True,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "batch_size": 8,
        "accumulation_steps": 1,
        "seed": 14_159_265,
    }
    observed = {key: cfg.get(key) for key in expected}
    for key, value in expected.items():
        if observed[key] != value:
            add_issue(issues, f"{label}_config_value", f"{key}={observed[key]!r}, expected {value!r}")
    data_expected = {
        "tsv": str(TRAIN_TSV),
        "gt_cache": str(CACHE),
        "npz_dir": str(NPZ_DIR),
    }
    observed_data = {key: train.get(key) for key in data_expected}
    for key, value in data_expected.items():
        if observed_data[key] != value:
            add_issue(issues, f"{label}_data_path", f"{key}={observed_data[key]!r}, expected {value!r}")
    evidence[label] = {
        "config": str(path),
        "config_sha256": sha256_file(path),
        "observed": observed,
        "data": observed_data,
    }


def tensor_tree_summary(value: Any) -> dict[str, Any]:
    stack = [value]
    count = 0
    numel = 0
    finite = True
    while stack:
        current = stack.pop()
        if torch.is_tensor(current):
            count += 1
            numel += current.numel()
            if current.is_floating_point() and not torch.isfinite(current).all().item():
                finite = False
        elif isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return {"tensor_count": count, "numel": numel, "finite": finite}


def audit_checkpoint(
    label: str,
    path: Path,
    expected_iteration: int,
    evidence: dict[str, Any],
    issues: list[dict[str, str]],
    skip_load: bool,
) -> None:
    if not path.is_file():
        add_issue(issues, f"{label}_missing", f"missing checkpoint {path}")
        evidence[label] = {"path": str(path), "present": False}
        return
    item: dict[str, Any] = {"path": str(path), "present": True, "bytes": path.stat().st_size}
    if skip_load:
        item["load_skipped"] = True
        evidence[label] = item
        return
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        add_issue(issues, f"{label}_unreadable", repr(exc))
        evidence[label] = item
        return
    if not isinstance(checkpoint, dict):
        add_issue(issues, f"{label}_schema", "checkpoint is not a dictionary")
        evidence[label] = item
        return
    item["keys"] = sorted(checkpoint)
    item["iteration"] = checkpoint.get("it")
    if checkpoint.get("it") != expected_iteration:
        add_issue(
            issues,
            f"{label}_iteration",
            f"iteration={checkpoint.get('it')!r}, expected {expected_iteration}",
        )
    for key in ("weights", "optimizer", "scheduler", "ema"):
        value = checkpoint.get(key)
        if value is None:
            add_issue(issues, f"{label}_{key}", f"checkpoint lacks {key}")
        if key != "scheduler":
            summary = tensor_tree_summary(value)
            item[key] = summary
            if value is not None and (summary["tensor_count"] == 0 or not summary["finite"]):
                add_issue(issues, f"{label}_{key}", f"{key} tensor tree is empty or non-finite")
    evidence[label] = item
    del checkpoint
    gc.collect()


def audit_ema(
    label: str,
    path: Path,
    evidence: dict[str, Any],
    issues: list[dict[str, str]],
    skip_load: bool,
) -> None:
    if not path.is_file():
        add_issue(issues, f"{label}_missing", f"missing EMA weights {path}")
        evidence[label] = {"path": str(path), "present": False}
        return
    item: dict[str, Any] = {"path": str(path), "present": True, "bytes": path.stat().st_size}
    if skip_load:
        item["load_skipped"] = True
        evidence[label] = item
        return
    try:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        summary = tensor_tree_summary(weights)
    except Exception as exc:
        add_issue(issues, f"{label}_unreadable", repr(exc))
        evidence[label] = item
        return
    item.update(summary)
    item["sha256"] = sha256_file(path)
    if summary["tensor_count"] == 0 or not summary["finite"]:
        add_issue(issues, f"{label}_tensors", "EMA tensor tree is empty or non-finite")
    evidence[label] = item
    del weights
    gc.collect()


def parse_metrics(path: Path) -> tuple[dict[str, str], dict[str, float]]:
    metadata: dict[str, str] = {}
    values: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in REQUIRED_METRICS:
            values[key] = float(value)
        else:
            metadata[key] = value
    return metadata, values


def audit_eval(evidence: dict[str, Any], issues: list[dict[str, str]]) -> None:
    item: dict[str, Any] = {
        "tsv": str(MUSICCAPS_TSV),
        "audio_dir": str(EVAL_DIR),
        "metrics": str(METRICS),
        "eval_log": str(EVAL_LOG),
    }
    if not MUSICCAPS_TSV.is_file():
        add_issue(issues, "eval_tsv_missing", f"missing {MUSICCAPS_TSV}")
        evidence["evaluation"] = item
        return
    rows = read_tsv(MUSICCAPS_TSV)
    ids = [row.get("id", "") for row in rows]
    expected_ids = set(ids)
    item["tsv_rows"] = len(rows)
    item["unique_ids"] = len(expected_ids)
    if len(rows) != EXPECTED_MUSICCAPS_ROWS or len(expected_ids) != len(rows) or "" in expected_ids:
        add_issue(
            issues,
            "eval_tsv_rows",
            f"MusicCaps rows={len(rows)}, unique_ids={len(expected_ids)}, expected={EXPECTED_MUSICCAPS_ROWS}",
        )

    actual_files = list(EVAL_DIR.glob("*.flac")) if EVAL_DIR.is_dir() else []
    actual_ids = {path.stem for path in actual_files}
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    item["audio_files"] = len(actual_files)
    item["missing_audio_count"] = len(missing)
    item["extra_audio_count"] = len(extra)
    item["missing_audio_examples"] = missing[:10]
    item["extra_audio_examples"] = extra[:10]
    if missing:
        add_issue(issues, "eval_audio_missing", f"missing {len(missing)} MusicCaps audio files")
    if extra:
        add_issue(issues, "eval_audio_extra", f"found {len(extra)} unexpected audio files")
    if len(actual_files) != len(actual_ids):
        add_issue(issues, "eval_audio_collision", "multiple FLAC paths collapse to the same stem")

    unreadable: list[str] = []
    bad_format: list[str] = []
    for path in actual_files:
        try:
            info = sf.info(path)
        except Exception:
            unreadable.append(path.name)
            continue
        if info.samplerate != 16_000 or info.channels != 1 or not (9.8 <= info.duration <= 10.2):
            bad_format.append(
                f"{path.name}: sr={info.samplerate}, channels={info.channels}, duration={info.duration:.4f}"
            )
    item["unreadable_audio_count"] = len(unreadable)
    item["bad_format_audio_count"] = len(bad_format)
    item["unreadable_audio_examples"] = unreadable[:10]
    item["bad_format_audio_examples"] = bad_format[:10]
    if unreadable:
        add_issue(issues, "eval_audio_unreadable", f"{len(unreadable)} FLAC files are unreadable")
    if bad_format:
        add_issue(issues, "eval_audio_format", f"{len(bad_format)} FLAC files have unexpected format")

    if not EVAL_LOG.is_file():
        add_issue(issues, "eval_log_missing", f"missing {EVAL_LOG}")
    else:
        with EVAL_LOG.open("rb") as handle:
            log_head = handle.read(512 * 1024).decode("utf-8", errors="replace")
        required_log_tokens = {
            # Q-trained repro must be evaluated in-support, never with --no_q
            # (see HISTORICAL_CLAP comment / reference_eval_q_flag_rule).
            "quality_level_9": "'quality_level': 9",
            "no_q_disabled": "'no_q': False",
            "no_text_attention_mask": "'no_text_attention_mask': True",
            "meanflow": "'use_meanflow': True",
            "num_steps": "'num_steps': 1",
            "cfg_strength": "'cfg_strength': 0.5",
            "model_path": str(S2_EMA),
            "tsv": str(MUSICCAPS_TSV),
        }
        item["eval_args_verified"] = {}
        for key, token in required_log_tokens.items():
            present = token in log_head
            item["eval_args_verified"][key] = present
            if not present:
                add_issue(issues, "eval_args", f"evaluation log does not prove {key}")

    if not METRICS.is_file():
        add_issue(issues, "metrics_missing", f"missing {METRICS}")
    else:
        try:
            metadata, values = parse_metrics(METRICS)
        except Exception as exc:
            add_issue(issues, "metrics_parse", repr(exc))
        else:
            item["metrics_metadata"] = metadata
            item["metric_values"] = values
            item["metrics_sha256"] = sha256_file(METRICS)
            for key in REQUIRED_METRICS:
                value = values.get(key)
                if value is None or not math.isfinite(value):
                    add_issue(issues, "metrics_value", f"missing/non-finite {key}")
            if metadata.get("Experiment") != f"{S2_EXP}_musiccaps":
                add_issue(issues, "metrics_experiment", "metrics experiment label is wrong")
            if metadata.get("Test TSV") != str(MUSICCAPS_TSV):
                add_issue(issues, "metrics_tsv", "metrics TSV path is wrong")
            if metadata.get("Test clips") != str(EXPECTED_MUSICCAPS_ROWS):
                add_issue(issues, "metrics_rows", "metrics did not cover all 5,521 MusicCaps rows")
            generated = metadata.get("Generated audio")
            if generated is None or Path(generated).resolve() != EVAL_DIR.resolve():
                add_issue(issues, "metrics_audio_dir", "metrics generated-audio path is wrong")
            clap = values.get("clap_score")
            if clap is not None and math.isfinite(clap):
                delta = clap - HISTORICAL_CLAP
                item["historical_comparison"] = {
                    "historical_clap": HISTORICAL_CLAP,
                    "reproduction_clap": clap,
                    "delta": delta,
                    "review_threshold_absolute_delta": MAX_CLAP_DELTA_WITHOUT_REVIEW,
                }
                if abs(delta) > MAX_CLAP_DELTA_WITHOUT_REVIEW:
                    add_issue(
                        issues,
                        "clap_needs_review",
                        f"CLAP delta from historical baseline is {delta:+.4f}, exceeding ±{MAX_CLAP_DELTA_WITHOUT_REVIEW:.2f}",
                    )
    evidence["evaluation"] = item


def audit_guard(evidence: dict[str, Any], issues: list[dict[str, str]]) -> None:
    state = read_json(STATE_FILE)
    alert = read_json(ALERT_FILE)
    evidence["guard"] = {
        "state_file": str(STATE_FILE),
        "state": state,
        "alert_file": str(ALERT_FILE),
        "alert": alert,
    }
    if state.get("phase") != "DONE":
        add_issue(issues, "guard_not_done", f"guard phase is {state.get('phase')!r}, expected 'DONE'")
    if alert:
        add_issue(issues, "guard_alert", "guard alert is present")


def main() -> None:
    args = parse_args()
    if args.skip_large_loads and not args.progress:
        raise SystemExit("--skip-large-loads is allowed only with --progress")

    evidence: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    audit_cache(evidence, issues)
    audit_config("stage1", S1_DIR, 1, evidence, issues)
    audit_config("stage2", S2_DIR, 2, evidence, issues)
    audit_checkpoint("stage1_checkpoint", S1_CKPT, 400_000, evidence, issues, args.skip_large_loads)
    audit_checkpoint("stage2_checkpoint", S2_CKPT, 600_000, evidence, issues, args.skip_large_loads)
    audit_ema("stage1_ema_final", S1_EMA, evidence, issues, args.skip_large_loads)
    audit_ema("stage2_ema_final", S2_EMA, evidence, issues, args.skip_large_loads)
    audit_eval(evidence, issues)
    audit_guard(evidence, issues)

    status = "passed" if not issues else ("in_progress" if args.progress else "failed")
    report = {
        "status": status,
        "audited_at": utc_now(),
        "historical_target": "Phase8 legacy reproduction (matched original captions, historical q training, in-support q=9 + NoMask eval)",
        "issues": issues,
        "evidence": evidence,
    }
    atomic_json(args.report, report)
    print(json.dumps({"status": status, "issues": issues, "report": str(args.report)}, indent=2))
    if issues and not args.progress:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
