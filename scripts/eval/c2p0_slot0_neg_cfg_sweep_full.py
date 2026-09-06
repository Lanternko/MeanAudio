#!/usr/bin/env python3
"""Strict MusicCaps-5521 secondary CFG sweep for the c2p0 slot0 full NoQ EMA."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path("/home/kojiek/MeanAudio")
PYTHON = Path("/home/kojiek/venvs/dac/bin/python")
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
CHECKPOINT = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
CLAP_CHECKPOINT = ROOT / "weights/music_speech_audioset_epoch_15_esc_89.98.pt"
CFG0_REFERENCE = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/novocal_reeval/c2p0_slot0_full_noq.json")
OUT = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/c2p0_slot0_neg_cfg_full5521")
NEGATIVE_PROMPT = "low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi"
CFGS = ("2.5", "4.0")
EXPECTED = 5521
EXPECTED_FRAMES = 159744
LABEL = "c2p0_slot0_full_noq"
EXP_ID = "phase8_qwen_caption10s_multisent_noq_full_stage2_200000"
SCORER = ROOT / "scripts/eval/negprompt_reeval_full_arms.py"
EVAL_ENTRYPOINT = ROOT / "eval.py"
EVAL_UTILS = ROOT / "meanaudio/eval_utils.py"
HARD_STOP_FREE_BYTES = 63_687_091_200
WARNING_FREE_BYTES = 80_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows() -> list[dict[str, str]]:
    with TSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    ids = [row["id"] for row in rows]
    if len(rows) != EXPECTED or len(set(ids)) != EXPECTED:
        raise SystemExit(f"[FAIL] MusicCaps identity mismatch: rows={len(rows)} unique={len(set(ids))}")
    if any(not item or "/" in item or item in {".", ".."} for item in ids):
        raise SystemExit("[FAIL] unsafe MusicCaps id")
    return rows


def storage_gate(phase: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(OUT)
    free = stats.f_bavail * stats.f_frsize
    print(f"[storage] phase={phase} free_bytes={free} warning={WARNING_FREE_BYTES} hard={HARD_STOP_FREE_BYTES}")
    if free < HARD_STOP_FREE_BYTES:
        raise SystemExit(f"[HOLD] storage hard stop before {phase}: free_bytes={free}")
    return free


def cfg_tag(cfg: str) -> str:
    return "cfg" + cfg.replace(".", "p")


def report_path(cfg: str) -> Path:
    return OUT / f"{LABEL}_{cfg_tag(cfg)}_neg.json"


def audio_dir(cfg: str) -> Path:
    return OUT / "_audio" / f"{LABEL}_{cfg_tag(cfg)}_neg"


def input_identities() -> dict[str, dict[str, str]]:
    return {
        "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT)},
        "evaluation_tsv": {"path": str(TSV), "sha256": sha256(TSV)},
        "clap_checkpoint": {"path": str(CLAP_CHECKPOINT), "sha256": sha256(CLAP_CHECKPOINT)},
        "scorer": {"path": str(SCORER), "sha256": sha256(SCORER)},
        "eval_entrypoint": {"path": str(EVAL_ENTRYPOINT), "sha256": sha256(EVAL_ENTRYPOINT)},
        "eval_utils": {"path": str(EVAL_UTILS), "sha256": sha256(EVAL_UTILS)},
        "paired_cfg0_reference": {"path": str(CFG0_REFERENCE), "sha256": sha256(CFG0_REFERENCE)},
    }


def protocol(cfg: str) -> dict[str, object]:
    return {
        "classification": "secondary_noncanonical", "dataset": "MusicCaps", "rows": EXPECTED,
        "solver": "MeanFlow", "steps": 25, "cfg_strength": float(cfg),
        "negative_prompt": NEGATIVE_PROMPT, "seed": 42, "mask": "NoMask",
        "precision": "full", "conditioning": "NoQ", "encoder_name": "t5_clap",
        "text_c_dim": 512,
    }


def generation_argv(cfg: str, directory: Path) -> list[str]:
    return [
        str(PYTHON), str(EVAL_ENTRYPOINT), "--variant", "meanaudio_s",
        "--model_path", str(CHECKPOINT), "--output", str(directory),
        "--tsv", str(TSV), "--use_meanflow", "--num_steps", "25",
        "--cfg_strength", cfg, "--negative_prompt", NEGATIVE_PROMPT,
        "--no_text_attention_mask", "--encoder_name", "t5_clap",
        "--text_c_dim", "512", "--seed", "42", "--full_precision", "--no_q",
    ]


def validate_finished_report(path: Path, cfg: str, rows: list[dict[str, str]]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        full = record["aggregates"]["full"]
        per_clip = record.get("per_clip") or {}
        expected_ids = {row["id"] for row in rows}
        finite = all(
            isinstance(metrics, dict)
            and all(isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
                    for key in ("clap", "CE", "CU", "PC", "PQ"))
            for metrics in per_clip.values()
        )
        return (
            record.get("label") == LABEL
            and record.get("exp_id") == EXP_ID
            and float(record.get("cfg_strength")) == float(cfg)
            and record.get("negative_prompt") == NEGATIVE_PROMPT
            and record.get("protocol") == protocol(cfg)
            and record.get("generation_argv") == generation_argv(cfg, audio_dir(cfg))
            and record.get("input_identities") == input_identities()
            and int(full.get("n")) == EXPECTED
            and all(math.isfinite(float(full[key])) for key in ("clap", "CE", "CU", "PC", "PQ"))
            and set(per_clip) == expected_ids
            and finite
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def remove_partial_audio(directory: Path) -> None:
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise SystemExit(f"[FAIL] unsafe audio directory: {directory}")
    children = list(directory.iterdir())
    for path in children:
        if path.is_symlink() or not path.is_file() or path.suffix != ".flac":
            raise SystemExit(f"[FAIL] unexpected partial artifact: {path}")
    for path in children:
        path.unlink()
    print(f"[resume] removed {len(children)} partial FLACs to preserve seed-42 sequence")


def validate_audio(directory: Path, rows: list[dict[str, str]]) -> None:
    expected = {f"{row['id']}.flac" for row in rows}
    actual = {path.name for path in directory.iterdir() if path.is_file() and not path.is_symlink()}
    if actual != expected:
        raise SystemExit(f"[FAIL] audio ID mismatch: actual={len(actual)} expected={len(expected)}")
    for name in sorted(expected):
        path = directory / name
        info = sf.info(path)
        if (info.samplerate != 16000 or info.channels != 1 or info.frames != EXPECTED_FRAMES
                or info.format != "FLAC"):
            raise SystemExit(f"[FAIL] invalid audio metadata: {path}")


def generate(cfg: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    command = generation_argv(cfg, directory)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"[FAIL] generation cfg={cfg} rc={completed.returncode}")


def score(rows: list[dict[str, str]], directory: Path) -> dict[str, dict[str, float]]:
    # Reuse the already-audited scorer implementation used for the CFG-1.5 full sweep.
    sys.path.insert(0, str(ROOT / "scripts/eval"))
    import negprompt_reeval_full_arms as base

    per = base.score(rows, directory)
    if len(per) != EXPECTED:
        raise SystemExit(f"[FAIL] scored rows={len(per)}/{EXPECTED}")
    for item, metrics in per.items():
        if any(not math.isfinite(float(metrics[key])) for key in ("clap", "CE", "CU", "PC", "PQ")):
            raise SystemExit(f"[FAIL] non-finite metric for {item}")
    return per


def aggregate(per: dict[str, dict[str, float]], ids: list[str]) -> dict[str, float | int]:
    return {
        "n": len(ids),
        **{key: float(np.mean([per[item][key] for item in ids])) for key in ("clap", "CE", "CU", "PC", "PQ")},
    }


def paired_delta(per: dict[str, dict[str, float]]) -> dict | None:
    if not CFG0_REFERENCE.is_file():
        return None
    reference = json.loads(CFG0_REFERENCE.read_text(encoding="utf-8")).get("per_clip") or {}
    shared = sorted(set(per).intersection(reference))
    if len(shared) != EXPECTED:
        return None
    result: dict[str, object] = {"n_paired": len(shared)}
    for key in ("clap", "CE", "CU", "PC", "PQ"):
        values = np.asarray([per[item][key] - reference[item][key] for item in shared])
        result[key] = {
            "mean_delta": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "frac_improved": float((values > 0).mean()),
        }
    return result


def cleanup_audio(directory: Path, rows: list[dict[str, str]]) -> None:
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise SystemExit(f"[FAIL] unsafe cleanup directory: {directory}")
    allowed = {f"{row['id']}.flac" for row in rows}
    children = list(directory.iterdir())
    for path in children:
        if path.name not in allowed or path.is_symlink() or not path.is_file():
            raise SystemExit(f"[FAIL] cleanup identity mismatch: {path}")
    for path in children:
        path.unlink()
    directory.rmdir()


def run_cfg(cfg: str, rows: list[dict[str, str]]) -> None:
    storage_gate(cfg_tag(cfg))
    destination = report_path(cfg)
    if validate_finished_report(destination, cfg, rows):
        print(f"[done] cfg={cfg} verified report exists: {destination}")
        directory = audio_dir(cfg)
        if directory.exists():
            cleanup_audio(directory, rows)
            print(f"[cleanup] removed residual verified-report audio for cfg={cfg}")
        return
    if destination.exists():
        raise SystemExit(f"[FAIL] stale or invalid report exists: {destination}")
    directory = audio_dir(cfg)
    if directory.exists():
        # A report-less phase never trusts old audio, even when all 5,521 names
        # exist: provenance or a final interrupted write cannot be proven.
        remove_partial_audio(directory)
    started = time.time()
    generate(cfg, directory)
    validate_audio(directory, rows)
    per = score(rows, directory)
    ids = [row["id"] for row in rows]
    payload = {
        "document_kind": "secondary_musiccaps_evaluation_report",
        "label": LABEL,
        "exp_id": EXP_ID,
        "classification": "secondary_noncanonical",
        "cfg_strength": float(cfg),
        "negative_prompt": NEGATIVE_PROMPT,
        "protocol": protocol(cfg),
        "generation_argv": generation_argv(cfg, directory),
        "input_identities": input_identities(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "tsv": str(TSV),
        "tsv_sha256": sha256(TSV),
        "elapsed_seconds": time.time() - started,
        "aggregates": {"full": aggregate(per, ids)},
        "paired_delta_vs_cfg0": paired_delta(per),
        "per_clip": per,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)
    if not validate_finished_report(destination, cfg, rows):
        raise SystemExit(f"[FAIL] final report validation failed: {destination}")
    cleanup_audio(directory, rows)
    full = payload["aggregates"]["full"]
    print(f"[complete] cfg={cfg} n={full['n']} CLAP={full['clap']:.4f} CE={full['CE']:.4f} CU={full['CU']:.4f} PC={full['PC']:.4f} PQ={full['PQ']:.4f}")


def main() -> None:
    rows = read_rows()
    for cfg in CFGS:
        run_cfg(cfg, rows)
    print("ALL CFG FULL-5521 EVALUATIONS DONE")


if __name__ == "__main__":
    main()
