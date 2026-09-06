#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/kojiek/MeanAudio")
CONTRACT = ROOT / "docs/experiments/caption2p0_quarter_cfg0_rerun_contract.json"
RUNNER = ROOT / "scripts/eval/eval_caption2p0_quarter_cfg0.sh"
WRAPPER = ROOT / "scripts/caption10s_pipeline/eval_musiccaps_mf25.sh"
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/experiment_harness"))

import validate_caption2p0_cfg0_report as strict  # noqa: E402
import validate_cfg0_output_path as output_path  # noqa: E402
import caption2p0_quarter_cfg0_harn as cfg0_harn  # noqa: E402


def main() -> None:
    payload = json.loads(CONTRACT.read_text())
    cells = payload["cells"]
    assert payload["experiment_id"] == "phase8-caption2p0-quarter-mf25-cfg0-rerun"
    assert payload["fixed_protocol"]["expected_rows"] == 5521
    assert payload["fixed_protocol"]["num_steps"] == 25
    assert payload["fixed_protocol"]["cfg_strength"] == 0
    assert len(cells) == len({cell["cell_id"] for cell in cells}) == 4
    assert len({cell["label"] for cell in cells}) == 4
    assert all("_mf25_cfg0_" in cell["label"] for cell in cells)
    assert sum(cell["conditioning_argv"] == ["--no_q"] for cell in cells) == 3
    assert sum(cell["conditioning_argv"] == ["--quality_level", "9"] for cell in cells) == 1
    source = WRAPPER.read_text()
    assert "--num_steps 25 --cfg_strength 0" in source
    assert '"${COND_ARGS[@]}"' in source
    assert '"$@"' not in source
    assert "requires CFG0_CONTRACT and CFG0_ARM" in source
    assert "rm -rf" not in source
    assert "_cfg4p5_" in source and "unsafe or historical label" in source
    assert "HOLD canonical CFG0 storage hard stop" in source
    test_wrapper_condition_label_mapping()
    runner = RUNNER.read_text()
    assert "unknown or duplicate arm" in runner
    assert "PREFLIGHT_OK cells=4 noq=3 q9=1 steps=25 cfg=0" in runner
    test_strict_completion_fixtures()
    test_preflight_binding_fixture()
    test_symlink_containment_fixture()
    print("PASS cfg0 four-cell contract and command manifest")


def test_wrapper_condition_label_mapping() -> None:
    env = os.environ.copy()
    env.update(CFG0_CONTRACT="fixture", CFG0_ARM="fixture")
    missing = "/home/kojiek/MeanAudio/.selftest-missing-checkpoint.pth"
    result = subprocess.run(
        [str(WRAPPER), "fixture_musiccaps_mf25_cfg0_noq", missing, "--no_q"],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "FAIL missing ckpt" in result.stderr
    assert "label conditioning does not match argv" not in result.stderr


def test_strict_completion_fixtures() -> None:
    with tempfile.TemporaryDirectory(dir="/home/kojiek") as raw:
        tmp = Path(raw)
        checkpoint = tmp / "model.pth"
        checkpoint.write_bytes(b"checkpoint")
        tsv = tmp / "musiccaps.tsv"
        tsv.write_text("id\tcaption\na\tone\n")
        label = "fixture_musiccaps_mf25_cfg0_noq"
        metrics = tmp / "metrics" / label / "metrics.txt"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("clap_score: 0.1\naes_CE: 1\naes_CU: 2\naes_PC: 3\naes_PQ: 4\n")
        report = tmp / "report.json"
        contract = tmp / "contract.json"
        sha = strict.digest
        contract.write_text(json.dumps({
            "fixed_protocol": {"tsv": str(tsv), "tsv_sha256": sha(tsv)},
            "runtime_storage": {"metrics_root": str(tmp / "metrics")},
            "cells": [{
                "cell_id": "fixture", "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint), "conditioning": "no_q",
                "label": label, "report": str(report),
            }],
        }))
        payload = {
            "status": "passed", "label": label, "protocol": strict.PROTOCOL,
            "cfg_strength": 0, "num_steps": 25, "seed": 42, "conditioning": "no_q",
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "tsv": str(tsv), "tsv_sha256": sha(tsv),
            "audio_validation": {"rows": 5521, "unique_ids": 5521, "sample_rate": 16000, "channels": 1},
            "metrics": {"clap_score": 0.1, "aes_CE": 1.0, "aes_CU": 2.0, "aes_PC": 3.0, "aes_PQ": 4.0},
            "metrics_path": str(metrics), "metrics_sha256": sha(metrics),
        }
        report.write_text(json.dumps(payload))
        try:
            strict.validate(contract, "fixture", report)
            mutations = [
                ("seed", 7), ("conditioning", "q9"),
                ("audio_validation", None), ("metrics_sha256", "0" * 64),
            ]
            for key, value in mutations:
                changed = dict(payload)
                changed[key] = value
                report.write_text(json.dumps(changed))
                try:
                    strict.validate(contract, "fixture", report)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"strict validator accepted mutation {key}")
            changed = dict(payload)
            changed["metrics"] = dict(payload["metrics"], clap_score=math.nan)
            report.write_text(json.dumps(changed))
            try:
                strict.validate(contract, "fixture", report)
            except ValueError:
                pass
            else:
                raise AssertionError("strict validator accepted NaN")
        finally:
            pass


def test_preflight_binding_fixture() -> None:
    contract = cfg0_harn.make_contract()
    original = cfg0_harn.harn.digest_file
    original_storage = cfg0_harn.harn.storage_check
    storage = original_storage()
    storage.update(verdict="pass", free_bytes=max(int(storage.get("required_bytes") or 0), 10**12))
    cfg0_harn.harn.storage_check = lambda: dict(storage)
    good = cfg0_harn.make_preflight(contract, "a" * 64, True)
    assert good["derived_verdict"] == "pass"
    target = str(cfg0_harn.WRAPPER)
    cfg0_harn.harn.digest_file = lambda path: "0" * 64 if str(path) == target else original(path)
    try:
        bad = cfg0_harn.make_preflight(contract, "a" * 64, True)
        checks = {item["check_id"]: item["verdict"] for item in bad["checks"]}
        assert checks["inputs_bound"] == "fail"
        assert bad["derived_verdict"] == "fail"
    finally:
        cfg0_harn.harn.digest_file = original
        cfg0_harn.harn.storage_check = original_storage


def test_symlink_containment_fixture() -> None:
    with (tempfile.TemporaryDirectory(dir="/home/kojiek") as root_raw,
          tempfile.TemporaryDirectory(dir="/home/kojiek") as external_raw):
        root = Path(root_raw)
        external = Path(external_raw)
        sentinel = external / "sentinel.flac"
        sentinel.write_bytes(b"preserve-me")
        redirected = root / "label"
        redirected.symlink_to(external, target_is_directory=True)
        try:
            output_path.validate_root_target(root, redirected / "audio")
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked label was accepted")
        assert sentinel.read_bytes() == b"preserve-me"
        normal = root / "normal" / "audio"
        output_path.validate_root_target(root, normal)
        os.chmod(root, 0o777)
        try:
            output_path.validate_root_target(root, normal)
        except ValueError:
            pass
        else:
            raise AssertionError("world-writable runtime root was accepted")


if __name__ == "__main__":
    main()
