#!/usr/bin/env python3
"""No-GPU provenance, resume, terminal, and notification tests for CFG full sweep."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock


ROOT = Path("/home/kojiek/MeanAudio")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = load("cfg_sweep_driver_test", ROOT / "scripts/eval/c2p0_slot0_neg_cfg_sweep_full.py")
guest = load("cfg_sweep_guest_test", ROOT / "scripts/experiment_harness/secondary_cfg_sweep_queue_guest.py")
sys.path.insert(0, "/home/kojiek/gpu_queue")
import lib_scheduler as scheduler


def write_tsv(path: Path, count: int) -> list[dict[str, str]]:
    rows = [{"id": f"id{i:04d}", "caption": f"caption {i}"} for i in range(count)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "caption"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def report_fixture(specification: dict, spec: dict, ids: list[str]) -> dict:
    inputs = {record["kind"]: record for record in specification["inputs"]}
    cfg = str(spec["cfg_strength"])
    tag = "cfg" + cfg.replace(".", "p")
    path = Path(spec["path"])
    directory = path.parent / "_audio" / f"c2p0_slot0_full_noq_{tag}_neg"
    identities = {
        key: {"path": inputs[kind]["path"], "sha256": inputs[kind]["sha256"]}
        for key, kind in {
            "checkpoint": "checkpoint", "evaluation_tsv": "evaluation_tsv",
            "clap_checkpoint": "clap_checkpoint", "scorer": "scorer_reference",
            "eval_entrypoint": "evaluation_entrypoint", "eval_utils": "evaluation_utils",
            "paired_cfg0_reference": "paired_cfg0_per_clip_reference",
        }.items()
    }
    protocol = {
        "classification": "secondary_noncanonical", "dataset": "MusicCaps", "rows": 5521,
        "solver": "MeanFlow", "steps": 25, "cfg_strength": float(cfg),
        "negative_prompt": specification["protocol"]["negative_prompt"], "seed": 42,
        "mask": "NoMask", "precision": "full", "conditioning": "NoQ",
        "encoder_name": "t5_clap", "text_c_dim": 512,
    }
    argv = [
        "/home/kojiek/venvs/dac/bin/python", inputs["evaluation_entrypoint"]["path"],
        "--variant", "meanaudio_s", "--model_path", inputs["checkpoint"]["path"],
        "--output", str(directory), "--tsv", inputs["evaluation_tsv"]["path"],
        "--use_meanflow", "--num_steps", "25", "--cfg_strength", cfg,
        "--negative_prompt", specification["protocol"]["negative_prompt"],
        "--no_text_attention_mask", "--encoder_name", "t5_clap", "--text_c_dim", "512",
        "--seed", "42", "--full_precision", "--no_q",
    ]
    metrics = {"clap": 0.2, "CE": 6.0, "CU": 7.0, "PC": 5.0, "PQ": 6.5}
    return {
        "label": spec["label"], "exp_id": spec["exp_id"], "cfg_strength": float(cfg),
        "negative_prompt": specification["protocol"]["negative_prompt"], "protocol": protocol,
        "generation_argv": argv, "input_identities": identities,
        "aggregates": {"full": {"n": 5521, **metrics}},
        "per_clip": {item: dict(metrics) for item in ids},
    }


def test_report_provenance_and_nonfinite() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tsv = root / "musiccaps.tsv"
        rows = write_tsv(tsv, 5521)
        kinds = ["checkpoint", "clap_checkpoint", "scorer_reference", "evaluation_entrypoint",
                 "evaluation_utils", "paired_cfg0_per_clip_reference"]
        inputs = [{"kind": "evaluation_tsv", "path": str(tsv), "sha256": "a" * 64}]
        for index, kind in enumerate(kinds):
            path = root / f"input{index}"
            path.write_bytes(kind.encode())
            inputs.append({"kind": kind, "path": str(path), "sha256": f"{index + 1:064x}"})
        reports = [
            {"label": guest_label, "exp_id": "exp", "cfg_strength": cfg,
             "path": str(root / f"cfg{str(cfg).replace('.', 'p')}.json")}
            for guest_label, cfg in (("c2p0_slot0_full_noq", 2.5), ("c2p0_slot0_full_noq", 4.0))
        ]
        contract = {"protocol": {"negative_prompt": driver.NEGATIVE_PROMPT},
                    "inputs": inputs, "secondary_reports": reports}
        payloads = []
        for spec in reports:
            payload = report_fixture(contract, spec, [row["id"] for row in rows])
            Path(spec["path"]).write_text(json.dumps(payload), encoding="utf-8")
            payloads.append(payload)
        assert len(guest.validate_reports(contract)) == 2

        cases = [
            ("checkpoint hash", lambda value: value["input_identities"]["checkpoint"].update(sha256="f" * 64)),
            ("ID set", lambda value: value["per_clip"].pop("id0000")),
            ("CLAP hash", lambda value: value["input_identities"]["clap_checkpoint"].update(sha256="e" * 64)),
            ("nonfinite", lambda value: value["per_clip"]["id0000"].update(PQ=float("nan"))),
        ]
        for name, mutate in cases:
            bad = json.loads(json.dumps(payloads[0]))
            mutate(bad)
            Path(reports[0]["path"]).write_text(json.dumps(bad), encoding="utf-8")
            try:
                guest.validate_reports(contract)
            except ValueError:
                pass
            else:
                raise AssertionError(f"tampered {name} report was accepted")
            Path(reports[0]["path"]).write_text(json.dumps(payloads[0]), encoding="utf-8")


def test_reportless_5521_audio_is_never_resumed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = [{"id": f"id{i:04d}", "caption": "x"} for i in range(5521)]
        old_out, old_expected = driver.OUT, driver.EXPECTED
        driver.OUT, driver.EXPECTED = root, 5521
        directory = driver.audio_dir("2.5")
        directory.mkdir(parents=True)
        for row in rows:
            (directory / f"{row['id']}.flac").touch()
        reached_generate = False

        def fake_generate(cfg, target):
            nonlocal reached_generate
            reached_generate = True
            assert target == directory and list(target.iterdir()) == []
            raise SystemExit("fixture stop before GPU")

        try:
            with (mock.patch.object(driver, "storage_gate", return_value=10**12),
                  mock.patch.object(driver, "generate", fake_generate)):
                try:
                    driver.run_cfg("2.5", rows)
                except SystemExit as error:
                    assert "fixture stop" in str(error)
                else:
                    raise AssertionError("fixture unexpectedly continued")
            assert reached_generate
        finally:
            driver.OUT, driver.EXPECTED = old_out, old_expected


def test_verified_report_residual_cleanup_is_restart_safe() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = [{"id": "a", "caption": "x"}, {"id": "b", "caption": "x"}]
        old_out, old_expected = driver.OUT, driver.EXPECTED
        driver.OUT, driver.EXPECTED = root, 2
        directory = driver.audio_dir("2.5")
        directory.mkdir(parents=True)
        (directory / "b.flac").touch()
        driver.report_path("2.5").write_text("{}", encoding="utf-8")
        try:
            with (mock.patch.object(driver, "storage_gate", return_value=10**12),
                  mock.patch.object(driver, "validate_finished_report", return_value=True)):
                driver.run_cfg("2.5", rows)
            assert not directory.exists()
        finally:
            driver.OUT, driver.EXPECTED = old_out, old_expected


def test_terminal_evidence_classifies_done_only_with_anchor() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = root / "029.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        ema = root / "ema.pth"
        ema.write_bytes(b"ema")
        report = root / "cfg0.json"
        report.write_text(json.dumps({
            "status": "passed", "cfg_strength": 0, "protocol": "MusicCaps 5521",
            "checkpoint": str(ema),
            "metrics": {"clap_score": 0.2, "aes_CE": 6, "aes_CU": 7, "aes_PC": 5, "aes_PQ": 6.5},
        }), encoding="utf-8")
        terminal = script.with_name("029.terminal.json")
        terminal.write_text(json.dumps({"status": "completed", "evidence": {
            "ema": str(ema), "cfg0_report": str(report), "secondary_reports": [{}, {}],
        }}), encoding="utf-8")
        assert scheduler.classify_exit("p2", script, 0) == "done"
        terminal.write_text(json.dumps({"status": "completed", "evidence": {
            "secondary_reports": [{}, {}],
        }}), encoding="utf-8")
        assert scheduler.classify_exit("p2", script, 0) == "held"


def test_notification_idempotence_and_ambiguity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        old_root, old_ledger = guest.STATE_ROOT, guest.EVENT_LEDGER
        guest.STATE_ROOT, guest.EVENT_LEDGER = root, root / "ledger.json"
        calls = []
        try:
            guest.ensure_state()
            with mock.patch.object(guest.subprocess, "run", side_effect=lambda *a, **k: calls.append(a) or mock.Mock(returncode=0)):
                guest.notify_once("event-a", "held", "start")
                guest.notify_once("event-a", "held", "start")
            assert len(calls) == 1
            guest.update_event("event-b", delivery="attempted")
            try:
                guest.notify_once("event-b", "held", "ambiguous")
            except RuntimeError:
                pass
            else:
                raise AssertionError("ambiguous notification was retried")
            with mock.patch.object(guest.subprocess, "run", return_value=mock.Mock(returncode=7)):
                try:
                    guest.notify_once("event-c", "held", "must hold")
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("notification failure did not fail closed")
            assert json.loads(guest.EVENT_LEDGER.read_text())["events"]["event-c"]["delivery"] == "failed"
        finally:
            guest.STATE_ROOT, guest.EVENT_LEDGER = old_root, old_ledger


def test_storage_and_stall_boundaries() -> None:
    assert guest.storage_verdict(100, 80, 60) == "pass"
    assert guest.storage_verdict(79, 80, 60) == "warning"
    assert guest.storage_verdict(59, 80, 60) == "hard_stop"
    snapshot = (0, 10, 123)
    assert not guest.is_stalled(snapshot, snapshot, 1799, 0, threshold=1800)
    assert guest.is_stalled(snapshot, snapshot, 1800, 0, threshold=1800)
    assert not guest.is_stalled((0, 11, 123), snapshot, 1800, 0, threshold=1800)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} no-GPU CFG sweep tests")


if __name__ == "__main__":
    main()
