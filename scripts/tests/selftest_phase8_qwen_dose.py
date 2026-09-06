#!/usr/bin/env python3
"""Structural regression tests for the Phase-8 Qwen caption-dose chain."""

from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

import phase8_qwen_dose_queue as queue  # noqa: E402
import phase8_qwen_dose_monitor as monitor  # noqa: E402
from phase8_qwen_dose_audit import parse_metrics  # noqa: E402
from phase8_qwen_dose_paired_report import validate_paired_bootstrap  # noqa: E402
from phase8_qwen_dose_provenance import (  # noqa: E402
    sha256_file as provenance_sha256,
    validate_nested_cache_provenance,
)
from phase8_qwen_parent_completion_gate import parent_completion_passed  # noqa: E402


def main() -> int:
    assert queue.IMPLEMENTATION_RELATIVE_PATHS == monitor.IMPLEMENTATION_RELATIVE_PATHS
    with tempfile.TemporaryDirectory(prefix="phase8-qwen-dose-test-") as raw:
        root = Path(raw)
        old_run_root = queue.RUN_ROOT
        old_parent_root = queue.PARENT_ROOT
        try:
            queue.RUN_ROOT = root / "dose"
            queue.PARENT_ROOT = root / "parent"
            steps = queue.build_queue("fresh")
            names = [name for name, _ in steps]
            assert len(steps) == 20
            assert names[:5] == [
                "50k_control_train", "50k_control_eval", "50k_control_metrics",
                "50k_control_audit", "50k_qwen_train",
            ]
            assert names[9] == "50k_paired_report"
            assert names[10] == "100k_control_train"
            assert names[-1] == "100k_paired_report"
            commands = dict(steps)
            control_50 = commands["50k_control_train"]
            qwen_100 = commands["100k_qwen_train"]
            assert "num_iterations=650000" in control_50
            assert "num_iterations=700000" in qwen_100
            assert "+use_q_conditioning=false" in control_50
            assert "+use_text_attention_mask=false" in control_50
            assert "+use_rope=False" in control_50
            assert any(value.endswith("phase8_qwen_official_matched_control_20k_ckpt_last.pth") for value in control_50)
            assert any(value.endswith("phase8_qwen_dose_qwen_50k_ckpt_last.pth") for value in qwen_100)
            assert queue.metrics_path("50k", "control") == root / "dose/50k/control/musiccaps_metrics/phase8_qwen_dose_control_50k/metrics.txt"
            report_100 = commands["100k_paired_report"]
            iteration_index = report_100.index("--iteration")
            assert report_100[iteration_index + 1] == "700000"
        finally:
            queue.RUN_ROOT = old_run_root
            queue.PARENT_ROOT = old_parent_root

        metrics = root / "metrics.txt"
        metrics.write_text(
            "clap_score: 0.2\naes_CE: 5.0\naes_CU: 6.0\naes_PC: 4.0\naes_PQ: 6.1\n",
            encoding="utf-8",
        )
        assert set(parse_metrics(metrics)) == {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}

        paired = {
            "n": 5521, "mean_delta_treatment_minus_baseline": 0.01,
            "ci95_low": 0.0, "ci95_high": 0.02, "baseline_mean": 0.18,
            "treatment_mean": 0.19, "paired_id_sha256": "a" * 64,
            "bootstrap_seed": 14159265, "bootstrap_replicates": 10000,
            "tsv": "/eval.tsv", "baseline_dir": "/control", "treatment_dir": "/qwen",
        }
        validate_paired_bootstrap(
            paired, expected_tsv="/eval.tsv", expected_baseline_dir="/control",
            expected_treatment_dir="/qwen", expected_id_sha256="a" * 64,
        )
        try:
            validate_paired_bootstrap({**paired, "n": 5520})
        except RuntimeError:
            pass
        else:
            raise AssertionError("paired bootstrap count drift was accepted")
        malformed = {**paired, "ci95_low": float("nan")}
        try:
            validate_paired_bootstrap(malformed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-finite bootstrap field was accepted")
        malformed = {**paired, "mean_delta_treatment_minus_baseline": "bad"}
        try:
            validate_paired_bootstrap(malformed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-numeric bootstrap field was accepted")
        malformed = {**paired, "paired_id_sha256": ""}
        try:
            validate_paired_bootstrap(malformed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid bootstrap digest was accepted")
        mismatch_cases = [
            {"bootstrap_seed": 7},
            {"bootstrap_replicates": 9999},
            {"tsv": "/wrong.tsv"},
            {"baseline_dir": "/wrong-control"},
            {"treatment_dir": "/wrong-qwen"},
            {"paired_id_sha256": "b" * 64},
            {"ci95_low": 0.03, "ci95_high": 0.02},
            {"treatment_mean": 0.195},
        ]
        for changes in mismatch_cases:
            try:
                validate_paired_bootstrap(
                    {**paired, **changes}, expected_tsv="/eval.tsv",
                    expected_baseline_dir="/control", expected_treatment_dir="/qwen",
                    expected_id_sha256="a" * 64,
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"bootstrap mismatch was accepted: {changes}")

        control_npz = root / "control_npz"
        control_npz.mkdir()
        canonical = root / "MANIFEST.tsv"
        canonical.write_text("row\n", encoding="utf-8")
        validation = root / "FULL_VALIDATION.json"
        validation.write_text(json.dumps({
            "status": "passed",
            "paths": {"manifest": str(canonical), "output_dir": str(control_npz)},
            "sha256": {"manifest": provenance_sha256(canonical)},
        }), encoding="utf-8")
        control_outer = {
            "validation_report": str(validation),
            "validation_report_sha256": provenance_sha256(validation),
        }
        validate_nested_cache_provenance("control", control_npz, control_outer)
        canonical.write_text("drift\n", encoding="utf-8")
        try:
            validate_nested_cache_provenance("control", control_npz, control_outer)
        except RuntimeError:
            pass
        else:
            raise AssertionError("control canonical manifest drift was accepted")
        canonical.write_text("row\n", encoding="utf-8")
        validation.write_text(validation.read_text(encoding="utf-8") + " ", encoding="utf-8")
        try:
            validate_nested_cache_provenance("control", control_npz, control_outer)
        except RuntimeError:
            pass
        else:
            raise AssertionError("control validation report drift was accepted")

        qwen_npz = root / "qwen_npz"
        qwen_npz.mkdir()
        mapper = root / "mapper.json"
        mapper.write_text("{}\n", encoding="utf-8")
        boundary = qwen_npz / "boundary.npz"
        boundary.write_bytes(b"fixed-boundary")
        qwen_outer = {
            "mapper_manifest": str(mapper),
            "mapper_manifest_sha256": provenance_sha256(mapper),
            "resume_boundary": {
                "name": boundary.name, "sha256": provenance_sha256(boundary),
            },
        }
        validate_nested_cache_provenance("qwen", qwen_npz, qwen_outer)
        mapper.write_text('{"drift":true}\n', encoding="utf-8")
        try:
            validate_nested_cache_provenance("qwen", qwen_npz, qwen_outer)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Qwen mapper drift was accepted")
        mapper.write_text("{}\n", encoding="utf-8")
        boundary.write_bytes(b"drifted-boundary")
        try:
            validate_nested_cache_provenance("qwen", qwen_npz, qwen_outer)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Qwen boundary drift was accepted")

        report = root / "parent_report.json"
        manifest = root / "parent_manifest.json"
        report.write_text('{"status":"passed"}\n', encoding="utf-8")
        manifest.write_text(
            '{"steps":{"paired_final_report":{"status":"passed","exit_code":0}}}\n',
            encoding="utf-8",
        )
        assert parent_completion_passed(report, manifest, queue_active=False)
        assert not parent_completion_passed(report, manifest, queue_active=True)
        manifest.write_text(
            '{"steps":{"paired_final_report":{"status":"running"}}}\n', encoding="utf-8"
        )
        assert not parent_completion_passed(report, manifest, queue_active=False)

        sol = root / "sol.json"
        sol.write_text('{"verdict":"approve_predeclared_dose_chain"}\n', encoding="utf-8")
        auth = root / "auth.json"
        auth.write_text(json.dumps({
            "status": "approved", "codex_reviewed": True, "run_mode": "fresh",
            "contract_sha256": queue.sha256_file(queue.CONTRACT),
            "implementation_sha256": queue.implementation_hashes(),
            "sol_verdict_path": str(sol), "sol_verdict_sha256": queue.sha256_file(sol),
        }), encoding="utf-8")
        queue.verify_authorization(auth, "fresh")
        drifted = json.loads(auth.read_text(encoding="utf-8"))
        drifted["implementation_sha256"] = {**drifted["implementation_sha256"], "bad": "hash"}
        auth.write_text(json.dumps(drifted), encoding="utf-8")
        try:
            queue.verify_authorization(auth, "fresh")
        except RuntimeError:
            pass
        else:
            raise AssertionError("implementation hash drift was accepted")
        auth.write_text(json.dumps({
            **drifted, "implementation_sha256": queue.implementation_hashes(),
        }), encoding="utf-8")
        try:
            queue.verify_authorization(auth, "resume")
        except RuntimeError:
            pass
        else:
            raise AssertionError("fresh authorization was accepted for resume")

        resume_sol = root / "resume_sol.json"
        resume_sol.write_text('{"verdict":"resume_identical_contract"}\n', encoding="utf-8")
        resume_auth = root / "resume_auth.json"
        resume_payload = {
            "status": "approved", "codex_reviewed": True, "run_mode": "resume",
            "same_prefix_and_contract": True,
            "contract_sha256": queue.sha256_file(queue.CONTRACT),
            "implementation_sha256": queue.implementation_hashes(),
            "sol_verdict_path": str(resume_sol),
            "sol_verdict_sha256": queue.sha256_file(resume_sol),
        }
        resume_auth.write_text(json.dumps(resume_payload), encoding="utf-8")
        queue.verify_authorization(resume_auth, "resume")
        resume_auth.write_text(json.dumps({
            **resume_payload, "same_prefix_and_contract": False,
        }), encoding="utf-8")
        try:
            queue.verify_authorization(resume_auth, "resume")
        except RuntimeError:
            pass
        else:
            raise AssertionError("resume without identical prefix/contract was accepted")

        provenance = queue.validate_data_provenance()
        assert provenance["arms"]["control"]["rows"] == 251599
        original_control_tsv = queue.ARMS["control"]["tsv"]
        bad_tsv = root / "bad.tsv"
        bad_tsv.write_text("id\tcaption\tq_level\nx\ty\t1\n", encoding="utf-8")
        queue.ARMS["control"]["tsv"] = bad_tsv
        try:
            queue.validate_data_provenance()
        except RuntimeError:
            pass
        else:
            raise AssertionError("training TSV provenance drift was accepted")
        finally:
            queue.ARMS["control"]["tsv"] = original_control_tsv
    print("[PASS] phase8 Qwen dose self-tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
