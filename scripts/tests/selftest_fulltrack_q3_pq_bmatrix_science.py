#!/usr/bin/env python3
"""CPU-only tests for the Fulltrack-Q3 scientific scorer and report.

Every fixture is synthetic.  The tests import the scientific modules directly,
replace model calls with deterministic fakes, and set ``CUDA_VISIBLE_DEVICES``
empty before any optional torch import could occur.  No live checkpoint, queue,
sealed staging tree, or GPU is touched.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
SCORE_PATH = ROOT / "scripts/eval/score_musiccaps_per_item.py"
REPORT_PATH = ROOT / "scripts/analysis/fulltrack_q3_paired_report.py"
TEST_FIXTURE_ROOT = Path(
    "/home/kojiek/cfg0_eval_runtime/fulltrack_q3_pq_bmatrix_v1/gate1_test_fixtures"
)


def fixture_temporary_directory() -> tempfile.TemporaryDirectory:
    """Allocate test-only state below the exact Gate-1 fixture root."""

    if TEST_FIXTURE_ROOT.is_symlink() or not TEST_FIXTURE_ROOT.is_dir():
        raise RuntimeError(f"approved test fixture root is not a real directory: {TEST_FIXTURE_ROOT}")
    metadata = TEST_FIXTURE_ROOT.stat()
    if metadata.st_uid != os.geteuid() or (metadata.st_mode & 0o777) != 0o700:
        raise RuntimeError(f"approved test fixture root has unsafe metadata: {TEST_FIXTURE_ROOT}")
    return tempfile.TemporaryDirectory(dir=str(TEST_FIXTURE_ROOT), prefix="science-")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score = load_module("ftq3_science_score", SCORE_PATH)
report = load_module("ftq3_science_report", REPORT_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fixture_tsv(path: Path, ids: list[str]) -> None:
    write_tsv(
        path,
        [{"id": item, "caption": f"caption {item}"} for item in ids],
        ["id", "caption"],
    )


def fixture_audio(directory: Path, ids: list[str]) -> None:
    directory.mkdir()
    for index, item in enumerate(ids):
        # 10 seconds is not needed for the scorer's structural validation; a
        # deterministic non-empty 16 kHz mono clip is sufficient.
        audio = np.full(1600, (index + 1) / 10.0, dtype=np.float32)
        sf.write(directory / f"{item}.flac", audio, 16000)


class FakeClap:
    def get_audio_embedding_from_filelist(self, paths, *, use_tensor):
        assert use_tensor is True
        return [[1.0, 0.0] for _ in paths]

    def get_text_embedding(self, captions, *, use_tensor):
        assert use_tensor is True
        return [[1.0, 0.0] for _ in captions]


class FakeAes:
    def forward(self, batch):
        return [
            {"CE": 6.0 + i, "CU": 7.0 + i, "PC": 5.0 + i, "PQ": 6.5 + i}
            for i, _ in enumerate(batch)
        ]


def test_scorer_exact_ids_and_mocked_models() -> None:
    with fixture_temporary_directory() as temporary:
        root = Path(temporary)
        ids = ["mc_a", "mc_b", "mc_c"]
        tsv = root / "musiccaps.tsv"
        audio = root / "audio"
        out = root / "per_item.tsv"
        fixture_tsv(tsv, ids)
        fixture_audio(audio, ids)
        records = score.read_musiccaps_tsv(tsv, expected_count=3)
        mapping = score.validate_audio_directory(audio, records)
        rows = score.score_records(records, mapping, FakeClap(), FakeAes(), batch_size=2)
        assert [row["id"] for row in rows] == ids
        assert rows[0]["clap_score"] == 1.0
        assert rows[2]["aes_PQ"] == 6.5
        score._atomic_write_tsv(out, rows)
        parsed = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
        assert list(parsed[0]) == ["id", *score.METRIC_KEYS]
        assert len(parsed) == 3
        try:
            score._atomic_write_tsv(out, rows)
        except score.ScoringInputError:
            pass
        else:
            raise AssertionError("stale metric output was overwritten")


def test_scorer_cli_cpu_mock_injection() -> None:
    """Exercise the public CLI while keeping both expensive loaders mocked."""
    with fixture_temporary_directory() as temporary:
        root = Path(temporary)
        ids = ["mc_a", "mc_b", "mc_c"]
        tsv = root / "musiccaps.tsv"
        audio = root / "audio"
        checkpoint = root / "clap.pt"
        snapshot = root / "audiobox"
        out = root / "metrics.tsv"
        fixture_tsv(tsv, ids)
        fixture_audio(audio, ids)
        checkpoint.write_bytes(b"synthetic clap checkpoint")
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"synthetic aes model")
        original_clap = score.load_clap_model
        original_aes = score.load_aes_predictor
        score.load_clap_model = lambda *args, **kwargs: FakeClap()
        score.load_aes_predictor = lambda *args, **kwargs: FakeAes()
        try:
            exit_code = score.main(
                [
                    "--tsv", str(tsv), "--audio-dir", str(audio), "--out", str(out),
                    "--clap-checkpoint", str(checkpoint), "--audiobox-snapshot", str(snapshot),
                    "--local-files-only", "--require-exact-count", "3", "--device", "cpu",
                    "--batch-size", "2",
                ]
            )
        finally:
            score.load_clap_model = original_clap
            score.load_aes_predictor = original_aes
        assert exit_code == 0
        assert out.is_file()
        assert len(list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))) == 3


def test_scorer_missing_extra_nonfinite_inputs() -> None:
    with fixture_temporary_directory() as temporary:
        root = Path(temporary)
        ids = ["a", "b"]
        tsv = root / "musiccaps.tsv"
        fixture_tsv(tsv, ids)
        records = score.read_musiccaps_tsv(tsv, expected_count=2)

        missing = root / "missing"
        fixture_audio(missing, ["a"])
        try:
            score.validate_audio_directory(missing, records)
        except score.ScoringInputError:
            pass
        else:
            raise AssertionError("missing audio ID accepted")

        extra = root / "extra"
        fixture_audio(extra, ids)
        sf.write(extra / "unexpected.flac", np.ones(20, dtype=np.float32), 16000)
        try:
            score.validate_audio_directory(extra, records)
        except score.ScoringInputError:
            pass
        else:
            raise AssertionError("extra audio ID accepted")

        bad_tsv = root / "bad.tsv"
        write_tsv(bad_tsv, [{"id": "../escape", "caption": "bad"}], ["id", "caption"])
        try:
            score.read_musiccaps_tsv(bad_tsv, expected_count=1)
        except score.ScoringInputError:
            pass
        else:
            raise AssertionError("unsafe ID accepted")

        bad_metrics = root / "bad_metrics.tsv"
        write_tsv(
            bad_metrics,
            [{"id": "a", **{key: "1" for key in report.METRIC_KEYS}},
             {"id": "b", **{key: "nan" if key == "aes_PQ" else "1" for key in report.METRIC_KEYS}}],
            ["id", *report.METRIC_KEYS],
        )
        try:
            report._parse_metric_rows(bad_metrics, ids)
        except report.AnalysisInputError:
            pass
        else:
            raise AssertionError("nonfinite metric accepted")


def test_decimal_rounding_and_classifier_boundaries() -> None:
    assert str(report.quantize_metric("1.23485")) == "1.2349"
    assert str(report.quantize_metric("-1.23485")) == "-1.2349"
    assert report.classify_contrast(0.05, 0.0001, 0.1, 0.05)["classification"] == "positive_supported"
    assert report.classify_contrast(0.05, 0.0, 0.1, 0.05)["classification"] == "small_or_uncertain"
    assert report.classify_contrast(-0.05, -0.1, -0.0001, 0.05)["classification"] == "negative_supported"
    assert report.classify_contrast(-0.0499, -0.2, -0.1, 0.05)["classification"] == "small_or_uncertain"
    assert report.classify_contrast(float("nan"), 0.1, 0.2, 0.05)["classification"] == "invalid"
    assert report.classify_contrast(0.1, 0.2, 0.1, 0.05)["classification"] == "invalid"


def test_bootstrap_is_deterministic() -> None:
    values = [0.1, 0.2, -0.1, 0.0, 0.3]
    first = report.paired_bootstrap(values, replicates=500, seed=20260828)
    second = report.paired_bootstrap(values, replicates=500, seed=20260828)
    assert first == second
    assert first[0] <= first[1]


def _write_arm_fixture(
    root: Path,
    arm_id: str,
    ids: list[str],
    metrics: dict[str, str],
) -> dict[str, Any]:
    arm = root / arm_id
    metrics_path = arm / "metrics" / "per_item.tsv"
    metrics_path.parent.mkdir(parents=True)
    rows = [{"id": item, **metrics} for item in ids]
    write_tsv(metrics_path, rows, ["id", *report.METRIC_KEYS])
    aggregate = {key: float(report.quantize_metric(value)) for key, value in metrics.items()}
    manifest_path = arm / "manifests" / "audio_sha256.tsv"
    manifest_path.parent.mkdir(parents=True)
    write_tsv(
        manifest_path,
        [{"id": item, "path": f"{item}.flac", "bytes": "10", "sha256": "a" * 64} for item in ids],
        ["id", "path", "bytes", "sha256"],
    )
    report_path = arm / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "metrics": aggregate,
                "per_item_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
                "audio_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "per_item_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
        "aggregate_report": {"path": str(report_path), "sha256": sha256(report_path)},
        "audio_hash_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
    }


def _historical_fixture(root: Path, arm_id: str, metrics: dict[str, float]) -> dict[str, str]:
    path = root / f"historical_{arm_id}.json"
    path.write_text(
        json.dumps({"status": "passed", "metrics": metrics}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"path": str(path), "sha256": sha256(path)}


def test_full_report_b1_b6_and_target_precedence() -> None:
    with fixture_temporary_directory() as temporary:
        root = Path(temporary)
        ids = ["a", "b", "c", "d"]
        vectors = {
            "B1": {"clap_score": 0.1821, "aes_CE": 6.8458, "aes_CU": 7.1468, "aes_PC": 5.3016, "aes_PQ": 6.9337},
            "B2": {"clap_score": 0.2145, "aes_CE": 6.2474, "aes_CU": 6.7019, "aes_PC": 5.1752, "aes_PQ": 6.5437},
        }
        arms: dict[str, Any] = {}
        for arm_id in report.ARM_IDS:
            if arm_id in vectors:
                values = {key: f"{value:.4f}" for key, value in vectors[arm_id].items()}
            elif arm_id == "B6":
                values = {
                    "clap_score": "0.2000", "aes_CE": "6.0000", "aes_CU": "6.0000",
                    "aes_PC": "5.0000", "aes_PQ": "6.9000",
                }
            else:
                values = {
                    "clap_score": "0.2000",
                    "aes_CE": "6.0000",
                    "aes_CU": "6.0000",
                    "aes_PC": "5.0000",
                    "aes_PQ": "6.0000",
                }
            arms[arm_id] = _write_arm_fixture(root, arm_id, ids, values)

        b1_hist = _historical_fixture(root, "B1", vectors["B1"])
        b2_hist = _historical_fixture(root, "B2", vectors["B2"])
        audit_path = root / "audit.json"
        audit_path.write_text(
            json.dumps({"historical_metrics": vectors["B1"]}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        contract_path = root / "contract.json"
        contract = {
            "plan_id": "FTQ3-BMATRIX-v1",
            "protocol": {"unique_ids": len(ids)},
            "expected_ids": ids,
            "arms": arms,
            "reproduction_gate": {
                "historical_vectors": vectors,
                "historical_reports": {"B1": b1_hist, "B2": b2_hist},
                "audit_source": {"path": str(audit_path), "sha256": sha256(audit_path)},
            },
        }
        contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
        payload = report.build_report(contract_path, replicates=100, seed=20260828)
        assert payload["status"] == "passed"
        assert payload["reproduction"]["verdict"] == "passed"
        target = payload["decisions"]["non_fulltrack_target"]
        assert target["status"] == "canonical_non_fulltrack_pq_ge_6_9_achieved"
        assert target["qualifying_canonical_arms"] == ["B6"]
        assert payload["contrasts"]["Q_inference_fulltrack"]["metrics"]["aes_PQ"]["n"] == len(ids)
        assert payload["contract"]["sha256"] == sha256(contract_path)


def test_reproduction_only_does_not_require_b3_b6() -> None:
    with fixture_temporary_directory() as temporary:
        root = Path(temporary)
        ids = ["a", "b"]
        vector = {
            "clap_score": 0.1821, "aes_CE": 6.8458, "aes_CU": 7.1468,
            "aes_PC": 5.3016, "aes_PQ": 6.9337,
        }
        arms = {}
        for arm_id in ("B1", "B2"):
            arms[arm_id] = _write_arm_fixture(
                root, arm_id, ids, {key: f"{value:.4f}" for key, value in vector.items()}
            )
        b1_hist = _historical_fixture(root, "B1", vector)
        b2_hist = _historical_fixture(root, "B2", vector)
        audit = root / "audit.json"
        audit.write_text(json.dumps({"historical_metrics": vector}) + "\n", encoding="utf-8")
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "protocol": {"unique_ids": len(ids)},
                    "expected_ids": ids,
                    "arms": {"B1": arms["B1"], "B2": arms["B2"]},
                    "reproduction_gate": {
                        "historical_vectors": {"B1": vector, "B2": vector},
                        "historical_reports": {"B1": b1_hist, "B2": b2_hist},
                        "audit_source": {"path": str(audit), "sha256": sha256(audit)},
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        payload = report.build_reproduction_report(contract)
        assert payload["decision"] == "passed"
        assert set(payload["arms"]) == {"B1", "B2"}


def test_invalid_precedence_and_secondary_target() -> None:
    arms = {
        arm_id: {
            "status": "passed",
            "aggregate_decimal": {"aes_PQ": "6.0"},
        }
        for arm_id in report.ARM_IDS
    }
    arms["B5"]["aggregate_decimal"]["aes_PQ"] = "6.9000"
    invalid = {"classification": "invalid"}
    positive = {"classification": "positive_supported"}
    contrasts = {
        "Q_inference_fulltrack": {"metrics": {"aes_PQ": invalid}},
        "Q_inference_segment_slot0": {"metrics": {"aes_PQ": positive}},
        "checkpoint_family_q9": {"metrics": {"aes_PQ": positive}},
        "checkpoint_family_q0": {"metrics": {"aes_PQ": positive}},
        "checkpoint_family_noq": {"metrics": {"aes_PQ": positive}},
    }
    decisions = report.classify_decisions(arms, contrasts, {"verdict": "passed"})
    assert decisions["q_inference_association"]["status"] == "invalid"
    assert decisions["non_fulltrack_target"]["status"] == "secondary_q0_non_fulltrack_pq_ge_6_9_only"


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tests = [
        test_scorer_exact_ids_and_mocked_models,
        test_scorer_cli_cpu_mock_injection,
        test_scorer_missing_extra_nonfinite_inputs,
        test_decimal_rounding_and_classifier_boundaries,
        test_bootstrap_is_deterministic,
        test_full_report_b1_b6_and_target_precedence,
        test_reproduction_only_does_not_require_b3_b6,
        test_invalid_precedence_and_secondary_target,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    if failures:
        raise SystemExit(f"{failures} science self-test(s) failed")
    print(f"[OK] all {len(tests)} science self-tests passed; CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}")


if __name__ == "__main__":
    main()
