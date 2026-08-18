#!/usr/bin/env python3
"""CPU-only self-tests for Phase-8 Fixed-Q / ATTM chain tooling.

Uses synthetic fixtures only.  Does not load the live 2.4 GB checkpoint and
does not touch GPU training artifacts.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PY = sys.executable


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_ok(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        **kwargs,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"cmd failed ({proc.returncode}): {cmd}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_fixedq9_tsv_builder() -> None:
    mod_path = SCRIPTS / "preprocess" / "make_phase8_fixedq9_tsv.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "catalog.tsv"
        out = tmp_path / "fixed.tsv"
        man = tmp_path / "fixed.manifest.json"
        rows = [
            {"id": f"track{i}_segment_0_0", "caption": f"cap {i}", "q_level": str(i % 10)}
            for i in range(12)
        ]
        write_tsv(src, rows, ["id", "caption", "q_level"])
        run_ok(
            [
                PY,
                str(mod_path),
                "--input",
                str(src),
                "--output",
                str(out),
                "--manifest",
                str(man),
                "--expected-rows",
                "12",
                "--fixed-q",
                "9",
            ]
        )
        out_rows = list(csv.DictReader(out.open(), delimiter="\t"))
        assert len(out_rows) == 12
        assert {int(r["q_level"]) for r in out_rows} == {9}
        for a, b in zip(rows, out_rows):
            assert a["id"] == b["id"]
            assert a["caption"] == b["caption"]
        payload = json.loads(man.read_text())
        assert payload["unique_q_support"] == [9]
        assert payload["rows"] == 12
        assert "input_sha256" in payload and "output_sha256" in payload
        # Fresh-only: second write must fail.
        proc = subprocess.run(
            [
                PY,
                str(mod_path),
                "--input",
                str(src),
                "--output",
                str(out),
                "--manifest",
                str(man),
                "--expected-rows",
                "12",
            ],
            text=True,
            capture_output=True,
        )
        assert proc.returncode != 0


def _synthetic_ckpt(path: Path, *, it: int = 600_000) -> None:
    dim = 8
    q = torch.randn(11, dim)
    # Distinct null row for fixedq9 distance checks.
    q[10] = torch.ones(dim)
    weights = {"q_embed.weight": q.clone(), "other.weight": torch.randn(4, 4)}
    ema = {
        "shadow0.q_embed.weight": q.clone() + 0.01,
        "shadow1.q_embed.weight": q.clone() - 0.01,
        "shadow0.other.weight": torch.randn(4, 4),
    }
    # Ensure EMA q rows are not already equal so fixedq9 mutates them.
    ema["shadow0.q_embed.weight"][10] = torch.full((dim,), 2.0)
    ema["shadow1.q_embed.weight"][10] = torch.full((dim,), 3.0)
    state = {
        "it": it,
        "weights": weights,
        "ema": ema,
        "optimizer": {"fake": True},
        "scheduler": {"fake": True},
    }
    torch.save(state, path)


def test_checkpoint_init_fixedq9_and_noq() -> None:
    init = SCRIPTS / "init_phase8_fixedq_attm_checkpoint.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.pth"
        _synthetic_ckpt(source)

        # fixedq9
        out_f = tmp_path / "fixed.pth"
        man_f = tmp_path / "fixed.json"
        run_ok(
            [
                PY,
                str(init),
                "--source",
                str(source),
                "--output",
                str(out_f),
                "--manifest",
                str(man_f),
                "--mode",
                "fixedq9",
                "--expected-it",
                "600000",
            ]
        )
        state = torch.load(out_f, map_location="cpu", weights_only=False)
        assert state["optimizer"] is None and state["scheduler"] is None
        w = state["weights"]["q_embed.weight"]
        assert torch.equal(w[:10], w[10].unsqueeze(0).expand_as(w[:10]))
        for key in sorted(k for k in state["ema"] if k.endswith("q_embed.weight")):
            ew = state["ema"][key]
            assert torch.equal(ew[:10], ew[10].unsqueeze(0).expand_as(ew[:10]))
        man = json.loads(man_f.read_text())
        assert man["initialization"] == "copy_q10_exactly_to_q0_through_q9"
        assert man["optimizer_reset"] is True
        assert len(man["audited_tensors"]) == 3
        assert all(v["exactly_equal_after"] is True for v in man["audited_tensors"].values())

        # noq preserve
        out_n = tmp_path / "noq.pth"
        man_n = tmp_path / "noq.json"
        before = torch.load(source, map_location="cpu", weights_only=False)
        run_ok(
            [
                PY,
                str(init),
                "--source",
                str(source),
                "--output",
                str(out_n),
                "--manifest",
                str(man_n),
                "--mode",
                "noq",
                "--expected-it",
                "600000",
            ]
        )
        after = torch.load(out_n, map_location="cpu", weights_only=False)
        assert after["optimizer"] is None and after["scheduler"] is None
        assert torch.equal(
            before["weights"]["q_embed.weight"], after["weights"]["q_embed.weight"]
        )
        man2 = json.loads(man_n.read_text())
        assert man2["initialization"] == "preserve_q_embed_matched_optimizer_reset"
        assert all(v.get("mutated") is False for v in man2["audited_tensors"].values())

        # wrong iteration fails
        bad_src = tmp_path / "bad.pth"
        _synthetic_ckpt(bad_src, it=123)
        proc = subprocess.run(
            [
                PY,
                str(init),
                "--source",
                str(bad_src),
                "--output",
                str(tmp_path / "x.pth"),
                "--manifest",
                str(tmp_path / "x.json"),
                "--mode",
                "noq",
            ],
            text=True,
            capture_output=True,
        )
        assert proc.returncode != 0


def test_bootstrap_ci_math() -> None:
    mod = load_module(
        "paired_clap_fixedq",
        SCRIPTS / "eval" / "paired_clap_bootstrap_phase8_fixedq_attm.py",
    )
    import numpy as np

    diff = np.asarray([0.01] * 100, dtype=np.float64)
    ci = mod.bootstrap_ci(diff, seed=1, samples=2000)
    assert ci[0] > 0 and ci[1] > 0
    zero = np.zeros(50, dtype=np.float64)
    ci0 = mod.bootstrap_ci(zero, seed=2, samples=1000)
    assert math.isclose(ci0[0], 0.0, abs_tol=1e-12)
    assert math.isclose(ci0[1], 0.0, abs_tol=1e-12)


def test_official_caption_inventory_cpu_only() -> None:
    inv = SCRIPTS / "preprocess" / "build_official_caption_inventory.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local = tmp_path / "local.tsv"
        write_tsv(
            local,
            [
                {"id": "00_100_segment_0_0", "caption": "local alpha"},
                {"id": "200_segment_1_0", "caption": "local beta"},
                {"id": "300_segment_0_0", "caption": "local gamma"},
            ],
            ["id", "caption"],
        )
        qwen = tmp_path / "qwen.json"
        qwen.write_text(
            json.dumps(
                [
                    {"path": "00/100.mp3", "caption": "local alpha"},
                    {"track_id": "200", "caption": "official different"},
                    {"track_id": "999", "caption": "only official"},
                ]
            )
        )
        mf = tmp_path / "mf.json"
        mf.write_text(json.dumps({"100": "local alpha", "400": "mf only"}))
        out = tmp_path / "inventory.json"
        run_ok(
            [
                PY,
                str(inv),
                "--local-tsv",
                str(local),
                "--official-qwen-json",
                str(qwen),
                "--official-musicflamingo-json",
                str(mf),
                "--output",
                str(out),
            ]
        )
        payload = json.loads(out.read_text())
        assert payload["gpu_used"] is False
        assert payload["captions_encoded"] is False
        assert len(payload["inventories"]) == 2
        qwen_inv = next(x for x in payload["inventories"] if x["name"] == "qwen")
        assert qwen_inv["intersection_tracks"] == 2
        assert qwen_inv["official_only_tracks"] == 1


def test_monitor_grad_health_logic() -> None:
    """Parse synthetic log lines for persistent vs isolated nonfinite grad."""
    # Inline replicate of trailing logic used by the monitor.
    values = [1.0, float("nan"), 1.2, 1.1]
    bad = [not math.isfinite(v) for v in values]
    trailing = 0
    for value in reversed(bad):
        if not value:
            break
        trailing += 1
    assert trailing == 0  # recovered
    assert sum(bad) == 1

    values2 = [1.0, float("nan"), float("nan")]
    bad2 = [not math.isfinite(v) for v in values2]
    trailing2 = 0
    for value in reversed(bad2):
        if not value:
            break
        trailing2 += 1
    assert trailing2 >= 2


def test_shell_syntax() -> None:
    shells = [
        SCRIPTS / "training_pipelines" / "train_pipeline_phase8_fixedq_attm_ft.sh",
        SCRIPTS / "training_pipelines" / "sequence_phase8_fixedq_attm.sh",
    ]
    for path in shells:
        run_ok(["bash", "-n", str(path)])


def test_python_syntax() -> None:
    py_files = [
        SCRIPTS / "preprocess" / "make_phase8_fixedq9_tsv.py",
        SCRIPTS / "preprocess" / "build_official_caption_inventory.py",
        SCRIPTS / "init_phase8_fixedq_attm_checkpoint.py",
        SCRIPTS / "audit_phase8_fixedq_attm_ft.py",
        SCRIPTS / "monitor_phase8_fixedq_attm_ft.py",
        SCRIPTS / "eval" / "paired_clap_bootstrap_phase8_fixedq_attm.py",
    ]
    for path in py_files:
        run_ok([PY, "-m", "py_compile", str(path)])


def main() -> None:
    tests = [
        test_python_syntax,
        test_shell_syntax,
        test_fixedq9_tsv_builder,
        test_checkpoint_init_fixedq9_and_noq,
        test_bootstrap_ci_math,
        test_official_caption_inventory_cpu_only,
        test_monitor_grad_health_logic,
    ]
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: {exc}")
    if failed:
        raise SystemExit(f"{failed} self-test(s) failed")
    print(f"[OK] all {len(tests)} self-tests passed")


if __name__ == "__main__":
    # Keep CPU-only for torch ops in synthetic ckpt tests.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
