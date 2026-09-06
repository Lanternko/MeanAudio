#!/usr/bin/env python3
"""Temporary-input self-tests for the official-Qwen probe deliverables."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "preprocess"))
sys.path.insert(0, str(SCRIPTS / "analysis"))

from phase8_qwen_probe_lib import (  # noqa: E402
    ContractError,
    atomic_save_npz,
    check_projected_free,
    load_npz,
    read_cache_list,
    write_json_atomic,
)
from phase8_qwen_full_npz import load_progress, output_arrays  # noqa: E402
from phase8_qwen_official_mapper import build_mapping, local_track_id  # noqa: E402
from phase8_qwen_cache_audit import audit  # noqa: E402
from phase8_qwen_neutral_qcopy import process_checkpoint, process_ema  # noqa: E402
from phase8_qwen_probe_queue import (  # noqa: E402
    BASE_NPZ,
    BASE_TSV,
    eval_args,
    execution_step_passed,
    load_execution_manifest,
    metrics_file_path,
    reject_duplicates,
    require_passed,
    train_args,
)
from phase8_qwen_monitor import checkpoint_snapshot  # noqa: E402


def fake_rows(path: Path, captions: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "caption", "q_level"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, caption in enumerate(captions):
            writer.writerow({"id": f"00_{1000 + index}_segment_1_0", "caption": caption, "q_level": str(index)})


def fake_npz(path: Path, clip_id: str, caption: str, index: int) -> None:
    atomic_save_npz(
        path,
        {
            "mean": np.full((312, 20), index, dtype=np.float32),
            "std": np.full((312, 20), index + 1, dtype=np.float32),
            "text_features": np.zeros((77, 1024), dtype=np.float32),
            "text_features_c": np.zeros((512,), dtype=np.float32),
            "text_attention_mask": np.ones((77,), dtype=bool),
            "clip_id": np.asarray(clip_id),
            "catalog_index": np.asarray(index, dtype=np.int64),
            "caption_sha256": np.asarray(hashlib.sha256(caption.encode()).hexdigest()),
        },
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase8-qwen-selftest-") as raw:
        root = Path(raw)
        local = root / "local.tsv"
        cache = root / "cache.txt"
        official = root / "official.json"
        captions = ["caption zero", "caption one", "caption two", "caption three"]
        fake_rows(local, captions)
        cache.write_text("a.npz\nb.npz\nc.npz\nd.npz\n", encoding="utf-8")
        official.write_text(
            json.dumps([
                {"path": f"00/{1000 + i}.mp3", "caption": f"official {i}"}
                for i in range(4)
            ]),
            encoding="utf-8",
        )
        mapped, names, stats = build_mapping(local, cache, official)
        assert len(mapped) == 4 and names[0] == "a.npz" and stats["coverage_exact"]
        assert local_track_id(mapped[0]["id"]) == ("00", "1000")
        qwen_tsv = root / "qwen.tsv"
        with qwen_tsv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "caption", "q_level"], delimiter="\t", lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(mapped)

        base = root / "base"
        rebuilt = root / "rebuilt"
        base.mkdir()
        rebuilt.mkdir()
        for index, name in enumerate(names):
            fake_npz(base / name, mapped[index]["id"], captions[index], index)
            encoded = {
                "text_features": np.full((77, 1024), index + 2, dtype=np.float32),
                "text_features_c": np.full((512,), index + 3, dtype=np.float32),
                "text_attention_mask": np.ones((77,), dtype=bool),
            }
            arrays = output_arrays(load_npz(base / name), encoded, mapped[index]["caption"])
            atomic_save_npz(rebuilt / name, arrays)
        result = audit(
            tsv=qwen_tsv,
            cache_list=cache,
            npz_dir=rebuilt,
            reference_npz_dir=base,
            limit=4,
            clap_checkpoint=None,
            audio_dir=None,
            seed=14159265,
            skip_semantic=True,
            allow_test_rows=True,
        )
        assert result["status"] == "passed" and result["semantic_gate"]["status"] == "skipped"
        assert np.array_equal(load_npz(rebuilt / "a.npz")["mean"], load_npz(base / "a.npz")["mean"])

        progress = root / "progress.json"
        contract = {"input": "fake", "limit": 4}
        state = load_progress(progress, contract, resume=False)
        state["completed_count"] = 1
        state["last_completed"] = {"name": "a.npz", "sha256": "fakehash"}
        write_json_atomic(progress, state)
        resumed = load_progress(progress, contract, resume=True)
        assert resumed["completed_count"] == 1
        assert resumed["last_completed"]["name"] == "a.npz"
        try:
            check_projected_free(51 * 1024**3, 60 * 1024**3, 11 * 1024**3)
        except ContractError:
            pass
        else:
            raise AssertionError("disk refusal did not fail")

        source = root / "source.pth"
        q = torch.arange(11 * 448, dtype=torch.float32).reshape(11, 448)
        state = {
            "it": 600000,
            "weights": {"q_embed.weight": q.clone(), "preserved": torch.ones(2)},
            "optimizer": {"state": {0: {"exp_avg": torch.ones(2)}}, "param_groups": []},
            "scheduler": {"last_epoch": 3},
            "ema": {
                "ema_models.0.ema_model.q_embed.weight": q.clone(),
                "ema_models.1.ema_model.q_embed.weight": q.clone(),
                "other": torch.ones(1),
            },
        }
        torch.save(state, source)
        neutral = root / "neutral.pth"
        report = process_checkpoint(source, neutral, 600000)
        assert report["invariant"]["changed_tensor_paths"] == [
            "ema.ema_models.0.ema_model.q_embed.weight",
            "ema.ema_models.1.ema_model.q_embed.weight",
            "weights.q_embed.weight",
        ]
        out_state = torch.load(neutral, map_location="cpu", weights_only=False)
        assert torch.equal(out_state["weights"]["q_embed.weight"][9], q[10])
        assert torch.equal(out_state["weights"]["q_embed.weight"][8], q[8])
        ema_source = root / "ema.pth"
        torch.save({"q_embed.weight": q.clone(), "preserved": torch.ones(1)}, ema_source)
        process_ema(ema_source, root / "ema_neutral.pth")

        finite = root / "finite.pth"
        torch.save({"it": 1, "weights": {"x": torch.ones(1)}}, finite)
        old = time.time() - 180
        os.utime(finite, (old, old))
        assert checkpoint_snapshot(finite)["status"] == "passed"
        reject_duplicates([])
        require_passed({"status": "passed"}, "fake")
        train_command = train_args("fake_exp", root / "run", BASE_TSV, BASE_NPZ, root / "init.pth")
        assert train_command[0].endswith("/torchrun")
        assert "--standalone" in train_command and any(arg.endswith("/train.py") for arg in train_command)
        eval_command = eval_args("fake_exp", root / "ema.pth", root / "eval")
        assert any(arg.endswith("/eval.py") for arg in eval_command)
        assert "--no_text_attention_mask" in eval_command and "--no-mask" not in eval_command
        assert metrics_file_path(root / "run", "fake_exp") == root / "run/musiccaps_metrics/fake_exp/metrics.txt"
        try:
            reject_duplicates(["duplicate"])
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate guard did not fail")
        try:
            require_passed({"status": "failed"}, "fake")
        except RuntimeError:
            pass
        else:
            raise AssertionError("gate guard did not fail")

        # An authorized post-training resume must preserve the original passed
        # train command for the final provenance audit.
        import phase8_qwen_probe_queue as queue  # noqa: E402

        prior_manifest_path = root / "execution_manifest.json"
        original_manifest_path = queue.EXECUTION_MANIFEST
        try:
            queue.EXECUTION_MANIFEST = prior_manifest_path
            prior = {
                "schema_version": 1,
                "contract_sha256": hashlib.sha256(queue.CONTRACT.read_bytes()).hexdigest(),
                "steps": {
                    "control_train": {
                        "command": ["torchrun", "exp_id=preserved"],
                        "status": "passed",
                        "exit_code": 0,
                    }
                },
            }
            prior_manifest_path.write_text(json.dumps(prior), encoding="utf-8")
            loaded = load_execution_manifest("resume")
            assert loaded["steps"]["control_train"] == prior["steps"]["control_train"]
            assert execution_step_passed("control_train")
            assert not execution_step_passed("missing_step")
            try:
                load_execution_manifest("fresh")
            except RuntimeError:
                pass
            else:
                raise AssertionError("fresh run accepted a stale execution manifest")
        finally:
            queue.EXECUTION_MANIFEST = original_manifest_path
    print("[PASS] phase8_qwen_probe self-tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
