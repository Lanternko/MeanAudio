#!/usr/bin/env python3
"""No-GPU fixtures for exact resume coordinates and text-overlay loading."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from meanaudio.data.extracted_audio import ExtractedAudio
from meanaudio.runner_meanflow import _atomic_torch_save
from meanaudio.utils.training_resume import remaining_batch_indices, resume_coordinates
from train import _write_pause_ack


def test_resume_coordinates() -> None:
    assert resume_coordinates(0, 7) == (0, 0)
    assert resume_coordinates(3, 7) == (0, 3)
    assert resume_coordinates(7, 7) == (1, 0)
    assert resume_coordinates(17, 7) == (2, 3)
    uninterrupted = list(range(7)) + list(range(7)) + list(range(7))
    completed = 10
    resumed = uninterrupted[:completed] + remaining_batch_indices(completed, 7)
    assert resumed == uninterrupted[:14]


def test_text_overlay() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio_dir, text_dir = root / "audio", root / "text"
        audio_dir.mkdir(); text_dir.mkdir()
        np.savez(audio_dir / "sample.npz", clip_id=np.asarray("x"),
                 mean=np.zeros((250, 32), np.float32),
                 std=np.ones((250, 32), np.float32),
                 text_features=np.full((77, 1024), -1, np.float32),
                 text_features_c=np.full((512,), -1, np.float32))
        np.savez(text_dir / "sample.npz", clip_id=np.asarray("x"),
                 text_features=np.full((77, 1024), 7, np.float32),
                 text_features_c=np.full((512,), 9, np.float32),
                 text_attention_mask=np.ones((77,), np.int64))
        tsv = root / "train.tsv"
        with tsv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "caption"], delimiter="\t")
            writer.writeheader(); writer.writerow({"id": "x", "caption": "fixed"})
        cache = root / "cache.txt"; cache.write_text("sample.npz\n")
        dataset = ExtractedAudio(
            tsv, concat_text_fc=False, npz_dir=audio_dir, text_npz_dir=text_dir,
            data_dim={"latent_seq_len": 250, "text_seq_len": 77,
                      "text_dim": 1024, "text_c_dim": 512},
            repa_npz_dir=None, exclude_cls=False, repa_version=1,
            gt_cache=cache, require_text_overlay=True, multi_cap=False,
            use_text_attention_mask=True,
        )
        sample = dataset[0]
        assert torch.all(sample["a_mean"] == 0)
        assert torch.all(sample["text_features"] == 7)
        assert torch.all(sample["text_features_c"] == 9)
        np.savez(text_dir / "sample.npz", clip_id=np.asarray("wrong-id"),
                 text_features=np.full((77, 1024), 7, np.float32),
                 text_features_c=np.full((512,), 9, np.float32),
                 text_attention_mask=np.ones((77,), np.int64))
        try:
            dataset[0]
        except ValueError as error:
            assert "text overlay clip_id mismatch" in str(error)
        else:
            raise AssertionError("mismatched text overlay ID was accepted")


def test_atomic_rng_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pth"
        generator = torch.Generator(device="cpu").manual_seed(14159265)
        torch.rand(5, generator=generator)
        _atomic_torch_save({"it": 5, "trainer_rng_state": generator.get_state()}, path)
        expected = torch.rand(8, generator=generator)
        restored = torch.Generator(device="cpu")
        payload = torch.load(path, weights_only=True)
        restored.set_state(payload["trainer_rng_state"])
        assert payload["it"] == 5
        assert torch.equal(expected, torch.rand(8, generator=restored))
        assert path.stat().st_size > 0


def test_pause_ack() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        exp_id = "fixture"
        checkpoint = root / f"{exp_id}_ckpt_last.pth"
        _atomic_torch_save({
            "it": 13, "weights": {}, "optimizer": {}, "scheduler": {}, "ema": {},
            "trainer_rng_state": torch.get_rng_state(),
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": [],
        }, checkpoint)
        request = root / "pause.request.json"
        request.write_text("{}\n")
        _write_pause_ack(str(request), str(root), exp_id, 13)
        ack = __import__("json").loads(Path(f"{request}.ack.json").read_text())
        assert ack["status"] == "paused" and ack["iteration"] == 13
        assert ack["checkpoint"] == str(checkpoint) and ack["checkpoint_bytes"] > 0
        report = root / "verified.json"
        subprocess.run([
            sys.executable,
            str(Path(__file__).parents[1] / "experiment_harness" / "verify_preemptible_checkpoint.py"),
            "--ack", f"{request}.ack.json", "--checkpoint", str(checkpoint),
            "--report", str(report), "--expected-iteration", "13",
        ], check=True)
        assert __import__("json").loads(report.read_text())["status"] == "passed"


def test_interrupted_matches_uninterrupted() -> None:
    batches_per_epoch, total, pause_at = 7, 23, 13

    def fresh():
        torch.manual_seed(17)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        generator = torch.Generator(device="cpu").manual_seed(14159265)
        ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
        return model, optimizer, scheduler, generator, ema

    def advance(state, start, stop):
        model, optimizer, scheduler, generator, ema = state
        epoch, offset = resume_coordinates(start, batches_per_epoch)
        step = start
        while step < stop:
            order = torch.randperm(batches_per_epoch, generator=torch.Generator().manual_seed(1000 + epoch))
            for batch_index in range(offset, batches_per_epoch):
                token = int(order[batch_index]) + epoch * batches_per_epoch
                x = torch.tensor([[token, token + 1, token + 2]], dtype=torch.float32) / 50
                target = torch.rand((1, 2), generator=generator)
                optimizer.zero_grad(); loss = torch.nn.functional.mse_loss(model(x), target)
                loss.backward(); optimizer.step(); scheduler.step()
                for name, value in model.state_dict().items():
                    ema[name].mul_(0.9).add_(value, alpha=0.1)
                step += 1
                if step >= stop:
                    return state
            epoch += 1; offset = 0
        return state

    continuous = advance(fresh(), 0, total)
    interrupted = advance(fresh(), 0, pause_at)
    model, optimizer, scheduler, generator, ema = interrupted
    payload = {
        "it": pause_at, "weights": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "ema": ema,
        "trainer_rng_state": generator.get_state(),
    }
    resumed = fresh()
    resumed[0].load_state_dict(payload["weights"])
    resumed[1].load_state_dict(payload["optimizer"])
    resumed[2].load_state_dict(payload["scheduler"])
    resumed[3].set_state(payload["trainer_rng_state"])
    resumed[4].update({name: value.clone() for name, value in payload["ema"].items()})
    resumed = advance(resumed, pause_at, total)

    def assert_nested_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                assert_nested_equal(left_item, right_item)
        else:
            assert left == right

    for left, right in zip(continuous[0].state_dict().values(), resumed[0].state_dict().values()):
        assert torch.equal(left, right)
    assert_nested_equal(continuous[1].state_dict(), resumed[1].state_dict())
    assert_nested_equal(continuous[2].state_dict(), resumed[2].state_dict())
    for name in continuous[4]:
        assert torch.equal(continuous[4][name], resumed[4][name])
    assert torch.equal(continuous[3].get_state(), resumed[3].get_state())


if __name__ == "__main__":
    test_resume_coordinates()
    test_text_overlay()
    test_atomic_rng_checkpoint()
    test_pause_ack()
    test_interrupted_matches_uninterrupted()
    print("[SELFTEST OK] exact resume coordinates and isolated text overlay")
