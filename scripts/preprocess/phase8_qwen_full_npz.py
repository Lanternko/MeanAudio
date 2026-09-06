#!/usr/bin/env python3
"""Resumable full NPZ rebuild with official Qwen text features.

Only names from ``--cache-list`` are opened.  The base directory is never
globbed or enumerated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase8_qwen_probe_lib import (  # noqa: E402
    DEFAULT_SCHEMA,
    EXPECTED_ROWS,
    ContractError,
    atomic_save_npz,
    array_sha256,
    iter_rows,
    load_npz,
    projected_free_space,
    read_cache_list,
    read_tsv,
    sha256_file,
    validate_row_cache_alignment,
    validate_schema,
    write_immutable_json,
    write_json_atomic,
)


DEFAULT_BASE_NPZ = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz")
DEFAULT_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched.tsv")
DEFAULT_CACHE = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt"
)
DEFAULT_OUTPUT = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz")
DEFAULT_MAPPER_MANIFEST = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_manifest.json"
)
DEFAULT_OUTPUT_MANIFEST = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_manifest.json"
)
DEFAULT_T5_MODEL = "google/flan-t5-large"
DEFAULT_CLAP_CKPT = Path(
    "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
)


class QwenTextEncoder:
    """Encode exactly the T5+LAION-CLAP schema used by the base cache."""

    def __init__(self, *, device: str, t5_model_name: str, clap_checkpoint: Path):
        import torch
        from transformers import AutoTokenizer, T5EncoderModel
        import laion_clap

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        self.t5 = T5EncoderModel.from_pretrained(t5_model_name).eval().to(self.device)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.clap.load_ckpt(str(clap_checkpoint), verbose=False)
        self.clap = self.clap.eval().to(self.device)

    @property
    def config(self) -> dict[str, str]:
        return {
            "t5_model": str(self.t5.config._name_or_path),
            "clap_checkpoint": str(self.clap_checkpoint),
            "device": str(self.device),
        }

    @property
    def clap_checkpoint(self) -> Path:
        # laion_clap does not expose the path; the caller replaces this field
        # after construction for a reproducible manifest.
        return getattr(self, "_clap_checkpoint", Path("<loaded>"))

    @clap_checkpoint.setter
    def clap_checkpoint(self, value: Path) -> None:
        self._clap_checkpoint = value

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    def encode(self, captions: Sequence[str]) -> list[dict[str, np.ndarray]]:
        torch = self.torch
        tokens = self.tokenizer(
            list(captions),
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        with torch.inference_mode():
            text_features = self.t5(input_ids=input_ids, attention_mask=attention_mask)[0]
            text_c = self.clap.get_text_embedding(list(captions), use_tensor=True)
        return [
            {
                "text_features": self._numpy(text_features[index]).astype(np.float32, copy=False),
                "text_features_c": self._numpy(text_c[index]).astype(np.float32, copy=False),
                "text_attention_mask": attention_mask[index].detach().cpu().numpy().astype(bool, copy=True),
            }
            for index in range(len(captions))
        ]


def source_manifest(
    *,
    base_npz: Path,
    tsv: Path,
    cache_list: Path,
    mapper_manifest: Path,
    output_dir: Path,
    limit: int | None,
    t5_model: str,
    clap_checkpoint: Path,
    base_identity_sample: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "phase8_qwen_official_matched_npz",
        "base_npz_dir": str(base_npz),
        "tsv": str(tsv),
        "tsv_sha256": sha256_file(tsv),
        "cache_list": str(cache_list),
        "cache_list_sha256": sha256_file(cache_list),
        "mapper_manifest": str(mapper_manifest),
        "mapper_manifest_sha256": sha256_file(mapper_manifest),
        "output_dir": str(output_dir),
        "base_identity_sample": dict(base_identity_sample),
        "limit": limit,
        "encoder": {
            "text_encoder": "t5_clap",
            "t5_model": t5_model,
            "clap_checkpoint": str(clap_checkpoint),
            "clap_checkpoint_sha256": sha256_file(clap_checkpoint),
            "max_length": 77,
        },
        "copy_policy": "all base arrays exact except four text/caption fields",
        "training_q_policy": "NoQ; q_level is provenance only",
    }


def output_arrays(base: Mapping[str, np.ndarray], encoded: Mapping[str, np.ndarray], caption: str) -> dict[str, np.ndarray]:
    validate_schema(base)
    arrays = {key: np.array(value, copy=True) for key, value in base.items()}
    arrays["text_features"] = np.asarray(encoded["text_features"], dtype=np.float32)
    arrays["text_features_c"] = np.asarray(encoded["text_features_c"], dtype=np.float32)
    arrays["text_attention_mask"] = np.asarray(encoded["text_attention_mask"], dtype=bool)
    arrays["caption_sha256"] = np.asarray(__import__("hashlib").sha256(caption.encode("utf-8")).hexdigest())
    validate_schema(arrays)
    for key in base:
        if key not in {"text_features", "text_features_c", "text_attention_mask", "caption_sha256"}:
            if base[key].dtype != arrays[key].dtype or base[key].shape != arrays[key].shape or not np.array_equal(base[key], arrays[key]):
                raise ContractError(f"base array changed before publish: {key}")
    return arrays


def write_one(base_path: Path, output_path: Path, caption: str, encoded: Mapping[str, np.ndarray]) -> int:
    base = load_npz(base_path)
    arrays = output_arrays(base, encoded, caption)
    atomic_save_npz(output_path, arrays)
    return output_path.stat().st_size


def state_path(output_dir: Path) -> Path:
    return output_dir / ".phase8_qwen_npz_progress.json"


def load_progress(path: Path, contract: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    if not path.exists():
        if resume:
            raise ContractError(f"--resume requested but progress state is missing: {path}")
        return {
            "contract": dict(contract),
            "completed_count": 0,
            "last_completed": None,
            "sample_sizes": [],
            "status": "running",
        }
    if not resume:
        raise ContractError(f"progress state exists; pass --resume: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("contract") != dict(contract):
        raise ContractError("NPZ progress state is bound to a different input/encoder contract")
    if not isinstance(current.get("completed_count"), int):
        raise ContractError("invalid NPZ progress completed_count")
    if current["completed_count"] < 0:
        raise ContractError("negative NPZ progress completed_count")
    return current


def check_complete_file(output_path: Path, recorded_hash: str) -> bool:
    if not output_path.is_file():
        return False
    return sha256_file(output_path) == recorded_hash


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_tsv(args.tsv)
    names = read_cache_list(args.cache_list)
    validate_row_cache_alignment(rows, names)
    if len(rows) != EXPECTED_ROWS:
        raise ContractError(f"full source must have {EXPECTED_ROWS} rows, got {len(rows)}")
    if not args.base_npz.is_dir():
        raise ContractError(f"base NPZ directory missing: {args.base_npz}")
    if not args.mapper_manifest.is_file():
        raise ContractError(f"mapper manifest missing: {args.mapper_manifest}")
    if not args.clap_checkpoint.is_file():
        raise ContractError(f"CLAP checkpoint missing: {args.clap_checkpoint}")
    first_base_path = args.base_npz / names[0]
    if not first_base_path.is_file():
        raise ContractError(f"cache-listed base NPZ missing: {first_base_path}")
    first_base = load_npz(first_base_path)
    validate_schema(first_base)
    base_identity_sample = {
        "name": names[0],
        "mean_sha256": array_sha256(first_base["mean"]),
        "std_sha256": array_sha256(first_base["std"]),
        "clip_id_sha256": array_sha256(first_base["clip_id"]),
        "catalog_index_sha256": array_sha256(first_base["catalog_index"]),
    }

    limit = args.limit
    if limit is not None and not (1 <= limit <= 512):
        raise ContractError("--limit is only for a separate probe and must be 1..512")
    planned = len(rows) if limit is None else min(limit, len(rows))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = source_manifest(
        base_npz=args.base_npz,
        tsv=args.tsv,
        cache_list=args.cache_list,
        mapper_manifest=args.mapper_manifest,
        output_dir=output_dir,
        limit=limit,
        t5_model=args.t5_model,
        clap_checkpoint=args.clap_checkpoint,
        base_identity_sample=base_identity_sample,
    )
    progress = load_progress(state_path(output_dir), contract, resume=args.resume)
    completed_count = int(progress["completed_count"])
    if completed_count > planned:
        raise ContractError("progress state exceeds the requested cache-list/limit")
    last = progress.get("last_completed")
    if completed_count:
        expected_name = names[completed_count - 1]
        if not isinstance(last, dict) or last.get("name") != expected_name:
            raise ContractError("progress boundary name drift")
        if not check_complete_file(output_dir / expected_name, str(last.get("sha256", ""))):
            raise ContractError("progress boundary file is missing or corrupt")
    if progress.get("status") == "passed":
        if completed_count != planned or not args.output_manifest.is_file():
            raise ContractError("completed cache state is missing its final manifest")
        final = json.loads(args.output_manifest.read_text(encoding="utf-8"))
        if final.get("status") != "passed" or final.get("completed_rows") != planned:
            raise ContractError("completed cache manifest is not passed/full")
        for key, value in contract.items():
            if final.get(key) != value:
                raise ContractError(f"completed cache manifest contract drift: {key}")
        return final

    encoder = QwenTextEncoder(
        device=args.device,
        t5_model_name=args.t5_model,
        clap_checkpoint=args.clap_checkpoint,
    )
    encoder.clap_checkpoint = args.clap_checkpoint

    sample_count = min(512, planned)
    sample_sizes: list[int] = [int(value) for value in progress.get("sample_sizes", [])]
    # The sample is measured from actual output files, not a base-file guess.
    for start in range(completed_count, sample_count, args.batch_size):
        batch = list(iter_rows(rows[start : min(start + args.batch_size, sample_count)], None))
        captions = [str(row["caption"]) for _, row in batch]
        features = encoder.encode(captions)
        for local_index, (index, row) in enumerate(batch):
            global_index = start + index
            name = names[global_index]
            destination = output_dir / name
            base_path = args.base_npz / name
            if not base_path.is_file():
                raise ContractError(f"cache-listed base NPZ missing: {base_path}")
            size = write_one(base_path, destination, str(row["caption"]), features[local_index])
            sample_sizes.append(size)
        completed_count = min(start + len(batch), sample_count)
        boundary = names[completed_count - 1]
        progress["completed_count"] = completed_count
        progress["last_completed"] = {
            "name": boundary,
            "sha256": sha256_file(output_dir / boundary),
        }
        progress["sample_sizes"] = sample_sizes
        write_json_atomic(state_path(output_dir), progress)

    if not sample_sizes:
        raise ContractError("no measured sample outputs")
    missing_count = planned - completed_count
    additional = max(sample_sizes) * max(0, missing_count)
    space = projected_free_space(output_dir, additional)
    progress["space_projection"] = space
    write_json_atomic(state_path(output_dir), progress)

    for start in range(max(sample_count, completed_count), planned, args.batch_size):
        batch = list(iter_rows(rows[start : min(start + args.batch_size, planned)], None))
        captions = [str(row["caption"]) for _, row in batch]
        features = encoder.encode(captions)
        for local_index, (index, row) in enumerate(batch):
            global_index = start + index
            name = names[global_index]
            destination = output_dir / name
            base_path = args.base_npz / name
            if not base_path.is_file():
                raise ContractError(f"cache-listed base NPZ missing: {base_path}")
            write_one(base_path, destination, str(row["caption"]), features[local_index])
        completed_count = min(start + len(batch), planned)
        boundary = names[completed_count - 1]
        progress["completed_count"] = completed_count
        progress["last_completed"] = {
            "name": boundary,
            "sha256": sha256_file(output_dir / boundary),
        }
        rows_done = start + len(batch)
        if rows_done >= planned or (rows_done - sample_count) % 512 == 0:
            write_json_atomic(state_path(output_dir), progress)

    if completed_count != planned:
        raise ContractError("NPZ build ended before every planned file was atomically completed")

    progress["status"] = "passed"
    write_json_atomic(state_path(output_dir), progress)
    final = dict(contract)
    final.update(
        {
            "status": "passed",
            "planned_rows": planned,
            "completed_rows": completed_count,
            "space_projection": space,
            "sample_output_bytes": {
                "count": len(sample_sizes),
                "min": min(sample_sizes),
                "max": max(sample_sizes),
                "mean": sum(sample_sizes) / len(sample_sizes),
            },
            "resume_boundary": progress["last_completed"],
        }
    )
    write_immutable_json(args.output_manifest, final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-npz", type=Path, default=DEFAULT_BASE_NPZ)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--cache-list", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mapper-manifest", type=Path, default=DEFAULT_MAPPER_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--limit", type=int, help="separate deterministic probe; maximum 512")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--t5-model", default=DEFAULT_T5_MODEL)
    parser.add_argument("--clap-checkpoint", type=Path, default=DEFAULT_CLAP_CKPT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    result = build(args)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
