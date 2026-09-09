#!/usr/bin/env python3
"""Attach Q-bucket labels to the c2p0 013 true-random rotation TSV.

The 2026-07-24 bucket grid already discretised the same 251,599 clips'
credibility mean_similarity at K = 2/3/5/10 under two strategies. That grid was
only ever wired to the single-caption official-matched Qwen corpus. This script
carries the identical q_level column onto the K=3 true-random rotation TSV so a
Q-resolution ablation can be run on top of a rotating caption pool.

Nothing is recomputed: q_level is copied row-for-row from the grid TSV, and the
join is by explicit id equality (the two files were independently verified to
carry the same 251,599 ids in the same order). Any drift is fatal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

GRID_MANIFEST = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_meansim_bucket_grid.manifest.json"
)
BASE_TSV = Path(
    "/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/"
    "c2p0_k3_true_fake_random/k3_true_random_train.tsv"
)
BASE_TSV_SHA = "5ec90b0f8d963df50546730384446bdca1b185ee4b2e21a4094cf60398b39999"
EXPECTED_ROWS = 251599


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"[FAIL] no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, delimiter="\t",
        lineterminator="\n", extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"[FAIL] existing output content drift: {path}")
        return "verified"
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["k3_balanced", "k10_balanced"])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/kojiek/MeanAudio/docs/experiments/"
                     "c2p0_truerandom_q_tsvs.manifest.json"),
    )
    args = parser.parse_args()

    for path in (GRID_MANIFEST, BASE_TSV):
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing input: {path}")
    base_sha = sha256(BASE_TSV)
    if base_sha != BASE_TSV_SHA:
        raise SystemExit(f"[FAIL] base TSV drift: {base_sha}")

    grid = json.loads(GRID_MANIFEST.read_text(encoding="utf-8"))
    if grid.get("status") != "passed" or grid.get("rows") != EXPECTED_ROWS:
        raise SystemExit("[FAIL] bucket grid manifest is not a passed 251,599-row grid")

    base_fields, base_rows = read_tsv(BASE_TSV)
    if base_fields != ["id", "caption"]:
        raise SystemExit(f"[FAIL] unexpected base schema: {base_fields}")
    if len(base_rows) != EXPECTED_ROWS:
        raise SystemExit(f"[FAIL] base rows={len(base_rows)}")

    outputs: dict[str, dict[str, object]] = {}
    for arm in args.arms:
        spec = grid.get("outputs", {}).get(arm)
        if not spec:
            raise SystemExit(f"[FAIL] grid has no arm {arm}")
        grid_tsv = Path(str(spec["path"]))
        grid_sha = sha256(grid_tsv)
        if grid_sha != spec["sha256"] or spec.get("status") != "passed":
            raise SystemExit(f"[FAIL] grid arm {arm} drifted: {grid_sha}")
        grid_fields, grid_rows = read_tsv(grid_tsv)
        if "q_level" not in grid_fields or len(grid_rows) != EXPECTED_ROWS:
            raise SystemExit(f"[FAIL] grid arm {arm} schema/cardinality")

        merged: list[dict[str, str]] = []
        histogram: Counter[str] = Counter()
        for index, (base, source) in enumerate(zip(base_rows, grid_rows)):
            if base["id"] != source["id"]:
                raise SystemExit(
                    f"[FAIL] id order mismatch at row {index}: "
                    f"{base['id']!r} vs {source['id']!r}"
                )
            q = source["q_level"]
            if not q.isdigit() or not 0 <= int(q) <= 9:
                raise SystemExit(f"[FAIL] invalid q_level at row {index}: {q!r}")
            merged.append({"id": base["id"], "caption": base["caption"], "q_level": q})
            histogram[q] += 1

        expected_hist = {str(k): v for k, v in spec["q_histogram"].items() if v}
        if dict(histogram) != expected_hist:
            raise SystemExit(
                f"[FAIL] {arm} histogram drift: {dict(histogram)} vs {expected_hist}"
            )

        out = BASE_TSV.with_name(f"k3_true_random_train_q{arm}.tsv")
        action = write_tsv(out, ["id", "caption", "q_level"], merged)
        outputs[arm] = {
            "action": action,
            "grid_source": str(grid_tsv),
            "grid_source_sha256": grid_sha,
            "occupied_q_codes": sorted(int(k) for k in histogram),
            "path": str(out),
            "q_histogram": {k: histogram[k] for k in sorted(histogram, key=int)},
            "rows": len(merged),
            "sha256": sha256(out),
        }
        print(f"[{action}] {out}  q_histogram={dict(sorted(histogram.items()))}")

    manifest = {
        "base_tsv": {"path": str(BASE_TSV), "rows": EXPECTED_ROWS, "sha256": base_sha},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "grid_manifest": {"path": str(GRID_MANIFEST), "sha256": sha256(GRID_MANIFEST)},
        "join": "positional, guarded by per-row id equality over all 251,599 rows",
        "outputs": outputs,
        "q_code_policy": grid["q_code_policy"],
        "rows": EXPECTED_ROWS,
        "schema_version": 1,
        "signal": grid["signal"],
        "status": "passed",
        "strategies": grid["strategies"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[manifest] {args.manifest}")


if __name__ == "__main__":
    main()
