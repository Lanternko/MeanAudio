#!/usr/bin/env python3
"""Inputs for the score-aware Beta timestep probe (stage A, arXiv 2606.07387 reproduction).

select: pick one segment per track from the c2p0 slot0 corpus -> 100 val + 2,000 train rows,
        drawn from disjoint tracks, plus row-aligned gt_cache lists (row i <-> cache line i).
attach: read PE-AV scores, normalise on the train split to S in [0, 1]
        (S = clip((s - p05) / (p75 - p05), 0, 1); top quartile -> 1 as in the paper),
        and write `t_score` plus a `t_score_shuffled` control column (same multiset, rows permuted).

Everything is deterministic from the seeds below; a manifest records every hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

csv.field_size_limit(10**9)

SELECT_SEED = 20260913
SHUFFLE_SEED = 424242
N_VAL = 100
N_TRAIN = 2000
LOW_PCT, HIGH_PCT = 5.0, 75.0
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def id_to_audio_path(clip_id: str) -> Path:
    # Same mapping the c2p0 captioner used (gen_qwen_caption_10s_multisent.py).
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    return AUDIO_ROOT / "_".join(parts[: seg_idx - 1]) / parts[seg_idx - 1] / f"segment_{parts[seg_idx + 1]}.mp3"


def read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def write_lines(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
    tmp.replace(path)


def cmd_select(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields, rows = read_tsv(Path(args.src_tsv))
    cache = [x.strip() for x in Path(args.cache_list).read_text().splitlines() if x.strip()]
    if len(cache) != len(rows):
        raise SystemExit(f"[FAIL] cache {len(cache)} != tsv {len(rows)}")

    by_track: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r["caption"].strip():
            by_track.setdefault(r["track_id"], []).append(i)
    rng = random.Random(SELECT_SEED)
    tracks = sorted(by_track)
    rng.shuffle(tracks)

    picked: list[int] = []
    for track in tracks:
        if len(picked) == N_VAL + N_TRAIN:
            break
        idx = rng.choice(by_track[track])
        if not id_to_audio_path(rows[idx]["id"]).is_file():
            continue
        picked.append(idx)
    if len(picked) != N_VAL + N_TRAIN:
        raise SystemExit(f"[FAIL] only {len(picked)} usable tracks")

    splits = {"val": picked[:N_VAL], "train": picked[N_VAL:]}
    manifest = {"select_seed": SELECT_SEED, "src_tsv": args.src_tsv, "src_tsv_sha256": sha256(Path(args.src_tsv)),
                "cache_list": args.cache_list, "cache_list_sha256": sha256(Path(args.cache_list)), "splits": {}}
    for name, idxs in splits.items():
        tsv = out / f"{name}.tsv"
        lst = out / f"{name}_cache.txt"
        write_tsv(tsv, fields, [rows[i] for i in idxs])
        write_lines(lst, [cache[i] for i in idxs])
        manifest["splits"][name] = {"rows": len(idxs), "tracks": len({rows[i]["track_id"] for i in idxs}),
                                    "tsv_sha256": sha256(tsv), "cache_sha256": sha256(lst)}
    overlap = {rows[i]["track_id"] for i in splits["val"]} & {rows[i]["track_id"] for i in splits["train"]}
    if overlap:
        raise SystemExit(f"[FAIL] val/train share {len(overlap)} tracks")
    manifest["selected_at"] = datetime.now(timezone.utc).isoformat()
    (out / "select_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=1))


def cmd_attach(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    scores = {}
    with Path(args.scores_jsonl).open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            scores[rec["id"]] = float(rec["peav_cos"])

    fields, train = read_tsv(out / "train.tsv")
    _, val = read_tsv(out / "val.tsv")
    missing = [r["id"] for r in train + val if r["id"] not in scores]
    if missing:
        raise SystemExit(f"[FAIL] {len(missing)} rows unscored, e.g. {missing[:3]}")

    raw = np.array([scores[r["id"]] for r in train], dtype=np.float64)
    lo, hi = np.percentile(raw, LOW_PCT), np.percentile(raw, HIGH_PCT)
    if not hi > lo:
        raise SystemExit(f"[FAIL] degenerate score spread p05={lo} p75={hi}")
    norm = lambda s: float(np.clip((s - lo) / (hi - lo), 0.0, 1.0))  # noqa: E731
    s_train = [norm(scores[r["id"]]) for r in train]
    shuffled = list(s_train)
    random.Random(SHUFFLE_SEED).shuffle(shuffled)

    new_fields = fields + [c for c in ("peav_cos", "t_score", "t_score_shuffled") if c not in fields]
    for r, s, sh in zip(train, s_train, shuffled):
        r.update(peav_cos=f"{scores[r['id']]:.6f}", t_score=f"{s:.6f}", t_score_shuffled=f"{sh:.6f}")
    for r in val:
        s = norm(scores[r["id"]])
        r.update(peav_cos=f"{scores[r['id']]:.6f}", t_score=f"{s:.6f}", t_score_shuffled=f"{s:.6f}")
    write_tsv(out / "train_scored.tsv", new_fields, train)
    write_tsv(out / "val_scored.tsv", new_fields, val)

    s_arr = np.array(s_train)
    moved = float(np.mean(np.array(s_train) != np.array(shuffled)))
    summary = {
        "normalisation": f"S = clip((s - p{LOW_PCT:g}) / (p{HIGH_PCT:g} - p{LOW_PCT:g}), 0, 1) on train split",
        "p05": lo, "p75": hi, "shuffle_seed": SHUFFLE_SEED,
        "raw": {"mean": float(raw.mean()), "std": float(raw.std()), "min": float(raw.min()), "max": float(raw.max())},
        "S": {"mean": float(s_arr.mean()), "at_1": int((s_arr >= 1).sum()), "at_0": int((s_arr <= 0).sum())},
        "shuffled_rows_moved_fraction": moved,
        "train_scored_sha256": sha256(out / "train_scored.tsv"),
        "val_scored_sha256": sha256(out / "val_scored.tsv"),
        "scores_jsonl_sha256": sha256(Path(args.scores_jsonl)),
        "attached_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "score_manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--src-tsv", required=True)
    s.add_argument("--cache-list", required=True)
    s.add_argument("--out-dir", required=True)
    a = sub.add_parser("attach")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--scores-jsonl", required=True)
    args = ap.parse_args()
    {"select": cmd_select, "attach": cmd_attach}[args.cmd](args)


if __name__ == "__main__":
    main()
