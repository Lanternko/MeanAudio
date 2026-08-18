#!/usr/bin/env python3
"""Score all official IDs with CLAP(caption, first-10s) and write best/worst jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

SR = 16000
WINDOW = SR * 10
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")
CLAP_CKPT = Path("/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt")


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    artist = "_".join(parts[: seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f"segment_{seg_num}.mp3"


def load_jsonl_caps(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec.get("id")
            cap = (rec.get("caption") or "").strip()
            if cid and cap:
                out[cid] = cap
    return out


def load_ids(path: Path) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id"):
                ids.append(rec["id"])
    return ids


def load_crop(cid: str) -> np.ndarray:
    full, _ = librosa.load(str(id_to_audio_path(cid)), sr=SR, mono=True, duration=WINDOW / SR)
    crop = np.asarray(full[:WINDOW], dtype=np.float32)
    if crop.shape[0] < WINDOW:
        crop = np.pad(crop, (0, WINDOW - crop.shape[0]))
    return crop


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") and rec.get("sim0") is not None:
                done[rec["id"]] = rec
    return done


def load_clap():
    import laion_clap

    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap.load_ckpt(str(CLAP_CKPT), verbose=False)
    clap.eval()
    return clap


@torch.inference_mode()
def embed_audio(clap, audios):
    e = np.asarray(clap.get_audio_embedding_from_data(x=audios, use_tensor=False), dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
    return e


@torch.inference_mode()
def embed_text(clap, texts):
    e = np.asarray(clap.get_text_embedding(texts, use_tensor=False), dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_jsonl", type=Path, required=True)
    ap.add_argument("--slot0", type=Path, required=True)
    ap.add_argument("--slot1", type=Path, required=True)
    ap.add_argument("--slot2", type=Path, required=True)
    ap.add_argument("--scores_jsonl", type=Path, required=True)
    ap.add_argument("--best_jsonl", type=Path, required=True)
    ap.add_argument("--worst_jsonl", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    ids = load_ids(args.ids_jsonl)
    print(f"ids={len(ids)} cuda={torch.cuda.is_available()}", flush=True)
    print("load captions...", flush=True)
    s0 = load_jsonl_caps(args.slot0)
    s1 = load_jsonl_caps(args.slot1)
    s2 = load_jsonl_caps(args.slot2)
    missing = [i for i in ids if not (i in s0 and i in s1 and i in s2)]
    if missing:
        raise SystemExit(f"incomplete 3-cap coverage: {len(missing)} / {len(ids)}")

    done = load_done(args.scores_jsonl)
    todo = [i for i in ids if i not in done]
    print(f"already_scored={len(done)} todo={len(todo)}", flush=True)

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if todo:
        clap = load_clap()
        with args.scores_jsonl.open("a", encoding="utf-8") as fout:
            for start in range(0, len(todo), args.bs):
                batch = todo[start : start + args.bs]
                crops = []
                ok = []
                for cid in batch:
                    try:
                        crops.append(load_crop(cid))
                        ok.append(cid)
                    except Exception as exc:
                        rec = {
                            "id": cid,
                            "error": str(exc),
                            "sim0": None,
                            "sim1": None,
                            "sim2": None,
                        }
                        fout.write(json.dumps(rec) + "\n")
                        print(f"[WARN] {cid}: {exc}", flush=True)
                if not ok:
                    fout.flush()
                    continue
                a = embed_audio(clap, crops)
                t0 = embed_text(clap, [s0[c] for c in ok])
                t1 = embed_text(clap, [s1[c] for c in ok])
                t2 = embed_text(clap, [s2[c] for c in ok])
                sim0 = (a * t0).sum(axis=1)
                sim1 = (a * t1).sum(axis=1)
                sim2 = (a * t2).sum(axis=1)
                for j, cid in enumerate(ok):
                    scores = [float(sim0[j]), float(sim1[j]), float(sim2[j])]
                    best_i = int(np.argmax(scores))
                    worst_i = int(np.argmin(scores))
                    rec = {
                        "id": cid,
                        "sim0": scores[0],
                        "sim1": scores[1],
                        "sim2": scores[2],
                        "best_slot": best_i,
                        "worst_slot": worst_i,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    done[cid] = rec
                fout.flush()
                n_done = min(start + args.bs, len(todo))
                if n_done % 256 == 0 or n_done == len(todo):
                    print(f"scored {n_done}/{len(todo)}", flush=True)

    slots = [s0, s1, s2]
    names = ["slot0", "slot1", "slot2"]
    best_n = [0, 0, 0]
    worst_n = [0, 0, 0]
    sims = [[], [], []]
    errors = 0
    with args.best_jsonl.open("w", encoding="utf-8") as fb, args.worst_jsonl.open(
        "w", encoding="utf-8"
    ) as fw:
        for cid in ids:
            rec = done.get(cid)
            if rec is None or rec.get("sim0") is None:
                errors += 1
                continue
            b = int(rec["best_slot"])
            w = int(rec["worst_slot"])
            best_n[b] += 1
            worst_n[w] += 1
            sims[0].append(float(rec["sim0"]))
            sims[1].append(float(rec["sim1"]))
            sims[2].append(float(rec["sim2"]))
            fb.write(
                json.dumps(
                    {
                        "id": cid,
                        "caption": slots[b][cid],
                        "source_slot": names[b],
                        "clap": float(rec[f"sim{b}"]),
                    }
                )
                + "\n"
            )
            fw.write(
                json.dumps(
                    {
                        "id": cid,
                        "caption": slots[w][cid],
                        "source_slot": names[w],
                        "clap": float(rec[f"sim{w}"]),
                    }
                )
                + "\n"
            )

    def stat(xs):
        a = np.asarray(xs, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": float(a.mean()) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
        }

    summary = {
        "n_ids": len(ids),
        "n_scored": len(ids) - errors,
        "n_error": errors,
        "best_slot_hist": {names[i]: best_n[i] for i in range(3)},
        "worst_slot_hist": {names[i]: worst_n[i] for i in range(3)},
        "slot0": stat(sims[0]),
        "slot1": stat(sims[1]),
        "slot2": stat(sims[2]),
        "scores_jsonl": str(args.scores_jsonl),
        "best_jsonl": str(args.best_jsonl),
        "worst_jsonl": str(args.worst_jsonl),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise SystemExit(f"CLAP errors: {errors}")


if __name__ == "__main__":
    main()
