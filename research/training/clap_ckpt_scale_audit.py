#!/usr/bin/env python3
"""Same clips, two CLAP checkpoints, every caption corpus we have.

Question 1 (checkpoint scale): is the gap between our Jamendo CLAP distribution
(mean ~0.31, p90 0.42, tail capped ~0.57) and the ATTM paper Fig.1 (mean 0.3380,
median 0.3535, visible mass to ~0.61) a checkpoint artifact or a corpus difference?

Question 2 (prompt shape): does a bare attribute list with no templated opening
beat the sentence-shaped prompts, and is the boilerplate opening itself the cost?

Common pool = the 1,215 clips present in both the seed42 2048 subset and the
Caption 2.0 corpus (ids there carry a _<slot> suffix, see
memory/reference_c2p0_id_slot_suffix).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

# laion_clap.get_audio_embedding_from_filelist loads at 48 kHz and quantizes
# through int16; get_audio_embedding_from_data must be fed the same thing or
# every score drops by ~0.10 across the board.
SR = 48000
WINDOW = SR * 10
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")
WEIGHTS = Path("/home/kojiek/MeanAudio/weights")
CKPTS = {
    "89.98_music_speech": WEIGHTS / "music_speech_audioset_epoch_15_esc_89.98.pt",
    "90.14_music": WEIGHTS / "music_audioset_epoch_15_esc_90.14.pt",
}
DATA = Path("/mnt/HDD/kojiek/phase4_jamendo_data")
RESEARCH = Path("/home/kojiek/research/meanaudio_training")
COMMON_IDS = RESEARCH / "clap_scale_common_ids.jsonl"
QWEN5_JSONL = DATA / "phase9_omni_captions.jsonl"
LP_TSV = DATA / "_QUARANTINED_phase7_v1_train.tsv"
C2P0_SHORT = DATA / "phase8_qwen_caption10s_train.tsv"
C2P0_MULTI = DATA / "phase8_qwen_caption10s_multisent_train.tsv"
NEW_ARMS = {
    "attrlist_nobp": RESEARCH / "prompt_probe_attrlist_nobp.jsonl",
    "attrlist_bp": RESEARCH / "prompt_probe_attrlist_bp.jsonl",
}


def base_id(i: str) -> str:
    return re.sub(r"(_segment_\d+)_\d+$", r"\1", i)


def id_to_audio_path(clip_id: str) -> Path:
    parts = base_id(clip_id).split("_")
    seg_idx = parts.index("segment")
    artist = "_".join(parts[: seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f"segment_{seg_num}.mp3"


def load_crop(cid: str):
    try:
        full, _ = librosa.load(str(id_to_audio_path(cid)), sr=SR, mono=True)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] {cid}: {e}", flush=True)
        return None
    crop = np.asarray(full, dtype=np.float32)[:WINDOW]
    if crop.shape[0] < WINDOW:
        crop = np.pad(crop, (0, WINDOW - crop.shape[0]))
    # match laion_clap's own int16 quantization step
    return np.clip(crop * 32768.0, -32768, 32767).astype(np.int16).astype(np.float32) / 32768.0


def tsv_caption_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        print(f"[SKIP] missing {path}", flush=True)
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.setdefault(base_id(r["id"]), r["caption"])
    return out


def jsonl_caption_map(path: Path, key: str = "caption") -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        print(f"[SKIP] missing {path}", flush=True)
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cid = base_id(str(d.get("id", "")))
            cap = d.get(key) or d.get("text") or ""
            if cid and cap:
                out.setdefault(cid, cap.strip())
    return out


def load_clap(ckpt: Path):
    import laion_clap

    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap.load_ckpt(str(ckpt), verbose=False)
    clap.eval()
    return clap


@torch.inference_mode()
def diag_sims(clap, audios, texts, bs=16) -> np.ndarray:
    out = []
    for i in range(0, len(audios), bs):
        a = np.asarray(clap.get_audio_embedding_from_data(x=audios[i : i + bs], use_tensor=False), dtype=np.float32)
        t = np.asarray(clap.get_text_embedding(texts[i : i + bs], use_tensor=False), dtype=np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-8
        out.extend((a * t).sum(axis=1).tolist())
    return np.asarray(out, dtype=np.float32)


def summarize(x: np.ndarray) -> dict:
    ps = [5, 10, 25, 50, 75, 90, 95, 99]
    q = np.percentile(x, ps)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        **{f"p{p}": float(v) for p, v in zip(ps, q)},
        "max": float(x.max()),
        "frac_ge_0.33": float((x >= 0.33).mean()),
        "frac_ge_0.50": float((x >= 0.50).mean()),
        "frac_ge_0.55": float((x >= 0.55).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", type=Path, default=RESEARCH / "clap_ckpt_scale_audit.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pool = [base_id(json.loads(l)["id"]) for l in COMMON_IDS.open() if l.strip()]
    print(f"common pool: {len(pool)}", flush=True)

    sets: dict[str, dict[str, str]] = {}
    qwen5: dict[str, list[str]] = {}
    for line in QWEN5_JSONL.open():
        d = json.loads(line)
        cid = base_id(str(d.get("id", "")))
        if cid in set(pool):
            qwen5[cid] = d["captions"]
    for s in range(5):
        sets[f"old5_slot{s}"] = {k: v[s] for k, v in qwen5.items() if len(v) == 5}
    sets["lp_mc"] = tsv_caption_map(LP_TSV)
    sets["c2p0_short"] = tsv_caption_map(C2P0_SHORT)
    sets["c2p0_multisent"] = tsv_caption_map(C2P0_MULTI)
    for name, path in NEW_ARMS.items():
        m = jsonl_caption_map(path)
        if m:
            sets[name] = m

    # ---- derived sets from c2p0_multisent (text-only, no regeneration) ----
    # CLAP truncates at 77 internally, so trunc77 is a null check, not a measurement;
    # the informative cuts are the ones below 77.
    base = sets.get("c2p0_multisent", {})
    if base:
        from transformers import AutoTokenizer

        rtok = AutoTokenizer.from_pretrained("roberta-base")

        def cut(text: str, n_tok: int) -> str:
            ids = rtok.encode(text, add_special_tokens=False)[:n_tok]
            return rtok.decode(ids).strip()

        def first_sentence(text: str) -> str:
            parts = re.split(r"(?<=[.!?])\s+", text.strip())
            return parts[0].strip() if parts else text.strip()

        BOILERPLATE = re.compile(
            r"^(the audio|the music|this music|the piece|this piece|the composition|this is)\b[^,.]*?"
            r"\b(features?|contains?|is|presents?)\s+(a|an|the)?\s*",
            re.IGNORECASE,
        )

        sets["c2p0_multi_trunc77"] = {k: cut(v, 77) for k, v in base.items()}     # null check
        sets["c2p0_multi_trunc23"] = {k: cut(v, 23) for k, v in base.items()}     # length-matched to c2p0_short
        sets["c2p0_multi_firstsent"] = {k: first_sentence(v) for k, v in base.items()}
        sets["c2p0_multi_nobp"] = {k: BOILERPLATE.sub("", v).strip() or v for k, v in base.items()}

    # ---- Jamendo official tag captions (no LLM: fixed templates over meta_all.json) ----
    # Closest reconstruction we can make of the ATTM validation caption style.
    META = Path("/home/kojiek/data/meta_all.json")
    if META.exists():
        def to_path(cid: str) -> str:
            q = cid.split("_")
            i = q.index("segment")
            return f"{'_'.join(q[:i-1])}/{q[i-1]}/segment_{q[i+1]}.mp3"

        want = {to_path(i): i for i in pool}
        tags: dict[str, dict] = {}
        for rec in json.loads(META.read_text()):
            if rec["path"] in want:
                tags[want[rec["path"]]] = rec.get("tags", {})

        def j(xs):
            xs = [x.replace("_", " ").strip() for x in xs if x]
            if not xs:
                return ""
            return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

        def art(w):
            return "An" if w[:1].lower() in "aeiou" else "A"

        def graceful(t):
            g, m, ins = j(t.get("genre", [])), j(t.get("mood/theme", [])), j(t.get("instrument", []))
            head = " ".join(x for x in [m, g] if x) or "music"
            out = f"{art(head)} {head} track"
            if ins:
                out += f" featuring {ins}"
            return out + "."

        def strict(t):
            g, m, ins = j(t.get("genre", [])), j(t.get("mood/theme", [])), j(t.get("instrument", []))
            if not (g and m and ins):
                return ""
            return f"{art(m)} {m} {g} track featuring {ins}."

        sets["tag_template"] = {k: graceful(v) for k, v in tags.items()}
        sets["tag_bare"] = {k: ", ".join(x.replace("_", " ") for x in v.get("all", [])) or "music"
                            for k, v in tags.items()}
        strict_map = {k: strict(v) for k, v in tags.items()}
        strict_map = {k: v for k, v in strict_map.items() if v}
        print(f"tag sets: template/bare n={len(tags)}, strict triplet n={len(strict_map)}", flush=True)
        # strict triplet covers only ~19% of clips: scored separately, never in the shared pool
        globals()["_STRICT_TRIPLET"] = strict_map

    keep = [i for i in pool if all(i in m and m[i] for m in sets.values())]
    print(f"clips with every caption set: {len(keep)} across {len(sets)} sets", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        crops = list(ex.map(load_crop, keep))
    ok = [(c, i) for c, i in zip(crops, keep) if c is not None]
    audios = [c for c, _ in ok]
    keep = [i for _, i in ok]
    print(f"audio loaded: {len(audios)}", flush=True)

    results = {"meta": {"n_clips": len(keep), "audio_root": str(AUDIO_ROOT), "window_s": 10, "sr": SR,
                        "caption_sets": sorted(sets)}}
    raw: dict[str, dict[str, list]] = {}
    for ck_name, ckpt in CKPTS.items():
        print(f"=== {ck_name} ===", flush=True)
        clap = load_clap(ckpt)
        results[ck_name] = {}
        raw[ck_name] = {}
        for cs_name, cmap in sets.items():
            s = diag_sims(clap, audios, [cmap[i] for i in keep])
            results[ck_name][cs_name] = summarize(s)
            raw[ck_name][cs_name] = s.tolist()
            r = results[ck_name][cs_name]
            print(f"  {cs_name:16s} mean={r['mean']:.4f} med={r['median']:.4f} std={r['std']:.4f} "
                  f"p90={r['p90']:.3f} max={r['max']:.3f} >=0.50={r['frac_ge_0.50']*100:5.2f}% "
                  f">=0.55={r['frac_ge_0.55']*100:5.2f}%", flush=True)
        del clap
        torch.cuda.empty_cache()

    strict_map = globals().get("_STRICT_TRIPLET", {})
    if strict_map:
        sub = [(a, i) for a, i in zip(audios, keep) if i in strict_map]
        s_aud = [a for a, _ in sub]
        s_txt = [strict_map[i] for _, i in sub]
        print(f"=== strict tag triplet subset n={len(sub)} ===", flush=True)
        results["strict_triplet_subset"] = {}
        for ck_name, ckpt in CKPTS.items():
            clap = load_clap(ckpt)
            sc = diag_sims(clap, s_aud, s_txt)
            results["strict_triplet_subset"][ck_name] = summarize(sc)
            r = results["strict_triplet_subset"][ck_name]
            print(f"  {ck_name:20s} n={r['n']} mean={r['mean']:.4f} med={r['median']:.4f} "
                  f"std={r['std']:.4f} p90={r['p90']:.3f} max={r['max']:.3f}", flush=True)
            del clap
            torch.cuda.empty_cache()

    args.out_json.write_text(json.dumps(results, indent=2))
    np.savez_compressed(args.out_json.with_suffix(".raw.npz"), ids=np.array(keep),
                        **{f"{k}__{c}": np.array(v) for k, d in raw.items() for c, v in d.items()})
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
