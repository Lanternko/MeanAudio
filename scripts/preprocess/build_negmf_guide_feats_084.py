"""084 NegMF: encode the fixed negative prompt that replaces the empty string in the
guidance branch of the MeanFlow CFG training target.

The features are produced by the *inference* text path (FeaturesUtils.encode_text for
t5_clap: T5 padded to max_length 77 with pad tokens, CLAP get_text_embedding), because
the thing being distilled is what eval.py's --negative_prompt feeds the negative branch.

Checks:
  G0a  (report only) how the stored null features relate to the encoders. Measured
       2026-09-27: weights/empty_string_t5.pth is one constant row (norm 1.32) repeated
       77 times, not T5('') under any padding (cos -0.158 / 0.496 / 0.050), while
       empty_string_clap_c.pth equals CLAP('') (cos 1.0000). So the null T5 branch was
       never an encoder output and there is nothing to reproduce; this is recorded,
       not gated (see project_negative_prompt_empty_string_trap).
  G0b  (gate; nothing is written on failure)  the training overlay path (reextract_text_inplace_caption10s.py encode_t5 /
       encode_clap) agrees with the inference path on the valid-token positions
       (min per-token cosine >= 0.999) and on the CLAP vector (cosine >= 0.999).
       Pad positions are reported, not gated: the overlay builder encodes captions in
       batches, so its pads hold pad-token outputs up to the batch's longest caption
       (e.g. slot0clean pad norm 2.7-3.0, same as the inference path's 2.74); only a
       single-text call like this one zero-fills them.

Usage:
  python scripts/preprocess/build_negmf_guide_feats_084.py --name fidelity8 \
      --text "low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi" \
      --out_dir weights/negmf_084
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import laion_clap
import torch
from transformers import AutoTokenizer, T5EncoderModel

REPO = Path(__file__).resolve().parents[2]
TRAIN_ENCODER = Path(
    "/home/kojiek/research/meanaudio_training/caption10s_pipeline/reextract_text_inplace_caption10s.py")
CLAP_CKPT = REPO / "weights/music_speech_audioset_epoch_15_esc_89.98.pt"
FIDELITY8 = "low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def encode_inference_path(tokenizer, t5, clap, text: str, device):
    # Mirrors meanaudio/model/utils/features_utils.py FeaturesUtils.encode_text (t5_clap).
    tokens = tokenizer([text], max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    input_ids, attention_mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
    text_f = t5(input_ids=input_ids, attention_mask=attention_mask)[0]
    text_f_c = clap.get_text_embedding([text], use_tensor=True)
    return text_f.float().cpu(), text_f_c.float().cpu(), int(attention_mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_dir", default=str(REPO / "weights/negmf_084"))
    ap.add_argument("--min_cos", type=float, default=0.999)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    t5 = T5EncoderModel.from_pretrained("google/flan-t5-large").eval().to(device, torch.float32)
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap.load_ckpt(str(CLAP_CKPT), verbose=False)
    clap = clap.to(device)

    # G0a (report only): relation of the stored null features to the inference path.
    e_f, e_c, _ = encode_inference_path(tokenizer, t5, clap, "", device)
    ref_f = torch.load(REPO / "weights/empty_string_t5.pth", weights_only=True).float()
    ref_c = torch.load(REPO / "weights/empty_string_clap_c.pth", weights_only=True).float()
    g0a = {
        "t5_max_abs_diff": float((e_f - ref_f).abs().max()),
        "t5_cos": float(torch.nn.functional.cosine_similarity(e_f.flatten(), ref_f.flatten(), dim=0)),
        "clap_cos": float(torch.nn.functional.cosine_similarity(e_c.flatten(), ref_c.flatten(), dim=0)),
    }
    g0a["t5_stored_rows_constant"] = bool(torch.allclose(ref_f[0, 1:], ref_f[0, :1].expand(76, -1), atol=1e-4))

    # Guide features through the inference path.
    g_f, g_c, n_tok = encode_inference_path(tokenizer, t5, clap, args.text, device)

    # G0b: training overlay path on the same text.
    spec = importlib.util.spec_from_file_location("train_enc", TRAIN_ENCODER)
    train_enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_enc)
    tr_f, tr_mask = train_enc.encode_t5(tokenizer, t5, [args.text], device)
    tr_c = train_enc.encode_clap(clap, [args.text])
    tr_f = torch.from_numpy(tr_f)
    tr_c = torch.from_numpy(tr_c)
    n_valid = int(tr_mask.sum())
    tok_cos = torch.nn.functional.cosine_similarity(g_f[0, :n_valid], tr_f[0, :n_valid], dim=-1)
    pad_norm_inference = float(g_f[0, n_valid:].norm(dim=-1).mean()) if n_valid < 77 else 0.0
    g0b = {
        "n_valid_tokens_inference": n_tok,
        "n_valid_tokens_training": n_valid,
        "valid_token_min_cos": float(tok_cos.min()),
        "clap_cos": float(torch.nn.functional.cosine_similarity(g_c.flatten(), tr_c.flatten(), dim=0)),
        "pad_positions": 77 - n_valid,
        "pad_mean_norm_inference_path": pad_norm_inference,
        "pad_mean_norm_training_path": float(tr_f[0, n_valid:].norm(dim=-1).mean()) if n_valid < 77 else 0.0,
    }
    g0b["pass"] = n_tok == n_valid and g0b["valid_token_min_cos"] >= args.min_cos and g0b["clap_cos"] >= args.min_cos

    report = {"name": args.name, "text": args.text, "G0a_empty_string_reproduction": g0a,
              "G0b_training_vs_inference_path": g0b}
    print(json.dumps(report, indent=2))
    if not g0b["pass"]:
        raise SystemExit("G0 failed: nothing written")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t5_path, clap_path = out / f"{args.name}_t5.pth", out / f"{args.name}_clap_c.pth"
    torch.save(g_f.contiguous(), t5_path)      # [1, 77, 1024], same layout as empty_string_t5.pth
    torch.save(g_c.contiguous(), clap_path)    # [1, 512], same layout as empty_string_clap_c.pth
    report.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "t5_path": str(t5_path), "t5_sha256": sha_file(t5_path), "t5_shape": list(g_f.shape),
        "clap_path": str(clap_path), "clap_sha256": sha_file(clap_path), "clap_shape": list(g_c.shape),
        "builder_sha256": sha_file(Path(__file__).resolve()),
        "training_encoder_sha256": sha_file(TRAIN_ENCODER),
        "clap_ckpt": str(CLAP_CKPT), "device": str(device),
    })
    (out / f"{args.name}_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {t5_path}, {clap_path}")


if __name__ == "__main__":
    main()
