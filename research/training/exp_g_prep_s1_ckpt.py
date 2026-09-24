"""
EXP-G prep: wrap P8 S1 ema_final.pth → synthetic ckpt_last.pth
for migrate_stage1_to_stage2_ckpt.py

P8 S1 ckpt_last.pth was deleted to save space; only ema_final.pth (state_dict)
survives. migrate expects {'it', 'weights', 'ema', 'optimizer', 'scheduler'}.

We reconstruct:
  weights = ema_final state_dict (EMA-averaged model weights — better than raw)
  ema     = {ema_models.{0,1}.ema_model.{k}: v} (both slots seeded from ema_final)
  it      = 400000
  optimizer/scheduler = None (migrate clears these anyway)
"""
import torch

SRC = "/home/kojiek/MeanAudio/exps/phase8_stage1_400000/phase8_stage1_400000_ema_final.pth"
DST = "/home/kojiek/exps_nvme/phase8_s1_synthetic_ckpt_last.pth"

print(f"Loading {SRC} ...")
sd = torch.load(SRC, map_location="cpu")

# ema_final.pth should be a flat state_dict
assert isinstance(sd, dict), f"Unexpected type: {type(sd)}"
top_keys = list(sd.keys())[:5]
print(f"Top-level keys sample: {top_keys}")
# Sanity: should NOT have nested 'it'/'weights' keys
assert "it" not in sd, "ema_final seems to be a full ckpt already — no wrapping needed"

# Build the EMA dict in the format migrate expects
# PostHocEMA also needs _extra_state per ema_model slot
ema_dict = {}
for k, v in sd.items():
    ema_dict[f"ema_models.0.ema_model.{k}"] = v.clone()
    ema_dict[f"ema_models.1.ema_model.{k}"] = v.clone()
ema_dict["ema_models.0._extra_state"] = {"initted": True, "step": 400000}
ema_dict["ema_models.1._extra_state"] = {"initted": True, "step": 400000}

synthetic_ckpt = {
    "it":        400000,
    "weights":   sd,
    "ema":       ema_dict,
    "optimizer": None,
    "scheduler": None,
}

torch.save(synthetic_ckpt, DST)
print(f"Saved synthetic ckpt → {DST}")

# Quick verify
chk = torch.load(DST, map_location="cpu")
print(f"it = {chk['it']}")
print(f"weights keys (sample): {list(chk['weights'].keys())[:4]}")
print(f"ema keys (sample): {list(chk['ema'].keys())[:4]}")
print("Done.")
