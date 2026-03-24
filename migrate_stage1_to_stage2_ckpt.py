"""
Stage 1 → Stage 2 Checkpoint 移植腳本
======================================
目的：將 FluxAudio (Stage 1) checkpoint 轉換為 MeanAudio (Stage 2) 相容格式。

核心差異：
  FluxAudio: 只有 t_embed
  MeanAudio: 有 t_embed + r_embed（r_embed 初始化為 t_embed 的複製）

使用方式：
  python3 migrate_stage1_to_stage2_ckpt.py \\
      --s1_ckpt exps/stage1/stage1_ckpt_last.pth \\
      --s2_out  exps/stage2/stage2_ckpt_last.pth
"""

import argparse
import torch
import shutil
import os


def parse_args():
    p = argparse.ArgumentParser(description='Stage 1 → Stage 2 Checkpoint 移植')
    p.add_argument('--s1_ckpt', required=True, help='Stage 1 最終 checkpoint 路徑')
    p.add_argument('--s2_out',  required=True, help='Stage 2 輸出 checkpoint 路徑')
    return p.parse_args()


def main():
    args = parse_args()
    S1_CKPT = args.s1_ckpt
    S2_OUT  = args.s2_out

    # ── 載入 Stage 1 checkpoint ─────────────────────────────
    print(f"[1/4] 載入 Stage 1 checkpoint：{S1_CKPT}")
    if not os.path.exists(S1_CKPT):
        raise FileNotFoundError(f"找不到 Stage 1 checkpoint：{S1_CKPT}")
    s1 = torch.load(S1_CKPT, map_location="cpu")
    print(f"      it = {s1['it']}")
    print(f"      keys = {list(s1.keys())}")

    # ── 複製 t_embed → r_embed（主模型 weights）────────────
    print("[2/4] 複製 t_embed → r_embed（主模型）")
    weights = s1["weights"]
    mapping = {
        "t_embed.mlp.0.weight": "r_embed.mlp.0.weight",
        "t_embed.mlp.0.bias":   "r_embed.mlp.0.bias",
        "t_embed.mlp.2.weight": "r_embed.mlp.2.weight",
        "t_embed.mlp.2.bias":   "r_embed.mlp.2.bias",
    }
    for src, dst in mapping.items():
        if src in weights:
            weights[dst] = weights[src].clone()
            print(f"      {src} → {dst}  shape={weights[dst].shape}")
        else:
            print(f"      [WARNING] {src} 不存在，跳過")

    # ── 複製 t_embed → r_embed（EMA models）────────────────
    print("[2b/4] 複製 t_embed → r_embed（EMA models）")
    ema = s1["ema"]
    new_ema_entries = {}
    for k, v in ema.items():
        if "t_embed" in k:
            new_key = k.replace("t_embed", "r_embed")
            new_ema_entries[new_key] = v.clone()
            print(f"      {k} → {new_key}  shape={v.shape}")
    ema.update(new_ema_entries)
    s1["ema"] = ema

    # ── q_embed（主模型）：Stage 1 已訓練則保留，否則隨機初始化 ──
    import torch.nn as nn
    hidden_dim = weights["t_embed.mlp.0.weight"].shape[0]
    if "q_embed.weight" in weights:
        print(f"[2c/4] q_embed 已存在 Stage 1 checkpoint，直接保留（Phase 6 V2+）")
        print(f"      q_embed.weight  shape={weights['q_embed.weight'].shape}")
    else:
        print("[2c/4] q_embed 不存在，隨機初始化（Phase 6 V1 以前的 checkpoint）")
        q_embed_weight = nn.Embedding(11, hidden_dim).weight.data
        weights["q_embed.weight"] = q_embed_weight
        print(f"      q_embed.weight 隨機初始化  shape={q_embed_weight.shape}")

    # ── q_embed（EMA models）：同上邏輯 ──────────────────────
    print("[2d/4] q_embed（EMA models）")
    for prefix in ["ema_models.0.ema_model.", "ema_models.1.ema_model."]:
        key = prefix + "q_embed.weight"
        if key in ema:
            print(f"      {key} 已存在，保留  shape={ema[key].shape}")
        else:
            q_embed_weight = nn.Embedding(11, hidden_dim).weight.data
            ema[key] = q_embed_weight.clone()
            print(f"      {key} 隨機初始化  shape={q_embed_weight.shape}")
    s1["ema"] = ema

    # ── 清除 optimizer / scheduler state ───────────────────
    print("[3/4] 清除 optimizer 與 scheduler state（架構新增 r_embed + q_embed，舊 state 不相容）")
    s1["optimizer"] = None
    s1["scheduler"] = None
    s1["weights"]   = weights

    # ── 輸出 ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(S2_OUT)), exist_ok=True)
    if os.path.exists(S2_OUT):
        backup = S2_OUT.replace(".pth", "_backup_before_migrate.pth")
        shutil.copyfile(S2_OUT, backup)  # copyfile 不複製權限，相容 NTFS
        print(f"      既有 checkpoint 備份至：{backup}")

    torch.save(s1, S2_OUT)
    print(f"[4/4] 移植完成 → {S2_OUT}")

    # ── 驗證 ────────────────────────────────────────────────
    print("\n===== 驗證 =====")
    check = torch.load(S2_OUT, map_location="cpu")
    print(f"it = {check['it']}")
    embed_keys = [k for k in check["weights"].keys() if "embed" in k]
    print(f"embed keys = {embed_keys}")
    print("移植完成，Stage 2 可從此 it 正確接續。")


if __name__ == "__main__":
    main()
