#!/usr/bin/env python3
"""
set_training_stage.py
---------------------
切換 mean_flow.py 的訓練模式（Stage 1 / Stage 2）。

用法：
    python set_training_stage.py --stage 1    # FluxAudio，不傳入 r 參數
    python set_training_stage.py --stage 2    # MeanAudio，傳入 r 參數
    python set_training_stage.py --check      # 顯示目前模式，不修改檔案

目標檔案：
    ~/MeanAudio/meanaudio/model/mean_flow.py

Stage 1 vs Stage 2 差異說明：
    Stage 1 (FluxAudio) — fn() 不接受 r 參數，使用 flow_matching runner
    Stage 2 (MeanAudio) — fn() 需要 r 參數，使用 meanflow runner

作者：Research Assistant
日期：2026-03-04
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime

# ── 目標檔案路徑 ──────────────────────────────────────────────────────────────
MEAN_FLOW_PATH = Path.home() / "MeanAudio/meanaudio/model/mean_flow.py"

# ── Stage 1 ↔ Stage 2 的文字替換對照表 ───────────────────────────────────────
#
# 每個 entry 格式：
#   (stage1_text, stage2_text)
#
# 切換至 Stage 1 時：將 stage2_text → stage1_text
# 切換至 Stage 2 時：將 stage1_text → stage2_text

PATCHES = [
    # patch 1：u_t 呼叫，移除 r=t（保留 q=null_token）
    (
        # Stage 1
        "            u_t = fn(latent=z, \n"
        "                     text_f=u_text_f,\n"
        "                     text_f_c=u_text_f_c,\n"
        "                     text_attention_mask=empty_text_attention_mask,\n"
        "                     t=t,\n"
        "                     q=torch.full_like(q, 10) if q is not None else None).detach().requires_grad_(False)",
        # Stage 2
        "            u_t = fn(latent=z, \n"
        "                     text_f=u_text_f,\n"
        "                     text_f_c=u_text_f_c,\n"
        "                     text_attention_mask=empty_text_attention_mask,\n"
        "                     r=t,\n"
        "                     t=t,\n"
        "                     q=torch.full_like(q, 10) if q is not None else None).detach().requires_grad_(False)",
    ),
    # patch 2：u_t_c 呼叫，移除 r=t（保留 q=q）
    (
        # Stage 1
        "            u_t_c = fn(latent=z, \n"
        "                       text_f=text_f_undrop,\n"
        "                       text_f_c=text_f_c_undrop,\n"
        "                       text_attention_mask=text_attention_mask_undrop,\n"
        "                       t=t,\n"
        "                       q=q).detach().requires_grad_(False)",
        # Stage 2
        "            u_t_c = fn(latent=z, \n"
        "                       text_f=text_f_undrop,\n"
        "                       text_f_c=text_f_c_undrop,\n"
        "                       text_attention_mask=text_attention_mask_undrop,\n"
        "                       r=t,\n"
        "                       t=t,\n"
        "                       q=q).detach().requires_grad_(False)",
    ),
    # patch 3：jvp lambda，移除 r_f
    (
        # Stage 1
        "            lambda z_f, t_f: model_partial(latent=z_f, t=t_f),",
        # Stage 2
        "            lambda z_f, r_f, t_f: model_partial(latent=z_f, r=r_f, t=t_f),",
    ),
    # patch 4：jvp primals tuple，移除 r
    (
        # Stage 1
        "            (z, t),",
        # Stage 2
        "            (z, r, t),",
    ),
    # patch 5：jvp tangents tuple，移除 torch.zeros_like(r)
    (
        # Stage 1
        "            (v_hat, torch.ones_like(t)),",
        # Stage 2
        "            (v_hat, torch.zeros_like(r), torch.ones_like(t)),",
    ),
]


def detect_current_stage(content: str) -> int:
    """
    根據所有切換片段判斷目前是 Stage 1 還是 Stage 2。

    新增參數曾讓單一舊字串的 detector 誤報 stage；因此五個片段必須
    全部屬於同一 stage。混合或未知狀態直接失敗，不能默默繼續訓練。
    """
    stage1_count = sum(stage1_text in content for stage1_text, _ in PATCHES)
    stage2_count = sum(stage2_text in content for _, stage2_text in PATCHES)
    expected = len(PATCHES)
    if stage1_count == expected and stage2_count == 0:
        return 1
    if stage2_count == expected and stage1_count == 0:
        return 2
    raise RuntimeError(
        "無法安全判定 training stage："
        f"stage1 fragments={stage1_count}/{expected}, "
        f"stage2 fragments={stage2_count}/{expected}"
    )


def apply_stage(content: str, target_stage: int) -> tuple[str, int]:
    """
    將 content 轉換至指定 stage。
    回傳 (新內容, 成功替換次數)。
    """
    count = 0
    for stage1_text, stage2_text in PATCHES:
        if target_stage == 1:
            src, dst = stage2_text, stage1_text
        else:
            src, dst = stage1_text, stage2_text

        if src in content:
            content = content.replace(src, dst)
            count += 1

    return content, count


def backup_file(path: Path) -> Path:
    """在相同目錄建立 .bak 備份（覆蓋舊備份）。"""
    bak_path = path.with_suffix(".py.bak")
    shutil.copy2(path, bak_path)
    return bak_path


def check_mode(path: Path) -> None:
    """僅顯示目前模式，不修改檔案。"""
    if not path.exists():
        print(f"[ERROR] 找不到檔案：{path}")
        return

    content = path.read_text(encoding="utf-8")
    stage = detect_current_stage(content)
    model = "FluxAudio（不傳入 r）" if stage == 1 else "MeanAudio（傳入 r）"
    print(f"目前模式：Stage {stage}  →  {model}")
    print(f"檔案路徑：{path}")


def switch_stage(path: Path, target_stage: int) -> None:
    """切換至指定 stage 並寫入檔案。"""
    if not path.exists():
        print(f"[ERROR] 找不到檔案：{path}")
        return

    content = path.read_text(encoding="utf-8")
    current_stage = detect_current_stage(content)

    if current_stage == target_stage:
        print(f"[INFO] 目前已是 Stage {target_stage}，無需修改。")
        return

    # 套用替換
    new_content, count = apply_stage(content, target_stage)

    if count != len(PATCHES):
        raise RuntimeError(
            f"拒絕部分 stage 切換：預期替換 {len(PATCHES)} 處，實際 {count} 處"
        )

    # 確認完整替換後才備份與寫入。
    bak_path = backup_file(path)
    print(f"[備份] {bak_path}")

    # 寫入
    path.write_text(new_content, encoding="utf-8")

    model = "FluxAudio（不傳入 r）" if target_stage == 1 else "MeanAudio（傳入 r）"
    print(f"[完成] Stage {current_stage} → Stage {target_stage}  ({model})")
    print(f"       修改 {count} 處  |  檔案：{path}")

    # 驗證
    verify_content = path.read_text(encoding="utf-8")
    verified_stage = detect_current_stage(verify_content)
    if verified_stage == target_stage:
        print(f"[驗證] ✓ 確認目前為 Stage {target_stage}")
    else:
        print(f"[驗證] ✗ 異常：偵測到 Stage {verified_stage}，請手動檢查檔案。")


def main():
    parser = argparse.ArgumentParser(
        description="切換 mean_flow.py 的訓練模式（Stage 1 / Stage 2）"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        help="目標 Stage（1 = FluxAudio，2 = MeanAudio）",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="顯示目前模式，不修改檔案",
    )
    args = parser.parse_args()

    if args.check:
        check_mode(MEAN_FLOW_PATH)
    else:
        switch_stage(MEAN_FLOW_PATH, args.stage)


if __name__ == "__main__":
    main()
