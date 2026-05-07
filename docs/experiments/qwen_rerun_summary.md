# Qwen Single-Cap Rerun Series (P8 / P7V1 / P4V2 with Qwen captions)

> 三組對照 captioner-only control。P8-Qwen + P7V1-Qwen 完成 (5/6, 5/7)，P4V2-Qwen 跑中 (ETA 5/8)。
> 結論翻轉 P9.5 V1 後的 multi-cap-collapse 主因解讀。

---

## 1. 動機（Codex 2026-05-05 review）

P9.5 V1 後我們提出工作假說「multi-cap random-pick 是 collapse 主因」。Codex review 指出唯一值得補的 isolated control 是 **Qwen static single-cap** — 拆「multi-cap 形式」vs「Qwen caption style」。

延伸成 3 組（同訓練配方，唯一變因 = caption source/selection）：

| Phase | Setup | 目的 |
|---|---|---|
| **P8-Qwen** | Qwen single-cap **random** (seed=42) NoQ | 對 P8 LP-MC random NoQ (0.185)；測 Qwen caption 本身是否健康 |
| **P7V1-Qwen** | Qwen single-cap random + **Q (Qwen-local mean_sim bin)** | 對 P7V1 LP-MC random + Q (0.198)；測 Q 條件能否救回 |
| **P4V2-Qwen** | Qwen single-cap **BestConsensus** NoQ | 對 P4 V2 LP-MC BC NoQ (0.191)；測 BC selection 是否救得回 |

每組 S1 fluxaudio_s + S2 meanaudio_s 從零（251K Jamendo segments，phase7_v1 ID 順序），eval = MC + JM seed42 + PE-AV q=9 + steering probe。

---

## 2. 結果

### 2.1 完整數字

| Phase | MC CLAP | JM s42 CLAP | MC PE-AV peav | MC t2a R@10 | Steering max |
|---|---|---|---|---|---|
| P7V1 (歷史 LP-MC) | **0.1975** | 0.1981 | **+0.052** | **5.4%** | **1.70** |
| P8 (歷史 LP-MC) | **0.1851** | 0.1986 | — | — | **1.72** |
| P9 V1 (LP-MC multi) | 0.0650 | — | — | — | 0.147 |
| P9.5 V1 (Qwen multi) | 0.0609 | 0.0594 | — | — | 0.044 |
| **P8-Qwen** (Qwen single NoQ) | **0.0611** | 0.0582 | **−0.038** | 0.25% | 0.120 |
| **P7V1-Qwen** (Qwen single +Q) | **0.0687** (q=9) | 0.0599 (q=9) | **−0.038** | 0.13% | 0.057 |
| P4V2-Qwen (Qwen BC NoQ) | 跑中 | 跑中 | — | — | — |

### 2.2 P7V1-Qwen q sweep on MusicCaps

| q | CLAP |
|---|---|
| q=6 | 0.0687 |
| q=9 | 0.0686 |

→ **Qwen-local q sweep flat** — Qwen mean_sim 條件沒有 in-support gating 行為（與歷史 P7V1 LP-MC q sweep 對 q=6/9 plateau 不同）。

### 2.3 P8-Qwen vs P7V1-Qwen vs P9.5 V1 比較

```
單純把 multi-cap 從 Qwen 拿掉:    0.0609 → 0.0611  (Δ +0.3%, 統計噪聲)
單純加 Q 條件給 Qwen single-cap:  0.0611 → 0.0687  (Δ +12%, 但仍 collapse)
單純把 captioner 換掉 (LP→Qwen): 0.1851 → 0.0611  (Δ −67%, 巨大)
```

→ **captioner regime 的影響 >> multi-cap 與 Q 的影響**

---

## 3. 解讀（嚴格分層）

### 3.1 已證明（observation）

- P8-Qwen single-cap NoQ MC CLAP 0.0611, JM 0.0582
- P7V1-Qwen single-cap +Q MC CLAP 0.0687 (q=9)
- 三組 Qwen variant (multi-cap V1 / single random / single +Q) 都落在 0.06-0.07 CLAP 區
- Qwen 三組 PE-AV peav_score 全部負值或接近 0
- 所有 Qwen 模型 steering max < 0.15（與歷史 LP-MC P7V1/P8 的 1.7 對比 collapse）
- Qwen-local q sweep flat (q=6 ≈ q=9 = 0.069)

### 3.2 高可信推論（behavior-level，非 mechanism）

1. **Qwen task-framed caption distribution 本身會獨立觸發 collapse**，不需要 multi-cap random-pick
2. multi-cap supervision 形式在 Qwen regime 加成微小（multi 0.061 ~ single 0.061）
3. Q conditioning 在 Qwen regime 不能救回（+12% 仍遠低於 healthy ~0.18）
4. P9.5 V1 collapse 主因**更可能**是 Qwen captioner-style mismatch 而不是 multi-cap 形式

### 3.3 對 P9.5 V1 解讀的影響

舊（P9.5 V1 alone, 5/4）：「multi-cap 跨 captioner 都 collapse → multi-cap random-pick 是主因」

**修正版（5/7）**：「P9.5 V1 collapse 主要是 Qwen captioner-style，multi-cap 形式在那裡加成不顯著」

### 3.4 LP-MC regime 仍成立的觀察（不被推翻）

- P8 LP single-cap NoQ: CLAP 0.1851 ✓ healthy
- P9 V1 LP multi-cap NoQ: CLAP 0.0650 ❌ collapse

→ LP-MC 上 single→multi 仍有 0.185 → 0.065 drop。multi-cap 對 LP-MC 是真實 collapse 觸發。但這個證據**不能再用「跨 captioner」來支撐 multi-cap 主因說**，因為 Qwen single 已經 collapse。

### 3.5 Paper narrative 主軸（5/7 修正版）

> Two factors can independently produce collapse in this training regime:
>   (a) multi-cap random-pick supervision (P8 LP single 0.185 → P9 V1 LP multi 0.065)
>   (b) caption distribution mismatch (P8 LP single 0.185 → P8-Qwen 0.061)
>
> The Qwen captioner regime collapses regardless of single/multi format
> and with or without Q conditioning. This means P9.5 V1 collapse is
> dominantly attributable to (b), not (a) as previously hypothesized.

### 3.6 不能宣稱

- ❌「multi-cap 完全無關」（LP+single → LP+multi drop 仍是事實）
- ❌「Qwen captions 本質不適合 audio generation」（mechanism 沒證；可能是 lr / data ratio / 訓練 hyperparameter mismatch）
- ❌「Qwen 比 LP-MC 差」（單方向比較，可能 Qwen 適合別的任務）
- ❌「BC selection 救不回」（等 P4V2-Qwen 結果）
- ❌「captioner-style mismatch is the only cause」（兩個因子都觸發 collapse）

### 3.7 Mechanism 工作假說（未證、不寫 paper claim）

Qwen task-framed captions 與訓練時的 caption distribution 可能有以下差異：
- Verbosity：Qwen captions 平均 20-25 詞 vs LP-MC ~15 詞
- Narrative structure：Qwen 用「It begins with... transitions to... finally...」temporal narrative
- Task framing residue：「This music features...」「This composition masterfully blends...」前綴
- 這些 style features 在 MusicCaps test prompt 那種短格式上可能 distribution shift 嚴重

要證 mechanism 需要：
- 跑 LP-MC vs Qwen caption embedding 距離分析
- 用 Qwen-style prompt 重 eval（看是否 CLAP 回升）
- 訓練 short-Qwen variant（截斷至 LP-MC 長度）

不在當前 paper scope。

---

## 4. P4V2-Qwen 預測（待結果確認）

最可能：P4V2-Qwen MC CLAP ~0.06, steering < 0.15 → 跟 P8-Qwen 同 collapse 區，不顯著回升。
- 確認「BC selection 在 Qwen 不能救回」
- 強化「Qwen captioner-style 就是 collapse 根因」

若 P4V2-Qwen 顯著高於 P8-Qwen（CLAP > 0.10）→ 更複雜 narrative，需要分析 BC vs random 在 Qwen 下選的 caption 有什麼系統性差異。

---

## 5. Artifacts

### Pipeline scripts
- `~/MeanAudio/train_pipeline_p8_qwen.sh`
- `~/MeanAudio/train_pipeline_p7v1_qwen.sh`
- `~/MeanAudio/train_pipeline_p4v2_qwen.sh`
- `~/MeanAudio/probe_v1_steering.sh`（Q-aware via PROBE_QUALITY env）

### Prep scripts
- `~/research/meanaudio_training/gen_qwen_singlecap_selections.py`
- `~/research/meanaudio_training/slice_qwen_singlecap_npz.py`

### Backfill / chain
- `~/qwen_rerun_chain.sh` / `~/qwen_rerun_chain_phase2.sh`（auto-trigger sequential）
- `~/qwen_phase_backfill.sh`（generic PE-AV + steering backfill）
- `~/qwen_backfill_watcher.sh`（auto backfill on done sentinel）

### Checkpoints
- `~/MeanAudio/exps/p8_qwen_stage1_400000/` `p8_qwen_stage2_200000/`
- `~/MeanAudio/exps/p7v1_qwen_stage1_400000/` `p7v1_qwen_stage2_200000/`
- `~/MeanAudio/exps/p4v2_qwen_stage1_400000/` `p4v2_qwen_stage2_200000/`（跑中）

### Eval audio + metrics
- `~/MeanAudio/eval_output/{phase}_stage2_200000_{musiccaps,jamendo_s42}/audio/*.flac`
- `~/MeanAudio/eval_output/metrics/{phase}_stage2_200000_*/metrics.txt`
- `~/MeanAudio/eval_output/metrics/{phase}_stage2_200000_*_peav.json`

### Steering probes
- `~/MeanAudio/eval_output/{phase}_stage2_200000_steering_probe/audio/`（24 wav each）
- `~/logs/{phase}_stage2_200000_steering.log`

### Selections + bin edges
- `~/research/meanaudio_training/qwen_singlecap_selections.json`（id → random/BC idx + mean_sim + q_level）
- `~/research/meanaudio_training/qwen_singlecap_bin_edges.json`（Qwen-local percentile bin edges）

---

## 6. Pipeline 漏跑事件記錄（P8-Qwen）

P8-Qwen pipeline 在 phase4_eval JM s42 完成後沒進 PE-AV/steering 段，tmux 退出。原因不明：
- P7V1-Qwen 同 pipeline 結構正常完成
- 可能 transient OOM、外部 SIGKILL、或某 race condition

對策：寫了 `qwen_phase_backfill.sh` 通用 backfill；P8-Qwen 結果已手動補齊；P4V2-Qwen 設了 auto-watcher。

未來的 pipeline 可考慮把 PE-AV / steering 包成獨立 backfill script，pipeline 主流程不要連到 eval cleanup。

---

## 7. 引用

- 設計討論：`phase9_design.md`
- P9.5 V1 結果：`phase9_5_summary.md`（§5.X 為這次 update 的修正）
- Codex review：commits `01443b3`, `d45c90e`, `f9f055e`, `13ac52e`
- Memory：
  - `feedback_p9_5_paper_wording_2026_05_05.md`（三層 wording 紅線）
  - `project_p9_5_v1_result_2026_05_04.md`（P9.5 V1 結果，已被新證據更新）
  - `project_qwen_rerun_finding_2026_05_07.md`（本次主要 finding）
