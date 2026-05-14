# Qwen Caption Rerun Series — 4-token 命名 + JMQ confound resolution

> 4-token 命名（2026-05-08 統一）：`{Caption}-{Sel}-{Q}` 預設 MC eval；非預設加括號如 `(JMQ)`。
>
> 完整 token 對照見 `phase_status.md` 頂端速查表。

---

## 1. 動機（Codex 2026-05-05 review）

P9.5 V1 (Qwen-Multi-NoQ) 後我們提出工作假說「multi-cap random-pick 是 collapse 主因」。
Codex review 指出唯一值得補的 isolated control 是 **Qwen static single-cap** — 拆「multi-cap 形式」vs「Qwen caption style」。

延伸成 3 組（同訓練配方，唯一變因 = caption source/selection）：

| 4-token 名 | 對 LP 對照 | 目的 |
|---|---|---|
| **Qwen-Rnd-NoQ** | LP-Rnd-NoQ (0.185) | 測 Qwen single-cap 是否健康 |
| **Qwen-Rnd-Q** | LP-Rnd-Q (0.198) | 測 Qwen + Q 條件能否救回 |
| **Qwen-BC-NoQ** | LP-BC-NoQ (0.191) | 測 BestConsensus selection 能否救回 |

每組 S1 fluxaudio_s + S2 meanaudio_s 從零（251K Jamendo segments，phase7_v1 ID 順序）。

---

## 2. 結果

### 2.1 預設 MC eval（MusicCaps human captions）

| Model | MC CLAP ↑ | MC CE ↑ | MC CU ↑ | MC PC ↑ | MC PQ ↑ |
|---|---|---|---|---|---|
| **LP-Rnd-Q** (歷史最佳) | **0.1975** | 6.27 | 7.07 | 5.07 | 6.98 |
| **LP-Rnd-NoQ** (歷史) | **0.1851** | 5.91 | 6.75 | 4.98 | 6.54 |
| LP-BC-NoQ (歷史 P4 V2) | 0.1909 | — | — | — | — |
| LP-Multi-NoQ (P9 V1) | 0.0650 | — | — | — | — |
| LP-Multi-Q (P9 V2) q=9 | 0.0403 | — | — | — | — |
| **Qwen-Multi-NoQ** (P9.5 V1) | **0.0609** | 6.07 | 6.63 | 5.42 | 6.52 |
| **Qwen-Rnd-NoQ** (P8-Qwen) | **0.0611** | 5.98 | 6.59 | 5.42 | 6.48 |
| **Qwen-Rnd-Q** (P7V1-Qwen) q=9 | **0.0686** | 5.78 | 6.63 | 5.47 | 6.42 |
| **Qwen-BC-NoQ** (P4V2-Qwen) | **0.0611** | 6.02 | 6.59 | 5.36 | 6.54 |

### 2.2 非預設 eval（confound resolution）

JM s42 audio (n=2,048) × 兩種 prompt（`JM` = LP captions, `JMQ` = Qwen captions）：

| Model | (JM) CLAP | (JMQ) CLAP | (JMQ) PE-AV | (JMQ) R@10 |
|---|---|---|---|---|
| **LP-Rnd-NoQ (JMQ)** ← REVERSE | 0.199 | **0.225** | **+0.193** | **10.30%** |
| Qwen-Rnd-NoQ (JM) | 0.058 | — | — | — |
| Qwen-Rnd-NoQ (JMQ) | — | 0.078 | +0.085 | 0.20% |
| Qwen-Rnd-Q (JMQ) q=9 | 0.060 | 0.079 | +0.083 | 0.49% |
| Qwen-Multi-NoQ (JMQ) | 0.059 | 0.080 | +0.086 | 0.54% |

### 2.3 Steering probe ratio (4 prompt pair × 3 seed × 2 prompt = 24 wav each)

```
LP-Rnd-Q (歷史 P7 V1):       1.07-1.70   ← prompt-dominant
LP-Rnd-NoQ (歷史 P8):         0.91-1.72   ← prompt-dominant
LP-Multi-NoQ (P9 V1):         0.025-0.147 ← noise-dominant (collapse)
LP-Multi-Q (P9 V2 q=8):       0.012-0.056 ← collapse
Qwen-Multi-NoQ (P9.5 V1):     0.022-0.044 ← collapse
Qwen-Rnd-NoQ (P8-Qwen):       0.033-0.120 ← collapse
Qwen-Rnd-Q (P7V1-Qwen) q=9:   0.017-0.057 ← collapse
```

### 2.4 Qwen-local q sweep（Qwen-Rnd-Q on MC）

| q | MC CLAP |
|---|---|
| q=6 | 0.0687 |
| q=9 | 0.0686 |

→ **Qwen-local q sweep flat** — 與歷史 LP-Rnd-Q (P7 V1) 在 in-support q=6/9 plateau 行為一致。Q=N 在 Qwen-trained 上仍是 coarse regime marker。

---

## 3. 解讀（嚴格分層）

### 3.1 已證明（observation）

1. **MC eval（預設）**：所有 4 個 Qwen-trained 模型 MC CLAP 0.061-0.069，遠低於 LP-trained 0.185-0.198
2. **JMQ eval（in-distribution Qwen prompts）**：3 個 Qwen 模型 0.078-0.080，仍 collapse；reverse control LP-Rnd-NoQ (JMQ) **0.225** healthy
3. **Steering ratio**：所有 Qwen 模型 max < 0.15，與 LP-Multi 同 collapse 區
4. **PE-AV peav_score**：Qwen 模型在 JMQ 都 ~+0.085；LP reverse on JMQ +0.193（**~2.3x**）
5. **t2a R@10**：Qwen 模型 0.20-0.54%（≈ random baseline 0.49%）；LP reverse on JMQ **10.3%** (~50x)
6. **Qwen vs LP single-cap drop**：0.1851 → 0.0611 = **−67%**

### 3.2 高可信推論（behavior-level，非 mechanism）

1. **Qwen training regime 真的沒學會 prompt-conditioning**（不只是 train-test prompt mismatch artifact）
   - 三層證據：CLAP, PE-AV peav, t2a R@10 都 collapse
   - **同樣 Qwen prompts 在 LP-MC trained 模型上 work 良好** → 排除「Qwen prompts 不適合 eval」假設
2. **Train-test prompt distribution mismatch 真實但 minor**：
   - 補 Qwen-prompt eval 後 Qwen 模型 CLAP +30%（0.061 → 0.078-0.080），但仍 collapse
   - 50× R@10 gap = 主因是 training 失敗，不是 distribution mismatch
3. **multi-cap 與 Q signal 在 Qwen regime 影響邊際小**：
   - Qwen-Multi-NoQ (0.061) ≈ Qwen-Rnd-NoQ (0.061) — multi-cap 形式無顯著加成
   - Qwen-Rnd-Q (0.069) vs Qwen-Rnd-NoQ (0.061) — Q 加 +12% 仍遠 below healthy
4. **P9.5 V1 collapse 主因應改寫**：從「multi-cap random-pick」改為「Qwen captioner regime」

### 3.3 不能宣稱

- ❌「multi-cap 在 LP regime 也無關」— LP-Rnd-NoQ (0.185) → LP-Multi-NoQ (0.065) drop 仍真實
- ❌「Qwen captions 本質不適合 audio generation」— mechanism 沒證
- ❌「Qwen 比 LP-MC 差」— 單方向比較，可能 Qwen 適合別的任務或別的 hyperparameter
- ❌「BC selection 救不回」— 等 Qwen-BC-NoQ 結果

### 3.4 Mechanism 工作假說（未證、不寫 paper claim）

Qwen task-framed captions 與訓練看到的 caption distribution 可能差異：
- Verbosity：Qwen 平均 20-25 詞 vs LP-MC ~15 詞
- Narrative structure：Qwen 用「It begins with... transitions to... finally...」temporal narrative
- Task framing residue：「This music features...」「This composition masterfully blends...」前綴

但這些 style features **理論上 confound resolution 已部分處理**（Qwen-prompt eval 仍 collapse）。
更可能的 mechanism 假說：
- Qwen captioning model 的 audio→text 映射與 CLAP/PE-AV evaluator 內部 audio-text alignment 不一致
- Qwen captions 描述「整體音樂」而 MusicCaps 風格描述「核心特徵」→ 訓練時 model 學到的 conditional distribution 結構不同

要證 mechanism 需要：
- LP-MC vs Qwen caption embedding 距離分析
- 訓練 short-Qwen variant（截斷至 LP-MC 長度）
- 訓練 LP-MC ↔ Qwen 混合 caption variant
- 不在當前 paper scope

---

## 4. Paper narrative 主軸（5/8 確定版）

> Two factors can independently produce collapse in this training regime:
>   (a) multi-cap random-pick supervision (LP-Rnd-NoQ 0.185 → LP-Multi-NoQ 0.065)
>   (b) caption distribution mismatch — specifically the Qwen2.5-Omni captioner regime
>       (LP-Rnd-NoQ 0.185 → Qwen-Rnd-NoQ 0.061)
>
> Confound check: in-distribution Qwen-prompt eval (`(JMQ)`) shows that the
> reverse control LP-Rnd-NoQ (JMQ) achieves CLAP 0.225, PE-AV +0.193, R@10 10.3%
> — **the Qwen prompts themselves are not the issue**. All Qwen-trained variants
> (Multi-NoQ, Rnd-NoQ, Rnd-Q) cluster at CLAP ~0.08, PE-AV ~+0.085, R@10 < 0.6%
> on the same Qwen prompts, confirming the collapse is at the model side, not
> the evaluation side.
>
> P9.5 V1 collapse is dominantly attributable to (b), not (a) as previously
> hypothesized. The multi-cap formal structure has marginal additional effect
> in the Qwen regime (Qwen-Multi-NoQ ≈ Qwen-Rnd-NoQ).

---

## 5. Qwen-BC-NoQ（P4V2-Qwen）✅ 完成 2026-05-08

MC CLAP **0.0611**，JM s42 0.0596 → Qwen-Rnd-NoQ (0.0611) 完全相同，進入 +0.020 Qwen-prompt boost cluster（第 7 個 collapsed 模型）。

BC selection 在 Qwen captioner 下無法救回 collapse：確認 collapse 不是 selection strategy 的問題，而是 Qwen caption distribution 本身。

---

## 6. Artifacts

### Pipeline scripts
- `~/MeanAudio/train_pipeline_p8_qwen.sh` (Qwen-Rnd-NoQ)
- `~/MeanAudio/train_pipeline_p7v1_qwen.sh` (Qwen-Rnd-Q)
- `~/MeanAudio/train_pipeline_p4v2_qwen.sh` (Qwen-BC-NoQ)
- `~/MeanAudio/probe_v1_steering.sh`（Q-aware via PROBE_QUALITY env）
- `~/qwen_prompt_eval_chain.sh`（JMQ confound resolution chain）

### Prep scripts
- `~/research/meanaudio_training/gen_qwen_singlecap_selections.py` (251K train)
- `~/research/meanaudio_training/slice_qwen_singlecap_npz.py`
- `~/research/meanaudio_training/gen_qwen_test_captions.py` (2,048 test set)
- `~/research/meanaudio_training/gen_qwen_test_eval_tsvs.py`

### Eval audio + metrics（per-model 完整路徑）
- `eval_output/{exp}_stage2_200000_{musiccaps,jamendo_s42,qwen_random_jamendo_s42}/audio/*.flac`
- `eval_output/metrics/{exp}_stage2_200000_*/metrics.txt`
- `eval_output/metrics/{exp}_stage2_200000_*_peav.json`

### Steering probes
- `~/MeanAudio/eval_output/{exp}_stage2_200000_steering_probe/audio/`（24 wav each）

---

## 7. 引用

- Codex review commits: `01443b3`, `d45c90e`, `f9f055e`, `13ac52e`, `9481414`
- Memory:
  - `feedback_p9_5_paper_wording_2026_05_05.md`（三層 wording 紅線）
  - `feedback_qwen_eval_prompt_mismatch_2026_05_07.md`（confound 識別）
  - `project_qwen_rerun_finding_2026_05_07.md`（本次主要 finding）
  - `feedback_qwen_rerun_naming_2026_05_08.md`（4-token 命名規則）
- 設計討論：`phase9_design.md`
- P9.5 V1 結果：`phase9_5_summary.md`（pre-Qwen-rerun，narrative 已被 5/7 update 修正）
