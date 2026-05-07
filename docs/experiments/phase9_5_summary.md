# Phase 9.5 — 意義、訓練設置、測試目標、結果

> **4-token 名稱**：`Qwen-Multi-NoQ` (V1) / `Qwen-Multi-Q` (V2 已 SKIP)
> 對外舊名：`JamendoFull-QwenOmni-TrueRandom-NoQ` / `...MeanSim-Q`
> 完成 2026-05-04。設計討論見 `phase9_design.md`，Codex 2 輪 review 見 commit history。
>
> ⚠️ **2026-05-08 update**：本檔 §5 narrative 在 5/4 寫成「multi-cap collapse cross-captioner」，
> 但後續 Qwen single-cap rerun (Qwen-Rnd-NoQ / Qwen-Rnd-Q / Qwen-BC-NoQ) + JMQ confound
> resolution 顯示 Qwen-Multi-NoQ 的 collapse **主因是 Qwen captioner regime**（非 multi-cap 形式）。
> 修正版 narrative 見 `qwen_rerun_summary.md` §3-4。本檔保留作 5/4 當下推論的歷史記錄。

---

## 1. 意義（為什麼跑）

### 1.1 起源：P9 失敗歸因不確定

Phase 9 V1（multi-cap NoQ，LP-MusicCaps 5 caps）跑出 MusicCaps CLAP **0.0650**，
遠低於 P8 single-cap NoQ（0.1851）和 P7 V1 Q（0.1975）。
失敗來源有兩個 working hypotheses：

- **H_paradigm**：`multi_cap=True` 訓練範式（每 iter 對每 clip 隨機抽 5-of-1 caption）的 supervision noise 本質傷害 prompt-conditioning
- **H_caption**：LP-MusicCaps 5 caps 的 seed-sampled decoding noise 太雜亂；換結構化 caption 應該能救

### 1.2 LP-MusicCaps Jamendo 5 caps 的真相（york135 4/18 澄清）

我們手上的 `results_20260119_043407.jsonl` **不是** LP-MusicCaps 論文原始 4-task pipeline 的產物，
而是 wei-jaw 用 LP-MC captioning model 跑 **5 個不同 seed** 出來的：

- 5 caps 之間 diversity 來自 **decoding 隨機性**（同 model 同輸入不同 seed）
- 描述同一段音訊，用詞不同
- LP-MusicCaps 論文的 4-task 設計**從未套用於 Jamendo**

### 1.3 P9.5 兩個目的（教授 4/17 確認）

1. 自家跑 caption（不依賴第三方），答審稿人「真實 caption 來源」質疑
2. 看 multi-cap collapse 是否在另一組 caption 來源 + 不同 diversity 機制下仍成立

### 1.4 Framing 紅線（Codex 5/3 review）

❌ 不能寫：「P9.5 是 cross-captioner control」/「驗證 cross-captioner diversity hypothesis」  
✅ 該寫：「task-framed Qwen multi-cap variant / stress test」

理由：同時改了 **captioner**（LP-MC → Qwen）和 **diversity 機制**（seed-sampled → task-framed），
兩個變因綁在一起 → stress test 不是 isolated control。
真正的 captioner-only control 需要：同 captioner 出 seed-sampled vs task-framed 兩組。

---

## 2. 訓練設置

### 2.1 Caption 來源：Qwen2.5-Omni-3B 5-task captioning

對 251,599 個 Jamendo segments 跑 5 次 Qwen2.5-Omni-3B caption，每次用不同 task framing：

| Slot | Task prompt 方向 | Mean words |
|---|---|---|
| 0 | Writing：詳細自然描述句 | ~21 |
| 1 | Summary：壓縮綜合短句 | ~21 |
| 2 | Paraphrase：豐富詞彙改寫 | ~24 |
| 3 | Attribute Prediction：屬性為主 | ~25 |
| 4 | NaturalProse：中性自然敘述 | ~20 |

每個 caption **皆 comprehensive**（涵蓋樂器+情緒+節奏+風格），不是 aspect 切片。
Diversity 來自 **task framing**（不同 prompt 引出不同視角），不是 seed noise。

**Captioning 規模**：251,599 × 5 = 1,257,995 captions，~88h wall clock（混 GPU 共用 + 獨佔），
產出 `phase9_omni_captions.jsonl`（182 MB，5/2 完成）。

**Sanity（n=200 抽樣）**：
- 100% 唯一率（無 caption collapse）
- CLAP text-c pairwise mean_sim 平均 0.644，range 0.418-0.839（健康多樣性）

### 2.2 Caption-source 對比（vs LP-MC）

| 維度 | LP-MC (P9) | Qwen P9.5 |
|---|---|---|
| Captioning model | LP-MusicCaps captioning model | Qwen2.5-Omni-3B |
| Diversity 機制 | **5 different seeds** | **5 different task prompts** |
| Caption 性質 | 同視角不同詞語 | 不同視角各自 comprehensive |
| 5 caps 結構 | 描述同一面向 | 5 個 framing |

### 2.3 訓練 pipeline（共通）

| 維度 | 設定 |
|---|---|
| Audio dataset | Jamendo 251,599 segments（同 P7/P8/P9 的 phase7_v1_train.tsv ID order）|
| Latent encoder/VAE | 同 historical（mean/std reuse 自原 single-cap NPZ）|
| Model | FluxAudio_s (S1) → MeanAudio_s (S2)（Flow Matching → MeanFlow）|
| Batch / lr | 8 / 1e-4 |
| Iter | S1 400K + S2 200K |
| LR schedule | S1 [320000, 360000]、S2 [999999, 999999]（無 decay）|
| Bug fixes 套用 | networks.py q=10 + runner_meanflow clone + runner_flowmatching q-passing |

### 2.4 V1 vs V2 差異

| 維度 | V1 (NoQ) | V2 (Q) — 已 SKIP |
|---|---|---|
| `multi_cap=True` | ✓ | ✓ |
| `use_q_conditioning` | false | true |
| Q signal | 無 | pairwise text-text CLAP cos sim of 5 Qwen caps，**Qwen-local** percentile bin 0..9 |
| Train TSV | `phase9_5_train.tsv` (slot 0 caption + dummy q=5) | `phase9_5_v2_train.tsv` (real q_level，未產出) |
| S1 + S2 | 從零 | 從零（Codex P1：不可 reuse P9 V1 LP S1，會混 prior）|
| Eval q | `--no_q` (null token) | q sweep {5..9} on MusicCaps（Codex P2：Qwen q 是 captioner-local，不沿用 P7 q=6/9 假設）|

### 2.5 multi_cap=True 機制（`extracted_audio.py`）

```python
if self.multi_cap:
    cap_idx = random.randint(0, n_caps - 1)
    text_features   = np_data['text_features'][cap_idx]    # NPZ [5,77,1024]
    text_features_c = np_data['text_features_c'][cap_idx]  # NPZ [5,512]
```

NPZ 每檔存 5 個 T5+CLAP 編碼，每 iter 隨機抽 1 個當 supervision。
**唯一改動**：NPZ 裡的 5 caps 從 LP-MC seed-sampled 換成 Qwen task-framed。

---

## 3. 要測什麼

### 3.1 主問題

> **Multi-cap collapse 在「換 captioner + 換 diversity 機制」之後是否還會發生？**

### 3.2 副問題

1. Qwen task-framed 是不是比 LP seed-sampled 更好？
2. 換高品質 caption 救得回 prompt conditioning 嗎？

### 3.3 V2 launch gate（Codex 5/3 sequential gating）

V1 結果若**任一**符合就跑 V2，否則 SKIP：
- MC CLAP > 0.0650（超過 P9 V1 baseline）
- 任一 prompt-pair same-seed steering ratio > 0.2（沒 collapse）

---

## 4. 結果（2026-05-04 完成）

### 4.1 V1 metrics

| Benchmark | CLAP ↑ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ |
|---|---|---|---|---|---|
| MusicCaps n=5521 | **0.0609** | 6.07 | 6.63 | 5.42 | 6.52 |
| Jamendo seed42 n=2048 | 0.0594 | 6.04 | 6.61 | 5.42 | 6.50 |

### 4.2 對照 baselines

| Model | MC CLAP | Steering max | 解讀 |
|---|---|---|---|
| P7 V1 (Q, single-cap) | 0.1975 | 1.702 | 健康 prompt-dominant |
| P8 (NoQ, single-cap) | 0.1851 | 1.723 | 健康 prompt-dominant |
| P9 V1 (NoQ, LP multi) | 0.0650 | 0.147 | LP collapse |
| **P9.5 V1 (NoQ, Qwen multi)** | **0.0609** | **0.044** | **Qwen collapse, 略劣於 P9 V1** |

> CLAP 全量 n=5521（`phase4_eval.py --num_samples` 只控 FAD，CLAP/AES 永遠跑全量）。
> 跨 phase 直接對照成立，無 sample-size confound。

### 4.3 Steering probe 細節（`probe_v1_steering.sh`）

4 prompt pairs × 3 seeds × 2 prompts = 24 wav，量 `(A-B L2) / (noise L2)` ratio：

| pair | A-B L2 | noise L2 | ratio |
|---|---|---|---|
| 01 instrument (piano vs EDM) | 2.703 | 79.846 | 0.034 |
| 02 vocals (instrumental vs pop vocal) | 3.461 | 80.389 | 0.043 |
| 03 drums (drumless vs techno) | 3.491 | 80.006 | 0.044 |
| 04 density (sparse violin vs dense orchestra) | 1.790 | 79.846 | 0.022 |

**Max ratio 0.044**，全 4 pair 都遠低於 noise 主導門檻（1.0）。

### 4.4 V2 verdict

| Codex gate | 數值 | 結果 |
|---|---|---|
| MC CLAP > 0.0650 | 0.0609 | ❌ |
| 任一 pair steering > 0.2 | max 0.044 | ❌ |

→ **V2 SKIP**。再跑 19h 只會確認失敗。Q variant 在 P9 V2 已驗證更差（CLAP 0.0403）。

---

## 5. 解讀（嚴格分層 — Codex 5/5 review 收斂版）

### 5.1 已證明（observation level）

- P9.5 V1（Qwen task-framed multi-cap NoQ）也落在 collapse 區：MusicCaps CLAP **0.0609**、Jamendo seed42 CLAP **0.0594**
- P9.5 V1 same-seed prompt steering ratio 全 4 pair 都很低：**0.022–0.044**
- 此 pattern 與 P9 V1（LP-MC seed-sampled multi-cap NoQ，CLAP 0.0650 / steering 0.025–0.147）一致
- P7 V1 / P8 single-cap 明顯健康（CLAP 0.18-0.20、steering 0.9-1.7）；P9 / P9.5 multi-cap 明顯不健康

### 5.2 高可信推論（behavior-level，非 mechanism 證明）

- Collapse 不是 LP-MusicCaps seed-sampling 特有
- Qwen task-framing diversity 沒救回 multi-cap random-pick 訓練
- 問題**更可能**來自 multi-cap supervision 的形式（random 1-of-5 caption per iter），而不是單一 captioner 的 caption 品質
- captioner-specific artifact 的可能性**降低，但尚未完全排除**

### 5.3 不能宣稱（已被 Codex 5/5 review 否決）

- ❌「證明 multi-cap 本質不可用」
- ❌「證明跟 caption source 無關」（沒掃多種 captioner 不能 categorical claim）
- ❌「證明是 random 1-of-5 的 causal effect」（沒 isolated control）
- ❌「V2 一定會失敗」（V2 SKIP 是 gate 觸發，不是預測）
- ❌「task-framed 不如 seed-sampled」/「Qwen 比 LP-MC 差」（單 run、6.3% 差距，可能 noise）

### 5.4 V2 SKIP 的正確 wording

> Because V1 failed the pre-defined launch gate (MC CLAP > 0.0650 OR any pair steering > 0.2),
> we did not spend additional GPU on the Q-conditioned variant.

**不要寫**「V2 跑只會確認失敗」。是 gate 觸發、不是預測。

### 5.5 對論文的價值

P9.5 把 P9 的「可能只是 LP-MC 雜亂」這條 reviewer 反駁路線變弱：

- 質疑「P9 失敗只是 LP 第三方 caption 問題」→ 可指 P9.5 用自家 Qwen 也落在 collapse 區間
- 質疑「seed-sampled diversity 不夠結構化」→ 可指 P9.5 task-framed 結構化 prompt 也沒救回

Negative finding 但 robust，比 P9 單一觀察強 — 但要當 robustness evidence 寫，不是 categorical proof。

### 5.6 工作假說（未證、不寫進 paper claim）

multi_cap=True 訓練機制（每 iter 隨機抽 5 cap 中 1）**可能**對 text encoder 有結構性傷害：
- T5 forward 接到 5 個非常不同的 text features 但都對同一 audio target → 等效 noise injection
- Random pick 把 caption→audio 變 1-to-many → conditional generative 學不出穩定 prior

這些是 hypothesis，不是 result。要證 mechanism 需要 isolated control（見 §7）。

---

## 5.X UPDATE 2026-05-07 — Qwen single-cap rerun 推翻上面的 multi-cap 主因解讀

> Codex 5/5 推薦的「Qwen static single-cap NoQ」isolated control 已執行。完整 P8/P7V1/P4V2-Qwen 三組對照見 `docs/experiments/qwen_rerun_summary.md`，這裡只記對 §5 的修正。

### 新證據

| Phase | Setup | MC CLAP | Steering max | PE-AV peav |
|---|---|---|---|---|
| P8 LP-MC NoQ (歷史) | LP single-cap | **0.1851** | **1.72** | — |
| **P8-Qwen NoQ** | **Qwen single-cap** | **0.0611** | 0.120 | **−0.038** |
| **P7V1-Qwen Q** | **Qwen single-cap + Q** | **0.0687** | 0.057 | **−0.038** |
| P9.5 V1 multi NoQ | Qwen multi-cap | 0.0609 | 0.044 | — |

### 對 §5.2 推論的修正

舊推論「問題**更可能**來自 multi-cap supervision 形式」**不再成立**：

- Qwen single-cap (P8-Qwen) MC CLAP 0.0611 ≈ Qwen multi-cap (P9.5 V1) 0.0609 → **單把 multi-cap 拿掉沒救回**
- Qwen single-cap + Q (P7V1-Qwen) MC CLAP 0.0687 → **加 Q 條件也沒救回**
- LP-MC single-cap (P8 歷史) MC CLAP 0.1851 → 同訓練配方換 caption 來源就健康

**新的 high-confidence 推論**：
- Qwen task-framed caption distribution 在這個訓練配方下會獨立觸發 collapse
- multi-cap random-pick 形式在 Qwen regime 沒有顯著加成
- multi-cap **在 LP-MC regime 仍然觸發 collapse**（P8 LP single 0.185 → P9 V1 LP multi 0.065）— 但這個觀察不能再用 P9.5 跨 captioner 證據支撐

### 對 §5.5 paper 價值的修正

舊：「P9.5 把『P9 是 LP-MC artifact』反駁路線變弱」 — 這個 framing **反過來了**：

- Qwen single-cap collapse 反而表示 **Qwen 也是某種 artifact 來源**（雖然不一定是 hallucination；可能是 verbosity / temporal narrative / task-framing 殘留）
- 真正能說的：「**multi-cap 失敗 + Qwen 失敗 是兩個獨立的 collapse trigger**，可以各自獨立發生」

### 仍不能宣稱

- ❌「multi-cap 完全無關」（LP single 0.185 → LP multi 0.065 仍是事實）
- ❌「Qwen captions 本質不適合 audio generation」（沒做 lr / data ratio / hyperparameter ablation）
- ❌「BC selection 在 Qwen 救得回」（P4V2-Qwen 跑中，5/8 才知）

### Paper narrative 主軸修正

舊主軸（P9.5 V1 alone）：multi-cap collapse cross-captioner

**新主軸**：

> Two factors can independently produce collapse:
> (a) multi-cap random-pick supervision (LP single 0.185 → LP multi 0.065)
> (b) caption distribution mismatch (LP single 0.185 → Qwen single 0.061)
>
> The Qwen captioner regime collapses regardless of single/multi format and
> Q conditioning. This means P9.5 V1 multi-cap collapse is dominantly
> attributable to (b), not (a) as previously hypothesized.

---

## 6. Artifacts

### Checkpoints
- `~/MeanAudio/exps/phase9_5_v1_stage1_400000/`（42G，含 80 個 intermediate ema）
- `~/MeanAudio/exps/phase9_5_v1_stage2_200000/`（24G）
- `_ema_final.pth` 各 459MB

### Eval audio
- `~/MeanAudio/eval_output/phase9_5_v1_stage2_200000_musiccaps/` (5527 wav)
- `~/MeanAudio/eval_output/phase9_5_v1_stage2_200000_jamendo_s42/` (2048 wav)

### Metrics
- `eval_output/metrics/phase9_5_v1_stage2_200000_musiccaps/metrics.txt` (n=5521)
- `eval_output/metrics/phase9_5_v1_stage2_200000_musiccaps_n2048/metrics.txt` (backfill, 同數字 — `--num_samples` 只控 FAD)
- `eval_output/metrics/phase9_5_v1_stage2_200000_jamendo_s42/metrics.txt`

### Probe
- `~/MeanAudio/eval_output/p9_5_v1_steering_probe/audio/` (24 wav)
- `~/logs/p9_5_v1_steering_probe.log`

### Provenance manifest
- `~/research/meanaudio_training/phase9_5_manifest.json`
  - JSONL sha256 + TSV sha256 + reference TSV first-1000 id hash
  - NPZ size mode {1638140 bytes: 251599 files}
  - 3 validations recorded（sanity_qwen_jsonl + npz_full_deep_validate + post_npz_sanity_50）

### Pipeline scripts
- `train_pipeline_phase9_5_v1.sh`（含 Pre-flight 0 overwrite guard，executed）
- `train_pipeline_phase9_5_v2.sh`（hardened with q_level preflight + q sweep 5..9，未執行）
- `probe_v1_steering.sh`

### Prep scripts
- `~/research/meanaudio_training/sanity_qwen_jsonl.py`
- `~/research/meanaudio_training/gen_phase9_5_v1_tsv.py`
- `~/research/meanaudio_training/gen_phase9_5_q_levels.py`（V2 用，未執行）

---

## 7. 後續優先順序（Codex 5/5 review）

### 為了當前 paper / meeting

1. **不開新訓練**（不跑 V2，不跑 Qwen 全套 phase replay — 太貴且不乾淨）
2. **鎖定主線 narrative**：
   - Phase 5：hard filtering 失敗（資料量 vs 品質）
   - Phase 7 V1：q_embed conditioning 有效，最佳輸出
   - Phase 8：NoQ single-cap baseline 健康
   - Phase 9 / 9.5：multi-cap random-pick 失敗，跨 LP-MC + Qwen 都 collapse
   - P8 V4：把數字寫進 prompt 失敗 → 支持「embedding condition」優於「prompt token」
3. **補最小圖表**（不補大實驗）：
   - **Steering ratio bar plot**：P7 V1 / P8 vs P9 V1 / P9.5 V1（最直接支撐 multi-cap 不健康主張）
   - Hard filtering table：P4V2 / P5V1 / P5V2
   - Main results table：P8 NoQ / P7 V1 / P9 V1 / P9.5 V1
4. **不放 Phase 7 full-Q ablation 進報告**（internal debugging，非 paper 主線）

### Deadline 後若要補 causal proof — 唯一值錢的實驗

**Qwen static single-cap NoQ**（從 5 caps 固定選一條 per clip，不做 random 1-of-5）

| 結果 | 解讀 |
|---|---|
| Qwen static single-cap **健康** | 問題就是 multi-cap random-pick 訓練形式 |
| Qwen static single-cap **也 collapse** | 問題可能是 Qwen caption style / task framing 本身（不是 multi-cap） |

這是**唯一能拆「multi-cap random-pick」vs「Qwen caption 本身」的 isolated control**。

不應該重做 Qwen 全套 phase（confounders 一起變：captioner、style、task framing、q 分布、文字長度）。

### 不建議的選項（Codex 否決）

- ❌「Qwen 重做全部實驗」（太貴 + 不乾淨）
- ❌「P9.5 V2」（V1 已 fail gate，無資訊量）
- ❌「Qwen mean_sim 分布 figure」優先（不如 steering bar plot 直接）

---

## 8. Paper / meeting 一句版

> P9.5 used a different captioner and task-framed diversity, yet the model still
> falls in the Phase 9 collapse regime: MusicCaps CLAP ~0.061 and same-seed
> prompt steering at most 0.044. This weakens the "LP-MC artifact" hypothesis
> and supports the view that the current multi-cap random-pick training recipe
> is strongly correlated with weak prompt conditioning. This is a strong
> association, not a complete causal proof.

---

## 引用

- 設計討論：`phase9_design.md` (5/3 framing 修正版)
- Codex 兩輪 review：commits `d7b586e`, `d45c90e`
- V1 結果定稿：commit `13ac52e`
- Provenance manifest：MIR_ssh commit `a198367`
- 相關 memory：
  - `feedback_p9_5_framing_2026_05_03.md`（framing 紅線）
  - `feedback_diversity_hypothesis.md`（multi-cap 必須 task-framing + comprehensive，注意 LP 是 seed-sampled）
  - `project_p9_5_v1_result_2026_05_04.md`（V1 結果 + V2 SKIP）
  - `project_p9_text_conditioning_dead.md`（4 模型 2x2 steering 分析的 P9 部分）
