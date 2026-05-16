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
- ✅「BC selection 救不回」— Qwen-BC-NoQ 完成，MC CLAP 0.0611，進入 +0.020 cluster（confirmed）

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

---

## 8. 2026-05-15 教授討論 follow-up

### 8.1 教授討論摘要

- **已定位 collapse 點**：Stage 1 訓練期就崩，非 Stage 2 才壞（與 Section 3.2 一致）
- **caption 可讀性 OK**：Qwen captions 當 eval prompt 餵 LP-trained 模型 → CLAP 0.225（Section 2.2 LP-Rnd-NoQ (JMQ)），確認 prompts 本身可解析
- **4 種訓練配置 (Rnd-NoQ / Rnd-Q / BC-NoQ / Multi-NoQ) 全 collapse** → 排除 selection / Q 信號的鍋
- **教授提案（採用）**：把 LP-MusicCaps caption 檔 quarantine，確保下次 Qwen sanity test 時，任何 code path 偷讀 LP-MC 會直接 `FileNotFoundError` — 排除「LLM 在生成 dataloader code 時 hallucinate 偷讀 LP-MC 路徑」這條 silent path
- **教授提案（拒絕）**：S1 LP-MC + S2 Qwen — 訊號模糊，好壞都不能 isolate 主因，不採用

### 8.2 LP-MC quarantine 狀態（已執行）

`/mnt/HDD` disk full (7.3T / 7.3T, 100%)，無法 mkdir 子資料夾。改用 rename-prefix：

```
/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_<原檔名>
```

16 個 LP-MC 檔案完成 rename：
- `meanaudio_captions.tsv` (216M)
- `npz.tsv` (217M)
- `phase4_train.tsv` (129M) + `phase4_val.tsv` (41M)
- `phase5_train_080.tsv` + `phase5_train_random117k.tsv`
- `phase6_train.tsv`
- `phase7_v1/v2/v3_train.tsv`
- `phase8_v2/v3/v4/v5_train.tsv`
- `phase8_v4_captions.jsonl`
- `_archive_aspect_slot0.jsonl`

**保留可讀**（Qwen + eval set + audio paths）：
- `phase9_5_train.tsv`
- `phase9_omni_captions.jsonl` + `phase9_omni_captions_slot{0-4}.jsonl`
- `musiccaps_test.tsv`、`clips.tsv`

**還原方法**：拿掉 `_QUARANTINED_` prefix（同 fs metadata 操作，秒級可逆）。

### 8.3 Qwen captions 中文 / prompt injection audit

`phase9_5_train.tsv` (251,600 行) 用 `grep -P '[^\x00-\x7f]'` 全掃：

| 類別 | 行數 | 比例 |
|---|---|---|
| Non-ASCII（總計） | 108 | 0.043% |
| 含 CJK Unified Ideographs | 57 | 0.023% |
| 含 Hiragana | 1 | <0.001% |
| Accented chars / curly quotes（其餘） | ~50 | 0.020% |

**內容形態**（57 行 CJK 中）：

1. **單詞滑出（佔多數）**——`合成器` 9×、`节奏` 5×、`流行音乐` 3×、`流水` 3×、`伴奏` 3×。Caption 主體仍英文，模型偶蹦中文詞，例 `drum节奏`、`synth流行`。

2. **🚨 Prompt injection / degenerate output（~12 行）**：caption 後面接完全無關的 LLM dialog 殘留：

```
49585: ...让人倍感欢乐。 Please write a short summary about the given news article.
83508: ...动感的氛围。 Human: What are the main instruments...? Synthesizer Compute the total number of consonants in
103034: ...激励人心。 Human: The city's streets are bustling with people and taxis...
149457: ...节奏缓慢而轻柔。 Human: 请根据以下场景描述回答问题：房间里有一张桌子,桌上有一支蜡烛...
151246: ...氛围,适合恐惧或惊悚类影视作品。 Human: Generate a single sentence that encapsulates...
183451: "{"Instrument": "Drones", ...} 生成器会根据我之前的回答进行一些小幅度的随机变化...
202301: ...宁静和谐的氛围。 In a small town, could a couple's heartbreak be the reason behind...
208693: ...营造出梦幻般的氛围。 Human: Translate the following sentence into Spanish: "I am not going..."
215280: ...Human: What genres are included in the music? ambient dubstep 请根据以上信息,对这段音乐进行
231637: ...合成器和人声采样,营造出欢快的氛围。 Human: In a quaint village nestled between the peaks of the Andes...
```

這是 **Qwen2.5-Omni-3B 在 generation 時跑進 chat template / 訓練 data 殘留**，不是隨機字元 noise。

3. **Degenerate loop**：
```
123760: ...played in拍手拍手拍手拍手拍手拍手拍手拍手拍手拍手拍手拍手拍手拍手拍手拍手
```
（`拍手` = clap，重複 16 次）

4. **日文殘漏**：
```
174651: ...a voice saying 'you gotを持って'  (を持って = "holding")
```

### 8.4 判讀

**單獨用中文字 / injection 量解釋 collapse 不足**：57/251,600 = 0.023% 污染率，FLAN-T5 處理這些行會產生 unknown token 但不會崩潰 250K 訓練集的整體 conditioning。

**但是 prompt injection 形態是 captioner failure 的 visible tip-of-iceberg**：
- 0.02% 「能看見」的失敗（中文、prompt 殘留、degenerate loop）通常對應顯著比例的「看不見」失敗（content hallucination、audio-caption mismatch、aspect 間互相矛盾的 silent 失敗）
- 教授當下指的 Heavy Metal 樣本 caption 「Heavy Metal Female Vocal Drums」短列舉形態，可能就是這類 silent failure case
- 這支持 Section 3.4 mechanism 工作假說中的「Qwen captioner audio→text 映射 quality 問題」

**對 paper narrative 的影響**：
- Section 4 主軸「(b) caption distribution mismatch — Qwen2.5-Omni captioner regime」**仍然成立**
- 可進一步補充：Qwen captioner 不只是 style distribution 不同，**還有 visible failure modes**（中文滑出 + prompt injection 殘留）→ captioner quality 本身有 systematic issue
- 但要謹守 wording 紅線（`feedback_p9_5_paper_wording_2026_05_05.md`）：observation 層報數字、推論層用 "supports captioner quality hypothesis"、不能寫「Qwen 不適合 audio captioning」

### 8.5 下一步

LP-MC quarantine 為下次 Qwen 訓練 sanity test 提供「無 silent path」環境。建議的 next-step 實驗（待你決定優先級）：

1. **LP-MC isolated Qwen sanity rerun**：在 quarantine 狀態下複跑某個 Qwen 配置（最便宜的選擇 = Qwen-Rnd-NoQ），確認 collapse 仍發生 → 排除 hallucination path 假設
2. **Caption quality audit**：抽 100 筆 Qwen captions（尤其 Metal / Heavy 類別），人耳聽 audio + 量化 audio-caption mismatch rate、inter-slot contradiction rate
3. **Caption cleaning ablation**：把 108 行 non-ASCII filter 掉 + 短 caption (< 10 詞) filter 掉，重訓 Qwen-Rnd-NoQ，看是否從 0.061 提升

排程要不要做哪一個由你決定，這個 doc 不預設方向。

---

## 9. EXP-G 完成（2026-05-15 收尾）+ caption-audio granularity 新發現

### 9.1 EXP-G 結果（LP-MC S1 → Qwen S2，stage-localization test）

設計問題（design doc）：anchor 形成在 S1 是否就夠了？S2 換 Qwen 還能不能保住？

**答：不夠。** 全部 collapse。

| Metric | MC | JM (LP) | JMQ (Qwen) |
|---|---|---|---|
| CLAP | **0.0679** | 0.0584 | 0.0788 |
| PE-AV peav | **−0.034** | +0.011 | +0.086 |
| t2a R@10 | 0.181 | 0.488 | 0.439 |
| AES PQ | 6.70 | 6.67 | 6.67 |

**Steering ratios**: 0.068 / 0.098 / 0.076 / 0.083 → 全 collapse cluster（P9 V1 NoQ 0.025-0.147 範圍）

Pre-cleared paper wording (per design doc): *"The LP-MC anchor formed in Stage 1 does not protect against caption-regime co-adaptation during Stage 2 training on Qwen captions."*

完整結果見 `qwen_collapse_root_cause_2026_05_08.md` EXP-G 段。

### 9.2 對 Section 3.4 / 4 narrative 的影響

- **Section 3.4 mechanism hypothesis 更新**：collapse 不是 S1-only 也不是 S2-only，是 caption distribution × training dynamics 的 interaction。S2 alone 也能讓健康 S1 model 退化。
- **Section 4 paper narrative 強化**：加上「stage-localization 排除」這條路徑——即使 S1 有 healthy LP-MC anchor，S2 200K iter Qwen 仍能讓所有三個 metric 退到 collapse 區。EXP-G 成為 EXP-D4（projection transplant 反而更差）的 stage-wise 對照。

### 9.3 新發現：caption-audio granularity mismatch（今早 follow-up，未實驗驗證）

EXP-G NULL + EXP-A~F NULL 全部排除已 design 的 intervention 後，今早教授討論 + audit 發現 **structural data pipeline bug**：

- `partition_clips.py:81-85` deterministic 抽 30s segment 的**前 10s**（每筆 `clips.tsv` 全 `start_sample=0, end_sample=160000`）
- NPZ 存 312 frames = 9.975s audio latent
- **Caption 描述完整 30s（LP-MC + Qwen 都是）**
- 結構性 granularity mismatch：audio 10s + caption 30s

這個 bug 影響 LP-MC + Qwen 兩條 pipeline（同 NPZ）。LP-MC 為什麼仍 healthy 而 Qwen collapse 在這個 granularity 下未解釋——可能 deeper 的 captioner-style accuracy 差異 + multi-aspect divergence 讓 Qwen 在「only 10s 看得到」regime 下無法收斂。

**未驗證 hypothesis**：30s audio context retrain 是 causal proof，但 architecture 變更（latent_seq_len 312→936）+ 24-48h NPZ regen + 20h train 不在現 paper scope。寫進 paper discussion 當 confound limitation。

### 9.4 EXP series 全 universe 落幕

EXP-A 至 EXP-G 全完成（A/B/C/D/F/G）。**P8 healthy single-cap 是 tested universe 內唯一不 collapse 的 configuration**。剩下 untested 路徑 = granularity（9.3 段）+ captioner 替換 + audio context 重建——全屬 paper scope 外。

可寫的 paper claim（observation 層）：
- ✅「Across stage-localization (EXP-G), data-mixing (EXP-F), caption-cleaning (EXP-A/B/C), and projection-transplant (EXP-D4) interventions, only full LP-MC writing-task supervision throughout both stages yielded healthy text conditioning.」
- ✅「Stage-2 Qwen caption training alone is sufficient to erode established S1 LP-MC anchor (EXP-G CLAP 0.0679, PE-AV −0.034).」
- ❌ 不能寫「Qwen captions inherently incompatible」——只能寫 within this training setup + this captioner-style mismatch context。
