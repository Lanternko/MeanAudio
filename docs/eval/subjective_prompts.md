# 主觀評估固定 prompt 與下載流程

25 steps + **cfg_strength 0.5**（pure conditional 分支，無 guidance 放大）。

> ⚠️ **歷史踩坑（2026-04-21 修正）**：原本這裡寫 `cfg 3.5`，結果 subjective_ab v3 全 48 檔用 cfg=3.5 生成，導致 P7V1 + 非 null Q + 高能量 prompt（如 EDM/Trap/Dubstep）有 2/24 嚴重波形飽和（crest < 2.0）。Ablation mc18_abl_A–J 證實：cfg ≥ 2.0 會觸發 guidance 過度放大分支，cfg < 1.0 回傳 pure conditional 即正常（crest 3.49）。與 CLAUDE.md eval 段 `--cfg_strength 0.5` 對齊。York135 2026-04-21 指出此矛盾。

## ⚠️ q 旗標選擇（同 `memory/reference_eval_q_flag_rule.md`）

`infer.py` **沒有 `--no_q` flag**（只有 `eval.py` 有）。NoQ 模型要用 **`--quality_level 10`**（null token 的 workaround）：

| 訓練時 q conditioning | Phase 例 | `infer.py` 旗標 |
|---|---|---|
| **true**（q 0~9 訓練） | 6 V2、7 V1/V2、9 V2、9.5 V2 | `--quality_level 9` |
| **false**（永遠 null token） | 8、9 V1、9.5 V1 | **`--quality_level 10`** |
| **pre-Phase 6**（無 q_embed 層） | 4 V2、5 V1/V2 | 兩者等價 |

**踩坑紀錄**：2026-04-21 對 P9 V1 誤用 `--quality_level 9` → 所有 5 首 pairwise cos-sim 1.0000 + 音量飽和 clipping（未訓練 q=9 embedding 的 artifact）。歷史腳本 `run_subjective_jamendo.sh` 已正確處理（Phase 8 用 q=10），但 `subjective_prompts.md` 範例原本只寫 q=9 誤導。

## 推理指令

```bash
python infer.py \
    --variant meanaudio_s \
    --model_path exps/EXPNAME/EXPNAME_ema_final.pth \
    --output eval_output/EXPNAME_subjective \
    --encoder_name t5_clap --text_c_dim 512 \
    --use_meanflow --num_steps 25 --cfg_strength 0.5 \
    --quality_level {9|10} --full_precision \
    --prompt "PROMPT"
```

`--output_name` 命名規則：`<phase>_<style>`。

## 五首固定 prompt

### 鋼琴
```
This is a piano cover of a glam metal music piece. The piece is being played gently on a keyboard with a grand piano sound. There is a calming, relaxing atmosphere in this piece.
```

### 重金屬
```
This is the recording of a heavy metal music piece. There is a male vocalist singing melodically in the lead. The main tune is being played by the distorted electric guitar while the bass guitar is playing in the background. The rhythmic background consists of a simple acoustic drum beat. The atmosphere is aggressive.
```

### Lo-Fi 民謠
```
The low quality recording features a live performance of a folk song that consists of an arpeggiated electric guitar melody played over groovy bass, punchy snare and shimmering cymbals. It sounds energetic and the recording is noisy and in mono.
```

### EDM
```
This is an electronic dance music piece. There is a synth lead playing the main melody. The beat consists of a kick drum, clap, hi-hat and synthesized bass. The atmosphere is energetic and euphoric.
```

### Cinematic
```
This is a cinematic orchestral piece. There are strings playing a sweeping melody with brass accents. The piece builds in intensity with a dramatic crescendo. The atmosphere is epic and emotional.
```

## 下載到本機（Mac terminal）

```bash
scp -P 22 -r kojiek@140.122.184.29:~/MeanAudio/eval_output/EXPNAME_subjective/ ~/Downloads/EXPNAME_subjective/
```

---

## 補充：MusicCaps-sampled subjective A/B（v3，n=30 paired, cfg=0.5）

除了上面 5 首固定 prompt 的聽測外，2026-04-21/22 另起一條 MusicCaps 隨機抽樣的 paired A/B，用於「對 P7V1 / P8V1 做小 n 但 paired 的 CLAP + AES 客觀量化」。

**方法**：
- Prompt pool：MusicCaps test TSV（5,521 rows），`random.Random(42).sample(pool, 24)` → 後追加 `random.Random(43).sample(remainder, 6)` → 共 30 prompts (mc01–mc30)
- 每 prompt × {P7 V1 Q=9, P8 V1 NoQ (q=10 workaround)} = 60 wav，固定 infer_seed=42, num_steps=25, **cfg=0.5**
- Peak-normalize 到 −1 dBFS 後再計算 CLAP / AES

**相關腳本**（uncommitted，位於 repo root）：
- `sample_musiccaps_v3.py` + `sample_musiccaps_v3_extend.py`：抽 prompt，寫 `sampled_prompts.tsv`
- `run_subjective_ab_v3.sh` + `run_subjective_ab_v3_extend.sh`：跑 infer.py 產生音檔
- `normalize_ab_v3.py`：peak-normalize 到 −1 dBFS
- `write_metadata_v3.py`：寫 `metadata.json`
- `aes_subjective_v4.py` + `clap_subjective_v4.py`：計算客觀分（⚠️ 檔名寫 v4，實際讀取 `subjective_ab_v3/audio/` 並將結果寫入 `subjective_ab_v4/`；v4 只是結果目錄命名，不是第四版實驗）

**Artifacts**：
- 音檔：`eval_output/subjective_ab_v3/audio/`（60 wav，cfg=0.5 修正後）
- 崩壞版：`eval_output/subjective_ab_v3/audio_cfg35_broken/`（cfg=3.5 歷史踩坑證據）
- 客觀分：`eval_output/subjective_ab_v4/{clap_scores.json, aes_scores.json}`

**結果摘要（n=30 paired）**：

| 指標 | P7 V1 (Q=9) | P8 V1 (NoQ) | Δ (P7−P8) |
|------|-------------|-------------|-----------|
| CLAP mean | **0.2285** | 0.2072 | **+0.0213** |
| AES CE | **6.473** | 6.294 | +0.179 |
| AES CU | 7.291 | **7.440** | −0.149 |
| AES PC | 4.603 | **4.774** | −0.171 |
| AES PQ | 7.001 | **7.104** | −0.103 |

- **Paired CLAP**：P7 wins 20/30、P8 wins 10/30（biggest P7 wins: mc12/mc13/mc18；biggest P8 wins: mc07/mc19/mc27）
- **方向一致**：CLAP 上 P7V1 領先 +0.0213（vs n=5,521 MusicCaps benchmark +0.0124），AES 上兩模型接近、P8 略高 CU/PC/PQ、P7 略高 CE
- 無波形飽和（最低 PQ 4.29 > cfg=3.5 廢棄版本的 saturation 標準）
