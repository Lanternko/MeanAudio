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
