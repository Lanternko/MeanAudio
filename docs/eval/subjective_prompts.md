# 主觀評估固定 prompt 與下載流程

25 steps + cfg_strength 3.5，音質比單步好。

## 推理指令

```bash
python infer.py \
    --variant meanaudio_s \
    --model_path exps/EXPNAME/EXPNAME_ema_final.pth \
    --output eval_output/EXPNAME_subjective \
    --encoder_name t5_clap --text_c_dim 512 \
    --use_meanflow --num_steps 25 --cfg_strength 3.5 \
    --quality_level 9 --full_precision \
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
