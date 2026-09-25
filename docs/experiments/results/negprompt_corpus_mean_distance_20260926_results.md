# 負向 prompt 與「語料平均 caption」的距離 vs PQ 增益（2026-09-26）

**問題**（Task A 延伸）：假說是負向分支 B ≈「訓練語料的平均樣貌」，CFG 把生成推離這個平均。若成立，負向 prompt 越像訓練 caption 的平均，ΔPQ 應該越大。

**資料**：09-03 negprompt 消融中 17 個不重複的 cfg3 負向 prompt，皆為同一 checkpoint（c2p0_slot0 = `phase8_qwen_caption10s_multisent_noq_full_stage2_200000`）、同一 MusicCaps subset1024。ΔPQ = PQ − none（6.579）。POS 槽的格排除。
**語料**：該 checkpoint 的訓練 TSV（`phase8_qwen_caption10s_multisent_train.tsv`）隨機 5,000 句（seed 20260926）。
**腳本**：`scripts/analysis/negprompt_corpus_mean_distance.py`
**輸出**：`~/nvme_experiment_artifacts/meanaudio/negprompt_corpus_mean_distance/negprompt_corpus_mean_distance_n5000.json`
**執行**：直接跑約 2 分鐘，掛 notify trap；時間太短，不值得走 queue。

## 預測子（負向 prompt 對某個平均的 cosine）

- `clap`（主）：訓練語料 L2 正規化 CLAP 文字嵌入的平均
- `t5_pool`：訓練語料 flan-t5-large masked mean-pool 的平均
- `t5_seq`：訓練語料 77×1024 padded T5 序列（NoMask 模型實際看到的輸入）攤平後的平均
- `clap_eval`：1024 句 MusicCaps eval caption 的 CLAP 平均

## 結果（Spearman ρ vs ΔPQ，n=17，prompt bootstrap 95% CI，置換 p）

| 預測子 | ρ | 95% CI | p | 控制 token 長度後的偏相關 |
|---|---|---|---|---|
| clap（主） | −0.22 | [−0.62, +0.27] | 0.38 | −0.07 |
| t5_pool | +0.45 | [−0.08, +0.84] | 0.07 | +0.06 |
| t5_seq | +0.48 | [−0.05, +0.82] | 0.05 | +0.16 |
| clap_eval | **−0.64** | [−0.86, −0.21] | 0.007 | **−0.48** |
| （參考）T5 token 長度 | +0.50 | — | 0.04 | — |

拿掉 fidelity8 家族的 2 個長 prompt 後（n=15），clap／t5_pool／t5_seq 的 ρ 分別為 −0.17／+0.23／+0.24，都不顯著。

## 讀法

1. **「越像語料平均，增益越大」不成立**。主預測子 CLAP 的方向相反，且不顯著。
2. **T5 的正相關其實是 prompt 長度**。T5 相似度和 token 長度的 ρ 是 0.79–0.86，控制長度後偏相關只剩 +0.06／+0.16。長 prompt 的 padded 序列比較像長 caption，這反映的是字數，不是語義。而長度本身和 ΔPQ 有 ρ +0.50，這和 negprompt 消融中「長版比短版好」一致。
3. **唯一站得住的訊號是反向的**：負向 prompt 越像 **eval caption（音樂內容描述）**，增益越小。控制長度後仍有 −0.48。像 "lo-fi"、"amateur"、"music"、"genre" 這種和音樂內容共享 CLAP 空間的字，放進負向槽會推離一部分內容本身，增益最小（+0.04～+0.27）。增益大的都是跟音樂內容正交的保真度片語。這和 N1（增益隨「談不談保真度」單調上升）以及槽位不對稱都相容，但本分析沒有獨立檢定因果。
4. 限制：n=17 且 prompt 並非隨機抽樣（是消融設計挑的），CI 很寬；只有一個 checkpoint、一個 cfg。
