# ATTM-protocol benchmark（2026-09-04）

對標 ICME 2026 ATTM Grand Challenge（arXiv 2605.21538）。腳本在 `scripts/attm/`，
產出在 `/home/kojiek/nvme_experiment_artifacts/meanaudio/attm/`。

> **這不是 ATTM 排行榜數字。** 他們的 100 條測試 prompt 與 1,000 首隱藏 FAD 參考集
> 從未公開，CCS judge（Qwen3-Omni-30B-A3B，需 ~79 GB VRAM）在本機跑不動。任何對外
> 敘述一律寫 "ATTM-protocol reproduction"，不得把下表任何一列放進他們的 Table II 排名。

## 協定

- **Prompt / 評估集**：器樂 MusicCaps 2,535 條（5,521 條中 45.9%，用既有 `VOCAL_RE` 過濾）
- **FAD 參考**：2,382 條器樂 MusicCaps 參考音檔；另有 disjoint 版本（hash 分半，prompt 取 A 半、參考取 B 半）
- **CLAP**：兩把尺同時算 —— ATTM 的 `music_audioset_epoch_15_esc_90.14` 與我們歷史用的 `music_speech_audioset_epoch_15_esc_89.98`，batch 32
- **FAD**：在 ATTM 的 CLAP embedding 空間算（他們用同一個 checkpoint 當 feature extractor，不是 VGGish）
- **CCS**：照 Eq. 1–2，judge 換成 Qwen2.5-Omni-3B thinker，概念取自 MusicCaps human `aspect_list`

## 主表

| arm | CLAP-ATTM | CLAP-ours | 比值 | FAD-all ↓ | FAD-disj ↓ | CCS raw | CCS 校正 | CE | PQ |
|---|---|---|---|---|---|---|---|---|---|
| ours c2p0_slot0 **cfg3+neg** | **0.3112** | 0.2806 | 1.109 | 0.2451 | 0.2523 | 0.8833 | **0.836** | 6.924 | **7.436** |
| ours c2p0_slot0 **cfg0** | 0.2801 | 0.2389 | 1.172 | **0.1993** | **0.2082** | 0.8841 | 0.830 | 5.943 | 6.487 |
| MeanAudio-S-Full topline | 0.1304 | 0.0951 | 1.371 | 0.4504 | 0.4527 | 0.5630 | 0.224 | 2.998 | 4.990 |
| MeanAudio-L-Full topline | 0.1149 | 0.0860 | 1.337 | 0.4887 | 0.5019 | 0.4927 | 0.220 | 2.882 | 4.999 |

「CCS 校正」= judgement-weighted chance-corrected，見下節。

## 五個發現

### 1. harness 通過獨立重現檢驗
完全重新生成音檔（舊的已刪），與 2026-08 `negprompt_reeval_cfg3.0` 的 novocal 紀錄比：
CLAP 0.2806 vs 0.2811、CE 6.924 vs 6.937、CU 7.488 vs 7.492、PC 4.756 vs 4.759、PQ 7.436 vs 7.443。
五項全部三位小數內吻合。

### 2. 重現了 ATTM 的 S > L 反直覺排序
他們：S CLAP 0.210 / FAD 0.649，L CLAP 0.202 / FAD 0.660。
我們（不同 prompt 集、不同參考集）：S CLAP 0.1304 > L 0.1149，S FAD 0.4504 < L 0.4887。同方向。
這是在無法重現絕對值的情況下，唯一能拿到的 harness 正確性證據。

### 3. CLAP 換算比值不是常數 —— 歷史表格不能重算
四個 arm 的 ATTM/ours 比值：1.109、1.172、1.337、1.371。跨度 24%，遠大於 arm 間真實差異。
ATTM 的純音樂 CLAP 對 general-audio topline 的加成遠大於對 Jamendo-trained 模型。
**每個 arm 都要實測，沒有單一係數。**

### 4. FAD 的 per-clip 重疊 confound 不是主因（推翻先前假設）
先前擔心「參考集包含 prompt 來源錄音」會灌水。disjoint 測試結果：
delta 只有 +0.0023 ~ +0.0132，四個 arm 同向平移，**排序完全不變**。
→ 我們 FAD 遠低於他們表上任何數字（0.21 vs 最佳 0.417），**不是**這個 confound 造成的。
剩下未測的解釋是 **domain match**（我們是 MusicCaps 參考 + MusicCaps caption；
他們是 Jamendo 參考 + 合成 tag prompt），那是分布層級而非 clip 層級的優勢，仍然使得
跨表比較無效。

### 5. CCS 的 yes-bias：ATTM 的 criterion 2 有一半缺口
他們只量 recall（該有的偵測得到嗎），不量 specificity（不該有的會不會也說有）。
我們補跑負向對照（每 tag 最多 40 個真負例，含同義詞護欄）：

| 類別 | n tags | mean Youden J | mean FPR |
|---|---|---|---|
| genre | 6 | 0.681 | 0.221 |
| instrument | 1 | 0.601 | 0.250 |
| **mood** | 17 | **0.413** | **0.519** |

`hypnotic` recall 1.000 / FPR 0.800、`fun` 1.000 / 0.675、`calming` 1.000 / 0.650 —— 
**完美 recall 的 tag 全部是高 FPR**，只量 recall 的篩選會全數放行。

chance correction `(rate − fpr)/(1 − fpr)` 後：

| arm | raw | 校正後 |
|---|---|---|
| ours cfg3+neg | 0.883 | 0.836 |
| ours cfg0 | 0.884 | 0.830 |
| MeanAudio-S-Full | 0.563 | **0.224** |
| MeanAudio-L-Full | 0.493 | **0.220** |

**topline 的 CCS 幾乎全是 yes-bias**，校正後崩到 0.22。原始 CCS 把差距說成 1.6×，
校正後是 3.7×。兩種加權（per-tag macro / per-judgement micro）都同向，結論穩健。

### 6. negative prompt 對 CCS 無可測效果
raw：0.8833 vs 0.8841（judgement-weighted delta −0.0020）。
per-tag 未加權 delta +0.0110 曾看似方向相反，逐 tag 檢查後確認是加權假影 ——
單 tag n 只有 18~141，`epic` −0.19 (n=26)、`cheerful` +0.18 (n=22) 都在二項式雜訊內。
**結論：無效果。**

## 對打 ATTM 的策略修正

ATTM 用 Borda count 等權合併 FAD / CLAP / CCS。negprompt 的完整帳目：

| 指標 | 效果 |
|---|---|
| CLAP | **+0.0311** 好 |
| FAD | **+0.0458** 差（FAD 越低越好） |
| CCS | 無效果 |

→ **在 ATTM 計分法下淨效果約為零甚至為負。** 原本把 negprompt 列為對打 ATTM 第一優先
的建議不成立。機制：negprompt 把音檔推向乾淨高保真，但參考分布是 YouTube 錄音
（器樂子集 aspect 統計：`low quality` 621 次、`noisy` 357、`amateur recording` 271），
變乾淨就是離參考分布更遠。CLAP 與 PQ 獎勵、FAD 懲罰。

**negprompt 本身的價值不變** —— 在我們自己的 CLAP + AES 體系裡仍是 +0.046 CLAP /
+0.95 PQ 的最佳免費槓桿。變的只是「拿它打 ATTM 榜」這個特定用途。

## 未解 / 下一步

- **instrument 軸不可用**：校準後只剩 `electronic drums` 一個 tag（n=47）。3B judge 對
  e-bass 3.3%、keyboard 4.8%、bass 10% recall —— 這是模型能力天花板，調 prompt 修不好。
  要有可用的 instrument CCS 必須換 judge。
- **domain-match confound 未測**：需要拿 held-out 器樂 Jamendo 當 FAD 參考重跑一次，
  才能知道我們的低 FAD 有多少來自 domain match。
- **從未做過 MOS**。ATTM 唯一的 MeanAudio-based 投稿客觀 rank 6 但 MOS 2.006 全場最低，
  這個風險我們完全沒量過。
