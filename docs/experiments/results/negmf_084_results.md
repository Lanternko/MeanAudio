# 084／085 NegMF 負向蒸餾 Stage A＋B 結果（2026-09-29）

> **Stage B（N100 × 3 seed）已過，見文末〈Stage B〉一節。** 以下 Stage A 各節保留單 seed 原文。

- 設計：`../negprompt_distill_meanflow_084_20260927.md`
- 原始數字：`negmf_084_summary.json`（`scripts/analysis/negmf_084_analysis.py`，085 結尾自動跑，2026-09-29 01:11 本地時間）
- Stage A 只有 seed 14159265。`cells_missing` 裡另外兩個 seed 的格子是 Stage B 尚未排隊的格子，不是缺漏。
- queue 狀態：084 於 2026-09-28 12:14Z `completed`，085 於 17:11Z `completed`。
- 訓練健康：兩個 arm 的 S2 log 都沒有 `loss:nan`，`grad_norm:nan` 為 0。guide log 行都在：N100 是 `for t > 0.0`，Nhi 是 `for t > 0.6667`。

## 一句話

**CFG 訓練目標的 ∅ 分支換成 fidelity8 之後，負向 prompt 的 PQ 效果幾乎整份被收進了權重。**
- CFG0 下，lvl30 PQ 比 control 高 +0.97（N100）與 +0.92（Nhi）。這是門檻 +0.31 的 3 倍，也超過 control 自己在推論期加負向的增益 +0.83（R = 1.18／1.11）。
- 1-NFE 也有 +0.95／+0.89。
- CLAP 過了非劣性界，但沒達到獨立支持的門檻。靜音反而減少。
- **代價是 FAD**：CFG0 從 3.81 升到 6.16（N100）與 6.46（Nhi），比 control 開推論期負向（5.23）還差。
- 依預登錄規則，E1＋E2 都過，可以進 Stage B。可寫層級只到「單 seed 正向訊號」。因為 FAD 變差，**不可寫**「不需要推論期負向 prompt」。

## 每格絕對分數（MusicCaps 5521 首；MF25 除 1-NFE 格外）

| 模型 | 格 | PQ raw | PQ lvl30 | CLAP | CE | PC | LUFS | crest | 靜音 | FAD ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| control | CFG0 | 6.430 | 6.648 | 0.1977 | 6.027 | 5.097 | −19.0 | 6.44 | 52 | **3.81** |
| control | CFG3+neg | 7.258 | 7.476 | 0.2243 | 6.726 | 4.683 | −16.3 | 5.88 | 67 | 5.23 |
| control | 1-NFE CFG0 | 6.210 | 6.442 | 0.1869 | 5.751 | 5.116 | −18.9 | 6.73 | 6 | — |
| **N100** | **CFG0** | **7.460** | **7.622** | 0.2047 | 7.175 | 5.245 | −18.1 | 6.10 | 15 | 6.16 |
| N100 | CFG3+neg | 7.618 | 7.811 | 0.2160 | 7.145 | 4.878 | −16.8 | 6.00 | 19 | 7.12 |
| N100 | 1-NFE CFG0 | 7.220 | 7.389 | 0.1957 | 7.016 | 5.203 | −17.6 | 6.17 | 3 | — |
| Nhi | CFG0 | 7.385 | 7.570 | 0.2055 | 7.068 | 5.153 | −18.9 | 6.52 | 22 | 6.46 |
| Nhi | CFG3+neg | 7.578 | 7.788 | 0.2196 | 7.046 | 4.761 | −18.0 | 6.52 | 34 | 7.82 |
| Nhi | 1-NFE CFG0 | 7.130 | 7.334 | 0.1953 | 6.840 | 5.115 | −18.4 | 6.63 | 3 | — |

- control = `phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265`，兩個 arm 都從它的 S1 `ckpt_last`（it 100000）分支。
- FAD 抽樣 2048，參考目錄 `/mnt/HDD/kojiek/musiccaps_reference`，只有單值、沒有 CI。
- 沒有任何格 peak ≥ 0.999。

## 預登錄端點（arm − control，同 clip 配對，bootstrap 10000）

| 端點 | 定義 | N100 ΔPQ lvl30 [95% CI] | Nhi ΔPQ lvl30 [95% CI] | N100 ΔCLAP raw | Nhi ΔCLAP raw | ΔLUFS（N100／Nhi） | 判定 |
|---|---|---|---|---|---|---|---|
| **E1（主）** | N@CFG0 − C@CFG0 | **+0.974** [+0.954, +0.993] | **+0.922** [+0.902, +0.942] | — | — | +0.93／+0.10 | 兩者都過 Stage A（≥ +0.31 且 CI > 0） |
| **E2** | CLAP，同上 | — | — | +0.0070 [+0.0051, +0.0091] | +0.0078 [+0.0058, +0.0098] | — | 非劣性過（界 −0.008）；未達獨立支持（+0.008） |
| E3 | N@CFG0 − C@CFG3+neg | +0.146 [+0.128, +0.164] | +0.094 [+0.077, +0.111] | −0.0196 | −0.0188 | −1.74／−2.57 | 一次前向的 PQ 勝過兩次前向，CLAP 輸 0.02 |
| E4 | N@CFG3+neg − C@CFG3+neg | +0.335 [+0.320, +0.352] | +0.313 [+0.296, +0.330] | −0.0084 | −0.0047 | −0.47／−1.67 | 疊加仍有 PQ，CLAP 小輸，未飽和 |
| E5 | 1-NFE CFG0，N − C | +0.947 [+0.928, +0.966] | +0.892 [+0.874, +0.911] | +0.0088 | +0.0085 | +1.37／+0.52 | 一步取樣也吃得到 |
| 參照 | arm 自己的推論期負向增益 | +0.190 | +0.218 | +0.0113 | +0.0141 | — | 約為 control（+0.828）的 1/4 |
| 參照 | control 推論期負向增益 G_neg | +0.828 [+0.807, +0.851] | 同左 | +0.0266 | 同左 | +2.67 | — |

- 全部格的 raw 與 lvl30 都同號。E1 的 raw 是 +1.030／+0.955。
- Nhi 幾乎不變大聲（+0.10 LU），但 PQ 只比 N100 少 0.05，所以 PQ 增益不是響度造成的。
- 響度閘門：沒有任何格的靜音超過 control 的 2 倍。arm 的靜音反而更少（CFG0：15／22 對 52），`silence_escape` 為空。

## 讀法

1. **負向效果被收進權重，而且收得比推論期還多。**
   - CFG0 的 arm 比 control 開推論期負向還高 0.15 PQ（E3）。
   - arm 再開推論期負向只多 +0.19，約為 control 的 1/4，表示大部分方向已經在權重裡。
   - 設計 doc 的證偽假說（NASA：老師端負向讓學生變差）在 MeanFlow 目標上**沒有**複現。
2. **Nhi 的區間限制沒有帶來預期的好處。**
   - 075 的分段 CFG 顯示 93–100% 的增益來自 t > 2/3，所以 Nhi 的 PQ 幾乎與 N100 相同（+0.92 vs +0.97），這一點符合預期。
   - 但 guidance interval 預測的「FAD 代價較小」不成立：Nhi CFG0 的 FAD 是 6.46，N100 是 6.16。
   - Nhi 唯一明確的優點是響度幾乎不動（+0.10 LU vs +0.93 LU）。
3. **FAD 是這條線的主要負帳。**
   - CFG0 的 FAD 從 3.81 升到 6.16，+2.35；control 開推論期負向才 +1.42。
   - 所以 PQ 的贏面伴隨著生成分布離 MusicCaps 參考更遠，而且比推論期負向更遠。
   - 這與 `project_negprompt_hurts_fad` 同向、幅度更大。
   - E3 的「一次前向勝兩次前向」只在 PQ 上成立；CLAP −0.02、FAD 更差。
4. **CLAP 在 CFG0 下小幅上升（+0.007）**，但未過獨立支持門檻。在 CFG3+neg 下 arm 輸 control（−0.008／−0.005）。
5. **與 081 的粗略對照**（不同 arm、seed 數不同，只作參考）：
   - 081 arm 的 HQ 前綴＋CFG0 是 lvl30 PQ 7.37（3 seed），084 N100 CFG0 是 7.62（1 seed）。
   - 兩者都是「CFG0 拿到 PQ」的介入，但 081 需要重訓 S1＋S2 並改 caption，084 只重訓 S2。
   - 081 沒有 FAD，無法比代價。

## 限制

1. ~~單一訓練 seed~~（Stage B 已補到 3 seed，見文末；Nhi 仍是單 seed）。訓練 seed 的 PQ 底線約 0.155，E1 是它的 6 倍，但 FAD 沒有 seed 底線可比。
2. **fidelity8 是依 AES PQ 選出來的**，E1 與它同一把尺，CLAP 是唯一獨立讀數。尚未試聽。
3. FAD 只有單值、抽樣 2048，無 CI。
4. arm 與 control 的差異只有 S2 的 CFG 目標，S1 完全相同，所以沒有額外的訓練預算混淆。

## 下一步（Stage A 當時）

- ~~Stage B~~：已跑完（087／088），見下。
- 五首固定 prompt 試聽：盲聽包已做好（`deliverables/negmf_084_listening_20260929/`，`scripts/eval/negmf_084_listening_pack.py`），**尚未試聽**。
- Stage C（reversed 文字放 guidance 分支）照設計要等 Stage B 過了才排。

## Checkpoint

- `~/exps_nvme/phase8_qwen_caption2p0_slot0clean_negmf{n100,nhi}_noq_quarter_s14159265_stage2_50000/`（S2 ema_final）。
- control 三個 seed 的 S1 `ckpt_last`：Stage B 已收，但 Stage C 若要排仍需要，先別刪。

## Stage B（2026-09-29，N100 × seed 16180339／27182818）

- queue：087（s16180339）、088（s27182818）皆 `completed`；兩者 S2 log 無 `loss:nan`，guide 行 `for t > 0.0` 都在。
- 分析重跑 `scripts/analysis/negmf_084_analysis.py`：`pass_stageA` 與 `pass_stageB` 皆為 true，loudness gate 通過，`silence_escape` 為空。

### 端點（arm − control，lvl30 PQ，三 seed 合併，bootstrap CI）

| 端點 | 合併 | CI | s14159265 | s16180339 | s27182818 |
|---|---:|---|---:|---:|---:|
| **E1** CFG0 arm vs ctrl | **+0.948** | [+0.931, +0.964] | +0.974 | +0.942 | +0.928 |
| E5 1-NFE arm vs ctrl | +0.955 | [+0.939, +0.970] | +0.947 | +0.996 | +0.922 |
| E3 arm CFG0 vs ctrl CFG3+neg | +0.111 | [+0.098, +0.124] | +0.146 | +0.088 | +0.099 |
| E4 CFG3+neg arm vs ctrl | +0.331 | [+0.318, +0.343] | +0.335 | +0.279 | +0.377 |
| 參考：ctrl 推論期負向增益 | +0.837 | [+0.818, +0.855] | +0.828 | +0.854 | +0.828 |
| 參考：arm 推論期負向增益 | +0.219 | [+0.209, +0.230] | +0.190 | +0.191 | +0.277 |

- E1 三 seed 都 > 0，合併 +0.948，是門檻 +0.31 的 3 倍；R = 1.13（超過 control 自己推論期負向的增益）。
- E2 CLAP（raw）+0.0090 [+0.0075, +0.0107]：非劣性過，這次 CI 下界也 > 0（`E2_clap_independent_support` true）。
- E3：N100 CFG0 在 PQ 上還比 control CFG3+neg 高 +0.11，但 CLAP 低 −0.021。
- arm 上再加推論期負向，只多 +0.22 PQ，表示增益大部分已經收進權重。

### FAD（2048 抽樣，單值）

| | s14159265 | s16180339 | s27182818 |
|---|---:|---:|---:|
| ctrl CFG0 | 3.81 | 3.93 | 3.82 |
| ctrl CFG3+neg | 5.23 | 6.36 | 6.68 |
| **N100 CFG0** | **6.16** | **6.71** | **6.90** |
| N100 CFG3+neg | 7.12 | 8.27 | 8.60 |

- N100 CFG0 的 FAD 三 seed 都比 ctrl CFG0 高 2.3～3.1，也都比同 seed 的 ctrl CFG3+neg 差（差距 +0.2～+0.9）。**FAD 代價跨 seed 穩健**，不是 seed 14159265 的個案。

### 響度與靜音（CFG0）

- LUFS：arm −18.1／−17.9／−19.1，ctrl −19.0／−19.3／−18.8，差 +0.69 LU（per-seed +0.93／+1.46／−0.32）。
- 靜音 clip：arm 15／10／15，ctrl 52／39／39，**三 seed 都比 control 少**，沒有靜音逃逸。
- crest：arm 6.10／6.04／6.73，ctrl 6.44／6.59／6.58。

### 可寫層級

- **3 seed 成立**：S2 CFG 訓練目標的 ∅ 分支換成 fidelity8，可以在 CFG0 與 1-NFE 拿到比推論期負向還大的 AES PQ 增益，且 CLAP 不降、靜音變少。
- **同樣 3 seed 成立**：FAD 變差，而且比推論期負向還差。
- 仍然**不可寫**「不需要推論期負向 prompt」或「品質提升」：PQ 與 fidelity8 同一把尺，FAD 反向，試聽還沒做。
- Nhi 沒複製，只能寫單 seed。

### 下一步

- 試聽五首盲聽包，看 PQ 與 FAD 在耳朵上站哪一邊。
- **Stage C 已排（2026-09-29）**：p2 092／093／094 = rev100 × 3 seed（reversed 文字放 guidance 分支），control 多一格 `cfg3_revneg`。判讀規則預先登記在設計 doc §6.1；因為 reversed 與 fidelity8 的 T5 cos 0.814，結論最多只能寫到「fidelity 領域文字」，不能寫「任何非 null 文字」。
