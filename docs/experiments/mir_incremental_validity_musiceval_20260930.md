# MIR 指標對 AES 的增量效度（MusicEval，2026-09-30）

## 動機

- 真實 MusicCaps 全量 AES PQ 6.90，但 081 HQ+LQ 在 87% 的 prompt 上 PQ 勝過同一首真實錄音，而試聽明顯不如真實錄音（`results/musiccaps_reference_real_metrics_20260930_results.md`）→ AES PQ 已經被我們的手法推過頭。
- PAM 上 AES 在生成音樂只剩 PQ r 0.55，人類差 < 0.5 的配對等於擲硬幣。
- 群組討論提出：拍子稍微不穩人類會本能察覺，傳統 MIR 工具（beat tracking 等）可能補到 AES 看不到的東西。

這份要問的是：**MIR 特徵在 AES 之外，能不能多解釋人類對生成音樂的品質判斷？** 不能的話，MIR 指標就不值得加進 eval。

## 資料

MusicEval（ICASSP 2025；AudioMOS 2025 Track 1；HF `BAAI/MusicEval`，CC-BY-NC 4.0），放在 `/mnt/seagate/kojiek_data/musiceval/`，repo 內 symlink `research/eval/musiceval`（gitignored）。

- 2,748 首、31 個 TTM 系統（Suno、Udio、MusicGen、AudioLDM、Tango、MAGNeT、Riffusion…），全部 16 kHz 單聲道 → 沒有 PAM 那種頻寬差異
- 每首由 5 位音樂專家（14 人池）評兩項（1–5）：OVL 整體音樂印象、REL 文字符合度；有逐評分者分數
- **主分析集**：25 系統 × 100 條共同 prompt 完整交叉 = 2,500 首；每條 prompt 300 個系統配對
- 另 248 首 demo 片段（6 系統、各自 prompt）只進逐首相關，不進配對
- 片段長 5–87 s（中位 25 s）

## 特徵（先定死，看結果前不改）

AES 基準：PQ、PC、CE、CU（`eval_metrics.score_aes`，10 s 窗加權平均）。干擾共變量：log 長度、LUFS。

MIR 特徵（12 個）：

| 名稱 | 工具 | 定義 | 預期方向 |
|---|---|---|---|
| beat_act | madmom RNNBeat | 偵測拍點上的 activation 平均（拍點顯著度） | + |
| pulse_clarity | madmom RNNBeat | activation 自相關在 40–220 BPM 延遲的最大值（歸一化） | + |
| ibi_cv | madmom DBN | 拍間距變異係數（拍穩定度） | − |
| tempo_drift | madmom DBN | 前後半段中位拍間距的 \|log 比\| | − |
| downbeat_act | madmom RNNDownBeat | 偵測小節首拍上的 downbeat activation 平均 | + |
| rhythm_conf | essentia RhythmExtractor2013 | multifeature 信心值 | + |
| danceability | essentia Danceability | DFA 式 danceability | + |
| key_strength | essentia KeyExtractor（edma） | 全段調性強度 | + |
| key_cnn_conf | madmom CNNKey | 24 類調性機率最大值 | + |
| key_stability | librosa chroma_cqt | 5 s 窗（hop 2.5 s）Krumhansl 調性與全段調性相同的比例 | + |
| chroma_entropy | librosa chroma_cqt | 逐幀正規化 chroma 熵的平均 | − |
| dissonance | essentia Dissonance | 逐幀感知不協和度平均 | − |

（共 12 個；拍點少於 4 個的片段 ibi_cv／tempo_drift 設缺值，用訓練折中位數補值加缺值指示變數）

## 分析

目標：OVL（5 人平均）；REL 只當次要。

1. **人類上限**：每首 5 位評分者隨機拆 2 vs 3，200 次，Spearman-Brown 到 5 人；配對勝負的人類兩半一致率。
2. **逐特徵**：對 OVL 的 r、控制 AES 四軸後的偏相關、系統內去平均後的偏相關（bootstrap 對 prompt 重抽的 CI）。
3. **增量效度（主端點）**：ridge（內層 RidgeCV 選 α），比較
   - B = AES4＋干擾共變量
   - F = B＋MIR 12 個
   - M = 只有 MIR＋干擾共變量（參考）
   兩種 CV：
   - **依 prompt 分組 10 折**（未見過的 prompt）
   - **留一系統**（25 折，未見過的系統 → 最像「評一個新模型」）
   讀數：out-of-fold R²、r；系統層級 Spearman（OOF 預測的系統平均 vs 人類系統平均）；**同 prompt 配對勝負一致率**（依人類差 \|Δ\| ≤ 0.4／0.6–0.8／≥ 1.0 分組）。
   CI：對 prompt（或系統）bootstrap 2,000 次，F−B 的差值取配對 CI。
   敏感度：HistGradientBoosting 取代 ridge（MIR 效果可能非線性，例如拍點只對節奏性曲風重要）。
4. **外部檢查**：同樣特徵算在 PAM 400 首生成音訊（5 s，拍點特徵會弱）與 AES-natural 522 首真實錄音，看偏相關的方向是否一致。
5. **探索**：MIR 特徵在真實 MusicCaps vs 081／084／control 各 arm 上的分布——AES 說 081 > 真實錄音，MIR 怎麼說。只描述，不當證據。

## 判讀門檻（預登錄）

- **「MIR 有增量效度」**需同時成立：
  1. 依 prompt 分組 CV：ΔR²(F−B) ≥ 0.02 且 95% CI 下界 > 0
  2. 同 prompt 配對一致率 Δ 的 CI 下界 > 0
- **「可用於比較模型」**還需要留一系統 CV 的 ΔR² CI 下界 > 0，且系統層級 Spearman 不降。
- 只有 1 成立 → 報「MIR 在已知系統內有訊號、但不能泛化到新系統」。
- 都不成立 → MIR 特徵不進標準 eval；「拍子人會察覺」這件事在這份資料上沒有 AES 以外的可測貢獻。

## 為什麼直接跑、不走 queue

CPU 特徵抽取（20 核平行）＋AES 約 3,300 首的 GPU 評分（估 < 15 分鐘），屬短 probe；腳本掛 notify trap。

## 腳本

- `research/eval/mir_features_madmom.py`（`~/venvs/madmom`，py3.9）
- `research/eval/mir_features_essentia.py`（`~/venvs/mir`：essentia＋librosa）
- `research/eval/musiceval_mir_incremental.py`（`~/venvs/dac`：AES 評分＋分析）
- 輸出：`research/eval/output/musiceval_mir/`

## 實作時的補充（看真結果前寫定）

- **主分析集只用 shared 2,500 首**：CV、配對、偏相關都在 shared set；248 首 demo 只進「全部 2,748 首」的逐特徵 r。
- **配對門檻用「全部非平手配對」**：門檻 2 的「配對一致率 Δ」指同 prompt 所有人類分數不相等的配對（`all`）；\|Δ\| 三個分組只描述。
- **一首超長片段**：S013_P013 長 349 s（次長 87 s），AES 單次 forward OOM（GPU 另有 13 GB 他人 job），改評前 90 s。它是 demo 片段，不進主分析。
- **探索 arm 改用 nmv2pair s14159265 的 CFG0／CFG3+neg**：081／084 的音檔已刪，重生要 GPU 時間；nmv2 同樣是 CFG3+neg AES 超過真實錄音的情形，足以回答「MIR 怎麼看」。1,000 條 prompt 隨機抽（seed 0）。
- 外部集的 PAM 5 s 片段上 `rhythm_conf`、`key_stability` 幾乎沒有變異（< 5 個相異值）→ 自動略過。
