# 075 預註冊：D2 雜訊負樣本訓練（defect negative samples）

2026-09-23 設計並啟動。方向來源：`docs/meetings/2026-09-22_new_directions_cfg_encdec_noise.md` D 線
（「negative sample：給很多雜訊音檔，然後 caption 叫做雜訊」）。
**已收線（2026-09-23）** → `results/d2_defect_negsample_075_results.md`：E2 反向不過（lab−unlab −0.365 PQ）、E1 只有雜訊族、E3 lab 過。

## 為什麼是這一條、為什麼是現在

- negprompt 消融定論（031）：增益來自 fidelity **領域詞彙**，不是**缺陷極性**（reversed 複製 51%）。
- D1（2026-09-23 收線）給了機制面的補強：**模型沒有文字可達的缺陷方向** —— 缺陷 prompt 只壓出
  籠統劣化、波形簽名對不上（真削波 crest −3.7、prompt 只 −0.19），沒音樂時模型以靜音逃逸。
- 一個從未被檢定的解釋：**訓練語料裡沒有「壞音訊 ＋ 被標成壞」的監督訊號**。D2 直接補這個缺口。
- 推論期那側（A 線 prenorm、073 ADG/APG）已跑完，D0/B 預檢是零 GPU。D2 是優先序表上下一個要 GPU 的項目。

## 設計

**兩個處理臂 ＋ 既有對照**，全部 quarter recipe（S1 100k ＋ S2 50k、NoQ、NoMask、LR 1e-4、BS 8、
`cap_index_fixed=0`、`require_text_overlay=true`），**訓練 seed 14159265**：

| 臂 | 乾淨列（251,596） | 額外列（25,000，≈9.9%） | 額外列的 caption |
|---|---|---|---|
| **control066**（既有） | slot0clean_nmv2matched | — | — |
| **defectunlab** | 同上，逐位元相同 | 劣化音訊 | 原 caption（**不點名**缺陷） |
| **defectlab** | 同上，逐位元相同 | **同一批**劣化音訊 | `"<缺陷句>" + " " + 原 caption` |

- control066 = `phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265`（066–071 那組 3-seed 對照的其中一顆）。
- **主對比是 lab vs unlab**：兩臂的音訊、列數、id 順序、曝光量完全相同，只差額外列的 caption。
  它把「學到缺陷的**名字**」和「只是看過壞音訊」分開。vs control 是次要對比（多了 25k 列 →
  乾淨列曝光降 ~9%，兩個處理臂同樣受影響）。

### 劣化（`scripts/preprocess/build_defect_negsample_arm_inputs.py`）

來源：從乾淨列隨機抽（seed 20260923），**排除 `segment_0`**（曲頭淡入靜音 4.5%）；每個來源 clip 用一次。
音訊路徑逐步複製 `training/extract_audio_latents.py`：整個 30 s 檔 peak-normalize 到 0.95 → 取前 160,000 樣本 →
`mel_converter('16k')` → v1-16 VAE → mean/std (312, 20)。五類各 5,000 列，嚴重度隨機：

| 類 | 參數 | 缺陷句範例（每類 4 句輪替） |
|---|---|---|
| noise | 白雜訊 SNR U[0, 15] dB | “A noisy recording buried in hiss and static.” |
| clip | 驅動 U[9, 21] dB 後硬削波在原 peak | “Overdriven audio with crunchy digital clipping.” |
| lowpass | 8 階 Butterworth，截止 U[1, 3] kHz | “A muffled recording that lacks high frequencies.” |
| bitcrush | 4–6 bit 量化 ＋ sample-and-hold ÷2–4 | “Grainy low-bit digital audio with poor fidelity.” |
| crackle | 雙極脈衝 20–80 /s、0.3–0.9×peak | “A recording full of clicks and crackles.” |

- **每片劣化後重新對回乾淨窗口的 integrated LUFS**（peak 保護 0.999）→ 教的是音色方向不是響度（063/065）。
- **缺陷句放在 caption 前面**：T5 77-token 窗口會截尾（`reference_caption_corpus_t5_truncation`），
  放後面標籤會被切掉。缺陷句 < 40 tokens 由 builder assert。
- **缺陷句刻意不用 D1 probe 的字串**（D1 的字串留作 held-out 操弄檢查）。與 fidelity8 負向 prompt
  的詞彙**有自然重疊**（noisy、distorted、muffled、hiss、lo-fi、poor fidelity、low quality）——事前寫明：
  若 lab 臂的 negprompt 增益變大，一部分可能是「負向槽裡的字終於有了對應的訓練方向」，這正是要測的東西，
  但不能寫成「任何負向 prompt 都變好」。

### 已過的建構閘門（smoke，n=40）

| 閘門 | 結果 |
|---|---|
| 重編碼路徑 vs 官方 NPZ（32 個乾淨窗口） | median corr 0.99999999、min 0.9999991 ✅（門檻 median ≥ 0.9999） |
| loader 讀得到兩臂的乾淨／額外列（`ExtractedAudio`，cap_index_fixed=0，require_text_overlay） | ✅ |
| action Step 1 驗證器（乾淨塊 = 066 輸入逐位元、兩臂 id 順序相同、unlab caption 未改、lab = 前綴＋原文、overlay 綁定抽樣） | 兩臂 ✅ |
| **劣化能不能活過 VAE**（decode→vocode 後 vs 乾淨來源） | noise：平坦度 +0.074、重心 +539 Hz；clip：crest −0.96；lowpass：重心 −210 Hz；crackle：crest +1.09；bitcrush 最弱（平坦度 +0.013、crest +0.38）✅ |

## 端點（事前寫定）

### 門檻來源：control 的 3-seed 分布（066–071 slot0clean 臂，MusicCaps 5521，CLAP batch 1）

| seed | CFG0 CLAP | CFG0 PQ | CFG3+neg CLAP | CFG3+neg PQ | negprompt 增益 ΔPQ |
|---|---:|---:|---:|---:|---:|
| 14159265 | 0.1977 | 6.4302 | 0.2243 | 7.2582 | 0.8280 |
| 16180339 | 0.1989 | 6.4879 | 0.2322 | 7.3829 | 0.8950 |
| 27182818 | 0.1997 | 6.5079 | 0.2302 | 7.4314 | 0.9235 |
| **全距** | 0.0020 | 0.0777 | 0.0079 | 0.1732 | **0.0955** |

門檻 = **2× 3-seed 全距**（`reference_training_seed_pq_noise_floor`：效果要與底線同協定量、門檻設 2×）。

### E1 — 操弄檢查（mechanism，最先讀）：模型現在有沒有缺陷方向？

在三顆 checkpoint 上重跑 D1 probe（`run_d1_noise_probe.sh`，14 條 held-out prompt × 128、cfg0 與 cfg3 null-neg）。
**不用 CLAP**（D1 已證純靜音在缺陷 caption 上刷分）。用波形簽名，全部對同 checkpoint 的 rock 乾淨 stem 配對：

| prompt | 預期方向（真劣化參考） | 讀的量 |
|---|---|---|
| rock ＋ noisy/hiss | 頻譜平坦度 ↑、重心 ↑ | Δflatness |
| rock ＋ distortion/clipping | crest ↓ | Δcrest |
| rock ＋ muffled | 重心 ↓ | Δcentroid |
| 純缺陷（white noise／static／clipping／muffled, no music） | 不以靜音逃逸 | 靜音比例、RMS |

**判準**：lab 臂在 ≥2 個軸上效果方向正確、CI 不跨零，且量級 ≥ unlab 與 control 的 2 倍 → 「缺陷監督建立了文字可達的缺陷方向」。
純缺陷 prompt 的靜音比例 lab < control（D1 基準 cfg3 靜音 79.7% 是 full checkpoint 的數字，quarter 基準以 control066 重跑為準）。

### E2 — 主端點：negprompt 增益有沒有變大？

`G = PQ(CFG3+neg) − PQ(CFG0)`，同 checkpoint 逐檔配對。

- **支持**：`G_lab − G_unlab ≥ +0.19`（2 × 0.0955），且 `G_lab − G_control ≥ +0.19`。
- 只過 vs control、不過 vs unlab → 效果來自「看過壞音訊」本身，不是標籤。
- 響度閘門：任兩格 `level_lufs_mean` 差 > 0.5 LU 時必報響度對齊後分數（063/065）。

### E3 — 非劣性：乾淨生成有沒有被污染？

CFG0 下 lab / unlab vs control：CLAP 下降 ≤ 0.004（2 × 0.0020）、PQ 下降 ≤ 0.155（2 × 0.0777），
且 `level_silent_n` 不超過 control 的 2×（`project_slot0nm_silence_mode`）。
失敗 → 寫成「缺陷樣本滲進了無負向 prompt 的生成」，這本身是一個結果。

### E4 — 次要：極性（跑完 E1–E3 後補，不在本次 chain 內）

031 的 reversed 負向 prompt 在 lab 臂上是否**失效**（缺陷極性出現）。若 E1 未過則不跑。

## 判定規則

| 觀察 | 寫法 |
|---|---|
| E1 過、E2 過、E3 過 | 支持「缺少缺陷監督是 negprompt 只吃領域詞彙的原因」；補主觀試聽前不對外宣稱 |
| E1 過、E2 不過 | 模型學到了缺陷方向，但 fidelity8 的增益不靠它 → 031「領域詞彙」解讀更強 |
| E1 不過 | 25k × quarter 不足以建立方向；**不是**「缺陷監督無效」。記錄劑量上限 |
| E3 不過 | 報污染模式（哪些 prompt、靜音／響度／crest），不論 E2 |
| 全部 CI 跨零 | 證據不足，不是等效 |

## 限制

1. 單一訓練 seed（14159265），對照也只用同 seed 那顆；E2 門檻借 3-seed 全距補，但處理臂自身的 seed 變異未量。
2. quarter budget（S1 只有 ~3 epoch）；25k 額外列在 S1 期間約被看 3 次。
3. 劣化全是程式化、LUFS 對齊的；真實世界低品質錄音（壓縮、殘響、窄頻麥克風）不在內。
4. MusicCaps 評估音訊本身含低品質 YouTube 錄音；AES/CLAP 都不是聽感 ground truth（`reference_quality_metric_validity_literature`）。

## 執行

- 產物根目錄：`~/exps_nvme/defect_negsample/`（latents、overlay、symlink farm、兩臂 TSV/manifest、`degradations.tsv`）
- chain：`scripts/training_pipelines/d2_defect_negsample_chain.sh`（tmux `d2_075`，持有 `gpu0.lock`）
  build → defectlab → defectunlab → D1 probe × {defectlab, defectunlab, control066}
- 日誌：`~/logs/075_d2_defect_negsample_chain.log`、`~/logs/075_build.log`、`~/logs/075_{defectlab,defectunlab}.log`
- 預估：build ~15 min ＋ 每臂 S1 3.1 h ＋ S2 1.7 h ＋ eval ~50 min ＋ D1 probe ×3 ~40 min ≈ **12.5 h**
- 可恢復：S1/S2 每 10k 存 ckpt，action 自帶 resume；chain 重跑會跳過已完成的 build／EMA／REPORT。
- GPU queue：p1/p2 pending 皆空，啟動時 `gpu0.lock` 空閒；chain 全程持鎖。
- **偏離 queue 政策**：chain 直接跑沒走 queue，也沒掛通知 → 啟動時 Discord 沒收到任何訊息。2026-09-23 補掛 watcher
  （tmux `notifywatch_075`，`scripts/notify_when_pid_exits.sh`，以 `[DONE] 075 chain` 判定成敗，log `~/logs/075_notify_watch.log`）。
