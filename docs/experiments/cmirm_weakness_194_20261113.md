# 194 CMI-RM 弱點：文字攻擊＋30 秒視窗（預登錄，2026-11-13）

狀態：設計登錄；p2 194，eval-only，約 1–1.5 h。上游：`docs/experiments/muq_ce_cmi_next_stage_20261111.md` §192（Part 0 已做：`cmirm_painpoint_mining_20261111_results.md`）。
授權：使用者 2026-11-13「go」（回應「191–192 CE、CMI-RM 弱點排進 queue」）。
CMI-RM 只評分，不訓練；無人類評分，結果只描述評分器方向，不寫「變好／變差」。

## 0. 要回答的問題

1. **Part A 文字攻擊**：音訊不動、只改送進 CMI-RM 的文字，quality 頭的分數會不會動？（quality 應只看音訊；若會被「High quality」字樣或不相干的 caption 推動，代表 quality 頭混入文字。）
2. **Part C 30 秒視窗**：CMI-RM 只取前 30 秒（`max_dur=30.0`）。30 秒以後的劣化是否逐位元看不到（同 190 F2 的 MuQ 10 秒裁切）？
3. Part B（低通階梯）與 Part D（189 循環音）**不重跑**：193 已有 lp7k…lp1k 的 CMI_q，190 F1 有高頻補噪，189 有循環音的 CMI_q，結果文件直接引用。

## 1. Part A：文字攻擊

- **底**：9 個 MusicCaps 底（真實、MeanAudio CFG3＋neg × 3 seed、CFG0 × 3 seed、MusicGen、MusicLDM），各 1,000 id，用 189 `load_base`（16 kHz、10 s、−30 LUFS）。沒有挑選步驟，不分 dev／test，全用。
- **文字（8 種）**：
  | key | 文字 |
  |---|---|
  | `cap` | 原始 caption（對照） |
  | `empty` | 空字串 |
  | `irr` | 固定錯配：同一底內另一個 id 的 caption（種子 20261113 的 derangement，9 個底共用同一個對應） |
  | `cap_hq` | caption ＋ " High quality, professional studio recording." |
  | `hq_only` | "High quality, professional studio recording." |
  | `cap_lq` | caption ＋ " Low quality, noisy." |
  | `hqpre` | "High quality recording. " ＋ caption |
  | `lq_only` | "Low quality, noisy." |
- 每格記 CMI-RM 的 quality 與 alignment 兩個輸出。
- **幅度**：m_q = 0.10 × S_q，S_q = 0.657（既有 caption 模式 S）→ m_q 0.0657。S_align = 9 個底 `cap` alignment 平均的全距，由 driver 算出寫進 summary，m_align = 0.10 × S_align。
- **主端點（逐段配對差，id bootstrap 5,000 次，種子 20261113）**：
  - **E1**（品質字樣推高 quality）：Δq = q(`cap_hq`) − q(`cap`)，95% CI 下界 > m_q 算該底成立。
  - **E2**（錯配 caption 拉低 quality）：Δq = q(`irr`) − q(`cap`)，CI 上界 < −m_q 算該底成立。
  - 9 個底的成立數：≥ 7 → 「quality 分數可被文字操弄」（E1）／「quality 頭混入文字對齊」（E2）；3–6 → 依底而定；≤ 2 → 不成立。
- **次要（描述）**：`cap_lq`、`hqpre` 對 `cap`；`hq_only`、`lq_only` 對 `empty`；每種文字的 alignment 變化；9 個系統在 `cap` 與 `cap_hq` 下 quality 排名的 Kendall τ；`irr` 的 alignment 降幅（操弄檢查：錯配 caption 的 alignment 應明顯下降，否則 alignment 頭本身不讀文字，E2 無從解讀）。

## 2. Part C：30 秒視窗

- **素材**：MUSDB18-HQ（`/mnt/seagate/kojiek_data/musdb18hq/musdb18hq.zip`），test＋train 中長度 ≥ 75 s 的曲目，取 15 s 起的 60 秒混音，16 kHz 單聲道、−30 LUFS；最多 150 首（依曲名排序取前 150）。
- **格**：
  | key | 內容 |
  |---|---|
  | `base60` | 60 秒原版 |
  | `base30` | 前 30 秒（截斷檢查：應與 `base60` 逐位元同分） |
  | `noise_pre` | 前 30 秒加白噪 SNR 10（SNR 以該半段 RMS 計） |
  | `noise_post` | 後 30 秒加白噪 SNR 10 |
  | `lp_pre` | 前 30 秒 2 kHz 低通（8 階 Butterworth sosfiltfilt） |
  | `lp_post` | 後 30 秒 2 kHz 低通 |
  加噪／低通後不再做響度正規化（與 190 F2 同；只改一半，整段 LUFS 會動，前後兩格同一處理所以可對比）。
- **評分**：CMI-RM empty 模式與固定通用文字 "A piece of music." 兩種；同時記 quality、alignment。
- **判定**：`*_post` 對 `base60` 的 |Δ| 在每段都 < 1e-4，且 `*_pre` 的平均 Δq CI 上界 < −m_q（m_q 用 empty 模式 S 1.450 × 0.10 = 0.145；"A piece of music." 用 caption 模式 0.0657）→ **「30 秒外盲點成立」**；`*_post` 有任何段 |Δ| ≥ 1e-4 → 「CMI-RM 看得到 30 秒外」，報幅度；`*_pre` 未過 → 「此劣化在視窗內也看不到」，盲點無從展示，只報描述。`base30` 對 `base60` 不一致時視為實作問題、job 失敗。

## 3. 不做的事

- 不改 CMI-RM 程式；不做文字的最佳化搜尋（只用固定 8 種文字，避免測試期偷看）。
- 不重跑 193／190／189 已有的格。
- 不把 Part A 的結果寫成「人覺得…」：無人類評分。

## 4. 偏差（預先寫明）

- D1：Part A 9 個底全量都用，不切 dev／test（無挑選）。
- D2：Part C 的素材只用 MUSDB（MusicCaps 片段只有 10 秒、生成音檔 10 秒，造不出 60 秒刺激）。
- D3（實作審查時補，作答／評分前）：`base30` 對 `base60` 不是逐位元同分——CMI-RM 內部 24 kHz 重採樣在 30 秒截斷處有邊界效應（smoke 實測 max |Δq| ≈ 1e-3、平均 ≈ 1e-3）。實作問題的失敗門檻改為平均 |Δq| > 0.01。
- D4：同理，`*_post` 的 1e-4 門檻低於此邊界效應；主判定用 max(1e-4, 該模式實測 max|base30−base60|) 當盲點容差，嚴格 1e-4 判定並列報告。
- D5（收線時補，2026-11-14）：Part C 實際 139 首（MUSDB 解碼後長度 ≥ 75 s 的只有 139 首；設計寫「最多 150 首」）。driver 收尾驗證寫成「剛好 150 首」→ queue 判 rc=1 failed；分數完整、未重跑，結果見 `results/cmirm_weakness_194_results.md`。
