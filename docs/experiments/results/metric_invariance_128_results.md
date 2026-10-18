# 128 評分不變性修補 × 論文結論重判：結果（128a 摘要＋128b／c／d）

> 收線 2026-10-18。
> - 128a：budget_r3 於 2026-10-17 17:16 完成，見 `metric_invariance_128_budget_r3_outcome_20261018.md`。
> - 128b：p2 129（nvme_r5）於 2026-10-18 06:04 完成。
> - 128c：p2 130 於 08:58 完成，exit 0。
> - 128d：CPU，隨 129 一起產出。
>
> 設計：`../metric_invariance_readjudication_128_20261017.md`（K1–K4 見 §4，節奏分層見 §5）。
> 數字來源：
> - `metric_invariance_128b_summary.json`（sha 4da578da…）
> - `metric_invariance_128c_summary.json`（30f8de3f…）
> - `metric_invariance_128d_summary.json`（65a25b2d…）
> - `metric_invariance_128{b,c}_identity_checks.json`
>
> 原始輸出在 `~/eval_output_nvme/metric_invariance_128_20261017_nvme_r5/{b,c}/`。

## 0. 一句話

**四個修補方案都沒有讓 AES／CLAP 通過 126 的平手要求（128a）。** 依預登錄規則，所有重判退回 C3（響度正規化＋平移／極性集成），並標為 `sensitivity_only`。各結論在這個敏感度分析下的結果：

| 結論 | 在 C3 下 |
|---|---|
| 表 B 的 PQ 優勢（K1） | 仍成立 |
| 表 B 的 CLAP 優勢（K2） | 仍成立 |
| CFG0 對 MusicLDM 的 PQ（K3） | 平手 |
| NegMF E1（K4） | 仍成立 |

**這些都不能當成「經不變性驗證的結論」**（`eligible_as_conclusion=false`），只能寫成「對響度與平移較不敏感的評分方式下方向不變」。

## 1. 128a 摘要（細節見 budget_r3 outcome doc）

- C0 raw、C1 響度正規化、C2 平移＋極性集成、C3 兩者：PQ、CE、CU、CLAP 在每個方案下都沒有通過完整的平手＋偵測門檻。
- 舉例來說，C3 下 PQ 的 lead 10 ms 平手率只有 66–82%，CLAP 的 lead 200 ms 只有 45–59%。
- Q3 預測失敗：blastnote ×1 的 PQ 判「變好」率在 C0 與 C1 都是 80.35%，響度正規化沒有修掉單音爆音被判變好的問題。
- 排除 quiet-LUFS fallback 的片段不改變任何登錄判定。

## 2. 128b：表 B 重判（K1–K3，模式 C3，seed 14159265／16180339／27182818）

判準（預登錄）。**still_holds** 需同時滿足：

- 3 seed 同號；
- 每個 seed 的 |Δ| > m；
- 每個 seed 的 better 率 > worse 率。

3 seed 平均 |Δ| ≤ m 判 **tie**。

| 結論 | 比較 | m | 逐 seed Δ | 超過 2·SD 比例 | better/tie/worse（逐 seed） | 判定 |
|---|---|---|---|---|---|---|
| K1 PQ | ours CFG3+neg − MusicGen | 0.0919 | +0.155／+0.164／+0.115 | 0.86／0.86／0.86 | .51/.11/.38、.53/.11/.36、.50/.12/.38 | **still_holds** |
| K2 CLAP | ours CFG3+neg − MusicGen | 0.0104 | +0.0161／+0.0195／+0.0215 | 0.61／0.62／0.62 | .52/.09/.39、.53/.09/.38、.53/.10/.37 | **still_holds** |
| K3 PQ | ours CFG0 − MusicLDM | 0.0919 | −0.031／−0.036／−0.074（平均 −0.047） | 0.88／0.87／0.87 | .44/.06/.49、.45/.07/.49、.43/.06/.51 | **tie** |

- **K1**：原始 lvl30 的差是 +0.165／+0.124／+0.176，C3 下大致不變；第三個 seed 縮到 +0.115，仍 > m。
  - 但逐段 better 只有約 51%，worse 有約 37%。
  - 「勝過」是群體平均差，不是大多數 prompt 都贏。
- **K2**：原始 CLAP 差 +0.0105 只比 m 大一點，C3 下反而變大（+0.016～+0.022），三 seed 都 > m。逐段 better 約 53%。
- **K3**：平均差小於 m，判平手。
- **集成與 fallback**：C2 集成的組內 SD 平均，PQ 約 0.042–0.068，CLAP 約 0.014–0.017。
  - 也就是說，CLAP 光是平移／極性造成的變動就和 m 同量級，所以 K2 的逐段判定對評分方式很敏感。全部結論都標 sensitivity_only，則是因為 128a 沒有方案通過。
  - 排除 LUFS fallback 片段後（K1／K2 n = 5512），K1 的 Δ 變成 +0.152／+0.162，判定不變。
- **身分閘**：前 200 段在所有 claim 指標上與歷史分數差 0（`identity_checks.json`）。

## 3. 128c：NegMF E1 重判（K4，CFG0 lvl30 PQ，NegMF N100 − control，C3）

| seed | Δ | 超過 2·SD 比例 | better/tie/worse |
|---|---|---|---|
| 14159265 | +0.965 | 0.933 | .928/.046/.027 |
| 16180339 | +0.940 | 0.935 | .925/.041/.034 |
| 27182818 | +1.004 | 0.941 | .939/.032/.029 |

- 判定 **still_holds**（sensitivity_only）。和原始 E1 +0.956 幾乎相同。
- 逐段有 93% 判較好：這是唯一一個「大多數 prompt 都贏」的結論。
- LUFS fallback 只有 1 段（meanaudio s16180339 cfg0）。
- 身分閘：meanaudio 格與歷史分數差 0。NegMF 格沒有歷史參照，只核對了登錄的輸入雜湊與正規化流程。
- **提醒**：E1 的端點本身就是 AES PQ。FAD 三 seed 都更差的結論（`negmf_full_120_results.md`）不受 128 影響。

## 4. 128d：PQ 優勢依真實錄音節奏清晰度分層（CPU，C3）

依真實參考錄音的 pulse_clarity 分三層（切點 0.4935、0.8158），n_reference = 5131。結果為 ours CFG3+neg − MusicGen 的 PQ 差。

| | 整體 | 低 pulse | 中 | 高 |
|---|---|---|---|---|
| 3 seed 平均 | 0.139 | 0.112 | 0.157 | 0.149 |
| s14159265 | 0.153 | 0.150 | 0.149 | 0.160 |
| s16180339 | 0.155 | 0.120 | 0.185 | 0.160 |
| s27182818 | 0.110 | 0.065 | 0.137 | 0.129 |

- 預登錄判讀為 `advantage_independent_of_reference_rhythm`：PQ 優勢沒有集中在某個節奏層。
- 低 pulse 層稍小、seed 間變動較大（0.065–0.150），但沒有反號。
- 各系統類底線（floor-like）輸出比例都 ≤ 0.45%（real 23/5129、MusicGen 0），分層結果不是靜音／底線片段造成的。

## 5. 判讀（可寫／不可寫）

**可寫：**

- 「在響度正規化＋平移／極性集成的敏感度分析下，表 B 的 PQ 與 CLAP 優勢方向與大小維持，三 seed 都超過平手邊界；CFG0 對 MusicLDM 的 PQ 差落在平手邊界內。」
- 「NegMF 的 CFG0 PQ 增益在同一分析下不變（逐段 93% 較好）。」
- 「表 B 的 PQ 優勢不隨參考錄音的節奏清晰度改變。」

**不可寫：**

- 「經不變性驗證」或「指標修補後確認」：128a 沒有任何方案通過。
- 「ours 在大多數 prompt 上勝過 MusicGen」：逐段 better 只有約 51–53%。
- 把 PQ／CLAP 差講成人類可感知的品質差：無人類評分，表 E 人類聽測 pilot 仍在等聽者。

## 6. 執行歷程（給重現用）

| 輪次 | 結果 | 文件 |
|---|---|---|
| smoke | AES 尾批次 1e-4 重現失敗 | `metric_invariance_128_smoke_incident_20261017.md` |
| identity_r1 | 未定義 LUFS（RDCXvJPzT5k_520）停機 | `metric_invariance_128_identity_r1_20261017.md` |
| quiet_r2 | 吞吐閘 8.694 h > 7 h | `metric_invariance_128_quiet_r2_20261017.md` |
| budget_r3 | 128a 完成；129 因 CU 尾批次差 0.0129 停 | `metric_invariance_128_budget_r3_20261017.md`、`..._outcome_20261018.md` |
| claims_r4 | 身分閘縮到 claim 指標（使用者 GO） | `metric_invariance_128_claims_r4_restart_20261018.md` |
| nvme_r5 | 熱資料改回 NVMe，129／130 完成 | `metric_invariance_128_nvme_r5_storage_20261018.md` |
