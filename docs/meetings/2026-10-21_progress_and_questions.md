# 2026-10-21 會議用：進度整理與問題清單

對應 2026-10-14 會議第 8 項（`2026-10-14_metric_controls_real_pieces_fad_lalm.md`）。整理日期 2026-10-15。
所有結論都沒有人類評分支撐，「與預期不一致」不等於「指標錯」。

## 一、上週指示的完成狀況

| 指示 | 做了什麼 | 結果一句話 | 文件 |
|---|---|---|---|
| 1–4 真實曲 MIDI 控制、兩類介入、紅綠燈、更差的底線 | p2 123：MAESTRO 150 段 × 2 SoundFont × 57 版本（應變差 25／不應變 11／底線 12）× AES／MEva／FAD／MIR | 見下方「123 重點」 | `../experiments/results/midi_realpiece_control_123_results.md` |
| 5 FAD 檢查 | 同 123 §5 | 見下方「FAD 檢查」 | 同上 |
| 6 LALM 當評審 | p2 124：Music Flamingo、Qwen2.5-Omni-3B 評 74,158 段 | **兩個都沒過最低門檻**（噪音階梯不單調）；後續 probe 125 已擱置 | `../experiments/results/lalm_quality_judge_124_results.md` |
| 7 人類聽測 | 20 prompt × 3 組比較的 pilot 評分頁已發布 | 等聽者 | `../experiments/human_listen_real_vs_ours_20261015.md` |
| 8 進度與問題 | 本文件 | — | — |

### 123 重點

**應變差的音符層介入**（錯音、亂音高、打亂順序、節奏亂掉）
- PQ 最多只動 0.18、MEva 最多只動 0.26。
- 這和「不應變」控制組（殘響、換音色、移調）同一量級。
- 對照：訊號層底線（噪音、削波、bitcrush）會扣 1–4 分。

**各指標**
- AES CE／CU 沒過敏感度門檻，而且 CU 對移調與殘響有反應，方向與預先登錄的預期相反。
- 過敏感度門檻的只有 MEva、`mir_pulse_clarity`、`mir_key_cnn_conf`。
- 全部錯音時 PQ −0.06，但 key_cnn_conf −0.49。
- 掉 60% 的音符，PQ 反而 +0.18。

**紅綠燈**
- MEva 把真實錄音放在比較靠近純白噪音的一端。
- 其他指標的位置見 123 結果 §4。

### FAD 檢查

| 比較 | FAD |
|---|---|
| MusicCaps 真實錄音切兩半互算 | 0.146 |
| ours 不同 seed 互算 | 0.08–0.32 |
| ours CFG0 vs CFG3＋neg | 2.06–2.26 |
| 換 SoundFont（換音色） | 4.41 |
| 音符層介入（中位數） | ≈ 0.16，等於真實錄音切兩半的底線 |

- 對「換節奏 vs 換音色」的回答：FAD 幾乎只看音色，對音符層介入沒有反應。
- 「0.47」的出處見上週會議記錄：n = 100 的差值，不是絕對 FAD。

### 其他這週完成的

- **p2 120–122 NegMF N100 full recipe 三 seed 全過**（`../experiments/results/negmf_full_120_results.md`）
  - CFG0 lvl30 PQ +0.956，三 seed +0.92～+0.99。
  - CLAP 非劣性過，沒有靜音逃逸。
  - FAD 三 seed 都更差：CFG0 3.2–3.3 → 5.8–6.1。
  - 不推論期負向就追平 control 的 CFG3＋neg PQ，但 CLAP 輸 0.023。
  - 屬於 PQ 導向訓練，AES 欄不進排名。
- **MQAScore／MRMAD 文獻整理**（`../literature/lalm_judge_mqascore_mrmad_2026_10_15.md`）
  - MRMAD 裡 Qwen2.5-Omni 的音樂劣化兩段比較是 49.6%，和我們 124 的 pair 任務 0.50–0.52 一致。
  - MQAScore 刻意排除音質。

## 二、TTM 表的進度

| 表 | 狀態 |
|---|---|
| A 覆蓋 | 完成：ours 三 seed、MusicGen、MusicLDM（±fidelity8）、真實錄音都 5,521／缺 0 |
| B 主比較 | 客觀欄完成（118）。ours CFG3＋neg 對 MusicGen：lvl30 PQ +0.155、CLAP +0.0105。CLAP 輸 MusicLDM |
| B「方法消融」列 | NegMF N100 三 seed 完成（上面） |
| C 指標效度 | 123 補了真實曲控制，124 的 LALM 不可用 |
| D 白噪音階梯 | 完成 |
| E 人類欄 | pilot 頁已發布，正式版未做 |

## 三、問題清單（需要教授／使用者決定）

1. **聽測的聽者**：哪裡找、幾個人？
   - pilot 想要 3–5 位，不可以是熟悉 MeanAudio 輸出的實驗室成員（包含作者本人）。
   - 每人約 25–35 分鐘。
   - 頁面要用 Contributor 權限分享。
2. **正式表 E 的規模**：200 prompt × 5 人是否可行？
   - 要不要加 MusicLDM？
   - 要不要拆成「技術音質／整體喜好／文字吻合」三題（pilot 是分開問的）？
3. **控制類介入的「預期方向」**：123 預先登錄的方向是我寫的。
   - 例：移調 ±2 半音、輕微殘響「不應變」。
   - 例：掉 60% 音符「應變差」。
   - 需要確認這些預期本身合不合理，例如稀疏的鋼琴是不是本來就可能被聽成比較乾淨。
4. **AES CE／CU 對音符層介入不敏感**，要怎麼寫？
   - 目前只寫「與預先登錄的預期不一致」。
   - 要不要做小規模人類對照：同一段的原版 vs 錯音版，人能不能分辨、偏好哪個？
5. **FAD 對音符層介入看不到**（≈ 真實錄音切兩半的底線），換音色卻有 4.41。
   - 論文裡 FAD 要降級成「音色／分布」指標，還是照常報？
6. **LALM 線是否正式收線**：124 兩個評審都失敗，125（二元劣化偵測 probe）已設計但擱置。
   - 如果要再試，只剩 Qwen3-Omni-30B（要量化，MRMAD 上也只有 52–62%）。
7. **NegMF full recipe 過了 PQ 端點但 FAD 更差**：論文裡怎麼擺？
   - 方法消融列？
   - 還是只當「PQ 導向訓練會傷分布指標」的例子？
8. **磁碟**（已處理，2026-10-15）：改成搬到 seagate 歸檔碟，原路徑留 symlink，沒有刪除。
   - 搬了 `midi_realpiece_123`（16 GB）、`midi_realpiece_control_123`（6 GB）、control s16180339 的 S1 400k 目錄（42 GB；另兩個 seed 先前已在 seagate）。
   - 三個 control 的 S2 目錄留在 NVMe（TTM 表 ours 列）。
