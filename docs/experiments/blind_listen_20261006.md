# 盲聽：AES／MEva 在我們自己 arm 上的地面真值（2026-10-06）

## 問題

AES 與 MEva 都沒有在「我們的生成音訊、同架構 arm 之間」被驗證過：
- AES 在 PAM 生成音樂上的 PQ r 只有 0.55；人類差 < 0.5 的配對，PQ 一致率 0.46。081 在 87% prompt 勝過真實錄音，PQ 可能已被 game。
- MEva（pooled-small-f03）沒過 PAM 外部效度閘門：系統排名與人類相反、系統內 r 0.05–0.24；lowpass 2k 比 1k 罰更重。
- 兩者在 defectlab CFG3+neg vs CFG0 上方向相反。

本研究用小規模人類盲聽當裁決。

## 設計

三組比較，共用同一組 24 個 MusicCaps prompt：

| 比較 | treatment（指標偏好的那邊） | reference | 指標狀態 |
|---|---|---|---|
| `nmv2` | nmv2pair quarter s14159265 CFG3+neg（fidelity8） | 同 ckpt CFG0 | AES、MEva 都偏 treatment |
| `dlab` | defectlab quarter s14159265 CFG3+neg（fidelity8） | 同 ckpt CFG0 | AES（lvl30）與 MEva 不一致 |
| `q081` | 081 arm，HQ 前綴＋負向 `Low quality recording.` | 081 control CFG3+neg（fidelity8） | AES PQ 偏 arm +0.37；FAD 反向 |

- **prompt 抽樣**：從四個既有 cell 都有音檔的 5521 個 id 中，用 seed 20261006 隨機抽 24 個，不做任何篩選。TSV 按 csv record 讀取。q081 arm 用 hqprefix TSV 的同一列。
- **音訊**：nmv2 與 dlab 直接取既有全量 cell 的音檔。q081 兩格的音檔已刪，用原協定只重生這 24 首：MeanFlow 25、cfg 3、seed 42、fp32、NoMask、`--no_q`。雜訊與全量 run 不同，但交付的檔案全部重新評分，所以預測對應的正是聽者聽到的檔案。
- **響度**：每一對做 pyloudnorm 整合響度對齊，目標 −23 LUFS。若任一檔的峰值會超過 −1 dBFS，就把這一對的共同目標往下調；實際目標落在 −25.4～−23.0 LUFS。交付格式為 16-bit WAV（artifact 不供應 FLAC），檔名是不透明的隨機 hex。條件對照表 `key.json` 只留在本機。
- **trial**：72 個（3 比較 × 24 prompt），順序隨機，A/B 位置隨機。另加 6 個重複 trial（每組比較 2 個），A/B 位置與原 trial 對調，排在後半段，且距原 trial 至少 10 個。總共 78 個 trial。
- **每個 trial 的問題**：
  - 畫面顯示原始 caption（不含 HQ 前綴）。
  - Q1 **音質**（清晰度，無雜訊、失真、悶）：A／B／聽不出差別。
  - Q2 **整體**（音質＋與描述相符）：A／B／聽不出差別。
- **評分頁**：https://claude.ai/artifact/BoPrTzrffhrTtqdng1xoQy（私有；其他評分者需經 Share 給 Contributor 權限才能存檔）。模板是 `scripts/eval/blind_listen_20261006_page.html`。
  - db 規則：`ratings` 只有 owner 可讀寫；`ratings/{self}` 開放 interact 讀寫。
  - 每位評分者一份文件 `ratings/<uid>` = `{rater, answers: {<trial>: {pq, ovl, ms, at}}, n, updated}`，其中 pq 與 ovl 的值為 `'A'|'B'|'same'`。
  - 事後用 ArtifactData 讀出。
- **指標**：交付檔逐檔算 AES（batch 1；響度已對齊，等同 lvl30 讀數）、MEva raw、CLAP（batch 1，對原始 caption）。

## 指標預測（交付檔，聽測前登錄）

數值為 treatment − reference 的平均差，括號內是 treatment 在 24 首中勝出的數目：

| 比較 | AES PQ | AES CE | MEva | CLAP |
|---|---|---|---|---|
| nmv2 | +0.981 (24/24) | +0.869 (22/24) | +0.202 (15/24) | +0.036 (17/24) |
| dlab | +1.043 (22/24) | +1.104 (21/24) | +0.187 (12/24) | +0.022 (13/24) |
| q081 | +0.701 (21/24) | +0.299 (18/24) | +0.019 (13/24) | −0.009 (13/24) |

- **dlab 的不一致沒有重現**：在這 24 首上，MEva 的平均差是正的，但只勝 12/24。逐首來看，MEva 與 AES 仍有分歧。
- **MEva 對編碼格式敏感**：同一組音訊從 FLAC-24 改成 WAV-16 後，dlab 的 MEva 勝數從 15 掉到 12。CLAP 也有變動（nmv2 從 16 變 17）。MEva 的逐首符號有一部分落在量化雜訊範圍內。

## 預登錄讀法

1. **人類偏好率**：每組比較 treatment 勝率 = treatment 票 /（A+B 票），「聽不出」另外報比例。用雙尾二項檢定對 0.5，分 Q1、Q2 兩題各自報。
2. **指標與人的一致率**：只看人有表態的配對，數「該指標的配對差符號 = 人的選擇」的比例，指標為 AES PQ、AES CE、MEva、CLAP。
   - Q1 對 PQ、MEva；Q2 對 CE、MEva、CLAP。
   - 三組合併報，也分組報；附 Wilson 95% CI。
   - 一致率 CI 下界 > 0.5，才可以說「這個指標在我們的 arm 上與人同向」。
3. **重複一致性**：6 個重複 trial 中選擇相同的比例（A/B 位置已對調後換算）。若 < 4/6，人類讀數本身不穩，第 1、2 點只當方向讀。
4. **裁決 dlab**：人類在 dlab 上的勝率方向，決定 AES 與 MEva 哪個對。

## 限制

- 單一或少數評分者，24 prompt，只能偵測大效果。勝率 0.75 在 n=24 時雙尾 p ≈ 0.02；勝率 0.6 偵測不到。
- 16 kHz 生成音訊、筆電或耳機播放條件不受控。
- q081 是重生的 24 首，與全量 REPORT 的 PQ 數字不是同一批音檔。

## 為什麼不走 queue

GPU 部分只有重生 48 首加評分 144 個檔，預期 < 15 分鐘。用 tmux 直接跑，腳本掛了 `notify_on_exit` trap。

## 檔案

- 腳本：`scripts/eval/blind_listen_20261006.py`（select／generate／prep／score）
- 產物：`~/nvme_experiment_artifacts/meanaudio/blind_listen_20261006/`
  - `selection.json`
  - `key.json`：盲化對照表，勿公開
  - `key_scored.json`
  - `page_trials.json`
  - `deliver/*.wav`（發布用副本在 session scratchpad `blind/audio/`）
