# 總評

**1/5 失敗（建議不採用 MERT 當錯音守門員；否證結果可寫進指標分析）**

**核心結論**：不給原版、只在 MIDI 鋼琴分布內，MERT 判不出 10% 的錯音——test（generaluser）`wrong_10` in＋out 合併 AUC 0.640（95% CI 0.625–0.659）沒過 0.80，也只比最強的非 oracle 對照 `rule_2d`（0.595）高 0.045（CI 0.023–0.069）而不到 0.05；in（0.603）與 out（0.677）分開看都離門檻很遠，所以不是「只學到調性」，是兩類都弱。prereg §4 兩條主判準都自動判 fail。

**關鍵數據與例子**

主端點（test 250 段 × generaluser；正例 = 該 wrong 版本、負例 = 4 個 ok 版本；CI 為 2,000 次演奏 cluster bootstrap）：

| 版本 | MERT AUC | 最強非 oracle 對照 | 對照 AUC | 符號 oracle |
|---|---|---|---|---|
| wrong_1_in | 0.510 | rule_key | 0.506 | 0.374 |
| wrong_1_out | 0.520 | rule_2d | 0.536 | 0.404 |
| wrong_5_in | 0.548 | rule_key | 0.519 | 0.361 |
| wrong_5_out | 0.594 | rule_2d | 0.614 | 0.472 |
| wrong_10_in | 0.603 | mert_random | 0.515 | 0.350 |
| wrong_10_out | 0.677 | rule_2d | 0.700 | 0.557 |
| wrong_30_in | 0.747 | aes | 0.547 | 0.289 |
| wrong_30_out | 0.852 | rule_2d | 0.907 | 0.738 |
| **wrong_10_in+out（主判準）** | **0.640** | **rule_2d** | **0.595** | **0.454** |

未見音色 fluidr3（test 另外 250 段，只跑 MERT）：

| 版本 | generaluser | fluidr3 | 差 |
|---|---|---|---|
| wrong_1_in | 0.510 | 0.512 | +0.002 |
| wrong_1_out | 0.520 | 0.525 | +0.005 |
| wrong_5_in | 0.548 | 0.538 | −0.010 |
| wrong_5_out | 0.594 | 0.580 | −0.014 |
| wrong_10_in | 0.603 | 0.585 | −0.018 |
| wrong_10_out | 0.677 | 0.637 | −0.040 |
| wrong_30_in | 0.747 | 0.688 | −0.059 |
| wrong_30_out | 0.852 | 0.777 | −0.075 |

T2 兩段挑原版（同一段 clean vs wrong_x，取 wrong 機率較低者為原版）：

| 版本 | in | out | in＋out 合併 |
|---|---|---|---|
| wrong_1 | 0.612 | 0.784 | 0.698 |
| wrong_5 | 0.760 | 0.916 | — |
| wrong_10 | 0.788 | 0.956 | 0.872 |
| wrong_30 | 0.864 | 0.964 | 0.914 |

T3 不算錯的不能誤判（門檻 p ≥ 0.5；附帶條件為 transpose／tempo 誤報 ≤ clean + 0.10）：

| 音色 | clean | transpose | tempo | gain −6 dB | 附帶條件 |
|---|---|---|---|---|---|
| generaluser | 0.780 | 0.820（+0.040） | 0.816（+0.036） | 0.780 | **過** |
| fluidr3 | 0.920 | 0.920（+0.000） | 0.928（+0.008） | 0.920 | 過 |

選模（全部由 val 決定，test 不參與）：MERT 第 11 層／C=0.001（val AUC 0.652）；隨機權重 MERT 第 4 層／C=100（0.505）；AES C=0.001（0.531）；chroma 規則 C=0.001（0.584）；madmom key C=0.001（0.570）；chroma＋key 2 維 C=100（0.590）；MLP 用第 11 層（val log-loss 0.645）。

wrong_10 合併的完整對照（test AUC）：MERT 0.640、MLP 0.620、rule_2d 0.595、rule_chroma 0.593、rule_key 0.556、AES 0.524、隨機權重 MERT 0.514、符號 oracle 0.454。

- **訊號是真的，但很小**：MERT 0.640 高於隨機權重 0.514（≈ 擲硬幣）與 AES 0.524，也高於所有規則基線；只是幅度不夠，且離 0.80 很遠。
- **強度單調**：in 0.510→0.548→0.603→0.747、out 0.520→0.594→0.677→0.852（1／5／10／30%）。要到 30% 才接近可用（out 0.852 過 0.80、in 0.747 仍不過），與 133 的「reference-based MERT 在 ≥30% 才偵測得到」同一個量級。
- **out 比 in 容易（+0.074），但規則在 out 更強**：`rule_2d` 0.700 > MERT 0.677，MERT 在 out 沒有優勢；in 上 MERT 明顯贏規則（0.603 vs 0.490）但兩者都不可用。prereg 預期的「out 較易、規則在 out 很強」成立。
- **T2 與門檻互相矛盾**：配對挑原版 wrong_10 合併 0.872，排序資訊確實有；但同一顆頭把 78% 的 clean 判成 wrong（p ≥ 0.5）→ 分數排序有點用，絕對門檻／校準不可用。T2 的高正確率不能拿來當「判得出來」的證據。
- **fluidr3 沒有崩、也沒有變好**：−0.018～−0.075，說明 10% 的弱訊號不是 generaluser 音色特有，但也沒有可轉移的增益。
- **最佳層不是預期的中間層**：val 選到第 11 層（13 層裡的倒數第二），prereg §5「中間層 4–8 應比最後一層好」不成立；MLP 同層 0.620 反而低於線性 0.640。
- **符號 oracle 的 AUC < 0.5 是定義造成**：oracle 是調外能量比，`in` 把音移到調內會降低調外比例，所以 `in` 讀數天生反向；它只是參考欄，不進判準，也不算「贏過的對照組」。

**要你決定的事**

- 建議：**採用這個否證結果，不把 MERT 當錯音守門員**，寫進指標分析（與 126d／133 一致：神經指標對音符層錯誤無感）。理由：兩條主判準都不過、in／out 都差很遠、最強訊號只贏規則 0.045。
- 選項 B（若還想救）：把頭換成成對差值輸入（候選與 clean 的 MERT 特徵差，即 133 的 reference-based 形式）或直接餵 chroma；但 133 已顯示 MERT 對 ≤10% 錯音無感，預期上限受同樣限制，成本約 0.5 天。
- 選項 C：只保留「≥30% 可偵測、10% 不可」這條邊界，往 OOD（生成音樂）另立設計，不在 MIDI 分布內再投。
- 選項 D：收線，不再投入錯音偵測。

## 設定與偏離

- 刺激：MAESTRO v3 官方 split，train 800／val 150／test 250 段，每段 12 個版本（4 ok＋8 wrong）；test 另加 fluidr3 250 段，共 17,400 clips。
- 特徵：MERT-v1-95M rev `12af15f…` 13 層時間平均；隨機權重 MERT（seed 0）；AES 四軸；chroma_cqt 調外能量比；madmom `key_cnn_conf`。
- 判斷頭：逐層 L2 logistic，C 與層都在 val 選；MLP 次要。
- **test 只評一次**：C／層／early stopping／最強對照全部在 val 或 prereg 決定，test 的 T1／T2／T3 在單次 analyze 內一次算完（run.log 只有一次 `[analyze] done`）；summary 無 NaN／inf。
- 實測時間（p2 178 全程 43 分鐘）：render 682 s／17,400 clips（0.78 s/clip）、MERT 309 s、AES 215 s、chroma 40 s、madmom 1,068 s（10 workers）、analyze 264 s。
- **偏離 D1（段數替代）**：MAESTRO 官方 test 只有 177 場演奏、validation 只有 137 場，湊不到 250／150 段且每段都來自不同演奏；保留預登錄的段數，先用每個演奏一段，不足的部分（test 73、val 13）用同一批演奏裡第二個互不重疊的 10 s 窗口補，bootstrap 的抽樣單位改成「演奏」而不是「段」。manifest 的 window 欄可查。解讀時不能當成「250／150 個完全不同的演奏」。
- 偏離 D2：T3 的誤報門檻固定在 p ≥ 0.5（prereg 沒指定門檻）。
- 偏離 D3：「最強的非 oracle 對照組」取 test 上同一組比較（wrong_10 合併）中 AUC 最高者；用 test 最大值因此偏嚴格（只會讓 MERT 更難過關）。
- 偏離 D4：符號 oracle 的調外音比例用該段 clean 版本的調性計算，不隨版本重估；transpose 版本的 oracle 會變成 ~0.6（整段被移到調外），這是字面定義的結果。
- 偏離 D5：madmom 只算 `key_cnn_conf`（不跑拍點），其餘約定與 `mir_metrics.py` 相同。
- 偏離 D6：MLP 次要頭用線性頭選出的同一層特徵，early stopping 以 val log-loss 選 epoch。
- 不可寫：即使過了也只能寫「MIDI 鋼琴分布內」；不可寫成能判斷 AI 生成音樂的錯音，OOD 另立設計。
