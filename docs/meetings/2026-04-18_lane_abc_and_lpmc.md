# 2026-04-18 — Lane A/B/C 優先序 + LP-MusicCaps 澄清

## Lane A/B/C 優先序、MusicCaps benchmark、ISMIR 投稿準備

**背景**：原本 Phase 9.5 Qwen captioning 跑中（slot 0 至 232,681/251,599 = 92.5%），預估還要 ~3.5 天才全量完成。教授指示**暫停 Qwen**，重新排序：

### 三 Lane 順序（進可攻退可守）

1. **Lane A — MusicCaps benchmark**（~2-4 hr）
   - 用 5 組既有 checkpoint（Phase 4 V2 / 6 V2 / 7 V1 / 7 V2 / 8）跑 MusicCaps test set（5521 clips，ISMIR 標準 benchmark）
   - 取得跨 benchmark 的數字給教授和 reviewer
   - **已完成（2026-04-18 凌晨）**，結果見 `../experiments/best_results.md`
2. **Lane B — Phase 9 V1/V2 (LP-MusicCaps 5 caps, true random)**（~1.7 天）
   - 990/1218.npz 修復完成（`gen_multicap_npz.py --resume`），251K 全齊
   - 先跑 V1 (no Q)，再跑 V2 (+Q)，兩者都做 MusicCaps eval
3. **Lane C — Phase 9.5 Qwen captioning resume**（~3.5 天）
   - 最後才做

### 為什麼這個順序

- Lane A 幾乎不花 GPU 時間（多半是 metrics 計算），但得到教授立即可看的 benchmark 數字 — **最小成本、最大價值**
- Lane B 的實驗是**論文必要對照組**（LP-MusicCaps + true random 驗證 Phase 7 V1 的貢獻能否複現到 dynamic random）
- Lane C（Qwen）雖然是主結果，但沒有 Lane B 的對照也能獨立講 diversity hypothesis — **最晚做風險可控**

### Lane A 結果解讀

- **Phase 7 V1 MusicCaps 全面最佳** → random caption + Q 策略在 out-of-domain 泛化性勝出
- **Phase 6 V2 Jamendo > Phase 7 V1 Jamendo 的優勢在 MusicCaps 上倒轉** → best-consensus 有 Jamendo 特化 overfitting 的暗示
- **Phase 8 (random, no Q) 在 CE/PQ 上已接近 Phase 6 V2** → random caption 本身就已提供大部分 diversity 效益

### Lane A 踩坑

原本 pipeline 對所有 5 組傳 `--quality_level 9`，但 Phase 8 是 no-Q trained → 啟用未訓練 q=9 embedding 污染輸出。第一版 Phase 8 CLAP=0.1907（污染）已封存於 `eval_output/phase8_musiccaps_q9_WRONG_UNTRAINED_Q/`，重跑 `--no_q` 才得正確 0.1851。

Eval 旗標選擇規則已寫入 `memory/reference_eval_q_flag_rule.md` 與 CLAUDE.md「Eval → q 旗標選擇規則」段。

---

## LP-MusicCaps Jamendo caption 生成方式澄清（york135）

> 「LP-musiccaps 那個生資料的流程，根本沒有做在 MTG-jamendo 上面過。如果你說的是我之前給的那個 google drive 連結，那個是 wei-jaw 自己用 lp-musiccaps 這個 model 跑的 captioning，以 5 個不同的 seed 跑出來的。」

### CC 先前的誤解

把 LP-MusicCaps 論文的 pseudo-caption 生成 pipeline（tag→GPT-3.5→caption）當成我們 Jamendo 檔案的產生方式，並規劃 Phase 9.5 Qwen「鏡像 LP 4-task 方法論」。

### 事實

- LP-MusicCaps 論文 4-task pipeline **從未套用於 Jamendo**
- 我們手上的 `results_20260119_043407.jsonl` 是 wei-jaw 用 LP-MusicCaps **captioning MODEL**，在 Jamendo audio 上用 5 個不同 seed 產生的
- 5 caps 間 diversity 來自 **seed-sampled decoding noise**，不是結構化 task prompts

### 對研究敘事的影響

Phase 9.5 Qwen 的 5-task 設計應定位為「我們自己的新貢獻」，不是 LP 複刻。論文 methodology section 要清楚區分：
- **Phase 9 V1/V2 (LP-MusicCaps Jamendo caps)**：diversity 來源 = seed noise
- **Phase 9.5 V1/V2 (Qwen 5-task caps)**：diversity 來源 = structured task framing

這兩種 diversity 機制的對比本身就是一個有意思的研究問題（seed-noise vs task-framing 哪個產生的 MeanSim 更能當品質 q 信號）。
