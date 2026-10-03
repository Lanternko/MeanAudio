# MEva 擴充比較：資料量、訓練預算與明確退化對照

固定使用已驗證的 MEva pooled / small / f03 SAE-only checkpoint，檢查它能否辨認我們實驗中的品質差異。所有比較與預期方向在 queue 註冊前固定；完整公開反向與不顯著結果，不依分數挑選模型或要求特定組必勝。這是評估模型的實驗，尚未取代 AES。

| 比較 | 控制與差異 | 每組配對數 | 可以回答什麼 |
|---|---|---:|---|
| slot0nm quarter／full | 相同 251,598 筆資料；S1/S2 步數 100k/50k 對 400k/200k | 5,521；歷史 CFG0 與 CFG3+negative 分開 | MEva 是否偏好較長訓練的輸出；不是資料量對照 |
| MusicFlamingo 10k／100k | 同方法的存留 checkpoint；步數也從 100k/50k 變為 200k/100k，歷史資料來源切分不同 | 5,521 | MEva 如何評價較大資料與較多訓練的組合 |
| LPMC 10k／100k | 同上，另一套 caption 方法 | 5,521 | 趨勢是否能在另一方法重現 |
| MusicFlamingo／LPMC | 在 10k、100k 兩種規模分開比較 | 5,521 | 方法間模型偏好，無預設真人贏家 |
| 原始／人工退化 | 相同來源音訊；9 種噪音、削波、低通、bitcrush 強度；嘗試匹配 LUFS，超峰值時縮放 | 192 × 9 | 嚴重退化是較強的品質 sanity check；輕度失真可能有風格主觀性 |
| 現存各組與 CFG0／CFG3+negative | 完整保留的 33 組 MusicCaps 輸出；同 prompt 配對 | 各 5,521 | 歷史模型排序，以及 CFG 與 negative prompt 共同改變的偏好 |

總計 33 組現存 MusicCaps、4 組重新生成 MusicCaps，以及 1,920 段原始／退化音訊，206,197 筆 MEva 分數。原 095 實驗的 99,378 筆只有在音訊與模型雜湊一致時才沿用。歷史缺失音訊與缺少現代 REPORT 的組別明列於 coverage，不冒充全歷史資料已恢復。

新生成的四組統一 MusicCaps 5,521、MeanFlow 25、CFG 3、固定 fidelity8 negative prompt、seed 42、NoMask、fp32、NoQ。沿用正式 wrapper，另產生 CLAP／AES PQ、CE、CU 的標準報告。已存留的 CFG0 與音量診斷保留原始標籤。

報告提供每組平均分、同 prompt 的平均分差、MEva 偏好左組的比例（平手算半票），以及 2,000 次 prompt/source 配對 bootstrap 的點對點 95% CI。這些比例不是「符合真人的正確率」，CI 也沒有涵蓋訓練 seed 不確定性或多重比較校正。更多資料／步數通常有利，但不構成必勝的真人 ground truth；真正驗證人類一致性仍需要對這些配對做盲評。

100k 歷史 pipeline 曾指向被隔離的來源 TSV，本次只描述存留 checkpoint 的行為，不重訓、不把它當成乾淨的資料量因果實驗或正式方法晉級證據。

執行順序為：全輸入 preflight → 現存音訊 MEva 評分與中期報告 → 四組 canonical 生成 → 新音訊 MEva 評分 → 完整報告與 postflight。每次擴大 GPU 工作前核對固定輸入与空間並發送 gate 通知。原子 per-clip cache 可續跑；完整新生成 REPORT 驗證後沿用，未完成的新生成組則從頭生成，以維持 seed 42 的 RNG 序列。

使用 P2 正式 queue，追加在 AES 101、102 之後，不改已有順序。48 小時累計 active budget、50 GiB 空間硬下限、80 GiB 警戒；包含通知、失敗、暫停、訊號、空間、過期／變動輸入與結束驗證。無自動修補、無 shared-host 修改、無刪除既有音訊。
