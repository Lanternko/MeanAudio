# 126c：MUSDB18-HQ 分軌編輯（去樂器／人聲、獨奏、重新混音、錯位、走音、換軌）

> **狀態**：2026-10-19 設計登錄＋排 p2 131。是 126 留下的「MUSDB18-HQ 去樂器／人聲 job 未排」（`results/metric_win_tie_126_results.md` §8）。
> 判定規則、m、通過門檻、S 全部沿用 126（`metric_win_tie_126_20261016.md` 登錄版）；這裡只登錄素材與介入。
> 沒有人類評分（唯一的人類讀數是 AES 公開的 MUSDB 分軌評分，見「人類錨點」，只能當描述）。

## 為什麼要做

126a／126b 的素材是 TTM 輸出、鋼琴 MIDI 和鋼琴／人聲錄音，**沒有多軌、有混音製作的真實歌曲**。
之前唯一的「去樂器」數字是 10/01 Codex 的 karaoke 12 首（`results/aes_codex_ablations_20261001_results.md`：去鼓 PQ −0.29、去人聲 −0.41），
用 demucs 分離，所以分不開「少了一個聲部」和「分離瑕疵」。MUSDB18-HQ 有真實分軌（mixture = 四軌相加），可以：

1. 做**真正的**去軌（不經分離），和 htdemucs 去軌逐窗對讀 → 量出分離瑕疵本身的分數。
2. 做只有多軌才能做的介入：單軌時間錯位、單軌走音、換成別首歌的人聲／鼓。這些是「一聽就錯」的編曲／演奏錯誤，
   126b 已經證實所有指標都聽不到錯音，這裡檢查**在完整混音裡**是否一樣。
3. 測「混音平衡」這種不該扣分的改動（某一軌 ±1／±3／±6 dB）。

## 素材

- MUSDB18-HQ（`/mnt/seagate/kojiek_data/musdb18hq/musdb18hq.zip`，直接從 zip 讀，不解壓）：train 100＋test 50 首，44.1 kHz 立體聲 → 單聲道 16 kHz（soxr VHQ）。
- 每首從 15 s 起切不重疊的 10 s 窗，只留**四軌都有聲音**的窗：每軌 LUFS ≥ −45 且 ≥ mixture − 20 LU，且 60% 的 1 s frame 該軌能量 ≥ mixture − 30 dB；mixture LUFS > −40。最多 4 窗，平均分布在可用窗之間。
- 每個 case 都**重新正規化到 −30 LUFS**（同 126 的 lvl30；峰值上限 0.999），所以去軌不是音量改變。
- CLAP caption 一律 `music`（MUSDB 沒有 caption；同 126b 的 floor）。
- htdemucs 在 MUSDB train 上訓練過，train 那組的分離品質偏樂觀；分離對讀要分 train／test 看（`by_group`）。

## 介入（每窗 65 個 case，base＝同窗 mixture）

| kind | case | 說明 |
|---|---|---|
| ref | `mix` | mixture |
| tie | `stemsum` | 四軌相加（smoke：與 mixture SNR 65–70 dB） |
| tie | `dmx_resum` | htdemucs 四軌相加（smoke：SNR 17–36 dB，是真的有殘差的平手測試） |
| tie | `rebal_<stem>_±1dB` | 單軌 ±1 dB（8） |
| soft | `rebal_<stem>_±3dB`、`±6dB` | 混音平衡選擇，判成不重要（16） |
| soft | `vocal_pv0` | 人聲經 rubberband +50 再 −50 cents（只有處理、沒有音高錯） |
| tie | 訊號層：`gain-1`、`gain-3`、`polarity`、`pcm16`、`lead10ms`、`mp3_64k` | 126a 同定義，套在 mix 上 |
| lose | `drums_lag{20,50,100,250}ms` | 鼓軌整體延遲（其他三軌不動） |
| lose | `vocals_lag{50,100,250}ms` | 人聲延遲 |
| lose | `vocals_detune{25,50,100}c` | 人聲升 25／50／100 cents（rubberband，時長不變），伴奏不動 |
| lose | `bass_semi+1` | 貝斯升半音 |
| lose | `swap_vocals`、`swap_drums` | 換成同 split 下一首歌、同窗序號的人聲／鼓 |
| lose | 訊號層：`noise_snr{40,30,20,10}`、`clip6db`、`lowpass4k`、`clicks5ps` | 126a 同定義 |
| descriptive | `no_<stem>` | 真實去軌（4） |
| descriptive | `only_<stem>` | 單軌獨奏（4） |
| descriptive | `dmx_no_<stem>` | htdemucs 去軌（4） |

去軌與獨奏不預設方向：少一個聲部是內容改變，不是品質錯誤。它們的讀數是描述，加上兩個對讀：

- **分離瑕疵**：`dmx_no_<stem>` 對 `no_<stem>`（同窗配對）。如果分數差主要來自這裡，Codex 10/01 的去鼓／去人聲數字要改讀成分離瑕疵。
- **走音扣掉處理**：`vocals_detune<c>` 對 `vocal_pv0`（同樣經過 rubberband），只剩音高錯。

## 人類錨點（描述）

AES 公開人類評分 `research/eval/aes_human_ratings/AES_natural_music.jsonl` 有 MUSDB **95 首（train＋test）的 mixture 與四軌**，各 10 位評分者 × 4 軸；以 (split, 曲名) 對到我們的窗。
每首算人類 mean(stem) − mean(mixture)，對照指標在該首各窗的平均 d(`only_<stem>`)；報各軸 Spearman（track × stem）、正負號一致率、各軌平均。
人類評的不是我們的 10 s 窗，只能當方向參考，不進通過判定。

## 讀數（同 126）

- d = score(case) − score(mix)；|d| ≤ m 判平，m = 0.10 S；S = 126a 的 S（神經指標）、本實驗 mix 段的 SD（MIR）；通過 = 80%。
- tie／soft 要判平；lose 只對有方向的指標（PQ、CE、CU、MEva、CLAP、MIR 四項）要判差。
- 階梯：鼓延遲、人聲延遲、走音報偵測門檻（第一個通過的幅度）；rebalance ±1→±6、gain、noise 報容忍度。
- 每格附 per-case FAD（VGGish，對同窗 mix）。
- MIR 只跑在 mix、stemsum、dmx_resum、±6 dB、去軌／獨奏、錯位、走音、換軌（訊號層與 ±1／±3 不跑）。

## 預期（登錄）

- 錯位／走音／換軌：依 126b，神經指標大多判不出差；`pulse_clarity`／`ibi_cv` 可能抓得到鼓延遲 ≥ 100 ms，`key_cnn_conf` 可能抓得到貝斯升半音與換人聲。
- `stemsum` 應全指標判平；`dmx_resum` PQ 可能過不了。
- `rebal ±1 dB` 應判平；±6 dB 預期 CE／CU 會動。
- 去鼓／去人聲：如果真實去軌也是 PQ −0.3 量級，Codex 的數字是內容效應；如果只有 `dmx_no_*` 掉，是分離瑕疵。

## 執行

- 刺激（CPU＋一次短 demucs）：`scripts/eval/metric_win_tie_126c_stimuli.py --stage prep|demucs|build`，
  輸出 `~/eval_output_nvme/metric_win_tie_126c_20261019/stimuli/`（NVMe；跑完可整包搬 seagate 留 symlink）。demucs 在 `~/venvs/karaoke-jp-align`（demucs 4.0.1，htdemucs）。
- 評分（p2 131，queue）：`scripts/eval/metric_win_tie_126c.py`（沿用 126b 的 unit 評分迴圈：AES、MEva、CLAP batch 1、VGGish frames；MIR 走 CPU subprocess）。
  contract：`metric_win_tie_126c_20261019_contract.json`；結果寫 `results/metric_win_tie_126c_summary.json`。
- smoke（4 首 16 窗、1,056 段）已跑通刺激與評分。
