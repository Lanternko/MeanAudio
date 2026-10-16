# MEva 擴充實驗比較（2026-10-04）

固定 pooled-small-f03；quarter/full 為步數對照。10k/100k 為資料量＋步數＋歷史資料來源共同變化，非純資料量因果證據。
所有預期方向事先登錄；不依結果改模型或篩選輸贏。無真人標籤，勝率指模型偏好。

| 對照（左−右） | n | MEva 平均差 [95% CI] | 左較高比例 | 方向 |
|---|---:|---:|---:|---|
| slot0nm full−quarter / mc_mf25_cfg3_neg | 5521 | +0.013 [-0.003, +0.029] | 52.0% | inconclusive |
| slot0nm full−quarter / musiccaps_mf25_cfg0_noq | 5521 | +0.000 [-0.013, +0.015] | 50.0% | inconclusive |
| clean−bitcrush_4 | 192 | +0.184 [+0.122, +0.248] | 67.7% | left higher |
| clean−bitcrush_8 | 192 | +0.050 [+0.012, +0.090] | 55.2% | left higher |
| clean−clip_18db | 192 | +0.313 [+0.247, +0.379] | 78.1% | left higher |
| clean−clip_6db | 192 | +0.037 [+0.001, +0.074] | 52.1% | left higher |
| clean−lowpass_1k | 192 | +0.289 [+0.205, +0.378] | 66.7% | left higher |
| clean−lowpass_2k | 192 | +0.473 [+0.396, +0.551] | 82.8% | left higher |
| clean−noise_snr0 | 192 | +0.366 [+0.300, +0.436] | 82.3% | left higher |
| clean−noise_snr10 | 192 | +0.177 [+0.117, +0.240] | 70.3% | left higher |
| clean−noise_snr20 | 192 | +0.080 [+0.024, +0.140] | 56.8% | left higher |
| phase8_qwen_caption2p0_slot0clean_defectlab_noq_quarter_s14159265_mc_mf25_cfg3_neg vs CFG0 | 5521 | +0.054 [+0.038, +0.069] | 53.7% | left higher |
| phase8_qwen_caption2p0_slot0clean_defectlab_noq_quarter_s14159265_mc_mf25_cfg3_neg_lvl30 vs CFG0 | 5521 | -0.026 [-0.041, -0.009] | 47.3% | right higher |
| phase8_qwen_caption2p0_slot0clean_defectlab_noq_quarter_s16180339_mc_mf25_cfg3_neg vs CFG0 | 5521 | -0.088 [-0.104, -0.071] | 44.8% | right higher |
| phase8_qwen_caption2p0_slot0clean_defectlab_noq_quarter_s27182818_mc_mf25_cfg3_neg vs CFG0 | 5521 | -0.041 [-0.057, -0.024] | 47.2% | right higher |
| phase8_qwen_caption2p0_slot0clean_defectunlab_noq_quarter_s14159265_mc_mf25_cfg3_neg vs CFG0 | 5521 | +0.151 [+0.135, +0.165] | 60.8% | left higher |
| phase8_qwen_caption2p0_slot0clean_defectunlab_noq_quarter_s14159265_mc_mf25_cfg3_neg_lvl30 vs CFG0 | 5521 | +0.099 [+0.083, +0.115] | 56.6% | left higher |
| phase8_qwen_caption2p0_slot0clean_defectunlab_noq_quarter_s16180339_mc_mf25_cfg3_neg vs CFG0 | 5521 | +0.088 [+0.074, +0.103] | 56.0% | left higher |
| phase8_qwen_caption2p0_slot0clean_defectunlab_noq_quarter_s27182818_mc_mf25_cfg3_neg vs CFG0 | 5521 | +0.104 [+0.090, +0.119] | 56.5% | left higher |
| phase8_qwen_caption2p0_slot0nmv2_nmv2pair_noq_quarter_s14159265_mc_mf25_cfg3_neg vs CFG0 | 5521 | +0.125 [+0.110, +0.139] | 59.0% | left higher |
| phase8_qwen_caption2p0_slot0nmv2_nmv2pair_noq_quarter_s16180339_mc_mf25_cfg3_neg vs CFG0 | 5521 | +0.129 [+0.114, +0.144] | 58.9% | left higher |
| phase8_qwen_caption2p0_slot0nmv2_nmv2pair_noq_quarter_s27182818_mc_mf25_cfg3_neg vs CFG0 | 5521 | +0.133 [+0.118, +0.149] | 58.5% | left higher |
| MusicFlamingo100k−10k | 5521 | +0.262 [+0.244, +0.279] | 66.5% | left higher |
| LPMC100k−10k | 5521 | +0.038 [+0.019, +0.055] | 52.0% | left higher |
| MusicFlamingo−LPMC / 10k | 5521 | -0.198 [-0.217, -0.179] | 37.4% | right higher |
| MusicFlamingo−LPMC / 100k | 5521 | +0.027 [+0.009, +0.044] | 51.3% | left higher |

## 限制

- Quarter/full slot0nm uses the same251598 rows; steps100k+50k vs400k+200k. This is a training-budget comparison.
- 10k/100k historical checkpoints also change update budgets100k+50k vs200k+100k and original corpus splits. No pure data-size causal claim.
- Historical100k scripts reference a quarantined source TSV. Checkpoints are evaluated descriptively, never retrained or promoted as clean method evidence.
- Many retained legacy cells have aggregate metrics but no modern REPORT; directory/source contracts identify protocol. Preserve historical labels, no canonical relabeling.
- No PromptCC human labels: model preference fractions are not human accuracy. More data/updates are expectations, not guaranteed winners.
- D1 degradations are preexisting paired controls; severe noise and clipping are strongest checks, stylistic effects of mild distortion are subjective.
- Pointwise prompt/source bootstrap CIs are descriptive and do not include training-seed variation or establish multiplicity-controlled adoption.
- Only retained complete cells and four selected surviving data-size checkpoints are covered. Missing historical audio is disclosed; not all historic experiments are recovered.
