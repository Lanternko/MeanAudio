# ISMIR 2026 · Paper 487 — Reviews Archive

**Title:** Improving Text-to-music Generation Model Training Through Prompt-consistency Conditioning  
**Author (CMT):** Tzu-Chieh Ko  
**Track:** Papers  
**Project:** MeanAudio  
**Archived:** 2026-07-15

---

## Location

```
MeanAudio/docs/reviews/ismir2026-487-promptcc/
```

Shortcut: [`../latest-487`](../latest-487)

---

## Links

### Paper

| Item | Path |
|------|------|
| PDF (symlink) | [paper.pdf](./paper.pdf) |
| PDF (local copy) | [Prompt_consistency_Conditioning_ISMIR2026.pdf](./Prompt_consistency_Conditioning_ISMIR2026.pdf) |
| PDF (ISMIR project root) | [../../../../Prompt_consistency_Conditioning_ISMIR2026.pdf](../../../../Prompt_consistency_Conditioning_ISMIR2026.pdf) |
| LaTeX source | [paper.tex](./paper.tex) → `ISMIR2026_meanaudio.tex` |

### Reviews (4)

| Reviewer | Overall / Final | File |
|----------|-----------------|------|
| Reviewer #1 | Weak reject | [R1.md](./R1.md) |
| Reviewer #2 | Weak reject | [R2.md](./R2.md) |
| Reviewer #3 | Weak accept | [R3.md](./R3.md) |
| Meta-Reviewer #1 | Weak accept (pre + post discussion) | [meta-review.md](./meta-review.md) |

### Follow-up analysis

| Item | Status | File |
|------|--------|------|
| Correctness validation plan | Human evaluation deferred; conservative claim and no-human diagnostics recorded | [CORRECTNESS_VALIDATION_PLAN.md](./CORRECTNESS_VALIDATION_PLAN.md) |
| Literature quality + gap survey (2026-07-20) | Article quality ratings, reviewer→paper mapping, CosyAudio/QA-MDT/MR-FlowDPO/Ding 一手補調查, writing+experiment checklist | [../../literature/PromptCC_Literature_Quality_and_Gaps_2026_07_20.md](../../literature/PromptCC_Literature_Quality_and_Gaps_2026_07_20.md) |

### Absolute path

```
/Users/kojiek/Documents/ISMIR/MeanAudio/docs/reviews/ismir2026-487-promptcc/
├── README.md
├── paper.pdf          → ../../../../Prompt_consistency_Conditioning_ISMIR2026.pdf
├── paper.tex          → ../../../../ISMIR2026_meanaudio.tex
├── Prompt_consistency_Conditioning_ISMIR2026.pdf
├── R1.md
├── R2.md
├── R3.md
└── meta-review.md
```

---

## Score summary

| Reviewer | Overall | Key stance |
|----------|---------|------------|
| R1 | WR | q 機制不清；客觀增益小且混雜；缺 demo / multi-seed |
| R2 | WR | 增量貢獻；單 backbone + 單 captioner；generality 不足 |
| R3 | WA | 方法清楚；minor（citation、demographics、q≥5） |
| Meta | WA → WA | split reviewers；若 accept 務必 camera-ready 整合 feedback |

### Meta-reviewer consensus weaknesses (post-discussion)

1. Consistency score = text-space agreement，非 audio-grounded correctness  
2. Hard filtering 砍 53% data，不公平 baseline  
3. 單一 TTM backbone + 單一 captioner；無 demo / audio examples  

### Meta-reviewer camera-ready / discussion highlights

- 把 music caption multi-validity 當 fundamental property（非僅 captioner limitation）  
- 解釋 w/o quantize 為何崩得很兇（可能是 broader continuous-conditioning 洞見）  
- 解釋為何 PromptCC 只在 stage-2 有效  
- Abstract 先定義 prompt consistency  
- 考慮用全部 5 captions 訓練而非 random sample one  
- 分析 low/high-q 與音樂類型 / 屬性的關係  

---

## Related MeanAudio paths

- Project root: [`../../../`](../../../)
- Experiments log: [`../../../EXPERIMENT_LOG.md`](../../../EXPERIMENT_LOG.md)
- Subjective prompts: [`../../eval/subjective_prompts.md`](../../eval/subjective_prompts.md)
