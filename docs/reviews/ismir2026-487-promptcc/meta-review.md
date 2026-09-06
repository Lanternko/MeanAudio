# Meta-Reviewer #1 — Meta-Review

- **Paper ID:** 487
- **Paper Title:** Improving Text-to-music Generation Model Training Through Prompt-consistency Conditioning
- **Track:** Papers
- **Venue:** ISMIR 2026
- **Pre-discussion overall:** Weak accept
- **Final recommendation:** Weak accept

---

## Scores

| # | Question | Answer |
|---|----------|--------|
| 2 | Expert on the topic | Strongly agree |
| 3 | Title and abstract reflect content | Strongly agree |
| 4 | Discusses, cites, compares with relevant related work | Agree |
| 6 | Readability and organization | Strongly agree |
| 7 | Adheres to ISMIR 2026 submission guidelines | Yes |
| 8 | Relevance to ISMIR | Strongly agree |
| 9 | Scholarly/scientific quality | Strongly agree |
| 11 | Novelty | Strongly agree |
| 12 | Reproducibility details | Agree |
| 13 | Pioneering proposals | Agree (Novel topic, task, or application) |
| 14 | Reusable insights | Strongly agree |
| 16 | AI Usage Policy | Agree |
| 21 | Potential to generate discourse | Agree |
| 22 | Overall evaluation (before discussion) | Weak accept |
| 24 | **Final recommendation (after discussion)** | **Weak accept** |

---

## Q5 / Q10 / Q17

n/a

---

## Q15 — Reusable insights

The idea of measuring feature variability/ambiguity is interesting and not heavily explored, in particular within music generation.

---

## Q18 — Take-home message

The variability of outputs of music captioning models is a useful signal for text-to-music generation models.

---

## Q23 — Main review (before discussion phase)

### Summary

This paper presents an improvement to text-to-music (TTM) models when ground-truth audio-caption pairs are missing. To address the inherent ambiguity in music captioning, the authors introduce Prompt-Consistency Conditioning (PromptCC), which models how much variability a captioning model produces for a given audio clip.

### Main Comments

- Overall the paper is well written, easy to follow, and experimentally solid.
- Its great to see a paired preference listening test included!
- In the paragraph starting on line 48, the claims could be framed much more strongly. (I believe) there is no such thing as a caption that "fully captures" every detail of a musical piece, and that there is no singular "ground truth" caption for music. There are naturally tons of different, equally valid captions for a single song—as your own Brahms example demonstrates—so I recommend leaning into this as a fundamental property of music rather than just a limitation of captioning models.
- The ablation results for "PromptCC w/o quantize" show surprisingly poor performance across almost all metrics compared to the quantized version. Why do you think omitting quantization degrades the model's prompt-following ability so severely? This feels like a major finding that could serve as a valuable broader learning for other continuous conditioning frameworks, such as CLAP or PQ conditioning, which might also perform significantly better if subjected to similar quantization steps.
- The note on line 299 is highly unexpected—why do you think applying prompt consistency conditioning during the first stage of training made performance worse? The text currently just says this was discovered empirically - I'd love to see some intuition for why the extra conditioning is only helpful in the second stage but not the first.

### Minor Comments

- Define prompt consistency already within the abstract itself rather than waiting for the introduction - I didn't understand the abstract until I'd read further.
- Why not train the TTM model on all five generated caption variants during training instead of randomly sampling just one? Utilizing all generated variations could drastically improve training data diversity and further protect the model against overfitting.
- It would be interesting to dig deeper into what specific types of music or musical attributes (e.g., dense textures, abstract electronic genres, solo instruments) tend to generate low versus high caption consistency. Do you believe certain musical styles are inherently more ambiguous to describe, and do you expect this behavior to remain uniform across different captioning models beyond LP-MusicCaps?

---

## Q25 — Meta-review and final comments for authors (after discussion)

This paper presents an improvement to text-to-music (TTM) models when ground-truth audio-caption pairs are missing. To address the inherent ambiguity in music captioning, the authors introduce Prompt-Consistency Conditioning (PromptCC), which models how much variability a captioning model produces for a given audio clip.

Overall, the paper has several strengths and some important weaknesses, detailed below, and the reviewers were split, including during the discussion phase. If accepted, I strongly recommend the authors read the reviews carefully and integrate feedback into the camera ready version.

### Main Strengths

- leveraging the stochastic agreement of an automatic captioning model as an auxiliary training signal—rather than utilizing it solely for rigid data filtering—is an intuitive, creative, and valuable perspective for unsupervised TTM training.
- The inclusion of a paired preference listening test provides solid qualitative evidence that the proposed framework improves over the baseline.
- The paper is generally well-structured, and easy to follow.

### Main Weaknesses

- Because the consistency score is computed entirely in a text-embedding space, it captures text-space agreement rather than true audio-grounded correctness. It remains unclear if high consistency represents a reliable label or merely a stable error/bias inherent to the captioning model.
- The "hard filtering" baseline discards 53% of the training dataset. As a result, its lower performance may simply stem from a lack of data volume rather than demonstrating the conceptual superiority of PromptCC over data-filtering strategies
- the evaluation is restricted to a single text-to-music backbone and a single captioner. Additionally, despite claiming perceptual improvements, the authors did not provide a demo page or audio examples for external verification
