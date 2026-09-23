#!/usr/bin/env python
"""B-line pilot: fuse the Qwen and Music Flamingo captions of the same 10 s clip into one
caption that fits the 77-token text window.

Qwen (c2p0 slot0) captions are T5-median 75 tokens (45% over 77); MF is median 194
(98.6% over), so plain concatenation truncates the second caption away. The only
viable fusion is an LLM merge under a length budget. This pilot writes the fused TSV;
score_caption_audio_alignment.py measures whether the merge is closer to the audio than
either source alone before any training is spent on it.

Rows: paired59k (same clip ids in both corpora), seed-0 sample.

    VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python \
        scripts/preprocess/fuse_qwen_mf_caption_pilot.py --n 2000 --out <dir>/fused_pilot.tsv
"""
import argparse
import csv
import random
import re

PAIRED = '/home/kojiek/exps_nvme/paired59k_mf_qwen'
QWEN_TSV = f'{PAIRED}/paired59k_qwen_slot0_train.tsv'
MF_TSV = f'{PAIRED}/arm_inputs/mf_recaption_train.tsv'   # the MF arm of paired59k

SYSTEM = (
    "You merge two descriptions of the SAME 10-second music clip, written by two different "
    "listeners, into one description for a text-to-music model.\n"
    "Rules:\n"
    "- Keep concrete audible facts: genre, instruments, timbre, vocals or their absence, "
    "tempo feel, rhythm, mood, production/recording character.\n"
    "- If the two descriptions agree, state the fact once. If a detail appears in only one "
    "and does not contradict the other, keep it. If they contradict, keep neither side.\n"
    "- Drop listening scenarios, use-case suggestions, and hedges like 'likely' or 'suggests'.\n"
    "- Do not add anything that is in neither description.\n"
    "- Write plain English prose, one paragraph, at most 45 words. Output only the description."
)


def load(tsv):
    with open(tsv, encoding='utf-8', newline='') as f:
        return {r['id']: r['caption'] for r in csv.DictReader(f, delimiter='\t')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    qwen, mf = load(QWEN_TSV), load(MF_TSV)
    ids = sorted(set(qwen) & set(mf))
    assert len(ids) == len(qwen) == len(mf), (len(ids), len(qwen), len(mf))
    ids = random.Random(a.seed).sample(ids, a.n)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    convs = [[{'role': 'system', 'content': SYSTEM},
              {'role': 'user', 'content': f"Description A: {qwen[i]}\n\nDescription B: {mf[i]}"}]
             for i in ids]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=120), use_tqdm=True)
    fused = [re.sub(r'\s+', ' ', o.outputs[0].text).strip() for o in outs]

    with open(a.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'qwen', 'mf', 'concat', 'fused', 'finish'])
        for i, c, o in zip(ids, fused, outs):
            w.writerow([i, qwen[i], mf[i], f'{qwen[i]} {mf[i]}', c, o.outputs[0].finish_reason])
    print(f'{a.out}: {len(ids)} rows; length-stopped {sum(o.outputs[0].finish_reason == "length" for o in outs)}')


if __name__ == '__main__':
    main()
