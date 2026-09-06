#!/usr/bin/env python3
"""Concept Coverage Score (CCS), ATTM arXiv 2605.21538 Eq. 1-2.

    D(x,t) = 1 if logit("Yes") > logit("No") else 0
    CCS    = (1/3N) * sum_i sum_{t in T_i} D(x_i, t)

Two deviations from the paper, both forced and both reported in the output JSON:

  judge   ATTM uses Qwen3-Omni-30B-A3B. That needs ~79 GB VRAM in bf16; this
          box is a single 32 GB RTX 5090 that is usually shared. We use the
          Qwen2.5-Omni-3B thinker instead. Absolute CCS is therefore NOT
          comparable to their table -- only rankings between our own arms, and
          the gap to the MeanAudio-S-Full topline we score ourselves, are.
  concepts ATTM synthesises (genre, instrument, mood) triplets and evaluates on
          those 100 prompts. Those prompts were never released, so we take the
          concepts from MusicCaps' human aspect_list, mapped onto a curated
          taxonomy, and evaluate on the instrumental MusicCaps subset.

The paper's tag-verifiability filter (criterion 2: recall >= 0.85 on ground
truth) IS reproduced -- see --calibrate, which runs the judge over the real
reference clips and drops tags the judge cannot detect even when they are
truly present.
"""
import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ATTM = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm')
MUSICCAPS_CSV = ATTM / 'musiccaps-public.csv'
REF_DIR = ATTM / 'musiccaps_instrumental_ref'
MODEL_ID = 'Qwen/Qwen2.5-Omni-3B'
TAXONOMY_PATH = ATTM / 'ccs_taxonomy.json'

VOCAL_RE = re.compile(
    r"\b(vocal|vocals|vocalist|vocalists|vocalisation|vocalization|singer|singers|"
    r"singing|sings|sung|sing|voice|voices|choir|chorus|chant|chanting|rap|rapper|"
    r"rapping|lyric|lyrics|acappella|harmonies|humming|hums|falsetto|soprano|tenor|"
    r"baritone|spoken|speaks|speech|narrate|narration|narrator)\b", re.I)

# Curated (genre, instrument, mood) taxonomy over MusicCaps aspects that occur
# at least 30x in the instrumental subset. Recording-quality aspects ("low
# quality", "mono", "amateur recording") are deliberately excluded: ATTM's CCS
# covers musical concepts only, and those terms are what the negative prompt
# manipulates, so scoring them would confound the two effects.
CATEGORIES = {
    'genre': ['classical', 'rock', 'jazz', 'electronic music', 'ambient', 'pop',
              'folk', 'blues', 'reggae', 'country', 'hip hop', 'funk', 'latin',
              'classical music', 'electronic', 'orchestral', 'dance', 'metal',
              'punk', 'soul', 'techno', 'house', 'edm', 'disco', 'march'],
    'instrument': ['electric guitar', 'acoustic guitar', 'piano', 'percussion',
                   'bass guitar', 'acoustic drums', 'electronic drums', 'e-bass',
                   'e-guitar', 'strings', 'orchestra', 'keyboard', 'synthesizer',
                   'violin', 'flute', 'saxophone', 'trumpet', 'drums', 'guitar',
                   'organ', 'harmonica', 'cello', 'banjo', 'ukulele', 'brass',
                   'woodwind', 'harp', 'accordion', 'sitar', 'marimba', 'bells',
                   'tambourine', 'congas', 'bongos', 'shaker', 'hi hats', 'snare',
                   'kick', 'cymbals', 'bass', 'keyboard harmony', 'synth'],
    'mood': ['passionate', 'emotional', 'energetic', 'groovy', 'spirited',
             'intense', 'relaxing', 'calming', 'happy', 'exciting', 'upbeat',
             'lively', 'mellow', 'cheerful', 'youthful', 'easygoing', 'soothing',
             'suspenseful', 'romantic', 'melancholic', 'fun', 'aggressive',
             'dark', 'dreamy', 'epic', 'playful', 'sad', 'uplifting', 'nostalgic',
             'tense', 'peaceful', 'meditative', 'hypnotic', 'triumphant'],
}
CANON = {t: cat for cat, tags in CATEGORIES.items() for t in tags}

PROMPTS = {
    'genre': ('You are a music genre classifier. Listen to the audio clip and '
              'answer with a single word. Is the genre of this music "{tag}"? '
              'Answer Yes or No.'),
    'instrument': ('You are an audio event detector. Listen to the audio clip and '
                   'answer with a single word. Can you hear any trace of "{tag}" '
                   'in this music? Answer Yes or No.'),
    'mood': ('You are a music mood classifier. Listen to the audio clip and '
             'answer with a single word. Would you describe the mood of this '
             'music as "{tag}"? Answer Yes or No.'),
}


def build_targets(min_count=30, max_per_category=1):
    """Per-clip target concepts from the human aspect_list, one per category."""
    rows = list(csv.DictReader(MUSICCAPS_CSV.open()))
    rows = [r for r in rows if not VOCAL_RE.search(r['caption'])]
    counts = Counter()
    parsed = {}
    for r in rows:
        cid = f"{r['ytid']}_{r['start_s']}"
        aspects = [a.strip().lower() for a in ast.literal_eval(r['aspect_list'])]
        parsed[cid] = aspects
        counts.update(a for a in aspects if a in CANON)
    keep = {t for t, n in counts.items() if n >= min_count}

    targets = {}
    for cid, aspects in parsed.items():
        by_cat = defaultdict(list)
        for a in aspects:
            if a in keep:
                by_cat[CANON[a]].append(a)
        picked = [v[0] for cat in ('genre', 'instrument', 'mood')
                  for v in [by_cat.get(cat, [])[:max_per_category]] if v]
        if picked:
            targets[cid] = picked
    return targets, keep, counts


class Judge:
    def __init__(self):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
        self.proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map='cuda',
            attn_implementation='sdpa').eval()   # sm_120 dev build has no FA2
        tok = self.proc.tokenizer
        # first-token ids for the two answers, matching Eq. 1's Yes/No logits
        self.yes = [tok.encode(s, add_special_tokens=False)[0] for s in ('Yes', ' Yes', 'yes')]
        self.no = [tok.encode(s, add_special_tokens=False)[0] for s in ('No', ' No', 'no')]

    @torch.no_grad()
    def detect(self, audio, sr, tag, category):
        text = PROMPTS[category].format(tag=tag)
        conv = [{'role': 'user', 'content': [{'type': 'audio', 'audio': audio},
                                             {'type': 'text', 'text': text}]}]
        prompt = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=prompt, audio=[audio], sampling_rate=sr,
                           return_tensors='pt', padding=True)
        inputs = {k: (v.to('cuda') if torch.is_tensor(v) else v) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0, -1].float()
        return float(max(logits[i] for i in self.yes)), float(max(logits[i] for i in self.no))


def load_audio(path, sr=16000):
    import librosa
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-dir', type=Path, help='generated clips (.flac) to score')
    ap.add_argument('--label', default=None)
    ap.add_argument('--calibrate', action='store_true',
                    help='run over the real reference wavs and write the tag '
                         'verifiability filter (ATTM criterion 2, recall >= 0.85)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-recall', type=float, default=0.85)
    args = ap.parse_args()

    targets, keep, counts = build_targets()
    print(f'clips with targets: {len(targets)}   candidate tags: {len(keep)}', flush=True)

    verifiable = None
    if TAXONOMY_PATH.exists() and not args.calibrate:
        verifiable = set(json.loads(TAXONOMY_PATH.read_text())['verifiable_tags'])
        print(f'verifiable tags (calibrated): {len(verifiable)}', flush=True)

    judge = Judge()

    if args.calibrate:
        hits, total = Counter(), Counter()
        items = sorted(targets)
        if args.limit:
            items = items[:args.limit]
        for n, cid in enumerate(items, 1):
            path = REF_DIR / f'{cid}.wav'
            if not path.exists():
                continue
            wav = load_audio(path)
            for tag in targets[cid]:
                y, no = judge.detect(wav, 16000, tag, CANON[tag])
                total[tag] += 1
                hits[tag] += int(y > no)
            if n % 100 == 0:
                print(f'  calibrated {n}/{len(items)}', flush=True)
        recall = {t: hits[t] / total[t] for t in total if total[t] >= 10}
        verifiable = sorted(t for t, r in recall.items() if r >= args.min_recall)
        TAXONOMY_PATH.write_text(json.dumps({
            'judge': MODEL_ID, 'min_recall': args.min_recall,
            'n_clips': len(items),
            'recall': {t: round(r, 4) for t, r in sorted(recall.items(), key=lambda kv: -kv[1])},
            'support': dict(total), 'verifiable_tags': verifiable}, indent=1))
        print(f'\nverifiable tags: {len(verifiable)}/{len(recall)} '
              f'(recall >= {args.min_recall})  -> {TAXONOMY_PATH}', flush=True)
        return

    assert args.audio_dir, 'need --audio-dir unless --calibrate'
    label = args.label or args.audio_dir.name
    detected = total = 0
    per_tag = defaultdict(lambda: [0, 0])
    per_cat = defaultdict(lambda: [0, 0])
    items = sorted(targets)
    if args.limit:
        items = items[:args.limit]
    for n, cid in enumerate(items, 1):
        path = args.audio_dir / f'{cid}.flac'
        if not path.exists():
            continue
        tags = [t for t in targets[cid] if verifiable is None or t in verifiable]
        if not tags:
            continue
        wav = load_audio(path)
        for tag in tags:
            y, no = judge.detect(wav, 16000, tag, CANON[tag])
            d = int(y > no)
            detected += d
            total += 1
            per_tag[tag][0] += d
            per_tag[tag][1] += 1
            per_cat[CANON[tag]][0] += d
            per_cat[CANON[tag]][1] += 1
        if n % 200 == 0:
            print(f'  {n}/{len(items)}  running CCS_micro {detected/max(total,1):.4f}', flush=True)

    out = {
        'label': label,
        'audio_dir': str(args.audio_dir),
        'judge': MODEL_ID,
        'judge_deviation': 'ATTM uses Qwen3-Omni-30B-A3B (~79GB VRAM); not runnable here',
        'concept_source': 'MusicCaps human aspect_list, curated taxonomy',
        'n_clips_scored': sum(1 for c in items if (args.audio_dir / f'{c}.flac').exists()),
        'n_concept_judgements': total,
        'CCS_micro': detected / total if total else None,
        # equal-weight over genre/instrument/mood. The micro average is
        # dominated by mood (MusicCaps annotators emit far more mood aspects
        # than genre ones), and mood is the category an LALM judge is least
        # reliable on, so the macro number is the one to compare arms with.
        'CCS_macro': (float(np.mean([v[0] / v[1] for v in per_cat.values()]))
                      if per_cat else None),
        'CCS_by_category': {c: {'rate': v[0] / v[1], 'n': v[1]}
                            for c, v in sorted(per_cat.items())},
        'per_tag_detection': {t: {'rate': v[0] / v[1], 'n': v[1]}
                              for t, v in sorted(per_tag.items())},
    }
    (ATTM / f'ccs_{label}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_tag_detection'}, indent=1))


if __name__ == '__main__':
    main()
