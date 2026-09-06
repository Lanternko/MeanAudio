#!/usr/bin/env python
"""Rewrite Caption 2.0 per-segment captions into a fixed modular template.

Motivation: `docs/experiments/results/phase8/fulltrack_pq_gap_mechanism_2026_08_28.md`
found that the only difference surviving every control between the fulltrack corpus
(AES-strong) and Caption 2.0 (AES-weak) is how narrow and templated the training text
distribution is. This script isolates that variable: it keeps each row's own content
(genre, tempo, instruments, mood, production) but re-renders it into one fixed frame,
so form narrows while per-segment content is preserved.

Slots that find no match get a fixed filler so every row has identical structure.

Usage:
  python scripts/preprocess/build_modular_template_captions.py --out <tsv> [--limit N]
"""
import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

SRC = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv')

GENRES = [
    'drum and bass', 'hip hop', 'trip hop', 'lo-fi', 'new age', 'post rock', 'indie funk',
    'electronic', 'classical', 'orchestral', 'ambient', 'acoustic', 'rock', 'metal', 'punk',
    'jazz', 'blues', 'folk', 'country', 'reggae', 'funk', 'soul', 'pop', 'techno', 'house',
    'trance', 'dubstep', 'disco', 'latin', 'flamenco', 'cinematic', 'soundtrack', 'chiptune',
    'experimental', 'industrial', 'gospel', 'world', 'celtic', 'bossa nova', 'swing', 'ska',
]
TEMPO = [
    (r'\b(fast[- ]paced|fast tempo|uptempo|up[- ]tempo|rapid|frantic|frenetic|driving|'
     r'high[- ]energy|quick tempo|fast)\b', 'fast'),
    (r'\b(upbeat|lively|bouncy|brisk|dance[- ]?able|groovy tempo)\b', 'upbeat'),
    (r'\b(slow[- ]paced|slow tempo|languid|sluggish|downtempo|leisurely|gentle pace|'
     r'sparse and slow|slow)\b', 'slow'),
    (r'\b(moderate|mid[- ]tempo|medium[- ]tempo|medium|steady|relaxed|laid[- ]back|'
     r'measured)\b', 'moderate'),
]
INSTRUMENTS = [
    'electric guitar', 'acoustic guitar', 'bass guitar', 'classical guitar', 'slide guitar',
    'grand piano', 'electric piano', 'piano', 'organ', 'harpsichord', 'accordion',
    'synthesizer', 'synth', 'drum machine', 'drum kit', 'drums', 'percussion', 'marimba',
    'xylophone', 'vibraphone', 'bass', 'double bass', 'cello', 'violin', 'viola', 'strings',
    'harp', 'flute', 'clarinet', 'oboe', 'saxophone', 'trumpet', 'trombone', 'brass',
    'banjo', 'mandolin', 'ukulele', 'sitar', 'bells', 'chimes', 'pad', 'arpeggiator',
    'guitar', 'keyboard', 'keys', 'drum', 'drumbeat', 'drum beat', 'snare', 'hi-hat',
    'hi hat', 'kick', 'cymbal', 'tambourine', 'conga', 'bongo', 'shaker', 'harmonica',
    'woodwind', 'horn', 'french horn', 'tuba', 'bassoon', 'lute', 'koto', 'kalimba',
    'steel drum', 'timpani', 'gong', 'triangle', 'whistle', 'strings section', 'choir pad',
]
MOODS = [
    'melancholic', 'contemplative', 'meditative', 'serene', 'reflective', 'dreamy', 'ethereal',
    'nostalgic', 'somber', 'dark', 'ominous', 'tense', 'aggressive', 'intense', 'energetic',
    'joyful', 'cheerful', 'carefree', 'playful', 'uplifting', 'triumphant', 'hopeful',
    'romantic', 'sensual', 'calm', 'relaxed', 'chill', 'peaceful', 'mysterious', 'dramatic',
    'epic', 'groovy', 'funky', 'warm', 'bright', 'spacious', 'soothing', 'haunting',
    'whimsical', 'urgent', 'anthemic', 'introspective', 'sombre', 'wistful', 'gritty',
    'laid-back', 'atmospheric', 'immersive', 'powerful', 'delicate', 'lush', 'raw',
    'sad', 'happy', 'emotional', 'cinematic mood', 'suspenseful', 'menacing', 'tranquil',
]
TEXTURE = ['sparse', 'dense', 'layered', 'lush', 'minimal', 'full-bodied', 'rich',
           'thick', 'airy', 'stripped-back', 'busy', 'spacious arrangement', 'repetitive']
RHYTHM = ['steady', 'syncopated', 'driving rhythm', 'loose', 'shuffling', 'swung',
          'four-on-the-floor', 'pulsing', 'staccato', 'flowing', 'marching', 'polyrhythmic']
DYNAMICS = ['soft', 'loud', 'dynamic', 'consistent', 'building', 'swelling', 'restrained',
            'explosive', 'gradual', 'punchy']
SPACE = ['reverb', 'dry', 'spacious', 'intimate', 'wide', 'echo', 'ambient space',
         'close-miked', 'cavernous', 'roomy']
TIMBRE = ['warm', 'bright', 'dark', 'crisp', 'distorted', 'clean', 'gritty', 'smooth',
          'fuzzy', 'shimmering', 'muted', 'sharp', 'mellow', 'metallic']

PROD_GOOD = re.compile(
    r'\b(high[- ]quality|excellent|impressive|professional|polished|pristine|clear|'
    r'well[- ]balanced|well[- ]mixed|balanced|crisp|good recording)\b', re.I)
PROD_BAD = re.compile(
    r'\b(low[- ]quality|poor|amateur|muddy|noisy|noise|distortion|distorted|muffled|'
    r'lo-fi|rough|thin|harsh|clipping)\b', re.I)

FILLER = '\x00EMPTY\x00'   # sentinel; sentences containing it are dropped, never emitted

# A bank of frames, not a single one. Calibration target is the fulltrack corpus
# (trigram repeat ~89%, opening-4gram variety ~0.03, pairwise Jaccard ~0.20): narrow
# enough to matter, not so narrow that every row opens identically -- a single frame
# overshoots to 99% repeat / 0.0001 opening variety, which is the shape that collapsed
# CLAP in the Phase 8 V4 `[consistency=]` prefix experiment.
FRAMES = [
    'The genre of the music is {genre}. The tempo is {tempo}. The instrumentation includes '
    '{instruments}. The rhythm is {rhythm} and the dynamics are {dynamics}. The texture is '
    '{texture} with a {timbre} timbre and a {space} sense of space. The mood is {mood}. '
    'The production is {production}.',
    'The music belongs to the genres {genre}. It is played at a {tempo} tempo and features '
    '{instruments}. The rhythm is {rhythm}, the dynamics are {dynamics}, and the texture is '
    '{texture}. The timbre is {timbre} and the space is {space}. The atmosphere is {mood}, '
    'and the production is {production}.',
    'This is a {tempo} {genre} piece. {Instruments_cap} carry the arrangement. The rhythm is '
    '{rhythm} with {dynamics} dynamics and a {texture} texture. The timbre is {timbre}, the '
    'space is {space}. It conveys a {mood} feeling, and the recording is {production}.',
    'The track is {genre} at a {tempo} tempo. The arrangement is built around {instruments}, '
    'with a {rhythm} rhythm and {dynamics} dynamics. The texture is {texture}, the timbre '
    '{timbre}, the space {space}. It has a {mood} character. The production is {production}.',
    'A {mood} {genre} recording. The tempo is {tempo}, and the instrumentation includes '
    '{instruments}. The rhythm is {rhythm} and the dynamics are {dynamics}. The texture is '
    '{texture}, the timbre is {timbre}, and the space is {space}. The production is {production}.',
    'The music is an amalgamation of {genre}, taken at a {tempo} tempo. {Instruments_cap} '
    'dominate the mix. The rhythm is {rhythm}, the dynamics {dynamics}, the texture {texture}. '
    'The timbre is {timbre} and the space is {space}. The overall mood is {mood} and the '
    'production is {production}.',
    'Instrumentation: {instruments}. The genre reads as {genre} and the tempo is {tempo}. '
    'The rhythm is {rhythm}, the dynamics are {dynamics}, the texture is {texture}. '
    'The timbre is {timbre} and the space is {space}. The piece feels {mood}, with '
    '{production} production.',
    'A {genre} arrangement featuring {instruments}. Taken at a {tempo} tempo, with a {rhythm} '
    'rhythm and {dynamics} dynamics. The texture is {texture}, the timbre {timbre}, the space '
    '{space}. It comes across as {mood}. The production is {production}.',
]


def _first_matches(text, vocab, limit):
    """Longest-first vocabulary match, returned in order of first appearance."""
    hits = []
    for term in sorted(vocab, key=len, reverse=True):
        m = re.search(r'\b' + re.escape(term) + r'\b', text)
        if m and not any(term in h[1] or h[1] in term for h in hits):
            hits.append((m.start(), term))
    hits.sort()
    return [t for _, t in hits[:limit]]


def _join(items):
    if not items:
        return FILLER
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def extract(caption):
    """Pull the slot values out of a Caption 2.0 row. Content only, no formatting."""
    low = caption.lower()
    tempo = ''
    for pattern, name in TEMPO:
        if re.search(pattern, low):
            tempo = name
            break
    bpm = re.search(r'(\d{2,3}(?:\.\d+)?)\s*bpm', low)
    if bpm:
        tempo = f'{tempo} {bpm.group(1)} BPM'.strip()
    good, bad = bool(PROD_GOOD.search(low)), bool(PROD_BAD.search(low))
    return {
        'genre': _first_matches(low, GENRES, 2),
        'tempo': [tempo] if tempo else [],
        'instruments': _first_matches(low, INSTRUMENTS, 5),
        'mood': _first_matches(low, MOODS, 3),
        'texture': _first_matches(low, TEXTURE, 2),
        'rhythm': _first_matches(low, RHYTHM, 2),
        'dynamics': _first_matches(low, DYNAMICS, 2),
        'space': _first_matches(low, SPACE, 2),
        'timbre': _first_matches(low, TIMBRE, 2),
        'production': (['clear and balanced'] if good and not bad else
                       ['rough and unpolished'] if bad and not good else
                       ['clear but uneven'] if good and bad else []),
    }


# When a slot finds nothing, keep the sentence and say so, rather than dropping it.
# Dropping empty sentences halved caption length (22 vs 44 words), which collapsed
# trigram entropy far below the fulltrack calibration target.
FALLBACKS = {
    'genre': ['not clearly defined', 'hard to place', 'ambiguous', 'not obvious'],
    'tempo': ['not clearly marked', 'hard to pin down', 'ambiguous', 'not steady enough to name'],
    'instruments': ['sounds that are hard to identify', 'unidentified sources',
                    'textures without a clear source', 'indistinct instrumentation'],
    'mood': ['hard to characterise', 'not strongly coloured', 'neutral', 'ambiguous'],
    'texture': ['unremarkable', 'hard to characterise', 'neither sparse nor dense', 'plain'],
    'rhythm': ['not clearly marked', 'loose', 'hard to follow', 'unemphatic'],
    'dynamics': ['fairly flat', 'unremarkable', 'hard to characterise', 'even'],
    'space': ['not clearly defined', 'unremarkable', 'hard to judge', 'plain'],
    'timbre': ['hard to characterise', 'unremarkable', 'neutral', 'plain'],
    'production': ['hard to judge', 'unremarkable', 'neither clean nor rough', 'plain'],
}


def rewrite(caption, clip_id):
    slots = extract(caption)
    digest = hashlib.sha256(str(clip_id).encode()).hexdigest()
    frame = FRAMES[int(digest[:8], 16) % len(FRAMES)]
    filled = {}
    for i, (k, v) in enumerate(sorted(slots.items())):
        if v:
            filled[k] = _join(v)
        else:
            bank = FALLBACKS[k]
            filled[k] = bank[int(digest[8 + i * 2:10 + i * 2], 16) % len(bank)]
    filled['Instruments_cap'] = filled['instruments'][:1].upper() + filled['instruments'][1:]
    out = frame.format(**filled)
    return re.sub(r'\s{2,}', ' ', out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    with args.src.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if args.limit:
        rows = rows[:args.limit]

    fields = list(rows[0].keys())
    filled = {k: 0 for k in ('genre', 'tempo', 'instruments', 'mood', 'texture', 'rhythm', 'dynamics', 'space', 'timbre', 'production')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for row in rows:
            new = rewrite(row['caption'], row['id'])
            for slot, values in extract(row['caption']).items():
                if values:
                    filled[slot] += 1
            writer.writerow({**row, 'caption': new})

    n = len(rows)
    print(f'wrote {n} rows -> {args.out}')
    for slot, count in filled.items():
        print(f'  slot {slot:12s} filled {count:6d} ({count / n * 100:5.1f}%)')


if __name__ == '__main__':
    sys.exit(main())
