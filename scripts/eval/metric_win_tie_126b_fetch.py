#!/usr/bin/env python3
"""126b material fetch: famous pieces as MIDI / real performance / poor performance.

Classical (piano): MIDI from Mutopia (public domain engravings), real performances and
presumed-poor performances (beginner / child / "mistakes" uploads) from YouTube search.
Nursery (singing): CSD (Children's Song Dataset, professional singer + melody MIDI, already
downloaded) plus presumed-poor child / amateur singing from YouTube search.
Modern: MUSDB18-HQ stems and ~/karaoke-jp versions are local; nothing fetched here.

"Poor" is presumed from the search query and title only. Every fetched item is listed in
manifest.json with its query, title, url and duration so it can be listened to and vetoed.
Audio stays on /mnt/seagate (cold storage); 126b reads it from there.
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

DST = Path('/mnt/seagate/kojiek_data/126_pieces')
YTDLP = '/home/kojiek/venvs/ytdlp_new/bin/yt-dlp'  # dac's yt-dlp 2026.07.04 gets 403 on every download
# YouTube needs a JS runtime for its signature challenge since 2026; without it downloads 403
YT_JS = ['--js-runtimes', 'node:/home/kojiek/.nvm/versions/node/v24.20.0/bin/node', '--remote-components', 'ejs:github']
MUT = 'https://www.mutopiaproject.org/ftp/'
BAD_TITLE = re.compile(r'tutorial|synthesia|lesson|how to|sheet|karaoke|lyrics|midi|easy piano|slow|instrumental only'
                       r'|hour|loop|8d|remix|orchestra|guitar|violin|flute|cello|1 hour|compilation|asmr|sleep|relax|for beginners', re.I)

CLASSICAL = {
    'fur_elise': ('Für Elise Beethoven', 'BeethovenLv/WoO59/fur_Elise_WoO59/fur_Elise_WoO59.mid'),
    'bach_prelude_c': ('Bach Prelude in C major BWV 846', 'BachJS/BWV846/wtk1-prelude1/wtk1-prelude1.mid'),
    'gymnopedie_1': ('Satie Gymnopédie No. 1', 'SatieE/gymnopedie_1/gymnopedie_1.mid'),
    'nocturne_op9_2': ('Chopin Nocturne Op. 9 No. 2', 'ChopinFF/O9/chopin_nocturne_op9_n2/chopin_nocturne_op9_n2.mid'),
    'clair_de_lune': ('Debussy Clair de Lune', 'DebussyC/L75/debussy_Ste_Bergamesq_Clair/debussy_Ste_Bergamesq_Clair.mid'),
    'alla_turca': ('Mozart Rondo Alla Turca', 'MozartWA/KV331/KV331_3_RondoAllaTurca/KV331_3_RondoAllaTurca.mid'),
    'k545_1': ('Mozart Sonata K. 545 first movement', 'MozartWA/KV545/K545-1/K545-1.mid'),
    'minuet_g': ('Bach Minuet in G major Anh. 114', 'BachJS/BWVAnh114/Minuet-xpose/Minuet-xpose.mid'),
    'ode_to_joy': ('Ode to Joy piano', 'BeethovenLv/ode/ode.mid'),
}
NURSERY = {  # CSD id -> song
    'en001': 'ABC song', 'en042': 'Twinkle Twinkle Little Star', 'en029': 'Mary Had a Little Lamb',
    'en028': 'London Bridge is Falling Down', 'en038': 'Row Row Row Your Boat', 'en021': 'Jingle Bells',
    'en041': 'Itsy Bitsy Spider', 'en046': "If You're Happy and You Know It", 'en011': 'Do-Re-Mi',
    'en035': 'Somewhere Over the Rainbow',
}
PRO_Q = ['{} piano performance', '{} piano concert live']
POOR_PIANO_Q = ['{} adult beginner piano first attempt', '{} piano recital beginner student', '{} learning piano month progress',
                '{} piano beginner mistakes']
POOR_SING_Q = ['toddler singing {}', 'my daughter singing {}', 'my son singing {}', '2 year old singing {}',
               '3 year old singing {}', 'kid singing {} off key']
# a poor pick must look amateur from its title and must not be a winner, a pro, or a teaching video
POOR_OK = re.compile(r'beginner|first (time|attempt|try)|learning|month|week|days|year[s -]?old|yo\b|toddler|kid|child'
                     r'|daughter|son\b|student|recital|attempt|progress|mistake|amateur|off[- ]key|bad|terrible|fail', re.I)
POOR_BAD = re.compile(r'win|prodigy|competition|lang lang|guide|things you|how |wrong|teach|tutorial|nightmare|track by track'
                      r'|cocomelon|little baby bum|pinkfong|nursery rhymes|kids songs|super ?simple|channel', re.I)
# singing: the title must say a specific child (or an off-key singer) is singing; kids-channel productions,
# talent shows and "amazes" videos are polished performances, not poor ones
POOR_SING_OK = re.compile(r'toddler|year[s -]?old|\byo\b|daughter|\bson\b|my (kid|child|baby|little)|off[- ]key|bad sing', re.I)
POOR_SING_BAD = re.compile(r'amaz|the voice|got talent|singer|nursery|kids? songs?|sing ?along|children\'s song|for (kids|children)'
                           r'|with actions|kidsongs|lingokids|mother goose|education|worship|animated|cartoon|ukulele|guitar|piano', re.I)


def search(query, n=8):
    out = subprocess.run([YTDLP, '--no-update', '--flat-playlist', '--print', '%(id)s\t%(duration)s\t%(title)s',
                          f'ytsearch{n}:{query}'], capture_output=True, text=True, timeout=120).stdout
    rows = []
    for line in out.splitlines():
        vid, dur, title = (line.split('\t', 2) + ['', ''])[:3]
        try:
            dur = float(dur)
        except ValueError:
            continue
        rows.append({'id': vid, 'duration': dur, 'title': title})
    return rows


def pick(queries, name, n_total, seen, max_dur=600, poor=False, ok=POOR_OK, bad=POOR_BAD):
    got = []
    for q in queries:
        for r in search(q.format(name), 15 if poor else 8):
            if r['id'] in seen or not 30 <= r['duration'] <= max_dur or BAD_TITLE.search(r['title']):
                continue
            if poor and (not ok.search(r['title']) or bad.search(r['title'])):
                continue
            got.append({**r, 'query': q.format(name), 'url': f'https://www.youtube.com/watch?v={r["id"]}'})
            seen.add(r['id'])
            break
        if len(got) >= n_total:
            break
    return got


def download(item, dst):
    if dst.is_file():
        return True
    tmpl = str(dst.with_suffix('')) + '.%(ext)s'
    p = subprocess.run([YTDLP, '--no-update', *YT_JS, '-q', '-f', 'bestaudio', '-x', '--audio-format', 'flac', '-o', tmpl,
                        item['url']], capture_output=True, text=True, timeout=900)
    if p.returncode != 0 or not dst.is_file():
        item['error'] = p.stderr[-300:]
        return False
    return True


def main():
    DST.mkdir(parents=True, exist_ok=True)
    mpath = DST / 'manifest.json'
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    seen = {it['id'] for p in manifest.values() for k in ('pro', 'poor') for it in p.get(k, [])}
    for key, (name, mid) in CLASSICAL.items():
        d = DST / 'classical' / key
        d.mkdir(parents=True, exist_ok=True)
        if not (d / 'mutopia.mid').is_file():
            urllib.request.urlretrieve(MUT + mid, d / 'mutopia.mid')
        p = manifest.setdefault(key, {'genre': 'classical', 'name': name, 'midi': MUT + mid})
        if 'pro' not in p:
            p['pro'] = pick(PRO_Q, name, 2, seen)
            p['poor'] = pick(POOR_PIANO_Q, name, 3, seen, poor=True)
            mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for csd, name in NURSERY.items():
        p = manifest.setdefault(csd, {'genre': 'nursery', 'name': name, 'csd': csd})
        if 'poor' not in p:
            p['pro'] = []
            p['poor'] = pick(POOR_SING_Q, name, 3, seen, poor=True, ok=POOR_SING_OK, bad=POOR_SING_BAD)
            mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for key, p in manifest.items():
        d = DST / p['genre'] / key
        d.mkdir(parents=True, exist_ok=True)
        for kind in ('pro', 'poor'):
            for k, it in enumerate(p[kind]):
                it['file'] = str(d / f'{kind}{k + 1}.flac')
                ok = download(it, Path(it['file']))
                print(f'{key} {kind}{k + 1} {"ok" if ok else "FAIL"} {it["title"][:70]}', flush=True)
        mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
