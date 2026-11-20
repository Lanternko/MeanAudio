"""Score the 24 key-matched e00 synonly stimuli of 196 (12 tonics x maj/min), -30 LUFS, 16 kHz."""
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.pop('MKL_NUM_THREADS', None)
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'MeanAudio/scripts/eval'))
import numpy as np, soundfile as sf
import ce_push_196 as D
C = D.C
OUT = Path(__file__).resolve().parent
D.setup(False)
D.build_syn(D.e00_params())
names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
keys = [(t, m) for m in (0, 1) for t in range(12)]
wavs = [D.SYN[(t, m, None)] for t, m in keys]
(OUT / 'wav').mkdir(exist_ok=True)
for (t, m), w in zip(keys, wavs):
    sf.write(OUT / 'wav' / f'{t:02d}_{names[t]}_{"min" if m else "maj"}.wav', w, D.SR)
X = C.Z(); X.load_all(); C.X = X
rows = X.score(wavs, None, 2, 'empty', sr=D.SR, clap=False)
res = []
for i, (t, m) in enumerate(keys):
    r = {'key': f'{names[t]} {"minor" if m else "major"}', 'tonic': t, 'mode': m}
    r.update({k: float(v) for k, v in rows[i].items() if isinstance(v, (int, float))})
    res.append(r)
(OUT / 'synonly_24keys.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
