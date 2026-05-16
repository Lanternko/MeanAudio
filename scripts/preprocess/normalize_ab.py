"""Peak-normalize all WAVs in eval_output/subjective_ab/audio to -1 dBFS."""
import soundfile as sf
import numpy as np
from pathlib import Path

OUT = Path("eval_output/subjective_ab/audio")
TARGET_DBFS = -1.0
TARGET_PEAK = 10 ** (TARGET_DBFS / 20.0)

wavs = sorted(OUT.glob("*.wav"))
print(f"Normalizing {len(wavs)} files to {TARGET_DBFS} dBFS...")

for wav in wavs:
    data, sr = sf.read(wav, always_2d=False)
    peak = np.max(np.abs(data))
    if peak < 1e-9:
        print(f"  SKIP (silent): {wav.name}")
        continue
    gain = TARGET_PEAK / peak
    data = data * gain
    sf.write(wav, data, sr, subtype="PCM_16")
    peak_before_db = 20 * np.log10(peak)
    print(f"  {wav.name}: {peak_before_db:+.2f} dBFS -> {TARGET_DBFS:+.2f} dBFS (x{gain:.3f})")

print("Done.")
