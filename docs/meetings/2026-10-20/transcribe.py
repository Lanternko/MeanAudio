"""Local meeting transcription utility; does not register or launch experiments."""
import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
MEETING_ROOT = ROOT
SOURCE = Path('/home/kojiek/.codex/attachments/3c363788-e7ad-432d-b397-caef45510143/Recording at 5-02 PM.wav')
MODEL = '/home/kojiek/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    p = ROOT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def timestamp(seconds, separator='.'):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02}{separator}{ms:03}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--breeze', action='store_true', help='Independent Mandarin Whisper review pass')
    args = parser.parse_args()
    if args.breeze:
        MODEL = '/home/kojiek/.cache/huggingface/hub/models--SoybeanMilk--faster-whisper-Breeze-ASR-25/snapshots/85be11de5f67aaa6a92e931622f1c2b55cc1dd3a'
        ROOT = MEETING_ROOT / 'breeze'
        ROOT.mkdir(exist_ok=True)
    own_lock = open(ROOT / 'transcribe.lock', 'a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'transcript.json').exists():
        raise SystemExit('Completed transcript already exists; refusing duplicate transcription.')
    gpu_lock = open('/home/kojiek/gpu_queue/gpu0.lock', 'a')
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Repeat the resource check after acquiring the queue's shared lock.
        compute = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'
        ], text=True).strip()
        if compute:
            raise RuntimeError('GPU already has a compute process; use CPU.')
        device, compute_type = 'cuda', 'float16'
    except (BlockingIOError, RuntimeError):
        gpu_lock.close()
        device, compute_type = 'cpu', 'int8'
    state = dict(status='running', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 pid=os.getpid(), purpose='User-requested meeting transcription, not a scientific evaluation',
                 source=str(SOURCE), source_sha256=digest(SOURCE),
                 recording=str(MEETING_ROOT / 'recording.flac'), recording_sha256=digest(MEETING_ROOT / 'recording.flac'),
                 model=MODEL, model_sha256=digest(Path(MODEL) / 'model.bin'),
                 faster_whisper=importlib.metadata.version('faster-whisper'),
                 ctranslate2=importlib.metadata.version('ctranslate2'),
                 device=device, compute_type=compute_type,
                 storage_free_bytes=shutil.disk_usage(ROOT).free,
                 parameters=dict(beam_size=5, vad_filter=True, condition_on_previous_text=False,
                                 word_timestamps=True, hallucination_silence_threshold=2.0,
                                 initial_prompt='以下是會議的逐字稿，請保留英文術語，中文使用繁體中文。',
                                 vad_parameters=dict(min_silence_duration_ms=700, threshold=0.2, speech_pad_ms=400)))
    save('transcription_state.json', state)
    started = time.monotonic()
    try:
        print(f'Loading {MODEL} on {device} ({compute_type})', flush=True)
        model = WhisperModel(MODEL, device=device, compute_type=compute_type,
                             cpu_threads=4, num_workers=1, local_files_only=True)
        segments, info = model.transcribe(str(MEETING_ROOT / 'recording.flac'), **state['parameters'])
        print(f'Language {info.language}, probability {info.language_probability:.3f}, duration {info.duration:.1f}s', flush=True)
        records = []
        with open(ROOT / 'segments.partial.jsonl', 'w') as partial:
            for s in segments:
                records.append(dataclasses.asdict(s))
                partial.write(json.dumps(records[-1], ensure_ascii=False) + '\n')
                partial.flush()
                state.update(last_segment_end=s.end, segments=len(records), elapsed_seconds=time.monotonic() - started)
                save('transcription_state.json', state)
                print(f'[{timestamp(s.start)} - {timestamp(s.end)}] {s.text.strip()}', flush=True)
        (ROOT / 'transcript.txt').write_text(''.join(
            f'[{timestamp(s["start"])} --> {timestamp(s["end"])}] {s["text"].strip()}\n' for s in records))
        (ROOT / 'transcript.srt').write_text(''.join(
            f'{i}\n{timestamp(s["start"], ",")} --> {timestamp(s["end"], ",")}\n{s["text"].strip()}\n\n'
            for i, s in enumerate(records, 1)))
        save('transcript.json', dict(language=info.language, language_probability=info.language_probability,
                                     duration_seconds=info.duration, segments=records))
        state.update(status='completed', elapsed_seconds=time.monotonic() - started,
                     language=info.language, language_probability=info.language_probability,
                     duration_seconds=info.duration,
                     outputs={n: digest(ROOT / n) for n in ['transcript.txt', 'transcript.srt', 'transcript.json']})
        save('transcription_state.json', state)
        (ROOT / 'segments.partial.jsonl').unlink()
        print(f'Completed {len(records)} segments in {state["elapsed_seconds"]:.1f}s', flush=True)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        save('transcription_state.json', state)
        raise
