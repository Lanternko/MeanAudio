"""
Download MusicCaps reference audio for FAD computation.

Reads musiccaps-public.csv, downloads each 10s YouTube segment as 16kHz mono WAV
to /mnt/HDD/kojiek/musiccaps_reference/{ytid}_{start_s}.wav.

Skips already-downloaded files. Logs failures to a TSV. Parallel workers.

Usage:
    python download_musiccaps_reference.py \
        [--csv <path>] [--out_dir <path>] [--workers 8] [--limit N]
"""
import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_CSV = "/home/kojiek/.cache/huggingface/hub/datasets--google--MusicCaps/snapshots/0a51889b340037bb75a9a0858af2e4ece21f7f89/musiccaps-public.csv"
DEFAULT_OUT = "/mnt/HDD/kojiek/musiccaps_reference"


def download_one(ytid: str, start_s: int, end_s: int, out_dir: Path, timeout: int = 90) -> tuple[str, bool, str]:
    """Download a single 10s segment. Returns (id, ok, msg)."""
    clip_id = f"{ytid}_{start_s}"
    final_wav = out_dir / f"{clip_id}.wav"
    if final_wav.exists() and final_wav.stat().st_size > 0:
        return clip_id, True, "skip-exists"

    url = f"https://www.youtube.com/watch?v={ytid}"
    tmp_prefix = out_dir / f".tmp_{clip_id}"

    # yt-dlp: extract audio, slice to [start_s, end_s], output as m4a
    cmd_dl = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "-q",
        "--no-warnings",
        "--no-playlist",
        "--socket-timeout", "20",
        "--retries", "2",
        "--sleep-requests", "1.5",
        "--sleep-interval", "2",
        "--max-sleep-interval", "6",
        "--download-sections", f"*{start_s}-{end_s}",
        "--force-keyframes-at-cuts",
        "-o", f"{tmp_prefix}.%(ext)s",
        url,
    ]
    try:
        r = subprocess.run(cmd_dl, capture_output=True, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        _clean_tmp(out_dir, clip_id)
        return clip_id, False, "timeout"
    if r.returncode != 0:
        _clean_tmp(out_dir, clip_id)
        msg = r.stderr.strip().replace("\n", " | ")[:200]
        return clip_id, False, f"dl_fail:{msg}"

    # Find the tmp file yt-dlp produced
    tmp_files = list(out_dir.glob(f".tmp_{clip_id}.*"))
    if not tmp_files:
        return clip_id, False, "no_output_file"
    tmp_audio = tmp_files[0]

    # ffmpeg → 16kHz mono wav
    cmd_ff = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_audio),
        "-ar", "16000", "-ac", "1",
        "-t", "10",
        str(final_wav),
    ]
    try:
        r = subprocess.run(cmd_ff, capture_output=True, timeout=30, text=True)
    except subprocess.TimeoutExpired:
        _clean_tmp(out_dir, clip_id)
        return clip_id, False, "ffmpeg_timeout"
    finally:
        try:
            tmp_audio.unlink()
        except FileNotFoundError:
            pass

    if r.returncode != 0 or not final_wav.exists() or final_wav.stat().st_size == 0:
        try:
            final_wav.unlink()
        except FileNotFoundError:
            pass
        return clip_id, False, f"ffmpeg_fail:{r.stderr.strip()[:200]}"

    return clip_id, True, "ok"


def _clean_tmp(out_dir: Path, clip_id: str):
    for p in out_dir.glob(f".tmp_{clip_id}.*"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--failure_log", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ytid = row["ytid"]
            start_s = int(float(row["start_s"]))
            end_s = int(float(row["end_s"]))
            tasks.append((ytid, start_s, end_s))
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"[dl] {len(tasks)} total tasks; out_dir={out_dir}")

    # Skip ones already present
    done_ids = {p.stem for p in out_dir.glob("*.wav") if p.stat().st_size > 0}
    pending = [t for t in tasks if f"{t[0]}_{t[1]}" not in done_ids]
    print(f"[dl] already have {len(done_ids)} wav files; {len(pending)} pending")

    failure_log = Path(args.failure_log) if args.failure_log else (out_dir / "failures.tsv")
    ok_count = skip_count = fail_count = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(failure_log, "a") as flog:
        futures = {ex.submit(download_one, *t, out_dir): t for t in pending}
        for i, fut in enumerate(as_completed(futures)):
            clip_id, ok, msg = fut.result()
            if ok:
                if msg == "skip-exists":
                    skip_count += 1
                else:
                    ok_count += 1
            else:
                fail_count += 1
                flog.write(f"{clip_id}\t{msg}\n")
                flog.flush()
            if (i + 1) % 50 == 0 or (i + 1) == len(pending):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1)
                remaining = (len(pending) - (i + 1)) / max(rate, 1e-6)
                print(
                    f"[dl] {i+1}/{len(pending)}  ok={ok_count} skip={skip_count} fail={fail_count}  "
                    f"{rate:.2f}/s  ETA={remaining/60:.1f} min",
                    flush=True,
                )

    print(f"\n[dl] DONE. ok={ok_count} skip={skip_count} fail={fail_count} in {(time.time()-t0)/60:.1f} min")
    print(f"[dl] Failures logged to: {failure_log}")


if __name__ == "__main__":
    main()
