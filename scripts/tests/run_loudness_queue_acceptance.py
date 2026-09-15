#!/usr/bin/env python3
"""Legacy host fixtures, two isolated notifier routes, plus this guest's idle rearm.

Legacy instantaneous bare guests expose an idle-rearm race. Production 051
explicitly rearms at its verified seat. The final idle fixture invokes that
same production function. No live host or scheduler source is changed.
"""
import os
import subprocess
import tempfile
from pathlib import Path
ROOT=Path('/home/kojiek/MeanAudio')
with tempfile.TemporaryDirectory(prefix='loudness-queue-acceptance-') as tmp:
    tmp=Path(tmp);mock=tmp/'queue_notify.py'
    mock.write_text('import os,sys\nfrom pathlib import Path\nwith (Path(os.environ["GPU_QUEUE_ROOT"])/"notify.log").open("a") as f: f.write(" ".join(sys.argv[1:])+"\\n")\n')
    rearm=tmp/'rearm.py'
    rearm.write_text('import sys\nsys.path.insert(0,"/home/kojiek/MeanAudio/scripts/experiment_harness")\nfrom loudness_aes_cfg3_20260911_guest import rearm_queue_idle\nrearm_queue_idle()\n')
    source=Path('/home/kojiek/gpu_queue/tests/test_scheduler.sh').read_text()
    source=source.replace('ROOT="$(cd "$(dirname "$0")/.." && pwd)"','ROOT=/home/kojiek/gpu_queue')
    source=source.replace('cp "$TMP/ok.sh" "$TMP/p1/pending/090_idle_next.sh"',
        'printf \'#!/bin/bash\\n/usr/bin/python3 "%s"\\nexit 0\\n\' "$LOUDNESS_REARM_FIXTURE" > "$TMP/p1/pending/090_idle_next.sh"\nchmod +x "$TMP/p1/pending/090_idle_next.sh"')
    # Re-evaluate the count on each poll (legacy test expands it only once).
    source=source.replace('wait_for 40 [ "$(grep -c "gpu-queue-idle" "$TMP/notify.log" || true)" = "2" ]',
        'wait_for 40 bash -c \'[ "$(grep -c gpu-queue-idle "$1" || true)" = 2 ]\' -- "$TMP/notify.log"')
    test=tmp/'test.sh';test.write_text(source)
    env=dict(os.environ,GPU_NOTIFY_QUEUE_STATUS=str(mock),GPU_REAL_FIXTURE='0',LOUDNESS_REARM_FIXTURE=str(rearm))
    raise SystemExit(subprocess.run(['bash',str(test)],env=env).returncode)
