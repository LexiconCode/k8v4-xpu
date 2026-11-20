#!/usr/bin/env python3
"""GPU busy share of the candidate container (vllm-exp2) between decode_step_timer.py's hostprof.start / hostprof.stop markers,
no profiler: xe driver per-client counters in /proc/<pid>/fdinfo (drm-cycles-ccs / drm-total-cycles-ccs), as
/root/rice/gpu_busy.py does for prod. Run as root on the inference guest. usage: gpu_busy_fdinfo.py MARKER_DIR [CONTAINER]"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

D = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "vllm-exp2"


def files():
    pids = subprocess.run(["docker", "top", NAME, "-eo", "pid"], capture_output=True, text=True).stdout.split()[1:]
    out = []
    for p in pids:
        try:
            for fd in os.listdir("/proc/%s/fd" % p):
                if os.readlink("/proc/%s/fd/%s" % (p, fd)).startswith("/dev/dri/"):
                    t = open("/proc/%s/fdinfo/%s" % (p, fd)).read()
                    if "drm-cycles-ccs" in t:
                        out.append("/proc/%s/fdinfo/%s" % (p, fd))
        except OSError:
            pass
    return out


def counters(fs):
    busy = total = 0
    for f in fs:
        t = open(f).read()
        m = re.search(r"drm-cycles-ccs:\s+(\d+)", t)
        n = re.search(r"drm-total-cycles-ccs:\s+(\d+)", t)
        if m:
            busy += int(m.group(1))
        if n:
            total = max(total, int(n.group(1)))
    return busy, total


while not os.path.exists(os.path.join(D, "hostprof.start")):
    time.sleep(0.05)
fs = files()
samples = []
while not os.path.exists(os.path.join(D, "hostprof.stop")):
    samples.append((time.time(),) + counters(fs))
    time.sleep(0.05)
samples.append((time.time(),) + counters(fs))
# decode_step_timer.py writes its all-decoding windows (wall clock) to D/windows.txt: "lo hi" per round
time.sleep(0.5)
wins = []
try:
    wins = [tuple(map(float, l.split())) for l in open(os.path.join(D, "windows.txt")) if l.strip()]
except OSError:
    pass


def share(lo, hi):
    s = [x for x in samples if lo <= x[0] <= hi]
    if len(s) < 2:
        return float("nan")
    return 100.0 * (s[-1][1] - s[0][1]) / max(1, s[-1][2] - s[0][2])


print("BUSY %s: %d fdinfo files, whole marker window %.2f%%" % (NAME, len(fs), share(samples[0][0], samples[-1][0])))
for i, (lo, hi) in enumerate(wins):
    print("BUSY round %d (all requests decoding, %.1f s): GPU busy %.2f%%" % (i, hi - lo, share(lo, hi)), flush=True)
