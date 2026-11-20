#!/usr/bin/env python3
"""Decode STEP time at 0.1 ms resolution: N users on warm ~63K prefixes (conc64.py prompts), all decoding at once,
REPS rounds. Each SSE chunk is one engine step for its request (MTP streams the step's accepted tokens together),
so the inter-chunk gaps inside the window where all N decode are step times. Prints p50 / mean / p10 / p90 per
round and over all rounds. Env URL, MODEL, PROMPTS_DIR. usage: decode_step_timer.py N [OUT=512] [CTX=63000] [REPS=3]"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import urllib.request
import uuid

URL = os.environ.get("URL", "http://127.0.0.1:8002").rstrip("/")
MODEL = os.environ.get("MODEL", "qwen38")
PROMPTS = os.environ.get("PROMPTS_DIR", "/root/rice/prompts")
N = int(sys.argv[1])
OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 512
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 63000
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
BASE = json.load(open("%s/p%d.json" % (PROMPTS, CTX)))["prompts"][1]["messages"]


def stream(messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
            "stream": True, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    times = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = d.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            if delta.get("content") or delta.get("reasoning_content"):
                times.append(time.perf_counter())
    return times


users = []
for i in range(N):
    m = [dict(x) for x in BASE]
    m[0]["content"] = "[agent %s]\n" % uuid.uuid4().hex + m[0]["content"]
    users.append(m)
    stream(m, 1)  # cold prefill, makes the prefix warm
MARK = os.environ.get("L3_MARK_DIR")  # diagnostic: hostprof.start / .stop markers around the decode rounds
if MARK:
    open(os.path.join(MARK, "hostprof.start"), "w").close()
allg = []
for rep in range(REPS):
    res = [None] * N

    def turn(i):
        m = users[i] + [{"role": "assistant", "content": "Understood."},
                        {"role": "user", "content": "Turn %d for agent %d: write a long, detailed status report."
                         % (rep + 2, i)}]
        res[i] = stream(m, OUT)

    th = [threading.Thread(target=turn, args=(i,)) for i in range(N)]
    [t.start() for t in th]
    [t.join() for t in th]
    lo = max(r[0] for r in res)
    hi = min(r[-1] for r in res)
    if MARK:  # wall-clock window for gpu_busy_fdinfo.py
        with open(os.path.join(MARK, "windows.txt"), "a") as wf:
            wf.write("%.3f %.3f\n" % (lo - time.perf_counter() + time.time(), hi - time.perf_counter() + time.time()))
    g = [b - a for r in res for a, b in zip(r, r[1:]) if a >= lo and b <= hi]
    g.sort()
    allg += g
    q = lambda v, p: v[min(len(v) - 1, int(p / 100 * len(v)))] * 1e3
    print("STEP N=%d rep %d: chunks %d  p50 %.1f ms  mean %.1f  p10 %.1f  p90 %.1f"
          % (N, rep, len(g), q(g, 50), statistics.mean(g) * 1e3, q(g, 10), q(g, 90)), flush=True)
if MARK:
    open(os.path.join(MARK, "hostprof.stop"), "w").close()
allg.sort()
q = lambda v, p: v[min(len(v) - 1, int(p / 100 * len(v)))] * 1e3
print("STEP N=%d all: chunks %d  p50 %.1f ms  mean %.1f  p10 %.1f  p90 %.1f"
      % (N, len(allg), q(allg, 50), statistics.mean(allg) * 1e3, q(allg, 10), q(allg, 90)), flush=True)
