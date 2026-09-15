#!/usr/bin/env python3
"""Measure decode throughput of a running Metal-backed FreeToken server.

Decode on Apple Silicon is memory-bandwidth bound: every token reads the whole
weight set, so tok/s is roughly (memory bandwidth) / (weight bytes). The report
prints the measured rate next to that ceiling so a slow run is recognisable as
contention rather than a bad model choice."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import urllib.request

PROMPT = "Write a 150-word explanation of how a write-ahead log works."

# Apple Silicon unified-memory bandwidth, GB/s, by chip family.
BANDWIDTH = {"M1": 68.25, "M1 Pro": 200.0, "M1 Max": 400.0, "M1 Ultra": 800.0,
             "M2": 100.0, "M2 Pro": 200.0, "M2 Max": 400.0, "M2 Ultra": 800.0,
             "M3": 100.0, "M3 Pro": 150.0, "M3 Max": 400.0,
             "M4": 120.0, "M4 Pro": 273.0, "M4 Max": 546.0}


def _chip() -> str:
    try:
        out = subprocess.run(["system_profiler", "SPHardwareDataType"],
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "Chip:" in line:
                return line.split("Chip:", 1)[1].strip()
    except Exception:
        pass
    return ""


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"), strict=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--weights-gb", type=float, default=0.0,
                    help="weight bytes read per token, for the bandwidth ceiling")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
        health = json.loads(r.read())
    model = health.get("model", "?")
    print(f"model {model}  backend {health.get('backend')}")

    rates = []
    for i in range(1, args.runs + 1):
        started = time.monotonic()
        body = _post(f"{base}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=600)
        elapsed = time.monotonic() - started
        usage = body.get("usage") or {}
        out_tokens = usage.get("completion_tokens", 0)
        rate = out_tokens / elapsed if elapsed else 0.0
        rates.append(rate)
        print(f"  run {i}: {out_tokens:4d} tok in {elapsed:5.1f}s = {rate:5.2f} tok/s")

    best, median = max(rates), statistics.median(rates)
    print(f"\nmedian {median:.2f} tok/s   best {best:.2f} tok/s")
    chip = _chip()
    bw = BANDWIDTH.get(chip)
    if bw and args.weights_gb:
        ceiling = bw / args.weights_gb
        print(f"{chip}: {bw:.0f} GB/s / {args.weights_gb:.1f} GB weights "
              f"= {ceiling:.1f} tok/s ceiling, reaching {100*best/ceiling:.0f}%")
    elif bw:
        print(f"{chip}: {bw:.0f} GB/s. Pass --weights-gb to print the ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
