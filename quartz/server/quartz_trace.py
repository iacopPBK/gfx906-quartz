#!/usr/bin/env python3
"""Read QZ_TRACE records. `quartz_trace.py [dir]` lists the requests, * marking the
flagged ones. `quartz_trace.py <file.json> [--prompt [N]]` shows one request."""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def one(path, prompt_chars=None):
    t = json.load(open(path))
    print("%s  %s  prompt %d (%d cached)  gen %d / max %d  finish %s  engine stop %s" % (
        t["time"], t["rid"], t["n_prompt"], t.get("n_cached", 0), t["n_gen"], t["max_new"], t["finish"], t["engine_stop"]))
    print("flags:", "; ".join(t["flags"]) or "none")
    if prompt_chars:
        print("\n--- rendered prompt (tail) ---\n" + (t.get("rendered") or "")[-prompt_chars:])
    print("\n--- raw output (special tokens kept) ---\n" + t["raw"])
    if t.get("tool_calls"):
        print("\n--- tool calls ---")
        for c in t["tool_calls"]:
            print(json.dumps(c["function"], ensure_ascii=False)[:400])


def listing(d):
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            t = json.load(open(p))
        except ValueError:
            continue
        tail = (t.get("content") or "").strip().replace("\n", " ")[-60:]
        print("%s %s  prompt %6d  gen %5d  %-10s %s  | %s" % (
            "*" if t["flags"] else " ", os.path.basename(p), t["n_prompt"], t["n_gen"], t["finish"],
            "; ".join(t["flags"]), tail))


a = sys.argv[1:]
if a and a[0].endswith(".json"):
    n = None
    if "--prompt" in a:
        i = a.index("--prompt")
        n = int(a[i + 1]) if len(a) > i + 1 else 2000
    one(a[0], n)
else:
    # default: logs/trace itself when it holds traces, else its newest subdirectory
    base = os.path.join(REPO, "logs/trace")
    subs = [p for p in glob.glob(os.path.join(base, "*")) if os.path.isdir(p)]
    d = a[0] if a else base if glob.glob(os.path.join(base, "*.json")) else max(subs, default=None, key=os.path.getmtime)
    if not d:
        sys.exit("no trace dir")
    print(d)
    listing(d)
