"""Summarize the time-dependence grid (results/heldout_hhs_td_s{sub}{ses}_{config}.json).

Configs: base (time-independent prior v2), and r{ratio}g{0|1}: time dependence with LTE/STE spectral
ratio r, gray-matter strengths off (g0) or on (g1). Errors are median RMS / noise sd of the simulator
prediction (1 = noise-limited), per held-out split and region.
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np

rows = defaultdict(dict)
for f in glob.glob("results/heldout_hhs_td_s*_*.json"):
    m = re.search(r"_td_s(\d{4})_([\w.]+?)\.json$", f)
    if not m:
        continue
    rows[m.group(2)][m.group(1)] = json.load(open(f))
cfgs = ["base"] + sorted(k for k in rows if k != "base")
sessions = sorted({s for c in rows.values() for s in c})
cells = [("run8", "WM-like"), ("run8", "GM-like"), ("run8", "CSF-ish"), ("run4", "WM-like"), ("run4", "GM-like"), ("run4", "CSF-ish")]
print(f"sessions: {', '.join('sub-' + s[:2] + ' ses-' + s[2:] for s in sessions)}")
print(f"{'config':8s} | " + " ".join(f"{sp[-1]}:{rg[:3]:>4s}" for sp, rg in cells)
      + " | mean | base on same sessions | n  (split 8 = STE b1500, 4 = LTE b2000)")
summary = {}
for cfg in cfgs:
    ss = [s for s in sessions if s in rows[cfg] and s in rows["base"]]
    vals = [np.mean([rows[cfg][s][sp][rg]["sim"] for s in ss]) for sp, rg in cells]
    base = [np.mean([rows["base"][s][sp][rg]["sim"] for s in ss]) for sp, rg in cells]
    summary[cfg] = {"sessions": ss, "vals": vals, "base_same_sessions": base}
    print(f"{cfg:8s} | " + " ".join(f"{x:7.2f}" for x in vals) + f" | {np.mean(vals):.3f} | {np.mean(base):.3f} | {len(ss)}")
complete = [c for c in cfgs if c != "base" and len(summary[c]["sessions"]) == len(sessions)]
best = min(complete, key=lambda c: np.mean(summary[c]["vals"]))
print(f"\nbest config by mean error: {best}")
print("per session, STE b1500 WM error: base -> best   (and GM)")
for s in sessions:
    if s in rows["base"] and s in rows[best]:
        b, t = rows["base"][s]["run8"], rows[best][s]["run8"]
        print(f"  sub-{s[:2]} ses-{s[2:]}: WM {b['WM-like']['sim']:.2f} -> {t['WM-like']['sim']:.2f}   "
              f"GM {b['GM-like']['sim']:.2f} -> {t['GM-like']['sim']:.2f}   "
              f"(LTE b2000 WM {rows['base'][s]['run4']['WM-like']['sim']:.2f} -> {rows[best][s]['run4']['WM-like']['sim']:.2f})")
json.dump({"sessions": sessions, "cells": [list(c) for c in cells], "summary": summary, "best": best},
          open("results/td_summary.json", "w"), indent=1)
