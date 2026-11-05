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
print(f"{'config':8s} | " + " ".join(f"{sp[-1]}:{rg[:3]:>4s}" for sp, rg in cells) + " | mean  (mean over sessions; split 8 = STE b1500, 4 = LTE b2000)")
summary = {}
for cfg in cfgs:
    vals = []
    for sp, rg in cells:
        v = [rows[cfg][s][sp][rg]["sim"] for s in sessions if s in rows[cfg]]
        vals.append(np.mean(v) if v else np.nan)
    summary[cfg] = vals
    print(f"{cfg:8s} | " + " ".join(f"{x:7.2f}" for x in vals) + f" | {np.nanmean(vals):.3f}")
best = min((c for c in cfgs if c != "base"), key=lambda c: np.nanmean(summary[c]))
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
