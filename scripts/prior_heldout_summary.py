"""Held-out prediction across sessions, by prior and time dependence (results/heldout_hhs_pv_s{sub}{ses}_{config}.json).

Configs: v2, v4 (time-independent brain priors) and v{N}r{ratio}: WM-only time dependence with LTE/STE
spectral ratio r. Produced by

    for each session and config:
        python scripts/heldout_hhs.py 100000 --prior N [--td r --tau-gm 0] --noise model --regions classical \
            --slice -1 --sub SS --ses EE --tag _pv_sSSEE_CONFIG

Errors are median RMS / noise sd of the simulator prediction (1 = noise-limited); bias is the median
signed residual in noise sd (negative: observed below predicted).
-> results/prior_heldout_summary.json
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np

rows = defaultdict(dict)
for f in glob.glob("results/heldout_hhs_pv_s*_*.json"):
    m = re.search(r"_pv_s(\d{4})_([\w.]+?)\.json$", f)
    if m:
        rows[m.group(2)][m.group(1)] = json.load(open(f))
order = ["v2", "v2r4", "v4", "v4r1.5", "v4r2", "v4r4"]
cfgs = [c for c in order if c in rows] + sorted(c for c in rows if c not in order)
sessions = sorted({s for c in rows.values() for s in c})
cells = [("run8", "WM-like"), ("run8", "GM-like"), ("run8", "CSF-ish"), ("run4", "WM-like"), ("run4", "GM-like"),
         ("run4", "CSF-ish")]
print("sessions: " + ", ".join(f"sub-{s[:2]} ses-{s[2:]}" for s in sessions))
print(f"\nmean over sessions of the median RMS error / noise sd (split 8 = STE b1500, 4 = LTE b2000)")
print(f"{'config':8s} " + " ".join(f"{sp[-1]}:{rg[:3]:>4s}" for sp, rg in cells) + "   WM STE bias   n")
out = {"sessions": sessions, "configs": {}}
for c in cfgs:
    ss = [s for s in sessions if s in rows[c]]
    err = [float(np.mean([rows[c][s][sp][rg]["sim"] for s in ss])) for sp, rg in cells]
    bias = float(np.mean([rows[c][s]["run8"]["WM-like"]["sim_bias"] for s in ss]))
    out["configs"][c] = {"sessions": ss, "mean_error": dict(zip([f"{sp}_{rg}" for sp, rg in cells], err)),
                         "mean_wm_ste_bias": bias,
                         "per_session_wm_ste": {s: [rows[c][s]["run8"]["WM-like"]["sim"],
                                                    rows[c][s]["run8"]["WM-like"]["sim_bias"]] for s in ss}}
    print(f"{c:8s} " + " ".join(f"{x:7.2f}" for x in err) + f"   {bias:+.2f}       {len(ss)}")
print("\nWM STE b1500 per session: error (bias)")
print(f"{'session':10s} " + " ".join(f"{c:>14s}" for c in cfgs))
for s in sessions:
    print(f"s{s:9s} " + " ".join(f"{rows[c][s]['run8']['WM-like']['sim']:6.2f} ({rows[c][s]['run8']['WM-like']['sim_bias']:+.2f})"
                                 if s in rows[c] else f"{'-':>14s}" for c in cfgs))
json.dump(out, open("results/prior_heldout_summary.json", "w"), indent=1)
print("-> results/prior_heldout_summary.json")
