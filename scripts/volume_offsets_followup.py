"""Follow-up on the per-volume gains of volume_offsets.py: drift or per-waveform, and do they inflate alpha?

  1. smoothness: lag-1 autocorrelation of each run's gains in acquisition order, against a permutation null
     (drift is smooth; a per-waveform or per-direction systematic is not)
  2. b-scaling: sd of the gains by b-value (100, 500, 1000, 1500, 2000), LTE and STE; a b-value error
     grows with b, a receive/transmit gain does not
  3. reproducibility: each session's gains against the mean of the other sessions (leave one session out)
  4. noise model: alpha(b) of sub-01 ses-01 (fullest slice, by region) refitted after dividing each STE
     volume by the gain template from the OTHER sessions; how much of alpha is a reproducible
     per-volume level rather than physiological noise

    python scripts/volume_offsets_followup.py [results/volume_offsets_v2.json]
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from qtilab import noise, paths  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "results/volume_offsets_v2.json"
d = json.load(open(src))
pv = d["per_volume"]
sessions = list(d["sessions"])
labels = list(d["runs"].values())
rng = np.random.default_rng(0)
out = {"source": src}

# 1 + 2 + 3
print("run           lag-1 r (null 95%)   gain sd by b (per mille)                    LOSO r with others' mean")
out["runs"] = {}
for lab in labels:
    G = np.array([pv[f"{s}|{lab}"]["gain"] for s in sessions])            # (n_sess, n_vol)
    bv = np.array(pv[f"{sessions[0]}|{lab}"]["bvals"])
    lag = []
    null = []
    for g in G:
        for b in np.unique(bv):                                            # within a b-value only (mixed runs)
            x = g[bv == b]
            if len(x) < 4:
                continue
            x = x - x.mean()
            lag.append(np.corrcoef(x[:-1], x[1:])[0, 1])
            for _ in range(200):
                y = rng.permutation(x)
                null.append(np.corrcoef(y[:-1], y[1:])[0, 1])
    lag_m = float(np.mean(lag))
    hi = float(np.percentile([np.mean(rng.choice(null, len(lag))) for _ in range(2000)], 97.5))
    sd_b = {int(b): 1000 * float(np.mean([G[i, bv == b].std() for i in range(len(G))])) for b in np.unique(bv) if (bv == b).sum() > 2}
    loso = []
    for i in range(len(G)):
        others = np.delete(G, i, 0).mean(0)
        r_ = [np.corrcoef(G[i, bv == b], others[bv == b])[0, 1] for b in np.unique(bv) if (bv == b).sum() > 3]
        loso.append(float(np.mean(r_)))
    out["runs"][lab] = {"lag1_mean": lag_m, "lag1_null_97.5": hi, "gain_sd_by_b_permille": sd_b, "loso_r": loso}
    print(f"{lab:12s}  {lag_m:+.2f} ({hi:+.2f})          " + ", ".join(f"b{b} {v:.1f}" for b, v in sd_b.items()).ljust(44)
          + " " + " ".join(f"{x:+.2f}" for x in loso))

# 4: alpha with and without the other sessions' STE gain template (sub-01 ses-01)
sub, ses = "01", "01"
key = f"{sub}-{ses}"
a = np.load(paths.aligned_session(sub, ses))
c = np.load(paths.classical_fit(sub, ses))
full = c["mask"]
z = int(np.argmax(full.sum((0, 1))[10:28])) + 10
slab = np.zeros_like(full); slab[:, :, z] = True
S = np.maximum(a["data"][full & slab].astype(np.float64), 1e-4)
bvals, shapes = a["bvals"], np.array([str(s) for s in a["shapes"]])
S = S / S[:, bvals < 50].mean(1, keepdims=True)
reg = noise.region_of(c["fa"][slab[full]], c["md"][slab[full]])
corr = np.zeros(len(bvals))
for lab in labels:
    if not lab.startswith("STE"):
        continue
    vols = np.array(pv[f"{key}|{lab}"]["volumes"])
    tmpl = np.mean([pv[f"{s}|{lab}"]["gain"] for s in sessions if s != key], 0)
    corr[vols] = tmpl
S_c = S / np.exp(corr)[None, :]
groups, levels = noise.ste_groups(bvals, shapes, exclude=bvals < 50)
out["alpha"] = {"levels": [float(x) for x in levels]}
print(f"\nalpha(b) for {key}, slice {z}, STE levels {np.round(levels).astype(int).tolist()}: raw vs after the other sessions' gain template")
for r, name in enumerate(("WM", "GM", "CSF")):
    row = {}
    for tag, X in (("raw", S), ("template-corrected", S_c)):
        m, v, dof = noise.group_moments(X[reg == r], groups)
        al, s2, _ = noise.fit_alpha_groups(m, v, dof)
        row[tag] = {"alpha": np.round(al, 4).tolist(), "snr": float(np.median(1 / np.sqrt(s2)))}
    out["alpha"][name] = row
    print(f"  {name}: raw {row['raw']['alpha']} (SNR {row['raw']['snr']:.0f})  ->  {row['template-corrected']['alpha']} (SNR {row['template-corrected']['snr']:.0f})")
json.dump(out, open("results/volume_offsets_followup.json", "w"), indent=1)
