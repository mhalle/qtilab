"""QTI+ maps from the DiB_217 brain with qtilab, plus where the unconstrained fit is non-physical."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, ".")
from qtilab import fit, scalars  # noqa: E402
from qtilab.data import load_dib  # noqa: E402
from qtilab.tensors import mat3, mat6  # noqa: E402

data, mask, aff, prot, *_ = load_dib("DiB_217_lte_pte_ste")
X = prot.design_matrix()
S = torch.as_tensor(np.maximum(data[mask].astype(np.float64), 1e-4))
pw = fit.wls(X, S)
pc, info = fit.constrained(X, S)
sc = scalars.scalars(pc)
neg_d = (torch.linalg.eigvalsh(mat3(pw[:, 1:7]))[:, 0] < 0).numpy()
neg_c = (torch.linalg.eigvalsh(mat6(pw[:, 7:]))[:, 0] < 0).numpy()

def vol(v):
    out = np.full(mask.shape, np.nan)
    out[mask] = v
    return out

z = mask.shape[2] // 2 + 2
panels = [("MD (µm²/ms)", sc["md"].numpy(), (0, 3)), ("FA", sc["fa"].numpy(), (0, 1)),
          ("µFA", sc["ufa"].numpy(), (0, 1)), ("C_c (orientation coherence)", sc["c_c"].numpy(), (0, 1)),
          ("MK_I = K_bulk", sc["k_bulk"].numpy(), (0, 1.5)), ("MK_A = K_µ", sc["k_mu"].numpy(), (0, 1.5)),
          ("C_MD (size variance)", sc["c_md"].numpy(), (0, 0.5)),
          ("unconstrained fit non-physical\n(gray: C not PSD, white: <D> not PSD)",
           np.where(neg_d, 2.0, np.where(neg_c, 1.0, 0.0)), (0, 2))]
fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), facecolor="black")
for ax, (title, v, (lo, hi)) in zip(axes.ravel(), panels):
    im = vol(v)[:, :, z]
    ax.imshow(np.rot90(im), cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title, color="white", fontsize=10)
    ax.axis("off")
fig.suptitle(f"DiB_217 (LTE+PTE+STE), slice {z}: qtilab constrained fit (QTI+ dc). "
             f"{100*neg_c.mean():.0f}% of brain voxels had a non-PSD covariance under WLS, "
             f"{100*neg_d.mean():.1f}% a non-PSD mean tensor.", color="white", fontsize=10)
plt.tight_layout()
plt.savefig("results/dib217_maps.png", dpi=110, facecolor="black")
print(f"saved; non-PSD C {neg_c.mean():.3f}, non-PSD <D> {neg_d.mean():.4f}")
