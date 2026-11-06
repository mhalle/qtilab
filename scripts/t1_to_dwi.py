"""Rigid registration of a session's T1w to its diffusion grid (mean b0), by mutual information.

Writes:
  QTILAB_WORK/hhs_sub{sub}_ses{ses}_t1_to_dwi.npz   reg (x_T1world = reg @ x_DWIworld, RAS mm, dipy's
                                                     convention), dwi_aff, dwi_shape
  results/t1_to_dwi_qc_s{sub}{ses}.png             T1 resampled into diffusion space, b0 edges on top

Rigid only: EPI susceptibility distortion (frontal sinuses, temporal poles) is not corrected.

    python scripts/t1_to_dwi.py [sub] [ses]          # default 01 01; needs prepare_session.py; ~10 s
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from dipy.align import affine_registration  # noqa: E402
from dipy.align.imaffine import AffineMap  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, ".")
from qtilab import hhs, paths  # noqa: E402

sub = sys.argv[1] if len(sys.argv) > 1 else "01"
ses = sys.argv[2] if len(sys.argv) > 2 else "01"

a = np.load(paths.aligned_session(sub, ses))
dwi_aff = a["affine"]
b0 = a["data"][..., a["bvals"] < 50].mean(-1)
mask = np.load(paths.classical_fit(sub, ses))["mask"]

t1 = nib.load(hhs.HHS_ROOT / f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_T1w.nii")
t1d = t1.get_fdata(dtype=np.float32)

_, reg = affine_registration(
    t1d, b0, moving_affine=t1.affine, static_affine=dwi_aff,
    pipeline=["center_of_mass", "translation", "rigid"],
    level_iters=[10000, 1000, 200], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1],
    static_mask=ndimage.binary_dilation(mask, iterations=2).astype(np.int32))
angle = np.degrees(np.arccos(np.clip((np.trace(reg[:3, :3]) - 1) / 2, -1, 1)))
print(f"rigid: rotation {angle:.2f} deg, translation {np.round(reg[:3, 3], 2)} mm")
np.savez(paths.t1_to_dwi(sub, ses), reg=reg, dwi_aff=dwi_aff, dwi_shape=np.array(b0.shape))

t1_in_dwi = AffineMap(reg, b0.shape, dwi_aff, t1d.shape, t1.affine).transform(t1d)
edges = ndimage.sobel(b0, 0) ** 2 + ndimage.sobel(b0, 1) ** 2
zs = np.linspace(0.3, 0.7, 4) * b0.shape[2]
fig, axs = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, z in zip(axs, zs.astype(int)):
    ax.imshow(t1_in_dwi[:, :, z].T, cmap="gray", origin="lower", vmax=np.percentile(t1_in_dwi[mask], 99))
    ax.contour(edges[:, :, z].T, levels=[np.percentile(edges[mask], 85)], colors="#ff3b30", linewidths=0.5)
    ax.set_title(f"z = {z}"); ax.axis("off")
    ax.text(0.02, 0.5, "R", transform=ax.transAxes, color="w", fontsize=8)
fig.suptitle(f"sub-{sub} ses-{ses}: T1 resampled into diffusion space; red = mean-b0 edges", fontsize=12)
plt.tight_layout()
out = f"results/t1_to_dwi_qc_s{sub}{ses}.png"
plt.savefig(out, dpi=140, facecolor="white")
print("wrote", paths.t1_to_dwi(sub, ses), "and", out)
