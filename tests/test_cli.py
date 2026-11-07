"""The click CLI on small synthetic NIfTI series (no downloaded data)."""
import json

import nibabel as nib
import numpy as np
import torch
from click.testing import CliRunner

from qtilab import fit, scalars, simulate
from qtilab.cli import main
from qtilab.protocol import Protocol


def _series(tmp_path, n_vox=60, seed=0):
    """Three series on one 5x4x3 grid: LTE and PTE (whole-series shapes) and STE given as a b_delta file."""
    rng = np.random.default_rng(seed)
    specs = []
    for shape, n in (("LTE", 18), ("PTE", 12), ("STE", 8)):
        b = np.r_[0, np.repeat([100, 700, 1400, 2000], n // 4 + 1)][: n + 1].astype(float)
        v = rng.normal(size=(3, len(b)))
        v /= np.linalg.norm(v, axis=0)
        v[:, b == 0] = 0
        specs.append((shape, b, v))
    prot = Protocol.from_table(np.concatenate([s[1] for s in specs]), np.concatenate([s[2] for s in specs], 1),
                               sum([[s[0]] * len(s[1]) for s in specs], []))
    S = simulate.simulate(prot, n_vox, g=torch.Generator().manual_seed(seed), prior=simulate.Prior(snr=(30, 60)))["signal"]
    S = (1000 * S).clamp_min(1e-3).numpy().astype(np.float32)
    grid = (5, 4, 3)
    args, off = [], 0
    for k, (shape, b, v) in enumerate(specs):
        vol = S[:, off:off + len(b)].reshape(*grid, len(b))
        off += len(b)
        nib.save(nib.Nifti1Image(vol, np.eye(4)), tmp_path / f"s{k}.nii.gz")
        np.savetxt(tmp_path / f"s{k}.bval", b[None], fmt="%g")
        np.savetxt(tmp_path / f"s{k}.bvec", v, fmt="%.8f")
        bt = shape
        if shape == "STE":
            (tmp_path / "ste_btens.txt").write_text(" ".join("0" for _ in b))
            bt = str(tmp_path / "ste_btens.txt")
        args += ["--dwi", str(tmp_path / f"s{k}.nii.gz"), "--bval", str(tmp_path / f"s{k}.bval"),
                 "--bvec", str(tmp_path / f"s{k}.bvec"), "--btens", bt]
    nib.save(nib.Nifti1Image(np.ones(grid, np.uint8), np.eye(4)), tmp_path / "mask.nii.gz")
    return args, prot, S


def test_cli_info_and_fit_match_the_library(tmp_path):
    args, prot, S = _series(tmp_path)
    r = CliRunner().invoke(main, ["info", *args])
    assert r.exit_code == 0, r.output
    assert "rank 28 of 28" in r.output and "PTE b =  1400" in r.output

    out = tmp_path / "out" / "sub"
    r = CliRunner().invoke(main, ["fit", *args, "--mask", str(tmp_path / "mask.nii.gz"), "-o", str(out),
                                  "--device", "cpu", "--maps", "md,ufa,c_c", "--params"])
    assert r.exit_code == 0, r.output
    rep = json.loads((tmp_path / "out" / "sub_fit.json").read_text())
    assert rep["protocol"]["rank"] == 28 and rep["solver"] == "ipm" and rep["voxels"] == 60
    p, _ = fit.constrained(prot.design_matrix(), torch.as_tensor(np.maximum(S.astype(np.float64), 1e-4)))
    ref = scalars.scalars(p)
    for k in ("md", "ufa", "c_c"):
        got = np.asarray(nib.load(f"{out}_{k}.nii.gz").dataobj).reshape(-1)
        assert np.allclose(got, ref[k].numpy().astype(np.float32), atol=1e-6), k
    assert nib.load(f"{out}_params.nii.gz").shape == (5, 4, 3, 28)


def test_cli_rejects_mismatched_inputs(tmp_path):
    args, _, _ = _series(tmp_path)
    r = CliRunner().invoke(main, ["info", *args[:-2]])          # three series, two --btens
    assert r.exit_code != 0
    (tmp_path / "bad.txt").write_text("LTE LTE")
    bad = args.copy()
    bad[bad.index("LTE")] = str(tmp_path / "bad.txt")
    r = CliRunner().invoke(main, ["info", *bad])
    assert r.exit_code != 0 and "entries for" in r.output
