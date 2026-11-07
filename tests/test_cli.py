"""The click CLI on small synthetic NIfTI series (no downloaded data), laid out the way BIDS (and tractline) expect."""
import json

import nibabel as nib
import numpy as np
import torch
from click.testing import CliRunner

from qtilab import fit, scalars, simulate
from qtilab.cli import main
from qtilab.protocol import Protocol


def _series(tmp_path, n_vox=60, seed=0):
    """Three series on one 5x4x3 grid with .bval/.bvec beside them: acq-linear and acq-spherical (shapes from the names),
    and acq-dir (PTE, given with --btens as a per-volume b_delta file)."""
    rng = np.random.default_rng(seed)
    specs = []
    for acq, shape, n in (("linear", "LTE", 18), ("dir", "PTE", 12), ("spherical", "STE", 8)):
        b = np.r_[0, np.repeat([100, 700, 1400, 2000], n // 4 + 1)][: n + 1].astype(float)
        v = rng.normal(size=(3, len(b)))
        v /= np.linalg.norm(v, axis=0)
        v[:, b == 0] = 0
        specs.append((acq, shape, b, v))
    prot = Protocol.from_table(np.concatenate([s[2] for s in specs]), np.concatenate([s[3] for s in specs], 1),
                               sum([[s[1]] * len(s[2]) for s in specs], []))
    S = simulate.simulate(prot, n_vox, g=torch.Generator().manual_seed(seed), prior=simulate.Prior(snr=(30, 60)))["signal"]
    S = (1000 * S).clamp_min(1e-3).numpy().astype(np.float32)
    grid, paths, off = (5, 4, 3), [], 0
    for acq, shape, b, v in specs:
        stem = tmp_path / f"sub-01_acq-{acq}_dwi"
        nib.save(nib.Nifti1Image(S[:, off:off + len(b)].reshape(*grid, len(b)), np.eye(4)), f"{stem}.nii.gz")
        np.savetxt(f"{stem}.bval", b[None], fmt="%g")
        np.savetxt(f"{stem}.bvec", v, fmt="%.8f")
        off += len(b)
        paths.append(f"{stem}.nii.gz")
    (tmp_path / "pte.txt").write_text(" ".join(["-0.5"] * len(specs[1][2])))
    btens = ["--btens", "LTE", "--btens", str(tmp_path / "pte.txt"), "--btens", "STE"]
    nib.save(nib.Nifti1Image(np.ones(grid, np.uint8), np.eye(4)), tmp_path / "mask.nii.gz")
    return paths, btens, prot, S


def test_info_finds_tables_and_shapes_from_bids_names(tmp_path):
    paths, btens, _, _ = _series(tmp_path)
    r = CliRunner().invoke(main, ["info", paths[0], paths[2], "--json", "-", "-q"])      # LTE + STE from acq-
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert [i["btens"] for i in out["inputs"]] == [["LTE"], ["STE"]] and out["protocol"]["rank"] < 28   # no PTE
    assert all(i["btens_from"].startswith("acq-") for i in out["inputs"])
    r = CliRunner().invoke(main, ["info", *paths])                          # acq-dir names no shape: one-line error
    assert r.exit_code != 0 and "no b-tensor shape" in r.output and "Traceback" not in r.output


def test_fit_matches_the_library_and_writes_provenance(tmp_path):
    paths, btens, prot, S = _series(tmp_path)
    out = tmp_path / "out" / "sub-01"
    r = CliRunner().invoke(main, ["fit", *paths, *btens, "--mask", str(tmp_path / "mask.nii.gz"), "--nifti", str(out),
                                  "--device", "cpu", "--maps", "md,ufa,c_c", "--params", "--json", "-", "-q"])
    assert r.exit_code == 0, r.output
    summ = json.loads(r.output)                                             # -q: standard output is the summary only
    assert summ["protocol"]["rank"] == 28 and summ["solver"] == "ipm" and summ["voxels"] == 60
    prov = json.loads((tmp_path / "out" / "sub-01.qtilab.json").read_text())
    assert [i["series"] for i in prov["inputs"]] == [p.split("/")[-1] for p in paths]   # no BIDS root: names only
    p, _ = fit.constrained(prot.design_matrix(), torch.as_tensor(np.maximum(S.astype(np.float64), 1e-4)))
    ref = scalars.scalars(p)
    for k in ("md", "ufa", "c_c"):
        got = np.asarray(nib.load(f"{out}_{k}.nii.gz").dataobj).reshape(-1)
        assert np.allclose(got, ref[k].numpy().astype(np.float32), atol=1e-6), k
    assert nib.load(f"{out}_params.nii.gz").shape == (5, 4, 3, 28)


def test_fit_rejects_mismatched_inputs_in_one_line(tmp_path):
    paths, btens, _, _ = _series(tmp_path)
    r = CliRunner().invoke(main, ["fit", *paths, *btens[:4], "--device", "cpu"])   # three series, two --btens
    assert r.exit_code != 0 and "one per series" in r.output
    (tmp_path / "bad.txt").write_text("LTE LTE")
    r = CliRunner().invoke(main, ["fit", *paths, "--btens", "LTE", "--btens", str(tmp_path / "bad.txt"), "--btens", "STE",
                                  "--device", "cpu"])
    assert r.exit_code != 0 and "entries for the" in r.output and "Traceback" not in r.output


def test_docs_prints_help_and_readme():
    r = CliRunner().invoke(main, ["docs"])
    assert r.exit_code == 0 and "## qtilab fit --help" in r.output and "## Command line" in r.output
