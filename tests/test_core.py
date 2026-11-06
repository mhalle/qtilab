"""Core correctness tests that need no downloaded data."""
import shutil

import numpy as np
import pytest
import torch

from qtilab import fit, scalars, simulate, tensors
from qtilab.protocol import Protocol


def _protocol(seed=0):
    rng = np.random.default_rng(seed)
    bvals, bvecs, shapes = [0.0], [[0, 0, 0]], ["LTE"]
    for b in (100, 700, 1400, 2000):
        for shape, n in (("LTE", 12), ("PTE", 12), ("STE", 4)):
            v = rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            bvals += [b] * n
            bvecs += v.tolist()
            shapes += [shape] * n
    return Protocol.from_table(bvals, np.array(bvecs).T, shapes)


def test_mandel_roundtrip_and_isometry():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(5, 3, 3, generator=g, dtype=torch.float64); A = A + A.transpose(-1, -2)
    C = torch.randn(5, 6, 6, generator=g, dtype=torch.float64); C = C + C.transpose(-1, -2)
    assert torch.allclose(tensors.mat3(tensors.vec6(A)), A)
    assert torch.allclose(tensors.mat6(tensors.vec21(C)), C)
    assert torch.allclose(tensors.vec6(A).norm(dim=-1), A.flatten(1).norm(dim=-1))
    assert torch.allclose(tensors.vec21(C).norm(dim=-1), C.flatten(1).norm(dim=-1))


def test_design_matrix_matches_dipy():
    dq = pytest.importorskip("dipy.reconst.qti")
    p = _protocol()
    Xd = dq.design_matrix(p.B.numpy())
    assert np.allclose(p.design_matrix().numpy(), Xd)


def test_b_delta():
    p = _protocol()
    bd = p.b_delta().numpy()
    shapes = np.array(p.shapes)
    nz = p.b.numpy() > 0
    assert np.allclose(bd[nz & (shapes == "LTE")], 1.0)
    assert np.allclose(bd[nz & (shapes == "PTE")], -0.5)
    assert np.allclose(bd[nz & (shapes == "STE")], 0.0, atol=1e-12)


def test_tiny_b_recovers_ground_truth():
    p = _protocol()
    tiny = Protocol(p.B * 1e-3)
    g = torch.Generator().manual_seed(1)
    sim = simulate.simulate(tiny, 2000, g=g, noise=False)
    est = scalars.scalars(fit.wls(tiny.design_matrix(), sim["clean"]))
    tru = scalars.scalars(sim["params"])
    for k in ("md", "fa", "ufa", "c_md"):
        assert np.nanmedian(np.abs((est[k] - tru[k]).numpy())) < 1e-3, k


def test_constrained_is_feasible_and_never_worse_than_projection():
    p = _protocol()
    g = torch.Generator().manual_seed(2)
    sim = simulate.simulate(p, 3000, g=g, prior=simulate.Prior(snr=(10, 20)))
    X = p.design_matrix()
    pw = fit.wls(X, sim["signal"])
    pc, info = fit.constrained(X, sim["signal"])
    assert info.converged_fraction > 0.99
    assert not bool(fit.violates(pc, tol=1e-8).any())
    # unconstrained-feasible voxels are untouched
    ok = ~fit.violates(pw)
    assert torch.allclose(pc[ok], pw[ok])


def test_rank_deficient_wls_float32_matches_float64():
    """LTE + STE only (rank 23): float32 WLS must solve in the 23-dim row space and match float64.

    Guards the row-space Cholesky solve: a float64 rank cutoff on a float32 design kept all 28 directions,
    silently sending every voxel to the batched pinv (5e-3 in µFA on CPU, 5e-2 on MPS).
    """
    p = _protocol()
    p = p.subset([s != "PTE" for s in p.shapes])
    assert p.rank() == 23
    torch.manual_seed(0)
    S = simulate.simulate(p, 4000, g=torch.Generator().manual_seed(1))["signal"].clamp_min(1e-4)
    X = p.design_matrix()
    X32 = X.float()
    assert fit._row_space(X32, int(torch.linalg.matrix_rank(X32))).shape == (28, 23)
    u64 = scalars.scalars(fit.wls(X, S))["ufa"]
    u32 = scalars.scalars(fit.wls(X32, S.float()).double())["ufa"]
    ok = torch.isfinite(u64) & (u64.abs() < 2)
    assert float((u32 - u64).abs()[ok].max()) < 1e-3


@pytest.mark.skipif(shutil.which("cc") is None, reason="torch.compile on CPU needs a C compiler")
def test_compiled_sampling_matches_eager():
    """PosteriorNet.sample(compile=True) equals eager on the same noise, and pads a short last chunk."""
    from qtilab import posterior

    torch.manual_seed(0)
    net = posterior.PosteriorNet(d_emb=16, transforms=2, hidden=(32,)).eval()
    x = posterior.simulate_batch(_protocol(), 300, g=torch.Generator().manual_seed(0), M=16).x
    kw = dict(n=4, chunk=128)
    eager = net.sample(x[:256], generator=torch.Generator().manual_seed(1), **kw)
    comp = net.sample(x[:256], generator=torch.Generator().manual_seed(1), compile=True, **kw)
    assert float((comp - eager).abs().max()) < 1e-4
    padded = net.sample(x, compile=True, **kw)             # 300 voxels: last chunk of 44 padded to 128
    assert padded.shape == (4, 300, len(posterior.TARGETS)) and bool(torch.isfinite(padded).all())
