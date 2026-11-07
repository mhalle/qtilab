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


def test_simulation_is_reproducible_from_generator():
    """Every random draw in the simulators goes through the generator (Dirichlet weights once did not)."""
    torch.manual_seed(1)
    a = (simulate.sample_brain(500, M=8, g=torch.Generator().manual_seed(7))[0],
         simulate.sample_dtd(500, M=8, g=torch.Generator().manual_seed(7)))
    torch.manual_seed(2)                                   # a different global state must not matter
    b = (simulate.sample_brain(500, M=8, g=torch.Generator().manual_seed(7))[0],
         simulate.sample_dtd(500, M=8, g=torch.Generator().manual_seed(7)))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    w = simulate._dirichlet(2, (200_000, 3), torch.Generator().manual_seed(0), torch.float64, "cpu")
    assert torch.allclose(w.sum(-1), torch.ones(200_000, dtype=torch.float64))
    assert abs(float(w.var(0).mean()) - 2 / 63) < 1e-3     # Dirichlet(2, 2, 2): var = 2 * 4 / (36 * 7)


def test_exact_free_water_weights():
    """exact_fw: one weighted free-water tensor carries exactly the voxel's free-water fraction."""
    from qtilab import posterior
    g = torch.Generator().manual_seed(3)
    d6, cls, free, w = simulate.sample_brain(400, M=16, g=g, return_free=True, version=4, exact_fw=True)
    assert d6.shape == (400, 17, 6) and free[:, -1].all() and not free[:, :-1].any()
    assert torch.allclose(w.sum(1), torch.ones(400, dtype=w.dtype))
    y = posterior.targets(d6, free, w)
    assert torch.allclose(y[:, posterior.TARGETS.index("f_fw")], w[:, -1])
    assert (y[:, 0] > 0).all()                                 # never an exact zero
    p = _protocol()
    S = simulate.signal(d6, p.b6.to(d6.dtype), weights=w)
    assert torch.allclose(S[:, p.b < 0.05], torch.ones_like(S[:, p.b < 0.05]))
    # free water alone: the weighted signal's high-b decay matches exp(-3 b) for the free-water part
    pure = torch.cat([torch.zeros(1, 1, 6, dtype=d6.dtype), d6[:1, -1:]], 1)
    S1 = simulate.signal(pure, p.b6.to(d6.dtype), weights=torch.tensor([[0.0, 1.0]], dtype=d6.dtype))
    assert torch.allclose(S1[0], torch.exp(-3 * p.b.to(d6.dtype)))


def test_forward_model_options():
    """simulate_batch's v5 options: time dependence at r = 1 is a no-op; noise terms have the requested size."""
    from qtilab import posterior
    p = _protocol()
    d6, cls, free, w, comp = simulate.sample_brain(200, M=16, g=torch.Generator().manual_seed(4), return_free=True,
                                                   version=4, exact_fw=True, return_comp=True)
    assert comp.shape == free.shape and (comp[:, -1] == 6).all()
    tau = simulate.sample_tau(cls, comp, torch.Generator().manual_seed(5), gm_scale=0.0)
    assert (tau[:, -1] == 0).all() and (tau[(cls == 1) | (cls == 3)] == 0).all()     # no free-water / GM td
    b6 = p.b6.to(d6.dtype)
    assert torch.allclose(simulate.signal_td(d6, b6, torch.ones(p.n), tau, weights=w), simulate.signal(d6, b6, weights=w))
    # multiplicative noise only (no thermal noise to speak of): relative sd at b = 0 equals alpha
    kw = dict(g=torch.Generator().manual_seed(6), snr=(1e6, 1e6 + 1), prior_version=4, exact_fw=True, M=16)
    b = posterior.simulate_batch(p, 4000, alpha=(0.02, 0.02), **kw)
    clean = posterior.simulate_batch(p, 4000, **kw)
    assert abs(float((b.S[:, 0] - 1).std()) - 0.02) < 0.002 and float((clean.S[:, 0] - 1).abs().max()) < 1e-4
    td = posterior.simulate_batch(p, 2000, td_ratio=1.5, gain_sd=0.01, pulsation=0.15, **kw)
    assert torch.isfinite(td.x).all() and torch.isfinite(td.y).all()


def test_multichannel_noise_floor():
    """channels = 1 is the Rician draw unchanged; channels = 2 raises the zero-signal floor to non-central chi's."""
    S = torch.zeros(20000, 4, dtype=torch.float64)
    sig = torch.ones(20000, dtype=torch.float64)
    a = simulate.rician(S, sig, torch.Generator().manual_seed(1))
    b = simulate.rician(S, sig, torch.Generator().manual_seed(1), channels=1)
    assert torch.equal(a, b)
    assert abs(a.mean().item() - (torch.pi / 2) ** 0.5) < 0.02            # Rayleigh mean
    c = simulate.rician(S, sig, torch.Generator().manual_seed(1), channels=2)
    assert abs(c.mean().item() / c.std().item() - 2.75) < 0.05              # chi, 4 degrees of freedom
