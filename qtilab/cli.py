"""Command-line interface: constrained QTI (QTI+ "dc") fits of NIfTI data.

    uv run qtilab info --dwi lte.nii.gz --bval lte.bval --bvec lte.bvec --btens LTE \\
                       --dwi ste.nii.gz --bval ste.bval --bvec ste.bvec --btens STE
    uv run qtilab fit  (same inputs) [--mask mask.nii.gz] -o out/sub-01

Each ``--dwi`` takes one ``--bval`` and one ``--bvec`` (FSL text files) and one ``--btens``, given in the same
order; a single ``--btens`` applies to every series. ``--btens`` is an encoding shape for the whole series
(LTE, PTE, STE) or a text file with one entry per volume: a shape name or b_delta (1 LTE, -0.5 PTE, 0 STE).
Series are concatenated along the 4th axis and must share a grid; they must already be motion- and
eddy-corrected and aligned (this tool fits, it does not register).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np

SHAPES = ("LTE", "PTE", "STE")
MAPS = ("md", "fa", "ufa", "c_md", "c_mu", "c_m", "c_c", "k_bulk", "k_shear", "mk", "k_mu", "v_md", "v_shear", "v_iso", "s0")
DEFAULT_MAPS = ("md", "fa", "ufa", "c_md", "c_mu", "c_c", "k_bulk", "k_shear", "mk", "k_mu")
UNITS = {"md": "um^2/ms", "v_md": "(um^2/ms)^2", "v_shear": "(um^2/ms)^2", "v_iso": "(um^2/ms)^2", "s0": "signal"}


def _shapes_for(spec: str, n: int, where: str) -> list[str]:
    """Encoding shape per volume from a shape name or a per-volume file (names or b_delta values)."""
    if spec.upper() in SHAPES:
        return [spec.upper()] * n
    p = Path(spec)
    if not p.exists():
        raise click.BadParameter(f"{spec!r} is neither LTE/PTE/STE nor a file", param_hint=f"--btens ({where})")
    tok = p.read_text().replace(",", " ").split()
    out = []
    for t in tok:
        if t.upper() in SHAPES:
            out.append(t.upper())
            continue
        try:
            v = float(t)
        except ValueError:
            raise click.BadParameter(f"{p}: entry {t!r} is not LTE/PTE/STE or a b_delta value", param_hint="--btens")
        near = {1.0: "LTE", -0.5: "PTE", 0.0: "STE"}
        k = min(near, key=lambda c: abs(c - v))
        if abs(k - v) > 0.1:
            raise click.BadParameter(f"{p}: b_delta {v} is not near 1, -0.5 or 0", param_hint="--btens")
        out.append(near[k])
    if len(out) != n:
        raise click.BadParameter(f"{p} has {len(out)} entries for {n} volumes", param_hint=f"--btens ({where})")
    return out


def _load_series(dwi, bval, bvec, btens, bval_units):
    import nibabel as nib

    if not (len(dwi) == len(bval) == len(bvec)):
        raise click.UsageError(f"give one --bval and one --bvec per --dwi ({len(dwi)} --dwi, {len(bval)} --bval, {len(bvec)} --bvec)")
    if len(btens) not in (1, len(dwi)):
        raise click.UsageError(f"give one --btens, or one per --dwi ({len(btens)} for {len(dwi)} series)")
    btens = list(btens) * len(dwi) if len(btens) == 1 else list(btens)
    data, bvals, bvecs, shapes, series = [], [], [], [], []
    affine = shape3 = None
    for k, (d, bv, bc, bt) in enumerate(zip(dwi, bval, bvec, btens)):
        img = nib.load(d)
        a = np.asarray(img.dataobj, dtype=np.float32)
        a = a[..., None] if a.ndim == 3 else a
        if affine is None:
            affine, shape3 = img.affine, a.shape[:3]
        elif a.shape[:3] != shape3 or not np.allclose(img.affine, affine, atol=1e-3):
            raise click.UsageError(f"{d}: grid {a.shape[:3]} / affine differs from the first series; resample first")
        b = np.atleast_1d(np.loadtxt(bv)).astype(float).ravel()
        v = np.atleast_2d(np.loadtxt(bc)).astype(float)
        v = v if v.shape[0] == 3 and v.shape[1] == len(b) else v.T
        n = a.shape[-1]
        if len(b) != n or v.shape != (3, n):
            raise click.UsageError(f"{d}: {n} volumes but {len(b)} b-values / bvec shape {v.shape}")
        data.append(a); bvals.append(b); bvecs.append(v)
        shapes += _shapes_for(bt, n, Path(d).name)
        series += [k] * n
    b = np.concatenate(bvals)
    unit = bval_units
    if unit == "auto":
        unit = "s/mm2" if b.max() > 100 else "ms/um2"
    b_smm = b if unit == "s/mm2" else b * 1000.0
    return {"data": np.concatenate(data, axis=-1), "affine": affine, "bvals": b_smm, "bvecs": np.concatenate(bvecs, axis=1),
            "shapes": shapes, "series": np.array(series), "bval_units": unit}


def _protocol_summary(s):
    import torch

    from .protocol import Protocol

    prot = Protocol.from_table(s["bvals"], s["bvecs"], s["shapes"])
    b = np.round(s["bvals"] / 50) * 50
    sh = np.array(s["shapes"])
    b0 = s["bvals"] < 50
    rows = []
    for shape in SHAPES:
        for bb in sorted(set(b[(sh == shape) & ~b0])):
            rows.append((shape, int(bb), int(((sh == shape) & (b == bb) & ~b0).sum())))
    rank = int(torch.linalg.matrix_rank(prot.design_matrix()))
    return prot, {"volumes": int(len(sh)), "b0_volumes": int(b0.sum()), "shells": rows, "rank": rank,
                  "bval_units_read_as": s["bval_units"]}


def _print_summary(summ):
    click.echo(f"{summ['volumes']} volumes ({summ['b0_volumes']} with b < 50 s/mm^2); b-values read as {summ['bval_units_read_as']}")
    for shape, bb, n in summ["shells"]:
        click.echo(f"  {shape} b = {bb:5d} s/mm^2: {n:3d} volumes")
    r = summ["rank"]
    click.echo(f"QTI design rank {r} of 28" + ("" if r == 28 else
               " (rank-deficient: part of the covariance is not observed; the fit picks the minimum-norm solution, "
               "and the usual scalars do not depend on the unobserved part)"))
    if r < 23:
        click.secho("warning: rank below 23; this protocol cannot separate the QTI covariance terms. "
                    "Consider a reduced model (e.g. RICE) instead.", fg="yellow", err=True)


def _auto_mask(data, bvals):
    b0 = data[..., bvals < 50].mean(-1) if (bvals < 50).any() else data[..., int(np.argmin(bvals))]
    try:
        from dipy.segment.mask import median_otsu
        _, m = median_otsu(b0, median_radius=2, numpass=1)
        return m, "median_otsu (dipy) on the mean b0"
    except ImportError:
        x = b0[b0 > 0]
        hist, edges = np.histogram(x, bins=256)
        w = np.cumsum(hist); mu = np.cumsum(hist * edges[:-1])
        between = (mu[-1] * w - mu * w[-1]) ** 2 / np.maximum(w * (w[-1] - w), 1)
        return b0 > edges[int(np.argmax(between))], "Otsu threshold on the mean b0 (install dipy for median_otsu)"


def _inputs(f):
    f = click.option("--bval-units", type=click.Choice(["auto", "s/mm2", "ms/um2"]), default="auto", show_default=True,
                     help="Units of the bval files; auto: s/mm^2 if any b > 100.")(f)
    f = click.option("--btens", multiple=True, required=True,
                     help="LTE, PTE or STE for a whole series, or a per-volume file (shape names or b_delta). One, or one per --dwi.")(f)
    f = click.option("--bvec", multiple=True, required=True, type=click.Path(exists=True, dir_okay=False),
                     help="FSL bvec file (3 x N), one per --dwi.")(f)
    f = click.option("--bval", multiple=True, required=True, type=click.Path(exists=True, dir_okay=False),
                     help="FSL bval file, one per --dwi.")(f)
    f = click.option("--dwi", multiple=True, required=True, type=click.Path(exists=True, dir_okay=False),
                     help="4D NIfTI series (repeatable; concatenated in order).")(f)
    return f


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="qtilab")
def main():
    """qtilab: batched constrained QTI fits (QTI+ "dc": mean tensor and covariance positive semidefinite)."""


@main.command()
@_inputs
def info(dwi, bval, bvec, btens, bval_units):
    """Summarize the protocol (shells, encoding shapes, QTI design rank) without fitting."""
    s = _load_series(dwi, bval, bvec, btens, bval_units)
    click.echo(f"grid {s['data'].shape[:3]}, {len(dwi)} series")
    _print_summary(_protocol_summary(s)[1])


@main.command()
@_inputs
@click.option("--mask", type=click.Path(exists=True, dir_okay=False), help="Brain mask NIfTI; default: automatic from the b0s.")
@click.option("-o", "--out", "out", required=True, type=click.Path(),
              help="Output prefix, e.g. out/sub-01 -> out/sub-01_md.nii.gz, ...; a trailing / writes into that directory.")
@click.option("--maps", default=",".join(DEFAULT_MAPS), show_default=True,
              help=f"Comma-separated scalars to write (any of: {', '.join(MAPS)}), or 'all'.")
@click.option("--params", "save_params", is_flag=True, help="Also write the 28 parameters as a 4D NIfTI ([ln S0, <D> Mandel 6, C Mandel 21], DIPY's order).")
@click.option("--method", type=click.Choice(["constrained", "wls"]), default="constrained", show_default=True,
              help="constrained: QTI+ dc (WLS where it already satisfies the constraints); wls: DIPY-equivalent WLS only.")
@click.option("--solver", type=click.Choice(["ipm", "admm"]), default="ipm", show_default=True, help="Constrained solver.")
@click.option("--device", type=click.Choice(["auto", "cpu", "mps", "cuda"]), default="auto", show_default=True,
              help="auto: cuda, else mps, else cpu.")
@click.option("--precision", type=click.Choice(["auto", "float64", "float32"]), default="auto", show_default=True,
              help="auto: float64 on CPU, float32 on GPUs (MPS has no float64).")
@click.option("--threads", type=int, default=None, help="CPU threads for torch (default: torch's choice).")
def fit(dwi, bval, bvec, btens, bval_units, mask, out, maps, save_params, method, solver, device, precision, threads):
    """Fit QTI to NIfTI data and write scalar maps (NIfTI, float32) plus a JSON report."""
    import nibabel as nib
    import torch

    from . import fit as qfit
    from . import scalars

    t0 = time.perf_counter()
    want = MAPS if maps.strip().lower() == "all" else tuple(m.strip() for m in maps.split(",") if m.strip())
    bad = [m for m in want if m not in MAPS]
    if bad:
        raise click.BadParameter(f"unknown map(s) {bad}; choose from {', '.join(MAPS)}", param_hint="--maps")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if precision == "auto":
        precision = "float64" if device == "cpu" else "float32"
    if device == "mps" and precision == "float64":
        raise click.UsageError("MPS has no float64; use --precision float32 or --device cpu")
    if threads:
        torch.set_num_threads(threads)
    dt = torch.float64 if precision == "float64" else torch.float32

    s = _load_series(dwi, bval, bvec, btens, bval_units)
    prot, summ = _protocol_summary(s)
    _print_summary(summ)
    data = s["data"]
    if mask:
        m = np.asarray(nib.load(mask).dataobj) > 0
        if m.shape != data.shape[:3]:
            raise click.UsageError(f"mask grid {m.shape} differs from the data {data.shape[:3]}")
        mask_how = str(mask)
    else:
        m, mask_how = _auto_mask(data, s["bvals"])
    S = data[m].astype(np.float64)
    S = np.maximum(S, 1e-4)                                     # DIPY's MIN_POSITIVE_SIGNAL, as in prepare_session.py
    click.echo(f"{int(m.sum())} voxels in the mask ({mask_how}); fitting on {device} in {precision}"
               + (f" with {solver}" if method == "constrained" else " (WLS)"))

    X = prot.design_matrix().to(device=device, dtype=dt)
    St = torch.as_tensor(S, dtype=dt, device=device)
    t = time.perf_counter()
    if method == "wls":
        p, info = qfit.wls(X, St), None
    else:
        p, info = qfit.constrained(X, St, solver=solver)
    if device != "cpu":
        getattr(torch, device).synchronize()
    t_fit = time.perf_counter() - t
    p = p.cpu().double()
    sc = scalars.scalars(p)
    collapsed = scalars.collapsed(sc).numpy()
    keep = ~collapsed
    if method == "constrained":
        click.echo(f"fit in {t_fit:.1f} s: {info.n_violating} voxels needed the constrained solve "
                   f"({100 * info.n_violating / max(1, len(S)):.0f}%), converged {info.converged_fraction:.4f}")
    else:
        click.echo(f"fit in {t_fit:.1f} s")
    if collapsed.any():
        why = "collapsed to <D> = 0 (no attenuation above the floor)" if method == "constrained" else "with MD <= 0 (unconstrained WLS)"
        click.echo(f"{int(collapsed.sum())} voxels {why}: excluded from the maps and the fit mask")

    out_s = str(out)
    prefix = Path(out_s) / "qti" if out_s.endswith("/") else Path(out_s)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fit_mask = m.copy()
    fit_mask[m] = keep
    written = []

    def save(arr, name):
        path = Path(f"{prefix}_{name}.nii.gz")
        img = nib.Nifti1Image(arr, s["affine"])
        img.header.set_xyzt_units("mm")
        nib.save(img, path)
        written.append(str(path))

    for k in want:
        vol = np.zeros(m.shape, np.float32)
        vol[fit_mask] = sc[k].numpy()[keep].astype(np.float32)
        save(vol, k)
    save(fit_mask.astype(np.uint8), "mask")
    if save_params:
        vol = np.zeros(m.shape + (28,), np.float32)
        vol[fit_mask] = p.numpy()[keep].astype(np.float32)
        save(vol, "params")
    report = {
        "qtilab_version": _version(), "inputs": {"dwi": list(dwi), "bval": list(bval), "bvec": list(bvec), "btens": list(btens)},
        "protocol": summ, "mask": mask_how, "voxels": int(m.sum()), "collapsed": int(collapsed.sum()),
        "method": method, "solver": solver if method == "constrained" else None, "device": device, "precision": precision,
        "fit_seconds": round(t_fit, 2), "total_seconds": round(time.perf_counter() - t0, 2),
        "units": {k: UNITS.get(k, "dimensionless") for k in want},
        **({"n_constrained": info.n_violating, "converged_fraction": info.converged_fraction,
            "solver_stats": {k: v for k, v in (info.extra or {}).items()}} if info is not None else {}),
        "outputs": written,
    }
    Path(f"{prefix}_fit.json").write_text(json.dumps(report, indent=1))
    click.echo(f"wrote {len(written)} NIfTI files and {prefix}_fit.json ({report['total_seconds']:.1f} s total)")


def _version():
    try:
        from importlib.metadata import version
        return version("qtilab")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
