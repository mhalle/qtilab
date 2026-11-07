"""The qtilab command: constrained QTI (QTI+ "dc") maps from tensor-valued dMRI, from the shell.

    qtilab fit sub-01/dwi/sub-01_acq-linear_dwi.nii.gz sub-01/dwi/sub-01_acq-spherical_dwi.nii.gz --nifti out/sub-01
    qtilab info sub-01/dwi/sub-01_acq-*_dwi.nii.gz        # the protocol: shells, shapes, QTI design rank
    qtilab docs                                           # the guide, the README and the commands' help as one Markdown text

Or run without installing:

    uvx --from "git+https://github.com/mhalle/qtilab@<tag>" qtilab fit ...

The command line follows tractline's (github.com/mhalle/tractline) so the two can be merged:
- series are positional NIfTI paths whose .bval and .bvec are found beside them or by BIDS inheritance (qtilab.bids);
- ``--mask median-otsu`` (the default; the same function as tractline's) or a mask file;
- ``--device auto`` picks the Apple GPU, then CUDA, then the CPU;
- progress goes to standard error, ``--json FILE`` (or ``-``, standard output) writes the run's summary, ``-q`` quiets it;
- an output writes a provenance sidecar beside it (``<prefix>.qtilab.json``, as ``<trx>.tractline.json``);
- input errors end in one line; ``QTILAB_DEBUG=1`` shows the traceback.

Several series (e.g. LTE and STE runs) are concatenated along the 4th axis and must share a grid: they must already
be motion- and eddy-corrected and aligned (this tool fits, it does not register; tractline's ``--motion-eddy``
corrects within a series). Each series' b-tensor shape comes from ``--btens``, else from its BIDS name
(``acq-linear``/``acq-planar``/``acq-spherical``) or a ``BTensorShape`` sidecar key.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import click
import numpy as np

MAPS = ("md", "fa", "ufa", "c_md", "c_mu", "c_m", "c_c", "k_bulk", "k_shear", "mk", "k_mu", "v_md", "v_shear", "v_iso", "s0")
DEFAULT_MAPS = ("md", "fa", "ufa", "c_md", "c_mu", "c_c", "k_bulk", "k_shear", "mk", "k_mu")
UNITS = {"md": "um^2/ms", "v_md": "(um^2/ms)^2", "v_shear": "(um^2/ms)^2", "v_iso": "(um^2/ms)^2", "s0": "signal"}


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("qtilab")
    except Exception:
        return "unknown"


def resolve_device(device="auto") -> str:
    """"auto": the Apple GPU when there is one, then CUDA, then the CPU (tractline.pipeline.resolve_device)."""
    import torch
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _one_line(fn):
    """Input, disk and format errors end in one line; QTILAB_DEBUG=1 shows the traceback (tractline's convention)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            from nibabel.filebasedimages import ImageFileError
        except ImportError:                                               # pragma: no cover
            ImageFileError = ValueError                                   # noqa: N806
        try:
            return fn(*a, **k)
        except (ValueError, OSError, ImageFileError) as e:
            if os.environ.get("QTILAB_DEBUG") == "1":
                raise
            raise click.ClickException(str(e)) from e
    return wrapper


def _shapes_for(spec, n: int, series: Path) -> list[str]:
    """Encoding shape per volume from a --btens value: a shape name for the whole series, or a per-volume file."""
    from .bids import SHAPES, shape_of
    if shape_of(spec) and not Path(spec).exists():
        return [shape_of(spec)] * n
    p = Path(spec)
    if not p.exists():
        raise ValueError(f"--btens {spec!r} (for {series.name}) is neither LTE/PTE/STE nor a file")
    tok = p.read_text().replace(",", " ").split()
    out = [shape_of(t) for t in tok]
    if None in out:
        bad = tok[out.index(None)]
        raise ValueError(f"{p}: entry {bad!r} is not one of {', '.join(SHAPES)} or a b_delta value (1, -0.5, 0)")
    if len(out) != n:
        raise ValueError(f"{p}: {len(out)} entries for the {n} volumes of {series.name}")
    return out


def load_series(dwi, bval=(), bvec=(), btens=(), bval_units="auto"):
    """The series (positional NIfTI paths) with their tables and shapes, concatenated: a dict of arrays."""
    import nibabel as nib
    from . import bids

    dwi = [Path(d) for d in dwi]
    for name, opt in (("--bval", bval), ("--bvec", bvec)):
        if opt and len(opt) != len(dwi):
            raise click.UsageError(f"give one {name} per series ({len(opt)} for {len(dwi)}), or none to find them beside the series")
    if btens and len(btens) not in (1, len(dwi)):
        raise click.UsageError(f"give one --btens, or one per series ({len(btens)} for {len(dwi)})")
    btens = (list(btens) * len(dwi) if len(btens) == 1 else list(btens)) if btens else [None] * len(dwi)
    data, bvals, bvecs, shapes, series, how = [], [], [], [], [], []
    affine = grid = None
    for k, d in enumerate(dwi):
        img = nib.load(d)
        a = np.asarray(img.dataobj, dtype=np.float32)
        a = a[..., None] if a.ndim == 3 else a
        if affine is None:
            affine, grid = img.affine, a.shape[:3]
        elif a.shape[:3] != grid or not np.allclose(img.affine, affine, atol=1e-3):
            raise ValueError(f"{d}: its grid {a.shape[:3]} or affine differs from {dwi[0].name}'s: align and resample first")
        b = np.atleast_1d(np.loadtxt(bval[k])).ravel() if bval else bids._table(d, ".bval")
        v = np.atleast_2d(np.loadtxt(bvec[k])) if bvec else bids._table(d, ".bvec")
        if b is None or v is None:
            raise ValueError(f"{d}: no .bval/.bvec (beside it or inherited); give --bval and --bvec")
        v = v if v.shape[0] == 3 else v.T
        n = a.shape[-1]
        if len(b) != n or v.shape != (3, n):
            raise ValueError(f"{d}: {n} volumes, {len(b)} b-values, bvec {v.shape}")
        if btens[k] is not None:
            sh, src = _shapes_for(btens[k], n, d), "--btens"
        else:
            s1, src = bids.btens_from_bids(d)
            if s1 is None:
                raise ValueError(f"{d}: no b-tensor shape in its name (acq-linear/planar/spherical) or sidecar "
                                 "(BTensorShape); give --btens LTE|PTE|STE or a per-volume file")
            sh = [s1] * n
        data.append(a); bvals.append(b.astype(float)); bvecs.append(v.astype(float))
        shapes += sh
        series += [k] * n
        how.append({"series": str(d), "volumes": n, "btens": sorted(set(sh)), "btens_from": src})
    b = np.concatenate(bvals)
    unit = ("s/mm2" if b.max() > 100 else "ms/um2") if bval_units == "auto" else bval_units
    return {"data": np.concatenate(data, axis=-1), "affine": affine, "bvals": b if unit == "s/mm2" else b * 1000.0,
            "bvecs": np.concatenate(bvecs, axis=1), "shapes": shapes, "series": np.array(series), "bval_units": unit,
            "inputs": how, "paths": dwi}


def protocol_summary(s):
    import torch
    from .protocol import Protocol

    prot = Protocol.from_table(s["bvals"], s["bvecs"], s["shapes"])
    b = np.round(s["bvals"] / 50) * 50
    sh = np.array(s["shapes"])
    b0 = s["bvals"] < 50
    shells = [{"shape": shape, "b": int(bb), "volumes": int(((sh == shape) & (b == bb) & ~b0).sum())}
              for shape in ("LTE", "PTE", "STE") for bb in sorted(set(b[(sh == shape) & ~b0]))]
    rank = int(torch.linalg.matrix_rank(prot.design_matrix()))
    return prot, {"volumes": int(len(sh)), "b0_volumes": int(b0.sum()), "shells": shells, "rank": rank,
                  "bval_units_read_as": s["bval_units"]}


def _describe(summ, say):
    say(f"{summ['volumes']} volumes ({summ['b0_volumes']} with b < 50 s/mm^2); b-values read as {summ['bval_units_read_as']}")
    for r in summ["shells"]:
        say(f"  {r['shape']} b = {r['b']:5d} s/mm^2: {r['volumes']:3d} volumes")
    r = summ["rank"]
    say(f"QTI design rank {r} of 28" + ("" if r == 28 else
        " (rank-deficient: part of the covariance is not observed; the fit picks the minimum-norm solution, "
        "and the usual scalars do not depend on the unobserved part)"))
    if r < 23:
        say("warning: rank below 23; this protocol cannot separate the QTI covariance terms. Consider a reduced model "
            "(e.g. RICE) instead.")


def _emit(summary, json_path):
    text = json.dumps(summary, indent=1, default=str)
    if json_path == "-":
        click.echo(text)
    elif json_path:
        Path(json_path).write_text(text)


def _series_options(f):
    """The inputs shared by fit and info, after the positional DWI paths."""
    for opt in reversed([
        click.option("--btens", multiple=True,
                     help="The b-tensor shape: LTE, PTE or STE for a whole series, or a file with one entry per volume "
                          "(a shape or b_delta 1, -0.5, 0). One for every series, or one per series in order. Default: "
                          "from the BIDS name (acq-linear/planar/spherical) or a BTensorShape sidecar key."),
        click.option("--bval", multiple=True, type=click.Path(exists=True, dir_okay=False),
                     help="A series' .bval when it is not beside it or inherited (one per series, in order)."),
        click.option("--bvec", multiple=True, type=click.Path(exists=True, dir_okay=False),
                     help="A series' .bvec (FSL, 3 x N) when it is not beside it or inherited (one per series, in order)."),
        click.option("--bval-units", type=click.Choice(["auto", "s/mm2", "ms/um2"]), default="auto", show_default=True,
                     help="Units of the b-values; auto: s/mm^2 if any b > 100."),
    ]):
        f = opt(f)
    return click.argument("dwi", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))(f)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_version(), prog_name="qtilab")
def main():
    """Tensor-valued dMRI to constrained QTI maps: QTI+ "dc" (mean tensor and covariance positive semidefinite),
    batched on the CPU, the Apple GPU or CUDA."""


@main.command()
@_series_options
@click.option("--json", "json_path", type=click.Path(dir_okay=False), default=None,
              help="Also write the protocol summary as JSON here ('-' for standard output).")
@click.option("-q", "--quiet", is_flag=True, help="Print nothing but errors (and --json -).")
@_one_line
def info(dwi, btens, bval, bvec, bval_units, json_path, quiet):
    """The protocol of one or more series (shells, encoding shapes, QTI design rank), without fitting. Errors end in one
    line; QTILAB_DEBUG=1 shows the traceback."""
    say = (lambda *a: None) if quiet else (lambda m: click.echo(m, err=True))     # noqa: E731
    s = load_series(dwi, bval, bvec, btens, bval_units)
    say(f"{len(dwi)} series on a {s['data'].shape[:3]} grid: " + "; ".join(
        f"{Path(i['series']).name} {i['volumes']} volumes, {'/'.join(i['btens'])} ({i['btens_from']})" for i in s["inputs"]))
    _, summ = protocol_summary(s)
    _describe(summ, say)
    _emit({"inputs": s["inputs"], "protocol": summ}, json_path)


@main.command()
@_series_options
@click.option("--mask", default="median-otsu", show_default=True,
              help="The brain mask: median-otsu (of the mean b0, DIPY's median_otsu with its defaults, as tractline's) "
                   "or a mask NIfTI on the series' grid.")
@click.option("--nifti", type=click.Path(), default=None,
              help="Write the maps here: a prefix (out/sub-01 -> out/sub-01_md.nii.gz, ..., out/sub-01_mask.nii.gz), or a "
                   "directory ending in /. Its provenance goes beside it (out/sub-01.qtilab.json).")
@click.option("--maps", default=",".join(DEFAULT_MAPS), show_default=True,
              help=f"Comma-separated scalars to write (any of: {', '.join(MAPS)}), or 'all'.")
@click.option("--params", "save_params", is_flag=True,
              help="Also write the 28 parameters as a 4D NIfTI ([ln S0, <D> Mandel 6, C Mandel 21], DIPY's order).")
@click.option("--method", type=click.Choice(["constrained", "wls"]), default="constrained", show_default=True,
              help="constrained: QTI+ dc (WLS where it already satisfies the constraints); wls: DIPY-equivalent WLS only.")
@click.option("--solver", type=click.Choice(["ipm", "admm"]), default="ipm", show_default=True,
              help="The constrained solver: the batched interior-point method, or ADMM (behind results before 2026-11-07).")
@click.option("--device", type=click.Choice(["auto", "mps", "cuda", "cpu"]), default="auto", show_default=True,
              help="Where to run: auto picks the Apple GPU, then CUDA, then the CPU.")
@click.option("--precision", type=click.Choice(["auto", "float64", "float32"]), default="auto", show_default=True,
              help="auto: float64 on the CPU, float32 on a GPU (MPS has no float64).")
@click.option("--workers", type=int, default=None, help="On the CPU: torch threads (default: torch's choice).")
@click.option("--record-absolute-paths", is_flag=True,
              help="Record full file paths in the provenance file, not paths relative to the dataset: for local "
                   "bookkeeping, not for sharing.")
@click.option("--json", "json_path", type=click.Path(dir_okay=False), default=None,
              help="Also write the run's summary as JSON here ('-' for standard output).")
@click.option("-q", "--quiet", is_flag=True, help="Print nothing but errors (and --json -).")
@_one_line
def fit(dwi, btens, bval, bvec, bval_units, mask, nifti, maps, save_params, method, solver, device, precision, workers,
        record_absolute_paths, json_path, quiet):
    """Fit QTI to one or more aligned series (NIfTI, BIDS) and write scalar maps (NIfTI, float32). Errors end in one
    line; QTILAB_DEBUG=1 shows the traceback."""
    import nibabel as nib
    import torch
    from . import fit as qfit
    from . import scalars

    t0 = time.perf_counter()
    say = (lambda *a: None) if quiet else (lambda m: click.echo(m, err=True))     # noqa: E731
    want = MAPS if maps.strip().lower() == "all" else tuple(m.strip() for m in maps.split(",") if m.strip())
    bad = [m for m in want if m not in MAPS]
    if bad:
        raise click.BadParameter(f"unknown map(s) {bad}; choose from {', '.join(MAPS)}", param_hint="--maps")
    if not nifti and not json_path:
        say("note: neither --nifti nor --json given: the fit runs and only its summary is printed")
    dev = resolve_device(device)
    if precision == "auto":
        precision = "float64" if dev == "cpu" else "float32"
    if dev == "mps" and precision == "float64":
        raise click.UsageError("MPS has no float64: use --precision float32 or --device cpu")
    if workers:
        torch.set_num_threads(workers)
    for folder in ([Path(nifti) if nifti.endswith("/") else Path(nifti).parent] if nifti else []) + \
            ([Path(json_path).parent] if json_path and json_path != "-" else []):
        folder.mkdir(parents=True, exist_ok=True)                    # before the fit, not after it

    s = load_series(dwi, bval, bvec, btens, bval_units)
    say(f"{len(dwi)} series: " + "; ".join(f"{Path(i['series']).name} ({i['volumes']} volumes, {'/'.join(i['btens'])} "
                                           f"from {i['btens_from']})" for i in s["inputs"]))
    prot, summ = protocol_summary(s)
    _describe(summ, say)
    data = s["data"]
    if mask == "median-otsu":
        from .mask import median_otsu
        b0 = data[..., s["bvals"] < 50].mean(-1) if (s["bvals"] < 50).any() else data[..., int(np.argmin(s["bvals"]))]
        m = median_otsu(b0)[1]
        mask_how = "median-otsu"
    elif Path(mask).is_file():
        m = np.asarray(nib.load(mask).dataobj) > 0
        if m.shape != data.shape[:3]:
            raise ValueError(f"{mask}: grid {m.shape}, the series' is {data.shape[:3]}")
        mask_how = str(mask)
    else:
        raise click.BadParameter(f"{mask!r} is neither median-otsu nor a file", param_hint="--mask")
    S = np.maximum(data[m].astype(np.float64), 1e-4)                 # DIPY's MIN_POSITIVE_SIGNAL
    say(f"running on {dev} ({precision}): {int(m.sum())} voxels in the mask ({mask_how}), "
        + (f"constrained fit ({solver})" if method == "constrained" else "WLS"))

    X = prot.design_matrix().to(device=dev, dtype=torch.float64 if precision == "float64" else torch.float32)
    St = torch.as_tensor(S, dtype=X.dtype, device=dev)
    t = time.perf_counter()
    p, info = (qfit.wls(X, St), None) if method == "wls" else qfit.constrained(X, St, solver=solver)
    if dev != "cpu":
        getattr(torch, dev).synchronize()
    t_fit = time.perf_counter() - t
    p = p.cpu().double()
    sc = scalars.scalars(p)
    collapsed = scalars.collapsed(sc).numpy()
    keep = ~collapsed
    fit_mask = m.copy()
    fit_mask[m] = keep
    if collapsed.any():
        why = "collapsed to <D> = 0 (no attenuation above the floor)" if method == "constrained" else "with MD <= 0 (WLS)"
        say(f"{int(collapsed.sum())} voxels {why}: left out of the maps and the fit mask")

    written = []
    prefix = None
    if nifti:
        prefix = Path(nifti) / "qti" if nifti.endswith("/") else Path(nifti)

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

    summary = {
        "series": [str(d) for d in dwi], "inputs": s["inputs"], "protocol": summ, "mask": mask_how,
        "voxels": int(m.sum()), "collapsed": int(collapsed.sum()), "method": method,
        "solver": solver if method == "constrained" else None, "device": dev, "precision": precision,
        "seconds": {"fit": round(t_fit, 2), "total": round(time.perf_counter() - t0, 2)},
        **({"n_constrained": info.n_violating, "converged_fraction": info.converged_fraction,
            "solver_stats": dict(info.extra or {})} if info is not None else {}),
        "maps": list(want), "units": {k: UNITS.get(k, "dimensionless") for k in want},
        "nifti": written, "provenance": f"{prefix}.qtilab.json" if prefix else None,
    }
    if prefix:
        Path(f"{prefix}.qtilab.json").write_text(json.dumps(_provenance(summary, s, record_absolute_paths), indent=1, default=str))
    if method == "constrained":
        say(f"fit in {t_fit:.1f} s: {info.n_violating} voxels needed the constrained solve "
            f"({100 * info.n_violating / max(1, len(S)):.0f}%), converged {info.converged_fraction:.4f}")
    else:
        say(f"fit in {t_fit:.1f} s")
    if prefix:
        say(f"{len(written)} NIfTI files at {prefix}_*.nii.gz; provenance {prefix}.qtilab.json; "
            f"{summary['seconds']['total']:.1f} s in all")
    _emit(summary, json_path)


def _provenance(summary, s, absolute):
    """What made the maps: the inputs (relative to their BIDS dataset unless ``absolute``), versions and settings."""
    import platform

    import torch
    from .bids import _root

    def rel(p):
        p = Path(p).resolve()
        if absolute:
            return str(p)
        root = _root(p)
        return str(p.relative_to(root)) if root else p.name

    inputs = [dict(i, series=rel(i["series"])) for i in s["inputs"]]
    keep = {k: summary[k] for k in ("protocol", "mask", "method", "solver", "device", "precision", "voxels", "collapsed",
                                   "maps", "units") if k in summary}
    if summary.get("mask") not in (None, "median-otsu"):
        keep["mask"] = rel(summary["mask"])
    return {"qtilab": _version(), "torch": torch.__version__, "python": platform.python_version(),
            "inputs": inputs, **keep, "outputs": [Path(o).name for o in summary["nifti"]],
            **({"converged_fraction": summary["converged_fraction"], "n_constrained": summary["n_constrained"]}
               if "converged_fraction" in summary else {})}


DOCS = {"readme": "README.md", "guide": "guide.md"}


def _doc(name: str) -> str:
    """A document's Markdown: the copy inside the installed package (qtilab/_docs, put there by the wheel build), else
    the repository's (a checkout)."""
    from importlib.resources import files
    packaged = files("qtilab") / "_docs" / DOCS[name]
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    local = root / (DOCS[name] if name == "readme" else f"docs/{DOCS[name]}")
    if local.is_file():
        return local.read_text(encoding="utf-8")
    raise click.ClickException(f"{DOCS[name]} is not in this installation")


def _helps() -> str:
    """Every command's --help, as the terminal shows it."""
    out = []
    for name, cmd in [("", main)] + sorted(main.commands.items()):
        prog = f"qtilab {name}".strip()
        with cmd.make_context(prog, [], resilient_parsing=True) as ctx:
            out.append(f"## {prog} --help\n\n```\n{cmd.get_help(ctx)}\n```")
    return "\n\n".join(out)


@main.command()
@click.argument("what", type=click.Choice(["all", "readme", "guide", "help"]), default="all")
def docs(what):
    """Print qtilab's documentation as one Markdown text, for reading or for an LLM: its version, every command's help,
    the README and the guide (all: the default), or one of them. Pipe it on, e.g. `qtilab docs | pbcopy`."""
    parts = []
    if what in ("all", "help"):
        parts.append(f"# qtilab {_version()}: the commands\n\n{_helps()}")
    if what in ("all", "readme"):
        parts.append(_doc("readme"))
    if what in ("all", "guide"):
        parts.append(_doc("guide"))
    click.echo("\n\n---\n\n".join(parts))


if __name__ == "__main__":
    sys.exit(main())
