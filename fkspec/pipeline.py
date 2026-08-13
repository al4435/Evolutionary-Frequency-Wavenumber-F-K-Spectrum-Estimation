"""Run the estimator at every centre of a field and stream the result to HDF5.

One local spectrum per grid point is far too much to hold in memory (a
50x50x131 field gives 327,500 spectra, several GB), so the driver walks the
field one time-slice at a time and appends each slice to disk.  Time slices are
independent, which also makes them the natural unit of parallelism.
"""

from dataclasses import dataclass
import multiprocessing as mp
import time

import h5py
import numpy as np

from .estimator import local_spectrum, taper_product
from .tapers import build_banks

__all__ = ["Config", "estimate"]


@dataclass(frozen=True)
class Config:
    """Estimation parameters.

    ``window`` is the full interior window ``(Nx, Ny, Nt)`` in samples and must
    be odd on every axis; ``spacing`` is ``(dx, dy, dt)`` in whatever physical
    units the field uses (they only set the plotting axes).
    """

    window: tuple[int, int, int] = (15, 15, 23)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    nw: float = 2.0
    n_iter: int = 5

    @property
    def npos(self) -> int:
        """Number of stored ``omega >= 0`` bins."""
        return self.window[2] // 2 + 1


# --- worker state -----------------------------------------------------------
# Set once per process by the pool initialiser, so the field and the taper
# banks are shipped to each worker a single time rather than per task.
_G: dict = {}


def _init_worker(R, banks, cfg):
    _G.update(R=R, banks=banks, cfg=cfg, cache={})


def _tapers_for(shape):
    """Separable taper products for a block shape, cached per process.

    Interior blocks all share one shape, so this is built a handful of times
    rather than once per centre.
    """
    cache = _G["cache"]
    if shape not in cache:
        bx, by, bt = _G["banks"]
        cache[shape] = taper_product(bx[shape[0]], by[shape[1]], bt[shape[2]])
    return cache[shape]


def _estimate_slice(r):
    """Every centre at one time index."""
    R, cfg = _G["R"], _G["cfg"]
    P, Q, M = R.shape
    nx, ny, _ = cfg.window
    hx, hy, ht = ((n - 1) // 2 for n in cfg.window)

    t0, t1 = max(0, r - ht), min(M - 1, r + ht)
    phi = np.zeros((P, Q, nx, ny, cfg.npos), np.float32)
    sig2 = np.zeros((P, Q))
    mu = np.zeros((P, Q))

    for q in range(Q):
        y0, y1 = max(0, q - hy), min(Q - 1, q + hy)
        for p in range(P):
            x0, x1 = max(0, p - hx), min(P - 1, p + hx)
            block = R[x0 : x1 + 1, y0 : y1 + 1, t0 : t1 + 1]
            H, lam = _tapers_for(block.shape)
            phi[p, q], sig2[p, q], mu[p, q] = local_spectrum(
                block, H, lam, cfg.window, cfg.n_iter
            )
    return r, phi, sig2, mu


def estimate(R, out_path, cfg=Config(), workers=None, compression=4, progress=True):
    """Estimate the evolutionary F-K spectrum of ``R`` and write it to HDF5.

    Parameters
    ----------
    R : ``(P, Q, M)`` real field, indexed ``(x, y, t)``.
    out_path : destination ``.h5`` (overwritten).
    cfg : :class:`Config`.
    workers : processes to use; ``None`` for all cores, ``1`` to run serially.
    compression : gzip level, or ``None`` to store uncompressed.

    Notes
    -----
    Uses :mod:`multiprocessing`, so on macOS and Windows a script calling this
    must guard its entry point with ``if __name__ == "__main__":``.

    Output layout, C-ordered::

        /Phi     (P, Q, M, Nx, Ny, Nt//2+1)  float32, native FFT bin order
        /sigma2  (P, Q, M)                   float64, local variance
        /mu      (P, Q, M)                   float64, local mean removed

    ``/mu`` is recorded because the estimator removes the local mean before
    transforming; it is needed to reconstruct the field from the spectra.
    """
    R = np.ascontiguousarray(R, dtype=np.float64)
    if R.ndim != 3:
        raise ValueError(f"field must be 3-D (x, y, t), got shape {R.shape}")

    P, Q, M = R.shape
    nx, ny, nt = cfg.window
    banks = build_banks(R.shape, cfg.window, cfg.nw)
    dx, dy, dt = cfg.spacing

    with h5py.File(out_path, "w") as f:
        kw = {"compression": "gzip", "compression_opts": compression} if compression else {}
        phi_d = f.create_dataset(
            "Phi", (P, Q, M, nx, ny, cfg.npos), dtype="float32",
            chunks=(1, 1, 1, nx, ny, cfg.npos), **kw,
        )
        sig_d = f.create_dataset("sigma2", (P, Q, M), dtype="float64")
        mu_d = f.create_dataset("mu", (P, Q, M), dtype="float64")

        f.attrs.update(
            dx=dx, dy=dy, dt=dt, nx=nx, ny=ny, nt=nt,
            nw=cfg.nw, n_iter=cfg.n_iter,
            layout="Phi[x, y, t, kx, ky, omega>=0]; native FFT bin order; "
                   "local mean removed (see /mu)",
        )

        t_start = time.perf_counter()
        if workers == 1:
            _init_worker(R, banks, cfg)
            results = map(_estimate_slice, range(M))
        else:
            results = _parallel(R, banks, cfg, M, workers)

        for done, (r, phi, sig2, mu) in enumerate(results, 1):
            phi_d[:, :, r] = phi
            sig_d[:, :, r] = sig2
            mu_d[:, :, r] = mu
            if progress:
                el = time.perf_counter() - t_start
                print(f"\r  {done:4d}/{M} slices  {el:6.1f}s "
                      f"(eta {el / done * (M - done):5.1f}s)", end="", flush=True)
        if progress:
            print()

    return out_path


def _parallel(R, banks, cfg, M, workers):
    """Yield finished slices from a worker pool, in completion order."""
    with mp.Pool(workers, initializer=_init_worker, initargs=(R, banks, cfg)) as pool:
        yield from pool.imap_unordered(_estimate_slice, range(M))
