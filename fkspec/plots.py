"""Figures for an estimation run.

Three groups, in the order you would actually read them:

* **variance structure** -- where and when the field is energetic;
* **spectral content** -- what the local spectra look like, and whether the
  energy propagates;
* **evolution** -- how the spectra change in time (non-stationarity) and across
  space (non-homogeneity).

Every function returns a matplotlib ``Figure`` and saves it if given ``path``.
"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "radial_bins", "radial_spectrum", "radial_profile", "bin_centres",
    "variance_maps", "intensification", "fk_slices", "dispersion",
    "radial_evolution", "radial_comparison", "dominant_wavenumber",
]


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --- radial helpers ---------------------------------------------------------

def radial_bins(res, n=8):
    """``(kr, edges)`` for radial averaging in the ``(kx, ky)`` plane.

    The outer edge reaches the *corner* radius, not the axis maximum, so power
    in the corners of the wavenumber grid is included rather than silently
    dropped.
    """
    KX, KY = np.meshgrid(res.kx, res.ky, indexing="ij")
    kr = np.hypot(KX, KY)
    return kr, np.linspace(0.0, kr.max(), n + 1)


def radial_spectrum(S, kr, edges):
    """Radially averaged spectrum ``(n_bins, npos)`` from a centred ``S``.

    Collapses the ``(kx, ky)`` plane onto ``|k|``, keeping the frequency axis.
    Empty bins come back NaN.
    """
    last = len(edges) - 2
    out = np.full((len(edges) - 1, S.shape[2]), np.nan)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (kr >= lo) & ((kr <= hi) if b == last else (kr < hi))
        if m.any():
            out[b] = S[m].mean(axis=0)
    return out


def radial_profile(S, kr, edges):
    """``omega``-integrated radial power ``P(|k|)``.

    Averaging over a bin and summing over ``omega`` commute, so this is just
    :func:`radial_spectrum` collapsed along the frequency axis.
    """
    return radial_spectrum(S, kr, edges).sum(axis=1)


def bin_centres(edges):
    return 0.5 * (edges[:-1] + edges[1:])


# --- variance structure -----------------------------------------------------

def variance_maps(res, times, labels=None, path=None):
    """``sigma^2(x, y)`` at selected times, on one shared colour scale."""
    s2 = res.sigma2
    labels = labels or [f"t={r}" for r in times]
    vmax = s2[:, :, list(times)].max()

    fig, axes = plt.subplots(1, len(times), figsize=(3.6 * len(times), 3.4))
    for ax, r, lab in zip(np.atleast_1d(axes), times, labels):
        im = ax.imshow(s2[:, :, r].T, origin="lower", vmin=0, vmax=vmax,
                       extent=[res.x[0], res.x[-1], res.y[0], res.y[-1]])
        ax.set_title(f"{lab}  ({res.t[r]:.2f})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label=r"$\sigma^2$")
    fig.suptitle("local variance structure")
    return _finish(fig, path)


def intensification(res, marks=(), path=None):
    """Domain-mean ``sigma^2`` against time."""
    curve = res.sigma2.mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(res.t, curve, "k-", lw=1.3)
    for r in marks:
        ax.axvline(res.t[r], ls="--", c="0.6")
    peak = int(curve.argmax())
    ax.plot(res.t[peak], curve[peak], "r^", label=f"peak t={peak}")
    ax.set_xlabel("t"); ax.set_ylabel(r"domain-mean $\sigma^2$")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title("evolution of local variance")
    return _finish(fig, path)


# --- spectral content -------------------------------------------------------

def fk_slices(res, p, q, r, n_slices=4, path=None):
    """``(kx, ky)`` maps at a few frequencies for one centre."""
    S = res.spectrum(p, q, r)
    om = res.omega
    idx = np.unique(np.linspace(0, len(om) - 1, n_slices).astype(int))

    fig, axes = plt.subplots(1, len(idx), figsize=(3.3 * len(idx), 3.2))
    for ax, iw in zip(np.atleast_1d(axes), idx):
        im = ax.imshow(S[:, :, iw].T, origin="lower",
                       extent=[res.kx[0], res.kx[-1], res.ky[0], res.ky[-1]])
        ax.axhline(0, c="r", lw=0.6); ax.axvline(0, c="r", lw=0.6)
        ax.set_title(rf"$\omega$={om[iw]:.2f}")
        ax.set_xlabel(r"$k_x$"); ax.set_ylabel(r"$k_y$")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"F-K spectrum at (x={p}, y={q}, t={r})")
    return _finish(fig, path)


def dispersion(res, p, q, r, path=None):
    """``k_x``-``omega`` power, ``k_y`` summed out.

    A field advecting at speed ``v_x`` concentrates power along
    ``omega = k_x v_x``, so a tilted ridge here *is* the propagation signature
    and its slope is the speed.
    """
    S = res.spectrum(p, q, r).sum(axis=1)  # (kx, omega)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(S, origin="lower", aspect="auto",
                   extent=[res.omega[0], res.omega[-1], res.kx[0], res.kx[-1]])
    ax.axhline(0, c="w", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$k_x$")
    ax.set_title(f"$k_x$-$\\omega$ dispersion at (x={p}, y={q}, t={r})")
    fig.colorbar(im, ax=ax)
    return _finish(fig, path)


# --- evolution --------------------------------------------------------------

def radial_evolution(res, p, q, times, labels=None, path=None):
    """``|k|``-``omega`` spectra at one location across several times."""
    kr, edges = radial_bins(res)
    kc = bin_centres(edges)
    labels = labels or [f"t={r}" for r in times]

    panels = [radial_spectrum(res.spectrum(p, q, r), kr, edges) for r in times]
    vmax = max(np.nanmax(P) for P in panels)
    fig, axes = plt.subplots(1, len(times), figsize=(3.6 * len(times), 3.4))
    for ax, RK, lab in zip(np.atleast_1d(axes), panels, labels):
        im = ax.imshow(RK, origin="lower", aspect="auto", vmin=0, vmax=vmax,
                       extent=[res.omega[0], res.omega[-1], kc[0], kc[-1]])
        ax.set_title(lab)
        ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$|k|$")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"radially averaged spectrum at (x={p}, y={q}): non-stationarity")
    return _finish(fig, path)


def radial_comparison(res, locs, r, labels=None, path=None):
    """Radial power profiles at several locations, same time."""
    kr, edges = radial_bins(res)
    kc = bin_centres(edges)
    labels = labels or [f"({p},{q})" for p, q in locs]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for (p, q), lab in zip(locs, labels):
        ax.plot(kc, radial_profile(res.spectrum(p, q, r), kr, edges), "-o", label=lab)
    ax.set_xlabel(r"$|k|$"); ax.set_ylabel(r"$\omega$-integrated power")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"spatial non-homogeneity at t={r}")
    return _finish(fig, path)


def dominant_wavenumber(res, r, path=None):
    """Map of the ``|k|`` carrying most power at each centre."""
    kr, edges = radial_bins(res)
    kc = bin_centres(edges)
    slab = res.slab(r)  # (P, Q, Nx, Ny, npos)

    P, Q = slab.shape[:2]
    dom = np.full((P, Q), np.nan)
    for p in range(P):
        for q in range(Q):
            prof = radial_profile(slab[p, q], kr, edges)
            if np.isfinite(prof).any():
                dom[p, q] = kc[np.nanargmax(prof)]

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(dom.T, origin="lower",
                   extent=[res.x[0], res.x[-1], res.y[0], res.y[-1]])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"dominant $|k|$ at t={r}")
    fig.colorbar(im, ax=ax, label=r"$|k|$")
    return _finish(fig, path)
