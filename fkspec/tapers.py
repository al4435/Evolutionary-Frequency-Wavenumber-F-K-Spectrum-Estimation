"""DPSS (Slepian) taper banks for the 3-D evolutionary F-K estimator.

A centred window at index ``c`` on an axis of length ``L`` spans
``c-h .. c+h`` with ``h = (N-1)//2``.  Near the edges it is truncated, so the
estimator has to work with several different window lengths on the same axis.

The key design choice is that the spectral half-bandwidth is held fixed in
**normalised** frequency, ``W = NW / N``.  A truncated window of length ``N'``
therefore uses ``NW' = W * N'`` and resolves the *same physical band* as a full
interior window -- it just supports fewer well-concentrated tapers.  That is
what makes spectra estimated from different window sizes comparable once they
are placed on a common grid (see :mod:`fkspec.estimator`).
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import dpss

__all__ = ["Taper", "window_lengths", "build_bank", "build_banks"]


@dataclass(frozen=True)
class Taper:
    """The DPSS bank for one axis at one realised window length."""

    N: int          #: window length in samples
    NW: float       #: time-bandwidth product used for this length
    E: np.ndarray   #: (K, N) unit-energy tapers
    lam: np.ndarray  #: (K,) concentration ratios in [0, 1]

    @property
    def K(self) -> int:
        """Number of tapers."""
        return self.E.shape[0]


def window_lengths(n_full: int, n_axis: int) -> list[int]:
    """Every window length that actually occurs on an axis of length ``n_axis``."""
    h = (n_full - 1) // 2
    return sorted({min(c, h) + min(n_axis - 1 - c, h) + 1 for c in range(n_axis)})


def build_bank(n_full: int, n_axis: int, nw: float) -> dict[int, Taper]:
    """Build the taper bank for one axis, keyed by realised window length.

    Parameters
    ----------
    n_full : Full (interior) window length.  **Must be odd** -- a centred
        window spans ``2h+1`` samples, so an even length can never be realised
        and would silently cost one taper on that axis while forcing every
        block through the common-grid interpolation.
    n_axis : Length of the field along this axis.
    nw : Time-bandwidth product for the full window.
    """
    if n_full % 2 == 0:
        raise ValueError(
            f"window length must be odd, got {n_full}: a centred window spans "
            f"2h+1 samples, so {n_full} can never be realised "
            f"(the largest is {n_full - 1})"
        )
    if n_full > n_axis:
        raise ValueError(f"window {n_full} exceeds axis length {n_axis}")

    W = nw / n_full  # normalised half-bandwidth, held fixed across lengths
    bank = {}
    for N in window_lengths(n_full, n_axis):
        nwp = W * N
        K = max(1, int(np.floor(2 * nwp)))
        E, lam = dpss(N, nwp, Kmax=K, return_ratios=True)
        bank[N] = Taper(N=N, NW=nwp, E=E, lam=lam)
    return bank


def build_banks(shape, window, nw: float) -> tuple[dict, dict, dict]:
    """Banks for all three axes.  ``shape`` and ``window`` are ``(x, y, t)``."""
    return tuple(build_bank(n, L, nw) for n, L in zip(window, shape))


def describe(bank: dict[int, Taper]) -> str:
    """One line per realised window length -- handy for a sanity check."""
    return "\n".join(
        f"  N={t.N:3d}  NW={t.NW:5.3f}  K={t.K}  "
        f"lambda=[{', '.join(f'{v:.4f}' for v in t.lam)}]"
        for t in bank.values()
    )
