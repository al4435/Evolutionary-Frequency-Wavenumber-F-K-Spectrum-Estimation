"""Thomson adaptive multitaper F-K spectrum of a single local block.

This is the numerical core.  Everything else in the package feeds it blocks or
consumes what it returns.

The estimate for one block is

    S(k) = sum_j |d_j(k)|^2 S_j(k) / sum_j |d_j(k)|^2 ,

where ``S_j`` is the eigenspectrum from the j-th separable 3-D taper and the
adaptive weights ``d_j`` trade each taper's concentration ``lambda_j`` against
the broadband leakage floor, as in Thomson (1982):

    d_j = sqrt(lambda_j) S / (lambda_j S + (1 - lambda_j) sigma^2).

Because ``numpy.fft.fftn`` is unnormalised and the DPSS tapers carry unit
energy, ``mean(S_j) == sigma^2`` already -- so the eigenspectra and the
leakage floor share one scale with no fudge factor.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = ["taper_product", "local_spectrum", "to_common_grid"]

_EPS = float(np.finfo(np.float64).eps)


def taper_product(bx, by, bt):
    """Separable 3-D tapers and their concentrations, flattened over taper index.

    Returns ``(H, lam)`` with ``H`` of shape ``(K, Nx, Ny, Nt)`` and ``lam`` of
    shape ``(K,)``, ``K = Kx*Ky*Kt``.  Depends only on the window-length triple,
    so callers should cache it (see :mod:`fkspec.pipeline`).
    """
    H = np.einsum("ai,bj,ck->abcijk", bx.E, by.E, bt.E)
    lam = np.einsum("a,b,c->abc", bx.lam, by.lam, bt.lam)
    return H.reshape(lam.size, *H.shape[3:]), lam.reshape(lam.size)


def local_spectrum(block, H, lam, shape_common, n_iter=5):
    """Adaptive multitaper F-K spectrum of one block, on the common grid.

    Parameters
    ----------
    block : ``(Np, Nq, Nr)`` local sub-volume of the field.
    H, lam : from :func:`taper_product`, matching ``block.shape``.
    shape_common : ``(Nx, Ny, Nt)`` grid every centre is mapped onto.
    n_iter : adaptive weighting iterations.

    Returns
    -------
    phi : ``(Nx, Ny, Nt//2+1)`` float32, the ``omega >= 0`` half in native FFT
        bin order, scaled so the full two-sided grid integrates to ``sigma2``.
    sigma2 : local variance after mean removal.
    mu : the local mean that was removed (needed to reconstruct the field).
    """
    shape_common = tuple(shape_common)
    npos = shape_common[2] // 2 + 1

    mu = float(block.mean())
    b = block - mu
    sigma2 = float((b * b).mean())
    if sigma2 <= 0.0:  # constant block: no fluctuation to resolve
        return np.zeros(shape_common[:2] + (npos,), np.float32), 0.0, mu

    # eigenspectra for every taper at once -- one batched 3-D FFT
    Sk = np.abs(np.fft.fftn(b * H, axes=(-3, -2, -1))) ** 2

    # Seed from the two most concentrated eigenspectra -- Thomson's two
    # lowest-order tapers.  Picking them by lambda rather than by position
    # keeps the result independent of how the separable tapers happen to be
    # flattened; seeding off an arbitrary neighbour instead leaves a ~1e-4
    # residue after five iterations.
    top = np.argsort(lam)[-2:] if lam.size > 1 else [0, 0]
    S = 0.5 * (Sk[top[0]] + Sk[top[1]])

    lam_ = lam[:, None, None, None]
    root = np.sqrt(lam_)
    for _ in range(n_iter):
        d2 = (root * S / (lam_ * S + (1.0 - lam_) * sigma2)) ** 2
        S = (d2 * Sk).sum(0) / np.maximum(d2.sum(0), _EPS)

    if S.shape != shape_common:
        S = to_common_grid(S, shape_common)

    # Discrete Parseval, enforced on the FULL two-sided grid so that
    # mean(S) == sigma2 exactly at every centre, boundaries included.
    S *= sigma2 * S.size / max(S.sum(), _EPS)

    return S[:, :, :npos].astype(np.float32), sigma2, mu


def to_common_grid(S, shape_common):
    """Resample a block's spectrum onto the common grid in normalised frequency.

    Boundary blocks are smaller than the interior window, so their FFT bins sit
    at different physical frequencies.  Interpolating in normalised frequency
    (cycles per sample) is what puts every centre on one shared ``(kx, ky,
    omega)`` sampling -- and it is only legitimate because the taper banks hold
    the half-bandwidth fixed in those same units.
    """
    padded = np.fft.fftshift(S)
    src = []
    for axis in range(padded.ndim):
        padded, coord = _wrap_pad(padded, axis)
        src.append(coord)

    interp = RegularGridInterpolator(
        tuple(src), padded, method="linear", bounds_error=False, fill_value=0.0
    )
    dst = tuple((np.arange(n) - n // 2) / n for n in shape_common)
    pts = np.stack(np.meshgrid(*dst, indexing="ij"), axis=-1)
    return np.maximum(np.fft.ifftshift(interp(pts)), 0.0)


def _wrap_pad(S, axis):
    """Extend one axis by a sample at each end, using the DFT's periodicity.

    A DFT spectrum is periodic in normalised frequency with period 1, so the
    bin just past each end is the one from the opposite end.

    This is not a cosmetic guard.  An even-length axis has its folded Nyquist
    bin at -1/2 with no +1/2 partner, so a bare grid is asymmetric about zero:
    the interpolator then zero-fills on one side only, the resampled spectrum
    stops being Hermitian, and the variance identity fails -- by ~10% when the
    even axis is time.  Wrapping restores the symmetry and removes the
    zero-fill entirely, so no edge power is fabricated or lost.
    """
    n = S.shape[axis]
    step = 1.0 / n
    coord = (np.arange(n) - n // 2) * step
    S = np.concatenate(
        [S.take([n - 1], axis=axis), S, S.take([0], axis=axis)], axis=axis
    )
    return S, np.concatenate([[coord[0] - step], coord, [coord[-1] + step]])


def hermitian_weights(nt):
    """Weights that recover a full-grid sum from the stored ``omega >= 0`` half.

    The spectrum of a real field obeys ``S(k) == S(-k)``, so once ``kx`` and
    ``ky`` are summed over, every positive-omega bin stands for two -- except DC,
    and the Nyquist bin when ``nt`` is even (both are their own conjugates).
    """
    npos = nt // 2 + 1
    w = np.full(npos, 2.0)
    w[0] = 1.0
    if nt % 2 == 0:
        w[-1] = 1.0
    return w
