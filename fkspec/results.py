"""Read-only accessor for an estimation run, with the physical axes attached.

The HDF5 file stores raw bins; this wraps it so callers work in rad/km and
rad/hr instead of indices, and never have to remember which axes need an
``fftshift``.
"""

from pathlib import Path

import h5py
import numpy as np

from .estimator import hermitian_weights

__all__ = ["Results", "load_field"]


class Results:
    """Open an estimation output.  Usable as a context manager.

    >>> with Results("Phi_FK.h5") as res:
    ...     S = res.spectrum(25, 25, 111)      # (kx, ky, omega>=0), centred
    ...     var_map = res.sigma2[:, :, 111]
    """

    def __init__(self, path):
        self._f = h5py.File(path, "r")
        a = self._f.attrs
        self.dx, self.dy, self.dt = float(a["dx"]), float(a["dy"]), float(a["dt"])
        self.nx, self.ny, self.nt = int(a["nx"]), int(a["ny"]), int(a["nt"])
        self.shape = self._f["Phi"].shape[:3]  # (P, Q, M)

    # --- lifecycle ---
    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- whole small datasets, read eagerly (a few MB) ---
    @property
    def sigma2(self):
        """``(P, Q, M)`` local variance."""
        return self._f["sigma2"][()]

    @property
    def mu(self):
        """``(P, Q, M)`` local mean that the estimator removed."""
        return self._f["mu"][()]

    # --- physical axes ---
    @property
    def kx(self):
        """Centred ``kx`` axis in rad per unit length."""
        return 2 * np.pi / (self.nx * self.dx) * (np.arange(self.nx) - self.nx // 2)

    @property
    def ky(self):
        return 2 * np.pi / (self.ny * self.dy) * (np.arange(self.ny) - self.ny // 2)

    @property
    def omega(self):
        """``omega >= 0`` axis in rad per unit time."""
        npos = self.nt // 2 + 1
        return 2 * np.pi / (self.nt * self.dt) * np.arange(npos)

    @property
    def x(self):
        return np.arange(self.shape[0]) * self.dx

    @property
    def y(self):
        return np.arange(self.shape[1]) * self.dy

    @property
    def t(self):
        return np.arange(self.shape[2]) * self.dt

    # --- spectra ---
    def spectrum(self, p, q, r, shift=True):
        """One local spectrum, ``(Nx, Ny, npos)``.

        With ``shift=True`` (the default) the wavenumber axes are ``fftshift``-ed
        so they line up with :attr:`kx` and :attr:`ky`.
        """
        S = self._f["Phi"][p, q, r].astype(np.float64)
        return np.fft.fftshift(S, axes=(0, 1)) if shift else S

    def slab(self, r, shift=True):
        """Every centre's spectrum at one time, ``(P, Q, Nx, Ny, npos)``.

        Tens of MB for a typical run -- much faster than looping ``spectrum``
        when a whole map is needed.
        """
        S = self._f["Phi"][:, :, r].astype(np.float64)
        return np.fft.fftshift(S, axes=(2, 3)) if shift else S

    def variance_check(self, p, q, r):
        """Recover ``sigma^2`` from the stored half-spectrum.

        Consistency check on the stored file, not a validation of the estimate:
        the estimator *enforces* this identity, so a mismatch means the data was
        written, read, or interpreted wrongly -- not that the physics is off.
        """
        S = self.spectrum(p, q, r, shift=False)
        total = (S * hermitian_weights(self.nt)).sum()
        return total / (self.nx * self.ny * self.nt)


def load_field(path, name="R"):
    """Load a real ``(x, y, t)`` field.

    Handles ``.npy``, ``.npz``, ``.h5``/``.hdf5`` and ``.mat``.  For the
    container formats, ``name`` selects the array; if it is absent the first
    array in the file is used.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.ascontiguousarray(np.load(path))

    if suffix == ".npz":
        with np.load(path) as z:
            return np.ascontiguousarray(z[name if name in z else list(z)[0]])

    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            return np.ascontiguousarray(np.array(f[name if name in f else list(f)[0]]))

    if suffix == ".mat":
        try:
            with h5py.File(path, "r") as f:  # v7.3 is HDF5, stored transposed
                return np.ascontiguousarray(
                    np.array(f[name if name in f else list(f)[0]]).T
                )
        except OSError:  # older MAT revisions are not HDF5
            from scipy.io import loadmat

            return np.ascontiguousarray(loadmat(path)[name])

    raise ValueError(f"unsupported field format {suffix!r}: {path}")
