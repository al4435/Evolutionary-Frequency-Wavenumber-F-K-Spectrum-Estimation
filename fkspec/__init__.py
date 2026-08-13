"""Evolutionary F-K spectrum estimation for non-homogeneous, non-stationary fields.

Estimates a local frequency-wavenumber spectrum ``Phi(kx, ky, omega; x, y, t)``
at every point of a real field ``R(x, y, t)``, using Thomson's adaptive
multitaper method on a sliding 3-D window.

Typical use::

    from fkspec import Config, estimate, Results, load_field

    R = load_field("field.npy")
    estimate(R, "Phi.h5", Config(window=(15, 15, 23), spacing=(1.0, 1.0, 5/60)))

    with Results("Phi.h5") as res:
        S = res.spectrum(25, 25, 111)
"""

from .estimator import hermitian_weights, local_spectrum, taper_product
from .pipeline import Config, estimate
from .results import Results, load_field
from .tapers import Taper, build_bank, build_banks, window_lengths

__version__ = "0.1.0"

__all__ = [
    "Config", "estimate", "Results", "load_field",
    "local_spectrum", "taper_product", "hermitian_weights",
    "Taper", "build_bank", "build_banks", "window_lengths",
]
