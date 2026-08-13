"""Tests on synthetic fields -- no external data needed."""

import numpy as np
import pytest

from fkspec import (
    Config, Results, build_bank, estimate, hermitian_weights,
    local_spectrum, taper_product, window_lengths,
)

WINDOW = (9, 9, 11)


# --- taper banks ------------------------------------------------------------

def test_even_window_rejected():
    """An even window can never be realised by a centred span, so it must fail."""
    with pytest.raises(ValueError, match="odd"):
        build_bank(24, 60, 2.0)


def test_full_window_is_realised():
    """The largest realised length must equal the requested window."""
    for n_full in (9, 15, 23):
        assert window_lengths(n_full, 60)[-1] == n_full


def test_tapers_are_unit_energy():
    bank = build_bank(15, 50, 2.0)
    for t in bank.values():
        assert np.allclose((t.E ** 2).sum(axis=1), 1.0)
        assert np.all((t.lam > 0) & (t.lam <= 1))


def test_bandwidth_fixed_in_normalised_frequency():
    """NW' / N' is constant, so every window resolves the same physical band."""
    bank = build_bank(15, 50, 2.0)
    ratios = [t.NW / t.N for t in bank.values()]
    assert np.allclose(ratios, 2.0 / 15)


# --- the estimator kernel ---------------------------------------------------

def _kernel(block, window, axis_len=60):
    """Run the kernel on one block, picking each axis' bank by block length."""
    bx = build_bank(window[0], axis_len, 2.0)[block.shape[0]]
    by = build_bank(window[1], axis_len, 2.0)[block.shape[1]]
    bt = build_bank(window[2], axis_len, 2.0)[block.shape[2]]
    H, lam = taper_product(bx, by, bt)
    return local_spectrum(block, H, lam, window)


def test_variance_identity_interior():
    """The stored half-spectrum must integrate back to sigma^2 exactly."""
    rng = np.random.default_rng(0)
    block = rng.standard_normal(WINDOW) * 2.5 + 7.0
    phi, sigma2, mu = _kernel(block, WINDOW)

    total = (phi * hermitian_weights(WINDOW[2])).sum()
    assert total / np.prod(WINDOW) == pytest.approx(sigma2, rel=1e-6)
    assert mu == pytest.approx(block.mean())
    assert sigma2 == pytest.approx(block.var())


def test_variance_identity_on_truncated_block():
    """Boundary blocks go through interpolation; the identity must still hold."""
    rng = np.random.default_rng(1)
    block = rng.standard_normal((5, 6, 7))
    phi, sigma2, _ = _kernel(block, WINDOW)

    total = (phi * hermitian_weights(WINDOW[2])).sum()
    assert total / np.prod(WINDOW) == pytest.approx(sigma2, rel=1e-6)


@pytest.mark.parametrize(
    "shape", [(9, 9, 10), (9, 6, 11), (5, 6, 7), (6, 6, 10), (8, 8, 8)]
)
def test_variance_identity_with_even_axes(shape):
    """Even-length source axes must not break the Hermitian symmetry.

    An even axis folds its Nyquist bin onto -1/2 with no +1/2 partner.  Without
    the periodic wrap in ``to_common_grid`` the interpolator zero-fills one side
    only, the resampled spectrum stops being Hermitian, and this identity is off
    by ~10% when the even axis is time.
    """
    rng = np.random.default_rng(abs(hash(shape)) % 2**32)
    phi, sigma2, _ = _kernel(rng.standard_normal(shape), WINDOW)
    total = (phi * hermitian_weights(WINDOW[2])).sum()
    assert total / np.prod(WINDOW) == pytest.approx(sigma2, rel=1e-6)


def test_interpolation_preserves_total_power():
    """Resampling must neither fabricate nor lose power at the grid edges."""
    from fkspec.estimator import to_common_grid

    rng = np.random.default_rng(7)
    S = np.abs(np.fft.fftn(rng.standard_normal((8, 9, 10)))) ** 2
    T = to_common_grid(S, (9, 9, 11))
    assert T.mean() == pytest.approx(S.mean(), rel=0.05)
    assert np.all(T >= 0)


def test_constant_block_is_zero():
    phi, sigma2, mu = _kernel(np.full(WINDOW, 3.0), WINDOW)
    assert sigma2 == 0.0 and mu == pytest.approx(3.0)
    assert np.all(phi == 0)


def test_white_noise_is_broadly_flat():
    """White noise has no preferred wavenumber; no bin should dominate."""
    rng = np.random.default_rng(2)
    phi, _, _ = _kernel(rng.standard_normal((15, 15, 15)), (15, 15, 15))
    assert phi.max() < 12 * phi.mean()


def test_propagating_wave_lands_on_the_right_bin():
    """A plane wave must put its power at the (kx, ky, omega) that generated it."""
    n = (15, 15, 15)
    px, py, pt = 2, 3, 4  # cycles across the window
    i, j, k = np.meshgrid(*[np.arange(v) for v in n], indexing="ij")
    field = np.cos(2 * np.pi * (px * i / n[0] + py * j / n[1] - pt * k / n[2]))

    phi, _, _ = _kernel(field, n)
    peak = np.unravel_index(phi.argmax(), phi.shape)
    # native FFT order: +px sits at index px, -py at n-py (the conjugate lobe)
    assert peak[2] == pt
    assert {peak[0], n[0] - peak[0]} == {px, n[0] - px}
    assert {peak[1], n[1] - peak[1]} == {py, n[1] - py}


# --- full pipeline ----------------------------------------------------------

def test_pipeline_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    R = rng.standard_normal((12, 11, 14))
    out = tmp_path / "phi.h5"
    cfg = Config(window=WINDOW, spacing=(1.0, 2.0, 0.5))

    estimate(R, out, cfg, workers=1, progress=False)

    with Results(out) as res:
        assert res.shape == R.shape
        assert res.spectrum(0, 0, 0).shape == (WINDOW[0], WINDOW[1], cfg.npos)
        # mu is the plain local mean, and every centre satisfies the identity
        # window (9,9,11) -> h=(4,4,5); centre (5,5,7) clipped to the field
        assert res.mu[5, 5, 7] == pytest.approx(
            R[1:10, 1:10, 2:13].mean(), rel=1e-9
        )
        for p, q, r in [(5, 5, 7), (0, 0, 0), (11, 10, 13), (0, 5, 13)]:
            assert res.variance_check(p, q, r) == pytest.approx(
                res.sigma2[p, q, r], rel=1e-5
            )


def test_parallel_matches_serial(tmp_path):
    rng = np.random.default_rng(4)
    R = rng.standard_normal((10, 10, 12))
    cfg = Config(window=WINDOW)

    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    estimate(R, a, cfg, workers=1, progress=False)
    estimate(R, b, cfg, workers=2, progress=False)

    with Results(a) as ra, Results(b) as rb:
        assert np.array_equal(ra.sigma2, rb.sigma2)
        assert np.array_equal(ra.spectrum(4, 4, 6), rb.spectrum(4, 4, 6))
