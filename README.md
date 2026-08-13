# fkspec — evolutionary F-K spectrum estimation

Estimates a local frequency–wavenumber spectrum

$$\Phi(k_x, k_y, \omega;\; x, y, t)$$

at **every point** of a real field $R(x,y,t)$, using Thomson's adaptive
multitaper method on a sliding 3-D window.

The point is to handle fields that are neither homogeneous nor stationary — a
storm that intensifies in time and is patchy in space — where a single global
spectrum would average away exactly the structure you care about.

```python
from fkspec import Config, Results, estimate, load_field

R = load_field("field.npy")                         # (x, y, t)
estimate(R, "Phi.h5", Config(window=(15, 15, 23), spacing=(1.0, 1.0, 5/60)))

with Results("Phi.h5") as res:
    S = res.spectrum(25, 25, 111)                   # (kx, ky, omega>=0), centred
    var_map = res.sigma2[:, :, 111]                 # local variance at t=111
```

## Install

```bash
pip install -e ".[test]"
```

Requires Python ≥3.10, numpy, scipy, h5py, matplotlib.

## Run the worked example

```bash
python examples/run_gard2002.py path/to/field.npy --out results/
```

Produces the full figure set and a variance animation. About 8 minutes for a
50×50×131 field on 10 cores; `--workers 1` runs serially.

## How it works

**Local windows.** A centred window at grid point $c$ spans $c-h \dots c+h$ with
$h=(N-1)/2$, clipped at the field edges. Window lengths must be **odd** — an
even $N$ can never be realised by a centred span, so `build_bank` rejects it
rather than silently giving you $N-1$.

**Taper banks.** Truncated boundary windows are shorter, so they get their own
DPSS bank. The half-bandwidth is held fixed in *normalised* frequency,
$W = NW/N$, so a window of length $N'$ uses $NW' = W N'$. Every window then
resolves the same physical band and just supports fewer well-concentrated
tapers — which is what makes spectra from different window sizes comparable.

**Adaptive weighting.** For separable 3-D tapers with concentrations
$\lambda_j$, eigenspectra $S_j$ are combined as

$$S = \frac{\sum_j |d_j|^2 S_j}{\sum_j |d_j|^2}, \qquad
d_j = \frac{\sqrt{\lambda_j}\, S}{\lambda_j S + (1-\lambda_j)\sigma^2},$$

iterated a few times (Thomson 1982). Because `numpy.fft.fftn` is unnormalised
and DPSS tapers carry unit energy, $\mathrm{mean}(S_j) = \sigma^2$ already, so
the eigenspectra and the leakage floor share one scale with no fudge factor.

**Common grid.** Boundary blocks produce spectra on a coarser frequency grid,
so they are resampled onto the interior grid in normalised frequency. The
resampling uses the DFT's periodicity to wrap at the edges rather than
zero-filling — see [Implementation notes](#implementation-notes).

**Variance normalisation.** Each spectrum is scaled so that, on the full
two-sided grid, $\mathrm{mean}(\Phi) = \sigma^2$ exactly at every centre. The
$\omega \ge 0$ half is stored; `Results.variance_check` recovers the full sum
using Hermitian weights.

Only the *shape* of the spectrum is estimated freely — its total is pinned to
the measured local variance. Note this makes `variance_check` a consistency
check on the stored file, not a validation of the estimate.

## Output layout

C-ordered HDF5:

| dataset | shape | meaning |
|---|---|---|
| `/Phi` | `(P, Q, M, Nx, Ny, Nt//2+1)` | float32, native FFT bin order |
| `/sigma2` | `(P, Q, M)` | local variance |
| `/mu` | `(P, Q, M)` | local mean removed before transforming |

`/mu` is recorded because the estimator removes the local mean; you need it to
reconstruct the field from the spectra.

## API

| | |
|---|---|
| `Config` | window, spacing, `nw`, `n_iter` |
| `estimate(R, out, cfg, workers=None)` | run everything, stream to HDF5 |
| `Results` | axes in physical units, `spectrum`, `slab`, `sigma2`, `mu` |
| `load_field(path)` | read a field from MAT ≤v7 or v7.3 |
| `plots` | variance maps, dispersion ridges, radial spectra, … |
| `animate.variance_slideshow` | animated $\sigma^2(x,y,t)$ |

Parallelism uses `multiprocessing`, so scripts calling `estimate` must guard
their entry point with `if __name__ == "__main__":`.

## Tests

```bash
pytest tests/
```

Synthetic fields only — nothing external needed. They pin the variance
identity (interior *and* truncated blocks), taper normalisation, fixed
normalised bandwidth, white-noise flatness, that a plane wave lands on the bin
that generated it, and that parallel and serial runs agree bit for bit.

## Implementation notes

Three details are easy to get wrong and are worth stating explicitly, because
each one silently corrupts results rather than raising an error.

**Even-length axes break Hermitian symmetry.** An even-length axis folds its
Nyquist bin onto $-1/2$ with no $+1/2$ partner, so the normalised-frequency
grid is asymmetric about zero. Resampling such an axis with plain zero-fill
extrapolation pads one side and not the other; the resampled spectrum stops
being Hermitian, and the variance identity fails — by **~10%** when the even
axis is time. Boundary blocks hit even lengths constantly, so this is the
common case, not an edge case. `_wrap_pad` extends each axis periodically
instead, which is what the DFT actually does: no edge power is fabricated or
lost, and the identity holds to ~1e-9 (float32 storage precision) at every
centre, boundaries included.

**Seed the adaptive iteration by concentration, not by position.** The
separable 3-D tapers are flattened into one array, and seeding the iteration
from the first two entries makes the result depend on the flattening order.
Selecting the two most concentrated eigenspectra instead — Thomson's two
lowest-order tapers — is order-independent. Seeding off an arbitrary
neighbour still converges, but leaves a ~1e-4 residue after five iterations.

**Radial bins must reach the corner radius.** Binning $|k|$ only out to the
axis maximum silently discards power in the corners of the wavenumber grid,
where $|k|$ reaches $\sqrt2$ times the axis maximum. `radial_bins` uses the
corner radius.

## Reference

Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
*Proceedings of the IEEE*, 70(9), 1055–1096.
