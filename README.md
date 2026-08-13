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

R = load_field("R_field.mat")                       # (x, y, t)
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
python examples/run_gard2002.py path/to/R_field.mat --out results/
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
zero-filling — see [Differences from the MATLAB original](#differences-from-the-matlab-original).

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

`tests/parity_matlab.py` is a separate local script for comparing against the
original MATLAB output.

## Differences from the MATLAB original

**Fixed: Hermitian symmetry on even-length blocks.** An even-length axis folds
its Nyquist bin onto $-1/2$ with no $+1/2$ partner, so the grid is asymmetric
about zero. The MATLAB version zero-filled the missing side, which cost the
resampled spectrum its Hermitian symmetry and broke the variance identity by
**~10%** when the even axis was time. Real boundary blocks hit even lengths
constantly. `_wrap_pad` extends each axis periodically instead, which restores
symmetry and removes the zero-fill, so no edge power is fabricated or lost.
The identity now holds to ~1e-9 (float32 storage precision) at every centre.

**Fixed: radial bins reached only the axis maximum**, dropping power in the
corners of the wavenumber grid. `radial_bins` now reaches the corner radius.

**Merged:** the MATLAB pipeline recomputed local means in a second pass over
every window. The estimator already has that number, so it returns it and it
is stored as `/mu`.

**Array order:** MATLAB wrote `Phi[kx, ky, w, x, y, t]` column-major; this
package writes `Phi[x, y, t, kx, ky, w]` C-ordered. Both keep one local
spectrum contiguous.

Verified against the MATLAB run: `/sigma2` agrees over all 327,500 centres to
float64 round-off (max relative error 5.8e-14).

## Reference

Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
*Proceedings of the IEEE*, 70(9), 1055–1096.
