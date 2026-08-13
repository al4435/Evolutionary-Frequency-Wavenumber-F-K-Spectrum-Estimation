"""End-to-end run on the Gard 2002 rainfall field.

Estimates the evolutionary F-K spectrum of a 50 x 50 x 131 field (1 km grid,
5 min steps), then writes the full figure set and the variance animation.

    python examples/run_gard2002.py path/to/R_field.mat --out results/

Roughly 8 minutes on 10 cores; pass ``--workers 1`` to run serially.
"""

import argparse
import sys
from pathlib import Path

# run from a fresh clone without installing first
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkspec import Config, Results, estimate, load_field
from fkspec.animate import variance_slideshow
from fkspec import plots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("field", type=Path, help=".mat file holding the field")
    ap.add_argument("--name", default="R", help="variable name inside the .mat")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--window", type=int, nargs=3, default=(15, 15, 23),
                    help="local window (Nx Ny Nt); all must be odd")
    ap.add_argument("--skip-estimate", action="store_true",
                    help="reuse an existing Phi.h5 and only redraw figures")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    h5 = args.out / "Phi.h5"

    cfg = Config(
        window=tuple(args.window),
        spacing=(1.0, 1.0, 5 / 60),   # km, km, hours
        nw=2.0,
        n_iter=5,
    )

    if not args.skip_estimate:
        R = load_field(args.field, args.name)
        print(f"field {R.shape}, window {cfg.window} -> {h5}")
        estimate(R, h5, cfg, workers=args.workers)

    with Results(h5) as res:
        # Characteristic times, taken from the data rather than hard-coded:
        # the storm's own peak, plus an early and a mid point.
        curve = res.sigma2.mean(axis=(0, 1))
        peak = int(curve.argmax())
        times = [20, res.shape[2] // 2, peak]
        labels = ["early", "mid", "peak"]

        # Characteristic locations: domain centre, the most and least
        # energetic cells at the peak time.
        s2_peak = res.sigma2[:, :, peak]
        P, Q = s2_peak.shape
        hot = divmod(int(s2_peak.argmax()), Q)
        quiet = divmod(int(s2_peak.argmin()), Q)
        locs = [(P // 2, Q // 2), hot, quiet]
        loc_labels = ["centre", "hotspot", "quiet"]

        print(f"peak t={peak}  hotspot={hot}  quiet={quiet}")

        plots.variance_maps(res, times, labels, args.out / "variance_maps.png")
        plots.intensification(res, times, args.out / "intensification.png")

        for (p, q), lab in zip(locs, loc_labels):
            plots.fk_slices(res, p, q, peak, path=args.out / f"fk_slices_{lab}.png")
            plots.dispersion(res, p, q, peak, path=args.out / f"dispersion_{lab}.png")

        plots.radial_evolution(res, *locs[0], times, labels,
                               args.out / "radial_evolution.png")
        plots.radial_comparison(res, locs[1:], peak, loc_labels[1:],
                                args.out / "non_homogeneity.png")
        plots.dominant_wavenumber(res, peak, args.out / "dominant_wavenumber.png")

        variance_slideshow(res, args.out / "variance_evolution.gif", fps=8)

    print(f"done -> {args.out}")


if __name__ == "__main__":   # required: the driver uses multiprocessing
    main()
