"""Compare a Python run against the original MATLAB output.

Not part of the pytest suite -- it needs the MATLAB HDF5, which is not in the
repo.  Run it locally to confirm the port reproduces the reference:

    python tests/parity_matlab.py MATLAB.h5 PYTHON.h5

Array order differs between the two.  MATLAB writes ``Phi[kx, ky, w, x, y, t]``
column-major, so h5py sees ``[t, y, x, w, ky, kx]``; the Python writer uses
``Phi[x, y, t, kx, ky, w]`` C-ordered, which h5py sees unchanged.

``sigma2`` and ``mu`` are window geometry only, so they should agree to
round-off for any window with the same half-widths.  ``Phi`` can only be
compared when both runs used the same ``Nt`` -- and expect the Python side to
differ near the boundaries, where it fixes a Hermitian-symmetry bug in the
MATLAB interpolation (see ``_wrap_pad`` in fkspec/estimator.py).
"""

import sys

import h5py
import numpy as np


def report(name, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.maximum(np.abs(a), 1e-30)
    rel = np.abs(a - b) / denom
    print(f"  {name:10s} max {rel.max():.3e}   median {np.median(rel):.3e}   "
          f"{'MATCH' if rel.max() < 1e-6 else 'DIFFERS'}")
    return rel


def main(mat_path, py_path):
    with h5py.File(mat_path, "r") as m, h5py.File(py_path, "r") as p:
        ms2 = m["sigma2"][()]                 # (M, Q, P)
        ps2 = p["sigma2"][()]                 # (P, Q, M)
        print(f"MATLAB sigma2 {ms2.shape} (t,y,x) | Python {ps2.shape} (x,y,t)")

        if ms2.shape[::-1] != ps2.shape:
            sys.exit(f"grid mismatch: {ms2.shape[::-1]} vs {ps2.shape}")

        print("\nlocal variance and mean:")
        report("sigma2", ms2.transpose(2, 1, 0), ps2)
        if "mu" in p:
            print("  mu         (MATLAB stores this separately, in block_means.mat)")

        mphi, pphi = m["Phi"], p["Phi"]
        m_npos, p_npos = mphi.shape[3], pphi.shape[5]
        print(f"\nspectra: MATLAB npos={m_npos}, Python npos={p_npos}")
        if m_npos != p_npos:
            print("  SKIPPED - different Nt, so the frequency grids are not "
                  "comparable.  Re-run MATLAB with the same window to compare.")
            return

        rng = np.random.default_rng(0)
        P, Q, M = ps2.shape
        interior, boundary = [], []
        for _ in range(200):
            x, y, t = rng.integers(P), rng.integers(Q), rng.integers(M)
            a = mphi[t, y, x].T                     # -> (kx, ky, w)
            b = pphi[x, y, t]
            rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
            edge = (min(x, P - 1 - x) < 7 or min(y, Q - 1 - y) < 7
                    or min(t, M - 1 - t) < 11)
            (boundary if edge else interior).append(rel)

        for name, vals in [("interior", interior), ("boundary", boundary)]:
            if vals:
                print(f"  {name:10s} n={len(vals):3d}  max rel {max(vals):.3e}  "
                      f"median {np.median(vals):.3e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
