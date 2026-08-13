"""Animated slideshow of the local variance field.

Shows ``sigma^2(x, y, t)`` beside the domain-mean curve with a marker tracking
the current frame, so each map is read against the field's overall life cycle
rather than in isolation.

The colour scale is fixed across all frames and computed once over the whole
cube.  That is the whole point: with per-frame autoscaling every map looks
equally intense and the evolution becomes invisible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

__all__ = ["variance_slideshow"]


def variance_slideshow(res, path, fps=8, stride=1, log=False, clip_pct=100.0,
                       cmap="viridis", writer=None):
    """Write an animation of ``sigma^2(x, y, t)``.

    Parameters
    ----------
    res : an open :class:`fkspec.results.Results`.
    path : output file; ``.gif`` uses Pillow, ``.mp4`` needs ffmpeg installed.
    stride : take every n-th time step (``2`` halves the frame count).
    log : plot ``log10(sigma^2)``; useful only if the field spans decades.
    clip_pct : upper colour limit as a percentile over all frames.  ``100`` is
        the true maximum; lower it (say ``99.5``) to bring out mid-range
        structure at the cost of saturating the brightest cells.
    """
    s2 = res.sigma2
    field = np.log10(np.maximum(s2, np.finfo(float).tiny)) if log else s2
    label = r"$\log_{10}\sigma^2$" if log else r"$\sigma^2$"

    lo = float(field.min())
    hi = float(np.percentile(field, clip_pct))
    if not hi > lo:
        hi = lo + 1.0

    curve = s2.mean(axis=(0, 1))
    peak = int(curve.argmax())
    t, frames = res.t, range(0, s2.shape[2], stride)

    fig, (ax_map, ax_cur) = plt.subplots(1, 2, figsize=(10.2, 4.3))
    im = ax_map.imshow(field[:, :, 0].T, origin="lower", vmin=lo, vmax=hi, cmap=cmap,
                       extent=[res.x[0], res.x[-1], res.y[0], res.y[-1]])
    ax_map.set_xlabel("x"); ax_map.set_ylabel("y")
    title = ax_map.set_title("")
    fig.colorbar(im, ax=ax_map, label=label)

    ax_cur.plot(t, curve, "k-", lw=1.2)
    ax_cur.plot(t[peak], curve[peak], "k^", mfc="0.6", label=f"peak t={peak}")
    marker, = ax_cur.plot(t[0], curve[0], "ro", ms=8)
    ax_cur.set_xlim(t[0], t[-1]); ax_cur.grid(alpha=0.3); ax_cur.legend()
    ax_cur.set_xlabel("t"); ax_cur.set_ylabel(r"domain-mean $\sigma^2$")
    ax_cur.set_title("evolution")
    fig.tight_layout()

    def update(r):
        im.set_data(field[:, :, r].T)
        marker.set_data([t[r]], [curve[r]])
        title.set_text(rf"local variance $\sigma^2$  |  t={r}  ({t[r]:.2f})")
        return im, marker, title

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    if str(path).endswith(".gif"):
        anim.save(path, writer=writer or PillowWriter(fps=fps))
    else:
        anim.save(path, writer=writer, fps=fps)
    plt.close(fig)
    return path
