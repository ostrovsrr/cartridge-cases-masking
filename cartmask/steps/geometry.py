"""Small geometric helpers shared by the pipeline stages.

All coordinates are in pixels of the working grid, ``x`` = column and
``y`` = row (image convention, y grows downward).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def disk(shape: tuple[int, ...], cx: float, cy: float, r: float) -> np.ndarray:
    """Boolean disk mask.

    Parameters
    ----------
    shape
        ``(rows, cols)`` of the output mask.
    cx, cy
        Centre in pixels (column, row).
    r
        Radius in pixels; the boundary is included.

    Returns
    -------
    numpy.ndarray
        Boolean array of ``shape``, True inside the disk.
    """
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit.

    Parameters
    ----------
    x, y
        1-D arrays of point coordinates in pixels.

    Returns
    -------
    tuple of float
        ``(cx, cy, r)``: centre and radius in pixels.
    """
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cy, float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-12)))


def robust_circle(x: np.ndarray, y: np.ndarray, iters: int = 6, k: float = 2.5,
                  min_tol: float = 1.0) -> tuple[float, float, float, np.ndarray]:
    """Circle fit with iterative MAD-based outlier rejection.

    Parameters
    ----------
    x, y
        1-D arrays of point coordinates in pixels.
    iters
        Maximum number of refit / re-select rounds.
    k
        Points farther than ``k`` x (MAD-estimated sigma) from the circle are dropped.
    min_tol
        Lower bound on the inlier tolerance in pixels, so a near-perfect fit
        does not reject everything.

    Returns
    -------
    tuple
        ``(cx, cy, r, keep)``; ``keep`` is the boolean inlier mask of the last
        fit.  The circle is NaN when fewer than 8 points were available.
    """
    keep = np.ones(len(x), bool)
    cx = cy = r = np.nan
    for _ in range(iters):
        if keep.sum() < 8:
            break
        cx, cy, r = fit_circle(x[keep], y[keep])
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        mad = np.median(res[keep]) * 1.4826
        new = res <= max(k * mad, min_tol)
        if (new == keep).all():
            break
        keep = new
    return cx, cy, r, keep


def sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of ``img`` at sub-pixel points (nearest at the border).

    Parameters
    ----------
    img
        2-D array to sample.
    xs, ys
        Column and row coordinates, any matching shape.

    Returns
    -------
    numpy.ndarray
        Interpolated values, same shape as ``xs``.
    """
    return ndi.map_coordinates(img, [ys, xs], order=1, mode="nearest")
