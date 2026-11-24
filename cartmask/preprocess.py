"""Resampling, gap filling and shaded rendering of height maps."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .x3p import Surface


def resample(surf: Surface, um_per_px: float) -> Surface:
    """Area-average the height map to a coarser isotropic grid.

    Invalid (NaN) samples are excluded from the average; an output pixel is
    invalid when fewer than half of its source samples were valid.
    """
    sx = surf.dx * 1e6 / um_per_px          # output/input size ratio per axis
    sy = surf.dy * 1e6 / um_per_px
    if sx >= 1.0 and sy >= 1.0 and abs(surf.dx - surf.dy) <= 1e-9 * surf.dx:
        return surf                          # already coarse enough and square: never upsample
    if sx > 1.0 or sy > 1.0:                 # anisotropic and coarser than target on one axis:
        um_per_px = max(surf.dx, surf.dy) * 1e6   # square the grid at the coarser spacing
        sx, sy = surf.dx * 1e6 / um_per_px, surf.dy * 1e6 / um_per_px
    h, w = surf.z.shape
    size = (max(1, round(w * sx)), max(1, round(h * sy)))
    valid = np.isfinite(surf.z)
    z0 = np.where(valid, surf.z, 0.0)
    num = cv2.resize(z0, size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(valid.astype(np.float64), size, interpolation=cv2.INTER_AREA)
    z = np.where(den >= 0.5, num / np.maximum(den, 1e-9), np.nan)
    return Surface(z=z, dx=um_per_px * 1e-6, dy=um_per_px * 1e-6, name=surf.name)


def fill_nearest(z: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest valid value (for filtering/gradients only)."""
    invalid = ~np.isfinite(z)
    if not invalid.any():
        return z.copy()
    idx = ndi.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def hillshade(z: np.ndarray, px_m: float, valid: np.ndarray | None = None,
              azimuth_deg: float = 315.0, altitude_deg: float = 35.0,
              exaggeration: float = 3.0) -> np.ndarray:
    """Render a height map (metres) as an 8-bit shaded relief image.

    This mimics the grey "virtual light" renderings produced by comparison
    microscopes (Evofinder, IBIS, TopMatch) that the brief's figures use.
    """
    if valid is None:
        valid = np.isfinite(z)
    zf = fill_nearest(z) if not valid.all() else z
    zf = ndi.gaussian_filter(zf, 0.7)
    gy, gx = np.gradient(zf * exaggeration, px_m)
    az, alt = np.deg2rad(azimuth_deg), np.deg2rad(altitude_deg)
    # light direction: compass azimuth (0 = up, 315 = upper left); image rows grow downward
    lx, ly, lz = np.cos(alt) * np.sin(az), -np.cos(alt) * np.cos(az), np.sin(alt)
    shade = (-gx * lx - gy * ly + lz) / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    v = shade[valid] if valid.any() else shade.ravel()
    lo, hi = np.percentile(v, [0.5, 99.5])
    img = np.clip((shade - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (img * 235 + 10).astype(np.uint8)
    img[~valid] = 0
    return img
