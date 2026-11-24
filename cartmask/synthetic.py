"""Synthetic cartridge-case head topography with known ground truth.

Used by the test-suite to check the pipeline end to end without real data:
a flat case head, a primer-pocket gap, a primer face with fine roughness, a
hemispherical firing-pin crater with an optional drag tail, and a raised
flow-back ring marking the firing-pin aperture.

Optional features with known masks (all off by default, so the default head is
unchanged):

* ``rect_aperture`` - a rectangular (slotted, Glock-like) flow-back plateau with
  straight parallel step edges instead of the round ring;
* ``aperture_step_um`` - a flat raised plateau inside the round aperture whose
  only trace is a step at the aperture edge (no ring profile);
* ``shear_um`` - a band of striations parallel to the drag axis at the
  aperture edge, on the side opposite the drag (aperture shear);
* ``rough_band_um`` - the same band filled with isotropic roughness instead
  (rough metal that is *not* shear).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .x3p import Surface


def make_head(size_mm: float = 7.0, um_per_px: float = 8.0, primer_center_mm=(0.15, -0.1),
              primer_radius_mm: float = 2.2, pin_offset_mm=(0.1, 0.05), pin_radius_mm: float = 0.7,
              crater_depth_um: float = 400.0, drag_angle_deg: float | None = 30.0,
              drag_length_mm: float = 0.45, aperture_radius_mm: float = 1.1,
              flowback_um: float = 60.0, seed: int = 0, invalid_corners: bool = True,
              rect_aperture: tuple[float, float] | None = None, rect_angle_deg: float = 0.0,
              aperture_step_um: float = 0.0, shear_um: float = 0.0, shear_angle_deg: float | None = None,
              shear_period_mm: float = 0.06, rough_band_um: float = 0.0) -> tuple[Surface, dict]:
    """Synthetic head scan and its ground truth.

    ``rect_aperture`` is ``(width_mm, length_mm)`` of a slot centred on the
    pin with its long axis at ``rect_angle_deg`` (image coordinates); the
    plateau stands ``flowback_um`` above the primer face and replaces the
    round ring.  ``shear_um`` is the amplitude of the striated band and
    ``rough_band_um`` the RMS of isotropic roughness in the same band; the
    band spans +-50 deg around ``shear_angle_deg`` (default: opposite the
    drag, or 180 deg without a drag) between 1.05 x the pin radius and the
    aperture radius.  The truth dict holds ``rect_mask`` and ``shear_mask``
    (boolean, empty when the feature is off) besides the geometry.
    """
    rng = np.random.default_rng(seed)
    n = int(round(size_mm * 1000 / um_per_px))
    px = um_per_px / 1000.0
    yy, xx = (np.indices((n, n)) - n / 2) * px  # mm, centred
    pcx, pcy = primer_center_mm
    fcx, fcy = pcx + pin_offset_mm[0], pcy + pin_offset_mm[1]
    r_primer = np.hypot(xx - pcx, yy - pcy)
    r_pin = np.hypot(xx - fcx, yy - fcy)

    z = np.zeros((n, n))                                     # case head at 0 um
    gap = (r_primer > primer_radius_mm) & (r_primer < primer_radius_mm + 0.35)
    z[gap] = -300.0                                          # primer-pocket gap
    primer = r_primer <= primer_radius_mm
    z[primer] = 15.0                                         # primer face
    # rounded primer-cup edge
    edge = primer & (r_primer > primer_radius_mm - 0.1)
    z[edge] -= (r_primer[edge] - (primer_radius_mm - 0.1)) / 0.1 * 150.0

    rect_mask = np.zeros((n, n), bool)
    if rect_aperture is not None:
        # slotted aperture: flat raised plateau between straight parallel steps
        ra_ = np.deg2rad(rect_angle_deg)
        along = (xx - fcx) * np.cos(ra_) + (yy - fcy) * np.sin(ra_)
        across = -(xx - fcx) * np.sin(ra_) + (yy - fcy) * np.cos(ra_)
        rect_mask = primer & (np.abs(across) <= rect_aperture[0] / 2) & (np.abs(along) <= rect_aperture[1] / 2)
        z[rect_mask] += flowback_um
    else:
        # flow-back ring inside the aperture
        ring = primer & (r_pin < aperture_radius_mm)
        z[ring] += flowback_um * np.clip(1.0 - (r_pin[ring] - pin_radius_mm) / (aperture_radius_mm - pin_radius_mm),
                                         0, 1)
        if aperture_step_um:
            z[ring] += aperture_step_um

    shear_mask = np.zeros((n, n), bool)
    if shear_um or rough_band_um:
        c_deg = shear_angle_deg if shear_angle_deg is not None else (
            (drag_angle_deg + 180.0) if drag_angle_deg is not None else 180.0)
        c = np.deg2rad(c_deg)
        phi = np.arctan2(yy - fcy, xx - fcx)
        shear_mask = (primer & (r_pin > 1.05 * pin_radius_mm) & (r_pin < aperture_radius_mm)
                      & (np.abs(np.angle(np.exp(1j * (phi - c)))) <= np.deg2rad(50)))
        if shear_um:
            # striations run along the shear axis (the line through the pin centre and the band)
            across_axis = -(xx - fcx) * np.sin(c) + (yy - fcy) * np.cos(c)
            z[shear_mask] += shear_um * np.sin(2 * np.pi * across_axis[shear_mask] / shear_period_mm)
        if rough_band_um:
            bump = ndi.gaussian_filter(np.random.default_rng(seed + 1).normal(0, 1, (n, n)), 1.5)
            z[shear_mask] += rough_band_um * bump[shear_mask] / bump.std()

    # hemispherical crater (spherical cap of radius R with given depth)
    R = (pin_radius_mm ** 2 + (crater_depth_um / 1000) ** 2) / (2 * crater_depth_um / 1000)
    cap = r_pin < pin_radius_mm
    crater = np.zeros((n, n))
    crater[cap] = -(np.sqrt(R ** 2 - r_pin[cap] ** 2) - (R - crater_depth_um / 1000)) * 1000
    # drag tail: elongated groove continuing from the crater
    tail_mask = np.zeros((n, n), bool)
    if drag_angle_deg is not None:
        a = np.deg2rad(drag_angle_deg)
        ux, uy = np.cos(a), np.sin(a)
        t = (xx - fcx) * ux + (yy - fcy) * uy                 # along drag
        s = -(xx - fcx) * uy + (yy - fcy) * ux                # across drag
        half_w = pin_radius_mm * 0.55 * np.clip(1 - (t - pin_radius_mm * 0.6) / (drag_length_mm + 0.4), 0, 1)
        tail = (t > 0) & (t < pin_radius_mm + drag_length_mm) & (np.abs(s) < half_w)
        depth = 0.45 * crater_depth_um * np.clip(1 - (t - pin_radius_mm) / drag_length_mm, 0, 1)
        tail_depth = np.where(tail, -depth * np.clip(1 - (s / np.maximum(half_w, 1e-6)) ** 2, 0, 1), 0)
        crater = np.minimum(crater, tail_depth)
        tail_mask = tail & (r_pin > pin_radius_mm) & (tail_depth < -20)
    z = np.where(primer, z + crater, z)

    z += ndi.gaussian_filter(rng.normal(0, 3.0, z.shape), 1.0)  # fine roughness
    z_m = z * 1e-6
    if invalid_corners:
        z_m[np.hypot(xx, yy) > size_mm / 2] = np.nan
    truth = dict(primer_center_px=((pcx / px) + n / 2, (pcy / px) + n / 2), primer_radius_px=primer_radius_mm / px,
                 pin_center_px=((fcx / px) + n / 2, (fcy / px) + n / 2), pin_radius_px=pin_radius_mm / px,
                 aperture_radius_px=aperture_radius_mm / px, drag_angle_deg=drag_angle_deg,
                 tail_mask=tail_mask, rect_mask=rect_mask, shear_mask=shear_mask)
    return Surface(z=z_m, dx=um_per_px * 1e-6, dy=um_per_px * 1e-6, name="synthetic"), truth


def write_x3p(surf: Surface, path) -> None:
    """Write a minimal X3P (float64, Y-up row order) - for tests and demos."""
    import zipfile

    z = surf.z[::-1]  # back to X3P row order (y up)
    ny, nx = z.shape
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:ISO5436_2 xmlns:p="http://www.opengps.eu/2008/ISO5436_2">
 <Record1><Revision>ISO5436 - 2000</Revision><FeatureType>SUR</FeatureType>
  <Axes>
   <CX><AxisType>I</AxisType><DataType>D</DataType><Increment>{surf.dx}</Increment><Offset>0</Offset></CX>
   <CY><AxisType>I</AxisType><DataType>D</DataType><Increment>{surf.dy}</Increment><Offset>0</Offset></CY>
   <CZ><AxisType>A</AxisType><DataType>D</DataType></CZ>
  </Axes></Record1>
 <Record3><MatrixDimension><SizeX>{nx}</SizeX><SizeY>{ny}</SizeY><SizeZ>1</SizeZ></MatrixDimension>
  <DataLink><PointDataLink>bindata/data.bin</PointDataLink></DataLink></Record3>
</p:ISO5436_2>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.xml", xml)
        zf.writestr("bindata/data.bin", z.astype("<f8").tobytes())
