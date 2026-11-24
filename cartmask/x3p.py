"""Minimal reader for ISO 25178-72 X3P surface-topography files.

An X3P file is a ZIP archive holding ``main.xml`` (metadata) and a binary
height matrix (usually ``bindata/data.bin``).  Only the subset used by
ballistics databases is supported: a single regular grid (FeatureType SUR)
whose Z values are stored as little-endian I/L/F/D numbers, X varying fastest.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {"I": "<i2", "L": "<i4", "F": "<f4", "D": "<f8"}


@dataclass
class Surface:
    """Height map with physical sampling.

    z        2-D array of heights in metres, shape (rows=Y, cols=X); NaN = invalid.
    dx, dy   pixel spacing in metres.
    """

    z: np.ndarray
    dx: float
    dy: float
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def um_per_px(self) -> float:
        return self.dx * 1e6


def _strip_ns(root: ET.Element) -> None:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def _text(root: ET.Element, path: str, default: str | None = None) -> str | None:
    el = root.find(path)
    return el.text.strip() if el is not None and el.text else default


def read_x3p(path: str | Path, image_orientation: bool = True) -> Surface:
    """Read an X3P file.

    X3P stores rows with the Y axis pointing up (row 0 = y = 0).  With
    ``image_orientation=True`` the rows are flipped so that the array displays
    the way the specimen is seen (headstamps read correctly).
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        xml_name = next(n for n in names if n.lower().endswith("main.xml"))
        root = ET.fromstring(zf.read(xml_name))
        _strip_ns(root)
        r3 = root.find("Record3")
        r1 = root.find("Record1")
        nx = int(_text(r3, "MatrixDimension/SizeX"))
        ny = int(_text(r3, "MatrixDimension/SizeY"))
        nz = int(_text(r3, "MatrixDimension/SizeZ", "1"))
        dx = float(_text(r1, "Axes/CX/Increment"))
        dy = float(_text(r1, "Axes/CY/Increment"))
        dtype = _DTYPES[_text(r1, "Axes/CZ/DataType", "D")]
        z_inc = float(_text(r1, "Axes/CZ/Increment", "1") or 1)
        z_off = float(_text(r1, "Axes/CZ/Offset", "0") or 0)

        link = _text(r3, "DataLink/PointDataLink", "bindata/data.bin")
        member = next((n for n in names if n.endswith(link)), link)
        data = np.frombuffer(zf.read(member), dtype=dtype).astype(np.float64)
        if data.size != nx * ny * nz:
            raise ValueError(f"{path.name}: expected {nx * ny * nz} values, got {data.size}")
        z = data.reshape(nz, ny, nx)[0]

        # Integer encodings: apply scale/offset. Floats are already in metres.
        if dtype in ("<i2", "<i4"):
            z = z * z_inc + z_off

        valid_link = _text(r3, "DataLink/ValidPointsLink")
        if valid_link:
            vm = next((n for n in names if n.endswith(valid_link)), None)
            if vm:
                bits = np.unpackbits(np.frombuffer(zf.read(vm), np.uint8), bitorder="little")
                if bits.size < nx * ny:  # short mask: missing bits count as valid
                    bits = np.concatenate([bits, np.ones(nx * ny - bits.size, np.uint8)])
                valid = bits[: nx * ny].reshape(ny, nx).astype(bool)
                z = np.where(valid, z, np.nan)

    if image_orientation:
        z = z[::-1].copy()
    return Surface(z=z, dx=dx, dy=dy, name=path.stem)
