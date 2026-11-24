import hashlib
import io
import zipfile
from pathlib import PurePosixPath

import pytest

from scripts import fetch_nbtrd


def example_scan():
    payload = b"good"
    scan = fetch_nbtrd.Scan(PurePosixPath("raw/example.x3p"), "unused", len(payload),
                           hashlib.sha256(payload).hexdigest())
    return scan, payload


def test_pinned_measurement_can_be_second_archive_member():
    scan, payload = example_scan()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("first.x3p", b"bad!")
        archive.writestr("second.x3p", payload)
    assert fetch_nbtrd._extract(buffer.getvalue(), scan) == payload
    assert fetch_nbtrd._extract(payload, scan) == payload


def test_verify_only_detects_same_size_corruption_without_network(tmp_path, monkeypatch):
    scan, _ = example_scan()
    corrupted = tmp_path / scan.path
    corrupted.parent.mkdir()
    corrupted.write_bytes(b"bad!")
    monkeypatch.setattr(fetch_nbtrd, "load_manifest", lambda manifest: [scan])
    monkeypatch.setattr(fetch_nbtrd, "get", lambda url: pytest.fail("offline mode made a request"))
    assert fetch_nbtrd.main(["--verify-only", "--data-root", str(tmp_path)]) == 1
