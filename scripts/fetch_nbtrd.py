"""Download pinned NIST X3P scans or verify local copies offline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/scan-manifest.csv"
DATA_SETS = {"default": "raw", "matching": "matching", "glock": "glock", "expanded": "expanded"}
DOWNLOAD_URL = re.compile(
    r"https://tsapps\.nist\.gov/NRBTD/Studies/(?:Cartridge/DownloadMeasurements|"
    r"CartridgeMeasurement/DownloadMeasurement)/[0-9a-f-]{36}"
)


@dataclass(frozen=True)
class Scan:
    path: PurePosixPath
    url: str
    size: int
    sha256: str


def load_manifest(path: Path = MANIFEST) -> list[Scan]:
    scans = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            name = PurePosixPath(row["file"])
            url, size, digest = row["download_url"], int(row["bytes"]), row["sha256"]
            if (len(name.parts) != 3 or name.parts[0] != "data" or name.parts[1] not in DATA_SETS.values()
                    or not re.fullmatch(r"[A-Za-z0-9_.-]+\.x3p", name.name)
                    or name in seen or not DOWNLOAD_URL.fullmatch(url)
                    or size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError(f"invalid or duplicate manifest entry: {row}")
            seen.add(name)
            scans.append(Scan(PurePosixPath(*name.parts[1:]), url, size, digest))
    if not scans:
        raise ValueError(f"manifest is empty: {path}")
    return scans


def get(url: str) -> bytes:
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "cartmask-forensic-research/2.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as exc:
            if attempt == 2:
                raise
            print(f"retry {url}: {exc}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _matches(data: bytes, scan: Scan) -> bool:
    return len(data) == scan.size and hashlib.sha256(data).hexdigest() == scan.sha256


def _extract(data: bytes, scan: Scan) -> bytes:
    if _matches(data, scan):
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if member.filename.lower().endswith(".x3p") and member.file_size == scan.size:
                payload = archive.read(member)
                if _matches(payload, scan):
                    return payload
    raise ValueError("download contained no X3P matching the manifest SHA-256")


def verify(scan: Scan, target: Path) -> None:
    if target.stat().st_size != scan.size:
        raise ValueError("byte count does not match the manifest")
    with target.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != scan.sha256:
            raise ValueError("SHA-256 does not match the manifest")


def fetch(scan: Scan, target: Path) -> None:
    payload = _extract(get(scan.url), scan)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            staging = Path(output.name)
            output.write(payload)
        staging.replace(target)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="default", choices=[*DATA_SETS, "all"],
                        help="all selects the 34 baseline files. Expanded is a separate cohort.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, help="override the destination folder for one set")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="list pinned URLs without making requests")
    mode.add_argument("--verify-only", action="store_true", help="check existing files without making requests")
    args = parser.parse_args(argv)
    if args.out and args.set == "all":
        parser.error("--out requires a single set. Use --data-root with --set all.")
    manifest = ROOT / "docs/expanded-scan-manifest.csv" if args.set == "expanded" else MANIFEST
    scans = [scan for scan in load_manifest(manifest)
             if args.set == "all" or scan.path.parts[0] == DATA_SETS[args.set]]
    failures = 0
    for scan in scans:
        target = args.out / scan.path.name if args.out else args.data_root / scan.path
        if args.dry_run:
            print(f"{target} <- {scan.url}")
            continue
        try:
            if not args.verify_only and not target.exists():
                fetch(scan, target)
            verify(scan, target)
            print(f"verified {target}")
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"{target}: {exc}", file=sys.stderr)
            failures += 1
    if not args.dry_run:
        print(f"{len(scans) - failures}/{len(scans)} scans verified")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
