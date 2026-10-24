"""Fetch the pinned original Ubisoft BVH archive without installing dependencies."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


REVISION = "94084601bacdf9cc3764b5c73daaeccae6035fac"
ARCHIVE_SHA256 = "ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"
ARCHIVE_BYTES = 144051503
REPOSITORY = "ubisoft/ubisoft-laforge-animation-dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    source = f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/lafan1/lafan1.zip"
    archive_path = arguments.output / "lafan1.zip"
    partial_path = archive_path.with_suffix(".zip.partial")
    checksum = hashlib.sha256()
    with urllib.request.urlopen(source, timeout=90) as response, partial_path.open("xb") as output:
        while block := response.read(1024 * 1024):
            checksum.update(block)
            output.write(block)
    if partial_path.stat().st_size != ARCHIVE_BYTES or checksum.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Original LAFAN1 archive failed its fixed size/SHA256 check")
    partial_path.rename(archive_path)
    license_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/license.txt"
    with urllib.request.urlopen(license_url, timeout=40) as response:
        (arguments.output / "license.txt").write_bytes(response.read())
    destination = arguments.output / "bvh"
    destination.mkdir()
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".bvh"):
                continue
            path = destination / Path(member).name
            contents = archive.read(member)
            with path.open("xb") as output:
                output.write(contents)
            records.append({
                "file": path.name, "sha256": hashlib.sha256(contents).hexdigest(),
                "bytes": len(contents),
            })
    if len(records) != 77:
        raise ValueError("Pinned original archive should contain 77 BVH clips")
    manifest = {
        "source": REPOSITORY, "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256, "archive_bytes": ARCHIVE_BYTES,
        "files": records,
    }
    (arguments.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified and extracted {len(records)} original BVH files")


if __name__ == "__main__":
    main()
