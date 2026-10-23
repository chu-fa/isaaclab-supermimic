"""Fetch or verify pinned official A3 assets without installing dependencies."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Offline copy of the pinned upstream tree")
    arguments = parser.parse_args()
    example = Path(__file__).resolve().parents[1] / "examples/a3"
    manifest_path = example / "official_asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_root = output / "official/robot_descriptions/models"
    for entry in manifest["files"]:
        relative_path = Path(entry["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Invalid manifest path: {relative_path}")
        if arguments.source:
            payload = (arguments.source / relative_path).read_bytes()
        else:
            address = (
                "https://raw.githubusercontent.com/AgibotTech/A3-A3U-robot-model/"
                f"{manifest['commit']}/{relative_path.as_posix()}"
            )
            with urllib.request.urlopen(address, timeout=60) as response:
                payload = response.read()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"Official asset checksum mismatch: {relative_path}")
        target = source_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    description = output / "a3_description"
    (description / "urdf").mkdir(parents=True)
    shutil.copyfile(source_root / "a3_t2d5/urdf/model.urdf", description / "urdf/model.urdf")
    shutil.copyfile(example / "runtime_profile.json", description / "runtime_profile.json")
    shutil.copyfile(manifest_path, output / "official_asset_manifest.json")
    print(json.dumps({"output": str(output), "commit": manifest["commit"], "verified_files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
