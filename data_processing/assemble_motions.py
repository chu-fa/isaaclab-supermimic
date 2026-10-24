"""Combine passing batches while requiring the complete requested clip set."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    required = json.loads(arguments.clips.read_text())
    selected = {}
    manifests = []
    contract = None
    for directory in arguments.batch:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete") or not manifest.get("quality_pass"):
            raise ValueError(f"Incomplete or failing batch: {directory}")
        current = {name: manifest[name] for name in (
            "joint_names", "fps", "urdf_sha256", "profile_sha256", "source_code_sha256",
            "source_manifest_sha256", "iterations", "torso_mapping", "method",
        )}
        if contract is not None and current != contract:
            raise ValueError("All batches must use identical sources and processing settings")
        contract = current
        manifests.append({"batch": directory.name, "sha256": digest(manifest_path)})
        for record in manifest["clips"]:
            name = record["clip"]
            if name in selected or name not in required:
                raise ValueError(f"Duplicate or unrequested clip: {name}")
            path = directory / (name + ".csv")
            if not record["quality_pass"] or record["quality_issues"] or digest(path) != record["sha256"]:
                raise ValueError(f"Clip failed verification: {name}")
            selected[name] = (path, record)
    if set(selected) != set(required) or len(required) != len(set(required)):
        raise ValueError("The complete requested clip set is required; no filtering is allowed")
    arguments.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name in sorted(required):
        source, record = selected[name]
        shutil.copyfile(source, arguments.output / source.name)
        records.append(record)
    result = {
        **contract, "intermediate_robot": None, "complete": True, "quality_pass": True,
        "clip_count": len(records), "excluded_clip_count": 0,
        "source_batches": manifests, "clips": records,
        "limitations": manifest["limitations"],
        "source": manifest["source"],
    }
    (arguments.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete_clip_set": len(records), "all_quality_checks_passed": True}))


if __name__ == "__main__":
    main()
