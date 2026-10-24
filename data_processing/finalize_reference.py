"""Finalize reference geometry on the exact 50 Hz training control grid."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import maximum_filter1d
import torch

from retarget_lafan1 import Retargeter
from supermimic_lab.motions import MotionLibrary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((arguments.source / "manifest.json").read_text())
    required = set(json.loads(arguments.clips.read_text()))
    if not manifest["complete"] or not manifest["quality_pass"]:
        raise ValueError("Source preparation has not passed")
    if {record["clip"] for record in manifest["clips"]} != required:
        raise ValueError("The complete requested motion set is required")
    model = Retargeter(arguments.asset_root, arguments.device)
    if digest(model.urdf) != manifest["urdf_sha256"] or digest(model.profile_path) != manifest["profile_sha256"]:
        raise ValueError("Finalization robot must match the retargeting robot")
    arguments.output.mkdir(parents=True, exist_ok=False)
    result = {
        **manifest,
        "fps": 50,
        "complete": False,
        "quality_pass": False,
        "clips": [],
        "source_preparation_manifest_sha256": digest(arguments.source / "manifest.json"),
        "preparation_metric_rate_hz": manifest["fps"],
        "runtime_control_rate_hz": 50,
        "finalization": "Exact training resampling followed by bounded whole-body floor clearance correction",
        "finalizer_sha256": digest(Path(__file__)),
    }
    for source_record in manifest["clips"]:
        path = arguments.source / (source_record["clip"] + ".csv")
        if digest(path) != source_record["sha256"]:
            raise ValueError(f"Source CSV checksum mismatch: {path}")
        library = MotionLibrary([path], len(model.names), "cpu", source_rate=manifest["fps"], control_period=0.02)
        states = library.states.numpy()
        root = states[:, :3].copy()
        quaternion = states[:, 3:7].copy()
        coordinates = states[:, 13:13 + len(model.names)].copy()
        _, _, heights = model.geometry(root, quaternion, coordinates)
        required_lift = np.maximum(0.001 - heights, 0)
        envelope = gaussian_filter1d(maximum_filter1d(required_lift, size=5, mode="nearest"), sigma=1, mode="nearest")
        lift = np.maximum(required_lift, envelope)
        if float(lift.max()) > 0.02:
            raise ValueError(f"Interpolation needs more than 2 cm floor correction: {path.name}")
        root[:, 2] += lift
        values = np.concatenate((root, quaternion, coordinates), axis=-1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite control-grid reference")
        maximum_speed_ratio = float(np.max(np.abs(np.diff(coordinates, axis=0)) * 50 / model.velocity_limits.cpu().numpy()))
        if maximum_speed_ratio > 0.86:
            raise ValueError("Control-grid joint velocity exceeds the quality limit")
        lower = model.tree.tree.lower_limits.cpu().numpy()
        upper = model.tree.tree.upper_limits.cpu().numpy()
        if np.any(coordinates < lower - 1e-6) or np.any(coordinates > upper + 1e-6):
            raise ValueError("Control-grid joint positions exceed official limits")
        destination = arguments.output / path.name
        np.savetxt(destination, values, delimiter=",", fmt="%.9f")
        record = {
            "clip": source_record["clip"], "sha256": digest(destination),
            "fps": 50, "output_frames": len(values),
            "quality_pass": True, "quality_issues": [],
            "preparation_30hz": source_record,
            "runtime_50hz": {
                "minimum_mesh_height_before_m": float(heights.min()),
                "minimum_mesh_height_after_m": float((heights + lift).min()),
                "additional_floor_lift_max_m": float(lift.max()),
                "additional_floor_lift_p95_m": float(np.percentile(lift, 95)),
                "joint_velocity_limit_ratio_max": maximum_speed_ratio,
                "quaternion_norm_error_max": float(np.max(np.abs(np.linalg.norm(quaternion, axis=-1) - 1))),
                "root_acceleration_p99_m_s2": float(np.percentile(np.linalg.norm(np.diff(root, n=2, axis=0) * 2500, axis=-1), 99)),
            },
        }
        result["clips"].append(record)
        (arguments.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"clip": record["clip"], **record["runtime_50hz"]}), flush=True)
    result["complete"] = True
    result["quality_pass"] = True
    (arguments.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshots = arguments.output / "source_snapshot"
    snapshots.mkdir()
    for name in ["bvh.py", "retarget_lafan1.py", "finalize_reference.py", "assemble_motions.py"]:
        shutil.copyfile(Path(__file__).parent / name, snapshots / name)


if __name__ == "__main__":
    main()
