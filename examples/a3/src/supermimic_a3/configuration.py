"""A3 task configuration using the public SuperMimic Lab Isaac Lab adapter."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree

from isaaclab.utils import configclass

from supermimic_lab.isaaclab.configuration import TrackingEnvironmentConfig
from supermimic_lab.isaaclab.configuration import configure_robot


@configclass
class A3EnvironmentConfig(TrackingEnvironmentConfig):
    # Override explicitly only when reproducing a historical asset contract.
    expected_urdf_sha256 = "68a6b337ca33cd1fe4fa08c80e15e45f8662d9b53cd752d1b6d4aa11670c3182"

    def configure_assets(self, directory, backend):
        """Read user-provided A3 data. No assets are bundled or downloaded."""
        root = Path(directory).resolve()
        urdf_path = root / "a3_description/urdf/model.urdf"
        profile_path = root / "a3_description/runtime_profile.json"
        usd_path = root / (
            "a3_usd_newton_v1/model.usda" if backend == "newton_mjwarp"
            else "a3_usd_physx_v2/model/model.usda"
        )
        for path in (urdf_path, profile_path, usd_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required external A3 asset: {path}")
        urdf_sha256 = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
        if urdf_sha256 != self.expected_urdf_sha256:
            raise ValueError("A3 demo requires the pinned official T2.5 URDF; see examples/a3/DATA.md")
        profile = json.loads(profile_path.read_text())
        # Physical assembly exclusions diagnosed previously for these assets.
        # These are robot data, not algorithm implementation; provenance is in docs.
        pairs = [
            ("pelvis_link", "torso_Link"),
            ("waist_yaw_Link", "torso_Link"),
            ("waist_yaw_Link", "torso_shell_Link"),
            ("waist_roll_Link", "torso_shell_Link"),
            ("imu_in_torso_Link", "torso_shell_Link"),
            ("left_wrist_roll_Link", "left_wrist_yaw_Link"),
            ("right_wrist_roll_Link", "right_wrist_yaw_Link"),
            ("left_knee_Link", "left_ankle_roll_Link"),
            ("right_knee_Link", "right_ankle_roll_Link"),
            ("left_shoulder_pitch_Link", "torso_shell_Link"),
            ("right_shoulder_pitch_Link", "torso_shell_Link"),
            ("torso_Link", "left_shoulder_roll_Link"),
            ("torso_Link", "right_shoulder_roll_Link"),
        ]
        # The official conversion merges fixed joints, including massless shells,
        # while keeping their geometry and inertia on the nearest moving parent.
        aliases = {}
        if urdf_sha256 == "68a6b337ca33cd1fe4fa08c80e15e45f8662d9b53cd752d1b6d4aa11670c3182":
            model = ElementTree.parse(urdf_path).getroot()
            aliases = {
                joint.find("child").get("link"): joint.find("parent").get("link")
                for joint in model.findall("joint") if joint.get("type") == "fixed"
            }
            for name in aliases:
                while aliases[name] in aliases:
                    aliases[name] = aliases[aliases[name]]
        elif backend == "newton_mjwarp":
            aliases = {"torso_shell_Link": "torso_Link"}
        pairs = [(aliases.get(first, first), aliases.get(second, second)) for first, second in pairs]
        pairs = sorted({tuple(sorted(pair)) for pair in pairs if pair[0] != pair[1]})
        tracked_links = [
            "pelvis_link", "torso_Link", "head_pitch_Link",
            "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
            "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link",
            "left_hip_yaw_Link", "left_knee_Link", "left_ankle_roll_Link",
            "right_hip_yaw_Link", "right_knee_Link", "right_ankle_roll_Link",
        ]
        specification = {
            "profile": profile, "urdf_path": str(urdf_path),
            "usd_path": str(usd_path), "tracked_links": tracked_links,
            "collision_exclusions": pairs, "action_scale_effort_fraction": 0.25,
            "asset_sha256": {
                "urdf": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                "profile": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                "usd": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
            },
        }
        configure_robot(self, specification, backend)
