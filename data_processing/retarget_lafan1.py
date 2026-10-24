"""Direct human LAFAN1 BVH to official AGIBOT A3; no intermediate robot data."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ElementTree

import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_opening
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import maximum_filter1d
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import torch
import trimesh

from bvh import read_bvh
from bvh import rigid_torso_frame
from supermimic_lab.geometry import multiply
from supermimic_lab.geometry import conjugate
from supermimic_lab.geometry import orientation_columns
from supermimic_lab.geometry import rotate
from supermimic_lab.geometry import rotation_vector_to_quaternion
from supermimic_lab.kinematics import KinematicTree


OFFICIAL_URDF_SHA256 = "68a6b337ca33cd1fe4fa08c80e15e45f8662d9b53cd752d1b6d4aa11670c3182"
BODY_NAMES = [
    "pelvis_link", "torso_Link", "head_pitch_Link",
    "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
    "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link",
    "left_hip_yaw_Link", "left_knee_Link", "left_ankle_roll_Link",
    "right_hip_yaw_Link", "right_knee_Link", "right_ankle_roll_Link",
]
FOOT_INDICES = [11, 14]
# BVH Hips/Spine/Head: local X points up, Z left, Y forward.
TORSO_BASIS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
# Toe: local X points forward, Y up, -Z left. Ankle-to-toe vectors
# are pitched down even in a flat stance, so they are not sole normals.
FOOT_BASIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
HAND_BASIS = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Degenerate human bone")
    return vectors / lengths


class PoseTree:
    """Evaluate independent branches together; shares the core URDF conventions."""

    def __init__(self, urdf, joint_names, device):
        links = [link.get("name") for link in ElementTree.parse(urdf).findall("link")]
        tree = KinematicTree(urdf, joint_names, links, device)
        self.tree = tree
        self.device = device
        depths = [0]
        for parent, *_ in tree.operations:
            depths.append(depths[parent] + 1)
        self.levels = []
        previous_indices = [0]
        order = [0]
        for depth in range(1, max(depths) + 1):
            indices = [index for index, value in enumerate(depths) if value == depth]
            operations = [tree.operations[index - 1] for index in indices]
            parent_indices = [previous_indices.index(operation[0]) for operation in operations]
            self.levels.append((
                torch.tensor(parent_indices, device=device),
                torch.stack([operation[1] for operation in operations]),
                torch.stack([operation[2] for operation in operations]),
                torch.tensor([
                    len(joint_names) if operation[3] is None else operation[3]
                    for operation in operations
                ], device=device),
                torch.stack([operation[4] for operation in operations]),
            ))
            previous_indices = indices
            order.extend(indices)
        self.link_names = [tree.link_names[index] for index in order]
        self.body_indices = [self.link_names.index(name) for name in BODY_NAMES]

    def forward(self, root_positions, root_rotations, joint_positions):
        coordinates = torch.cat((joint_positions, torch.zeros_like(joint_positions[:, :1])), -1)
        positions = [root_positions[:, None]]
        rotations = [root_rotations[:, None]]
        for parents, offsets, origins, coordinates_indices, axes in self.levels:
            parent_positions = positions[-1][:, parents]
            parent_rotations = rotations[-1][:, parents]
            positions.append(parent_positions + rotate(parent_rotations, offsets))
            local_rotations = multiply(
                origins[None],
                rotation_vector_to_quaternion(coordinates[:, coordinates_indices, None] * axes),
            )
            rotations.append(multiply(parent_rotations, local_rotations))
        return torch.cat(positions, 1), torch.cat(rotations, 1)


def collision_hulls(urdf, link_names):
    """Convex-hull vertices preserve exact extrema of each source mesh."""
    model = ElementTree.parse(urdf).getroot()
    records = []
    for link in model.findall("link"):
        vertices = []
        for collision in link.findall("collision"):
            mesh_element = collision.find("geometry/mesh")
            if mesh_element is None:
                raise ValueError("This official A3 importer expects mesh collision geometry")
            filename = mesh_element.get("filename")
            if filename.startswith("package://"):
                mesh_path = urdf.parents[2] / "official" / filename.removeprefix("package://")
            else:
                mesh_path = urdf.parent / filename
            if not mesh_path.is_file():
                # Also accept the documented flat description layout, provided
                # the original mesh basenames are retained.
                mesh_path = urdf.parent.parent / "meshes" / Path(filename).name
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            points = mesh.convex_hull.vertices.copy()
            points *= np.asarray([float(value) for value in mesh_element.get("scale", "1 1 1").split()])
            origin = collision.find("origin")
            if origin is not None:
                angles = [float(value) for value in origin.get("rpy", "0 0 0").split()]
                translation = [float(value) for value in origin.get("xyz", "0 0 0").split()]
                points = points @ Rotation.from_euler("xyz", angles).as_matrix().T + translation
            vertices.append(points)
        if vertices:
            records.append((link_names.index(link.get("name")), np.concatenate(vertices).astype(np.float32)))
    return records


class Retargeter:
    def __init__(self, asset_root, device):
        self.urdf = asset_root / "a3_description/urdf/model.urdf"
        if digest(self.urdf) != OFFICIAL_URDF_SHA256:
            raise ValueError("Expected the pinned, unmodified official A3 URDF")
        self.profile_path = asset_root / "a3_description/runtime_profile.json"
        self.profile = json.loads(self.profile_path.read_text())
        self.names = self.profile["joint_names"]
        self.device = device
        self.tree = PoseTree(self.urdf, self.names, device)
        self.hulls = collision_hulls(self.urdf, self.tree.link_names)
        directions = np.asarray([direction for direction in itertools.product([-1, 0, 1], repeat=3) if any(direction)], dtype=float)
        support_hulls = []
        for _, points in self.hulls:
            selected = points[np.argmax(points @ directions.T, axis=0)]
            support_hulls.append(selected)
        self.support_hulls = self.tensor(np.stack(support_hulls))
        self.support_indices = [index for index, _ in self.hulls]
        self.foot_hulls = [
            self.tensor(next(points for index, points in self.hulls
                             if self.tree.link_names[index] == BODY_NAMES[foot]))
            for foot in FOOT_INDICES
        ]
        self.sole_centers = torch.stack([
            hull[hull[:, 2] < hull[:, 2].min() + 0.01].mean(0)
            for hull in self.foot_hulls
        ])
        self.lower = self.tree.tree.lower_limits + 0.005
        self.upper = self.tree.tree.upper_limits - 0.005
        self.velocity_limits = self.tensor([
            self.profile["joints"][name]["velocity_limit_sim"] for name in self.names
        ])
        neutral = torch.zeros((1, len(self.names)), device=device)
        positions, _ = self.tree.forward(
            torch.zeros((1, 3), device=device), self.tensor([[0, 0, 0, 1]]), neutral
        )
        self.rest_positions = positions[0, self.tree.body_indices].cpu().numpy()
        self.default = self.tensor([
            self.profile["joints"][name]["default_position"] for name in self.names
        ])

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def targets(self, motion):
        lookup = {name: index for index, name in enumerate(motion.names)}
        positions = motion.positions
        rotations = motion.rotations

        def human(name):
            return positions[:, lookup[name]]

        def orientation(name, basis=TORSO_BASIS):
            return rotations[:, lookup[name]] @ basis

        human_leg_length = sum(np.linalg.norm(motion.offsets[lookup[name]]) for name in ["LeftLeg", "LeftFoot"])
        robot_leg_length = sum(np.linalg.norm(self.rest_positions[child] - self.rest_positions[parent])
                               for parent, child in [(9, 10), (10, 11)])
        scale = robot_leg_length / human_leg_length
        root_rotation = orientation("Hips")
        torso_rotation = rigid_torso_frame(
            human("Hips"), human("Neck"), human("LeftArm"), human("RightArm")
        )
        hip_offset = (self.rest_positions[9] + self.rest_positions[12]) * 0.5
        root = human("Hips") * scale - np.einsum("fij,j->fi", root_rotation, hip_offset)
        root[:, :2] -= root[0, :2]
        targets = np.zeros((len(root), len(BODY_NAMES), 3))
        targets[:, 0] = root
        targets[:, 1] = root + np.einsum("fij,j->fi", root_rotation, self.rest_positions[1])
        targets[:, 2] = targets[:, 1] + np.einsum(
            "fij,j->fi", torso_rotation, self.rest_positions[2] - self.rest_positions[1]
        )
        for side, shoulder, elbow, hand, hip, knee, foot in [
            ("Left", 3, 4, 5, 9, 10, 11), ("Right", 6, 7, 8, 12, 13, 14)
        ]:
            targets[:, hip] = root + np.einsum("fij,j->fi", root_rotation, self.rest_positions[hip])
            targets[:, shoulder] = targets[:, 1] + np.einsum(
                "fij,j->fi", torso_rotation, self.rest_positions[shoulder] - self.rest_positions[1]
            )
            for parent, child, human_parent, human_child in [
                (hip, knee, side + "UpLeg", side + "Leg"),
                (knee, foot, side + "Leg", side + "Foot"),
                (shoulder, elbow, side + "Arm", side + "ForeArm"),
                (elbow, hand, side + "ForeArm", side + "Hand"),
            ]:
                bone_length = np.linalg.norm(self.rest_positions[child] - self.rest_positions[parent])
                targets[:, child] = targets[:, parent] + unit(human(human_child) - human(human_parent)) * bone_length
        foot_rotations = np.stack([orientation(side + "Toe", FOOT_BASIS) for side in ["Left", "Right"]], 1)
        toe_positions = np.stack([human(side + "Toe") for side in ["Left", "Right"]], 1)
        foot_positions = np.stack([human(side + "Foot") for side in ["Left", "Right"]], 1)
        floor = np.percentile(toe_positions[..., 2], 2)
        toe_speed = np.linalg.norm(np.gradient(toe_positions, motion.frame_time, axis=0), axis=-1)
        foot_speed = np.linalg.norm(np.gradient(foot_positions, motion.frame_time, axis=0), axis=-1)
        # A stationary toe alone can be a rolling heel/toe pivot. Only lock a
        # rigid foot when BOTH human ankle and toe are nearly stationary.
        contacts = (toe_positions[..., 2] < floor + 0.045) & (np.maximum(toe_speed, foot_speed) < 0.15)
        for side_index in range(2):
            contacts[:, side_index] = binary_opening(binary_closing(contacts[:, side_index], iterations=1), iterations=1)
        # One constant floor offset preserves the human root motion; the IK base
        # correction subsequently adapts it to A3's different limb proportions.
        sole_minima = np.stack([
            (foot_rotations[:, side_index, 2, :] @ hull.cpu().numpy().T).min(-1)
            for side_index, hull in enumerate(self.foot_hulls)
        ], 1)
        contact_heights = (targets[:, FOOT_INDICES, 2] + sole_minima)[contacts]
        floor_shift = float(np.median(contact_heights)) if contact_heights.size else float(np.percentile(targets[:, FOOT_INDICES, 2] + sole_minima, 2))
        targets[..., 2] -= floor_shift - 0.003
        root = targets[:, 0].copy()
        foot_targets = targets[:, FOOT_INDICES].copy()
        sole_offsets = np.einsum("fsij,sj->fsi", foot_rotations, self.sole_centers.cpu().numpy())
        for side_index in range(2):
            active = contacts[:, side_index]
            boundaries = np.flatnonzero(np.diff(np.r_[False, active, False]))
            for start, end in boundaries.reshape(-1, 2):
                anchor = np.median((foot_targets + sole_offsets)[start:end, side_index, :2], axis=0)
                foot_targets[start:end, side_index, :2] = anchor - sole_offsets[start:end, side_index, :2]
            foot_targets[active, side_index, 2] = 0.003 - sole_minima[active, side_index]
        orientation_targets = np.stack([
            torso_rotation, orientation("Head"),
            *[foot_rotations[:, side] for side in range(2)],
            orientation("LeftHand", HAND_BASIS), orientation("RightHand", HAND_BASIS),
        ], 1)
        seed = np.broadcast_to(self.default.cpu().numpy(), (len(root), len(self.names))).copy()
        for side in ["Left", "Right"]:
            for limb, parent_name, middle_name, end_name, reference_rotation in [
                ("hip", side + "UpLeg", side + "Leg", side + "Foot", root_rotation),
                ("shoulder", side + "Arm", side + "ForeArm", side + "Hand", torso_rotation),
            ]:
                upper = unit(human(middle_name) - human(parent_name))
                lower = unit(human(end_name) - human(middle_name))
                bend_axis = np.cross(upper, lower) * (1 if limb == "hip" else -1)
                fallback = rotations[:, lookup[parent_name], :, 2] * (-1 if limb == "hip" else 1)
                bend_axis = np.where(np.linalg.norm(bend_axis, axis=-1, keepdims=True) > 0.05, bend_axis, fallback)
                bend_axis -= (bend_axis * upper).sum(-1, keepdims=True) * upper
                bend_axis = unit(bend_axis)
                upward = -upper
                forward = unit(np.cross(bend_axis, upward))
                anatomical_rotation = np.stack((forward, bend_axis, upward), -1)
                relative_rotation = np.swapaxes(reference_rotation, -1, -2) @ anatomical_rotation
                angles = Rotation.from_matrix(relative_rotation).as_euler("YXZ")
                for coordinate, suffix in enumerate(["pitch", "roll", "yaw"]):
                    seed[:, self.names.index(f"{side.lower()}_{limb}_{suffix}_joint")] = angles[:, coordinate]
                bend = np.arccos(np.clip((upper * lower).sum(-1), -1, 1))
                joint_name = side.lower() + ("_knee_joint" if limb == "hip" else "_elbow_joint")
                seed[:, self.names.index(joint_name)] = bend if limb == "hip" else np.pi / 2 - bend
        seed = np.clip(seed, self.lower.cpu().numpy(), self.upper.cpu().numpy())
        return {
            "body_positions": targets, "root_positions": root,
            "root_rotations": Rotation.from_matrix(root_rotation).as_quat(),
            "orientation_targets": Rotation.from_matrix(orientation_targets.reshape(-1, 3, 3)).as_quat().reshape(-1, 6, 4),
            "foot_targets": foot_targets, "sole_targets": foot_targets + sole_offsets, "contacts": contacts,
            "scale": scale, "human_floor_m": float(floor), "joint_seed": seed,
        }

    def solve(self, targets, iterations):
        body_targets = self.tensor(targets["body_positions"])
        root = self.tensor(targets["root_positions"])
        root_rotation = self.tensor(targets["root_rotations"])
        sole_targets = self.tensor(targets["sole_targets"])
        contacts = self.tensor(targets["contacts"])
        orientation_targets = orientation_columns(self.tensor(targets["orientation_targets"]))
        joint_positions = self.tensor(targets["joint_seed"]).clone().requires_grad_()
        root_correction = torch.zeros_like(root, requires_grad=True)
        rotation_correction = torch.zeros_like(root, requires_grad=True)
        optimizer = torch.optim.Adam([joint_positions, root_correction, rotation_correction], lr=0.035)
        body_weights = self.tensor([0.1, 0.2, 0.5, 0.5, 1, 1.5, 0.5, 1, 1.5, 0.5, 2, 3, 0.5, 2, 3])
        for iteration in range(iterations):
            fraction = iteration / max(1, iterations - 1)
            optimizer.param_groups[0]["lr"] = 0.0003 + 0.0347 * 0.5 * (1 + np.cos(np.pi * fraction))
            optimizer.zero_grad(set_to_none=True)
            corrected_rotation = multiply(root_rotation, rotation_vector_to_quaternion(rotation_correction))
            positions, rotations = self.tree.forward(root + root_correction, corrected_rotation, joint_positions)
            bodies = positions[:, self.tree.body_indices]
            orientations = rotations[:, self.tree.body_indices]
            fit = ((bodies - body_targets).square().sum(-1) * body_weights).mean()
            orientation_fit = (orientation_columns(orientations[:, [1, 2, 11, 14, 5, 8]]) - orientation_targets).square().mean()
            sole_positions = bodies[:, FOOT_INDICES] + rotate(orientations[:, FOOT_INDICES], self.sole_centers)
            contact_fit = ((sole_positions - sole_targets).square().sum(-1) * contacts).mean()
            persistent_contact = contacts[1:] * contacts[:-1]
            contact_velocity = (sole_positions[1:] - sole_positions[:-1]) * 30
            contact_motion = (contact_velocity.square().sum(-1) * persistent_contact).mean()
            foot_minima = torch.stack([
                (bodies[:, foot_index, None, 2] + rotate(orientations[:, foot_index, None], hull)[..., 2]).amin(-1)
                for foot_index, hull in zip(FOOT_INDICES, self.foot_hulls)
            ], -1)
            penetration = torch.relu(0.001 - foot_minima).square().mean()
            support_rotations = rotations[:, self.support_indices]
            with torch.no_grad():
                local_up = rotate(conjugate(support_rotations), self.tensor([0, 0, 1]))
                support_heights = torch.einsum("flc,lvc->flv", local_up, self.support_hulls)
                vertex_indices = support_heights.argmin(-1)
                selected_vertices = self.support_hulls[
                    torch.arange(len(self.hulls), device=self.device)[None], vertex_indices
                ]
            lowest_points = positions[:, self.support_indices, 2] + rotate(support_rotations, selected_vertices)[..., 2]
            whole_body_penetration = torch.relu(0.004 - lowest_points).square().amax(-1).mean()
            joint_delta = joint_positions[1:] - joint_positions[:-1]
            acceleration = joint_delta[1:] - joint_delta[:-1]
            base_acceleration = root_correction[2:] - 2 * root_correction[1:-1] + root_correction[:-2]
            excess_speed = torch.relu(joint_delta.abs() * 30 / self.velocity_limits - 0.85).square().mean()
            loss = (fit + 0.025 * orientation_fit + 30 * contact_fit + 20 * penetration
                    + 0.10 * contact_motion
                    + 100 * whole_body_penetration
                    + 0.2 * acceleration.square().mean() + 0.05 * excess_speed
                    + 2.0 * base_acceleration.square().mean()
                    + 0.03 * root_correction.square().mean()
                    + 0.005 * rotation_correction.square().mean()
                    + 0.2 * (rotation_correction[2:] - 2 * rotation_correction[1:-1] + rotation_correction[:-2]).square().mean()
                    + 0.00002 * (joint_positions - self.default).square().mean())
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite inverse-kinematics objective")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                joint_positions.clamp_(self.lower, self.upper)
                root_correction.clamp_(-0.18, 0.18)
                rotation_correction.clamp_(-0.6, 0.6)
        corrected_rotation = multiply(root_rotation, rotation_vector_to_quaternion(rotation_correction))
        return (root + root_correction).detach().cpu().numpy(), corrected_rotation.detach().cpu().numpy(), joint_positions.detach().cpu().numpy()

    @torch.no_grad()
    def geometry(self, root, quaternion, coordinates):
        body_positions, body_rotations, minima = [], [], []
        for start in range(0, len(root), 256):
            positions, rotations = self.tree.forward(
                self.tensor(root[start:start + 256]), self.tensor(quaternion[start:start + 256]),
                self.tensor(coordinates[start:start + 256]),
            )
            lowest = torch.full((len(positions),), torch.inf, device=self.device)
            for link_index, points in self.hulls:
                heights = positions[:, link_index, None, 2] + rotate(rotations[:, link_index, None], self.tensor(points))[..., 2]
                lowest = torch.minimum(lowest, heights.amin(-1))
            minima.append(lowest.cpu().numpy())
            body_positions.append(positions[:, self.tree.body_indices].cpu().numpy())
            body_rotations.append(rotations[:, self.tree.body_indices].cpu().numpy())
        return np.concatenate(body_positions), np.concatenate(body_rotations), np.concatenate(minima)

    def convert(self, source, output, iterations):
        started = time.monotonic()
        motion = read_bvh(source)
        if not np.isclose(motion.frame_time, 1 / 30, atol=1e-6):
            raise ValueError("Expected LAFAN1 at 30 Hz")
        targets = self.targets(motion)
        root, quaternion, coordinates = self.solve(targets, iterations)
        # Temporal regularization is part of IK. Independently filtering joints
        # after solving breaks the foot/base coupling and reintroduces slip.
        # Retiming is local: a brief fast transition must not slow every other
        # frame in a minutes-long clip. A conservative smooth envelope keeps
        # each interval at or below the same 85% joint-speed limit.
        interval_requirements = np.maximum(
            1.0, (np.abs(np.diff(coordinates, axis=0)) * 30 / self.velocity_limits.cpu().numpy()).max(-1) / 0.85
        )
        smooth_envelope = gaussian_filter1d(maximum_filter1d(interval_requirements, size=9, mode="nearest"), sigma=2, mode="nearest")
        interval_stretch = np.maximum(interval_requirements, smooth_envelope)
        retimed_keys = np.r_[0.0, np.cumsum(interval_stretch)]
        time_scale = float(retimed_keys[-1] / (len(root) - 1))
        if time_scale > 3.0:
            raise ValueError(f"Excessive time dilation: {time_scale:.3f}")
        source_times = np.arange(len(root), dtype=float)
        output_ticks = np.minimum(np.arange(int(np.ceil(retimed_keys[-1])) + 1), retimed_keys[-1])
        sample_times = np.interp(output_ticks, retimed_keys, source_times)
        if time_scale > 1:
            root = np.stack([np.interp(sample_times, source_times, column) for column in root.T], -1)
            coordinates = np.stack([np.interp(sample_times, source_times, column) for column in coordinates.T], -1)
            quaternion = Slerp(source_times, Rotation.from_quat(quaternion))(sample_times).as_quat()
        contacts = targets["contacts"][np.rint(sample_times).astype(int)]
        bodies, orientations, minimum_height = self.geometry(root, quaternion, coordinates)
        # Final whole-body mesh check includes hands and torso during floor motions.
        # This is a geometric floor correction, not a dynamics feasibility claim.
        lift = np.maximum(0.001 - minimum_height, 0.0)
        root[:, 2] += lift
        bodies[..., 2] += lift[:, None]
        output_values = np.concatenate((root, quaternion, coordinates), -1)
        if not np.isfinite(output_values).all():
            raise ValueError("Non-finite retarget output")
        sampled_targets = np.stack([
            np.interp(sample_times, source_times, column)
            for column in targets["body_positions"].reshape(len(source_times), -1).T
        ], -1).reshape(len(sample_times), -1, 3)
        relative_errors = np.linalg.norm((bodies - bodies[:, :1]) - (sampled_targets - sampled_targets[:, :1]), axis=-1)
        support_positions = []
        foot_minimum_height = []
        for foot_index, hull in zip(FOOT_INDICES, self.foot_hulls):
            vertices = hull.cpu().numpy()
            sole_center = vertices[vertices[:, 2] < vertices[:, 2].min() + 0.01].mean(0)
            matrices = Rotation.from_quat(orientations[:, foot_index]).as_matrix()
            support_positions.append(bodies[:, foot_index] + matrices @ sole_center)
            foot_minimum_height.append(bodies[:, foot_index, 2] + (matrices[:, 2, :] @ vertices.T).min(-1))
        support_positions = np.stack(support_positions, 1)
        foot_minimum_height = np.stack(foot_minimum_height, 1)
        planar_speed = np.linalg.norm(np.diff(support_positions[..., :2], axis=0) * 30, axis=-1)
        persistent = contacts[1:] & contacts[:-1]
        support_speed = planar_speed[persistent]
        max_velocity_ratio = float(np.max(np.abs(np.diff(coordinates, axis=0)) * 30 / self.velocity_limits.cpu().numpy()))
        statistics = {
            "clip": source.stem, "source_sha256": digest(source),
            "source_frames": len(source_times), "output_frames": len(root), "fps": 30,
            "time_scale": time_scale, "leg_scale": targets["scale"],
            "retiming": "Per-interval smooth joint-speed-limited time map, shared by root and joints",
            "maximum_interval_stretch": float(interval_stretch.max()),
            "body_fit_mean_m": float(relative_errors.mean()),
            "body_fit_p95_m": float(np.percentile(relative_errors, 95)),
            "body_fit_p95_by_link_m": dict(zip(BODY_NAMES, np.percentile(relative_errors, 95, axis=0).tolist())),
            "joint_velocity_limit_ratio_max": max_velocity_ratio,
            "contact_fraction": float(contacts.mean()),
            "contact_slip_mean_m_s": float(support_speed.mean()) if support_speed.size else None,
            "contact_slip_p95_m_s": float(np.percentile(support_speed, 95)) if support_speed.size else None,
            "contact_slip_definition": "Rigid sole-centre velocity when BOTH human ankle and toe have 3D speed below 0.15 m/s and toe is near ground; rolling contacts are excluded",
            "contact_foot_clearance_p95_m": float(np.percentile(foot_minimum_height[contacts], 95)) if contacts.any() else None,
            "whole_body_floor_lift_max_m": float(lift.max()),
            "whole_body_floor_lift_p95_m": float(np.percentile(lift, 95)),
            "minimum_mesh_height_m": float((minimum_height + lift).min()),
            "root_acceleration_p99_m_s2": float(np.percentile(np.linalg.norm(np.diff(root, n=2, axis=0) * 900, axis=-1), 99)),
            "wall_seconds": time.monotonic() - started,
        }
        problems = []
        if statistics["body_fit_p95_m"] > 0.12:
            problems.append("body_fit_p95_above_0.12m")
        if max_velocity_ratio > 0.86:
            problems.append("joint_velocity_above_0.86_limit")
        if statistics["contact_slip_p95_m_s"] is not None and statistics["contact_slip_p95_m_s"] > 0.20:
            problems.append("contact_slip_p95_above_0.20m_s")
        if statistics["contact_foot_clearance_p95_m"] is not None and statistics["contact_foot_clearance_p95_m"] > 0.05:
            problems.append("contact_foot_clearance_p95_above_0.05m")
        statistics["quality_pass"] = not problems
        statistics["quality_issues"] = problems
        np.savetxt(output, output_values, delimiter=",", fmt="%.9f")
        statistics["sha256"] = digest(output)
        return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True, help="JSON list of BVH basenames")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", nargs="*")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    arguments.output.mkdir(parents=True, exist_ok=False)
    names = json.loads(arguments.clips.read_text())
    if arguments.only:
        names = arguments.only
    source_manifest = json.loads((arguments.source / "manifest.json").read_text())
    expected_sources = {record["file"]: record["sha256"] for record in source_manifest["files"]}
    for name in names:
        if Path(name).name != name or digest(arguments.source / "bvh" / (name + ".bvh")) != expected_sources[name + ".bvh"]:
            raise ValueError(f"Original BVH source hash mismatch: {name}")
    retargeter = Retargeter(arguments.asset_root, arguments.device)
    manifest = {
        "method": "direct-human-BVH-to-official-A3-constrained-IK",
        "torso_mapping": "Rigid landmark frame from hip-to-neck chord and shoulder lateral axis",
        "intermediate_robot": None, "source_manifest_sha256": digest(arguments.source / "manifest.json"),
        "source": json.loads((arguments.source / "manifest.json").read_text()),
        "urdf_sha256": digest(retargeter.urdf), "profile_sha256": digest(retargeter.profile_path),
        "joint_names": retargeter.names, "iterations": arguments.iterations, "fps": 30,
        "seed": 0, "clips": [], "complete": False, "quality_pass": False,
        "limitations": ["Kinematic fitting is not a dynamics or actuator-effort feasibility certificate", "No finger articulation", "Whole-body mesh floor check does not certify self-collision clearance"],
        "source_code_sha256": {name: digest(Path(__file__).parent / name) for name in ["retarget_lafan1.py", "bvh.py"]},
    }
    manifest_path = arguments.output / "manifest.json"
    for name in names:
        record = retargeter.convert(arguments.source / "bvh" / (name + ".bvh"), arguments.output / (name + ".csv"), arguments.iterations)
        manifest["clips"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    manifest["complete"] = True
    manifest["quality_pass"] = all(record["quality_pass"] for record in manifest["clips"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["quality_pass"]:
        raise SystemExit("Retargeting completed with quality failures; do not start training")


if __name__ == "__main__":
    main()
