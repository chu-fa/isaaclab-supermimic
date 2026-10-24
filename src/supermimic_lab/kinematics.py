"""URDF link-frame kinematics and spatial velocities implemented in Torch."""

from __future__ import annotations

import xml.etree.ElementTree as element_tree

import torch

from .geometry import multiply, rotate, rotation_vector_to_quaternion


class KinematicTree:
    def __init__(self, urdf_path, joint_names, tracked_links, device):
        document = element_tree.parse(urdf_path).getroot()
        self.joint_names = list(joint_names)
        self.tracked_links = list(tracked_links)
        self.device = torch.device(device)
        joint_lookup = {name: index for index, name in enumerate(joint_names)}
        links = [link.attrib["name"] for link in document.findall("link")]
        descriptions = document.findall("joint")
        child_links = {joint.find("child").attrib["link"] for joint in descriptions}
        roots = set(links) - child_links
        if len(roots) != 1:
            raise ValueError("URDF must describe one connected rooted tree")
        self.root_link = roots.pop()
        self.link_names = [self.root_link]
        self.operations = []
        limits = {}
        unresolved = list(descriptions)
        while unresolved:
            remaining = []
            for description in unresolved:
                parent = description.find("parent").attrib["link"]
                if parent not in self.link_names:
                    remaining.append(description)
                    continue
                child = description.find("child").attrib["link"]
                kind = description.attrib["type"]
                if kind not in ("fixed", "revolute") or description.find("mimic") is not None:
                    raise ValueError("Only fixed and bounded revolute URDF joints are supported")
                origin = description.find("origin")
                position = self._vector(
                    "0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0")
                )
                roll, pitch, yaw = self._vector(
                    "0 0 0" if origin is None else origin.attrib.get("rpy", "0 0 0")
                )
                rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
                for axis_index, angle in ((2, yaw), (1, pitch), (0, roll)):
                    axis_angle = torch.zeros(3, device=device)
                    axis_angle[axis_index] = angle
                    rotation = multiply(rotation, rotation_vector_to_quaternion(axis_angle))
                coordinate_index = None
                axis = torch.tensor([1.0, 0.0, 0.0], device=device)
                if kind == "revolute":
                    name = description.attrib["name"]
                    if name not in joint_lookup:
                        raise ValueError(f"Active joint missing from configured order: {name}")
                    coordinate_index = joint_lookup[name]
                    axis_element = description.find("axis")
                    axis = self._vector("1 0 0" if axis_element is None else axis_element.attrib["xyz"])
                    if float(axis.norm()) < 1e-12:
                        raise ValueError(f"Zero URDF joint axis: {name}")
                    axis = axis / axis.norm()
                    limit = description.find("limit")
                    limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
                self.operations.append(
                    (
                        self.link_names.index(parent),
                        position,
                        rotation,
                        coordinate_index,
                        axis,
                    )
                )
                self.link_names.append(child)
            if len(remaining) == len(unresolved):
                raise ValueError("URDF tree contains a cycle or disconnected link")
            unresolved = remaining
        if set(limits) != set(joint_names):
            raise ValueError("Configured joint names do not exactly match active URDF joints")
        self.lower_limits = torch.tensor([limits[name][0] for name in joint_names], device=device)
        self.upper_limits = torch.tensor([limits[name][1] for name in joint_names], device=device)
        self.selected_indices = [self.link_names.index(name) for name in tracked_links]
        self._build_levels()

    def _vector(self, text):
        return torch.tensor([float(value) for value in text.split()], device=self.device)

    def _build_levels(self):
        """Group the tracked links' ancestors by tree depth for batched evaluation.

        Eager per-joint evaluation launches thousands of tiny kernels per call,
        which dominated learner time. Links at one depth share a set of batched
        operations; branches with no tracked descendant are skipped.
        """
        parents = [None] + [operation[0] for operation in self.operations]
        depths = [0]
        for parent in parents[1:]:
            depths.append(depths[parent] + 1)
        required = set()
        for index in self.selected_indices:
            while index is not None and index not in required:
                required.add(index)
                index = parents[index]
        levels = [[0]]
        for depth in range(1, max(depths[index] for index in required) + 1):
            levels.append([index for index in sorted(required) if depths[index] == depth])
        slots = {index: slot for level in levels for slot, index in enumerate(level)}
        self.levels = []
        for parent_level, level in zip(levels[:-1], levels[1:]):
            operations = [self.operations[index - 1] for index in level]
            for parent, *_ in operations:
                if parent not in parent_level:
                    raise AssertionError("Tree depth grouping is inconsistent")
            # Fixed joints use a zero axis: no joint rotation or rate contribution.
            self.levels.append((
                torch.tensor([slots[parent] for parent, *_ in operations], device=self.device),
                torch.stack([position for _, position, *_ in operations]),
                torch.stack([rotation for _, _, rotation, *_ in operations]),
                torch.tensor([
                    0 if coordinate is None else coordinate for *_, coordinate, _ in operations
                ], device=self.device),
                torch.stack([
                    torch.zeros_like(axis) if coordinate is None else axis
                    for *_, coordinate, axis in operations
                ]),
            ))
        offsets = [0]
        for level in levels:
            offsets.append(offsets[-1] + len(level))
        level_of = {index: depth for depth, level in enumerate(levels) for index in level}
        self.output_indices = torch.tensor([
            offsets[level_of[index]] + slots[index] for index in self.selected_indices
        ], device=self.device)

    def forward(self, state):
        """State order: root xyz, xyzw, world linear/angular velocity, q, qdot."""
        joint_count = len(self.joint_names)
        joint_positions = state[..., 13 : 13 + joint_count]
        joint_velocities = state[..., 13 + joint_count :]
        position = state[..., None, :3]
        rotation = state[..., None, 3:7]
        velocity = state[..., None, 7:10]
        angular_velocity = state[..., None, 10:13]
        levels = [(position, rotation, velocity, angular_velocity)]
        for parents, origin_position, origin_rotation, coordinates, axes in self.levels:
            parent_rotation = rotation[..., parents, :]
            offset = rotate(parent_rotation, origin_position)
            position = position[..., parents, :] + offset
            velocity = velocity[..., parents, :] + torch.cross(
                angular_velocity[..., parents, :], offset, dim=-1
            )
            rotation = multiply(parent_rotation, origin_rotation.expand_as(parent_rotation))
            angular_velocity = angular_velocity[..., parents, :] + rotate(rotation, axes) * (
                joint_velocities[..., coordinates].unsqueeze(-1)
            )
            rotation = multiply(rotation, rotation_vector_to_quaternion(
                axes * joint_positions[..., coordinates].unsqueeze(-1)
            ))
            levels.append((position, rotation, velocity, angular_velocity))
        return tuple(
            torch.cat([level[component] for level in levels], dim=-2)[..., self.output_indices, :]
            for component in range(4)
        )
