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

    def _vector(self, text):
        return torch.tensor([float(value) for value in text.split()], device=self.device)

    def forward(self, state):
        """State order: root xyz, xyzw, world linear/angular velocity, q, qdot."""
        joint_count = len(self.joint_names)
        positions = [state[..., :3]]
        rotations = [state[..., 3:7]]
        linear_velocities = [state[..., 7:10]]
        angular_velocities = [state[..., 10:13]]
        joint_positions = state[..., 13 : 13 + joint_count]
        joint_velocities = state[..., 13 + joint_count :]
        for parent, origin_position, origin_rotation, coordinate, axis in self.operations:
            offset = rotate(rotations[parent], origin_position)
            position = positions[parent] + offset
            origin = origin_rotation.expand_as(rotations[parent])
            rotation = multiply(rotations[parent], origin)
            velocity = linear_velocities[parent] + torch.cross(angular_velocities[parent], offset, dim=-1)
            angular_velocity = angular_velocities[parent]
            if coordinate is not None:
                world_axis = rotate(rotation, axis)
                angular_velocity = angular_velocity + world_axis * joint_velocities[..., coordinate, None]
                rotation = multiply(
                    rotation,
                    rotation_vector_to_quaternion(
                        axis * joint_positions[..., coordinate, None]
                    ),
                )
            positions.append(position)
            rotations.append(rotation)
            linear_velocities.append(velocity)
            angular_velocities.append(angular_velocity)
        return tuple(
            torch.stack([values[index] for index in self.selected_indices], dim=-2)
            for values in (positions, rotations, linear_velocities, angular_velocities)
        )
