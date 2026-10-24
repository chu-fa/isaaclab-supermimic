"""LAFAN1 BVH reader; centimetres/Y-up in, metres/Z-up out.

This parser is independent of the Ubisoft Python utilities. Rotation channels
are applied in their declared intrinsic order. LAFAN1 root position channels
contain absolute translations and replace the root OFFSET.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Motion:
    names: list[str]
    parents: list[int]
    offsets: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    frame_time: float


def rigid_torso_frame(hips, neck, left_shoulder, right_shoulder):
    """Fit a rigid torso to landmark chords instead of a terminal spine joint.

    Human spine curvature is distributed over multiple joints. Extending the
    last vertebra's orientation over the complete robot torso exaggerates lean.
    Inputs are world positions in metres; output columns are forward/left/up.
    """
    upward = neck - hips
    length = np.linalg.norm(upward, axis=-1, keepdims=True)
    if np.any(length < 1e-8):
        raise ValueError("Degenerate hip-to-neck chord")
    upward = upward / length
    lateral = left_shoulder - right_shoulder
    forward = np.cross(lateral, upward)
    length = np.linalg.norm(forward, axis=-1, keepdims=True)
    if np.any(length < 1e-8):
        raise ValueError("Shoulders do not define a torso plane")
    forward = forward / length
    lateral = np.cross(upward, forward)
    return np.stack((forward, lateral, upward), axis=-1)


def read_bvh(path):
    text = Path(path).read_text()
    hierarchy, motion_text = text.split("MOTION", 1)
    tokens = iter(hierarchy.replace("{", " { ").replace("}", " } ").split())
    if next(tokens) != "HIERARCHY" or next(tokens) != "ROOT":
        raise ValueError("Expected one BVH ROOT")
    names, parents, offsets, channels = [], [], [], []

    def read_joint(name, parent):
        joint_index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append([0.0, 0.0, 0.0])
        channels.append([])
        if next(tokens) != "{":
            raise ValueError("Missing opening BVH brace")
        for token in tokens:
            if token == "OFFSET":
                offsets[joint_index] = [float(next(tokens)) for _ in range(3)]
            elif token == "CHANNELS":
                channels[joint_index] = [next(tokens) for _ in range(int(next(tokens)))]
            elif token == "JOINT":
                read_joint(next(tokens), joint_index)
            elif token == "End":
                if next(tokens) != "Site" or next(tokens) != "{":
                    raise ValueError("Invalid End Site")
                while next(tokens) != "}":
                    pass
            elif token == "}":
                return
            else:
                raise ValueError(f"Unexpected BVH token: {token}")
        raise ValueError("Unclosed BVH joint")

    read_joint(next(tokens), -1)
    if list(tokens) or len(names) != len(set(names)):
        raise ValueError("Trailing hierarchy or duplicate joint names")
    lines = motion_text.strip().splitlines()
    frame_count = int(lines[0].split(":", 1)[1])
    frame_time = float(lines[1].split(":", 1)[1])
    channel_count = sum(map(len, channels))
    values = np.fromstring("\n".join(lines[2:]), sep=" ")
    if values.size != frame_count * channel_count or frame_time <= 0:
        raise ValueError("BVH frame count, channel count or frame time is invalid")
    values = values.reshape(frame_count, channel_count)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite BVH channels")
    offsets = np.asarray(offsets) * 0.01
    local_positions = np.broadcast_to(offsets, (frame_count, len(names), 3)).copy()
    local_rotations = np.empty((frame_count, len(names), 3, 3))
    channel_start = 0
    for joint_index, joint_channels in enumerate(channels):
        rotation_axes, rotation_values = [], []
        for relative_index, channel in enumerate(joint_channels):
            column = values[:, channel_start + relative_index]
            axis_index = "XYZ".index(channel[0])
            if channel.endswith("position"):
                local_positions[:, joint_index, axis_index] = column * 0.01
            elif channel.endswith("rotation"):
                rotation_axes.append(channel[0])
                rotation_values.append(column)
            else:
                raise ValueError(f"Unsupported BVH channel {channel}")
        if len(rotation_axes) != 3 or len(set(rotation_axes)) != 3:
            raise ValueError("Expected three distinct intrinsic rotation channels")
        local_rotations[:, joint_index] = Rotation.from_euler(
            "".join(rotation_axes), np.stack(rotation_values, axis=-1), degrees=True
        ).as_matrix()
        channel_start += len(joint_channels)
    global_positions = np.empty_like(local_positions)
    global_rotations = np.empty_like(local_rotations)
    for joint_index, parent_index in enumerate(parents):
        if parent_index == -1:
            global_positions[:, joint_index] = local_positions[:, joint_index]
            global_rotations[:, joint_index] = local_rotations[:, joint_index]
        else:
            global_positions[:, joint_index] = global_positions[:, parent_index] + np.einsum(
                "fij,fj->fi", global_rotations[:, parent_index], local_positions[:, joint_index]
            )
            global_rotations[:, joint_index] = (
                global_rotations[:, parent_index] @ local_rotations[:, joint_index]
            )
    # Proper rotation, not a reflection: (x, y, z)_BVH -> (x, -z, y)_world.
    basis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    global_positions = global_positions @ basis.T
    global_rotations = basis @ global_rotations
    return Motion(names, parents, offsets, global_positions, global_rotations, frame_time)
