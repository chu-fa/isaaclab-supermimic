"""Explicit CSV motion import, resampling and clip-boundary-aware indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .geometry import (
    conjugate,
    interpolate_quaternions,
    multiply,
    quaternion_to_rotation_vector,
)


class MotionLibrary:
    def __init__(
        self, paths, joint_count, device, *, source_rate=30.0, control_period=0.02
    ):
        if not np.isfinite(source_rate) or source_rate <= 0:
            raise ValueError("source_rate must be finite and positive")
        if not np.isfinite(control_period) or not 0 < control_period <= 1:
            raise ValueError("control_period must be in (0, 1] seconds")
        self.names = []
        self.control_period = control_period
        self.device = torch.device(device)
        self.states = []
        lengths = []
        expected_columns = 7 + joint_count
        for path in paths:
            raw = np.loadtxt(path, delimiter=",", ndmin=2)
            if (
                raw.shape[0] < 2
                or raw.shape[1] != expected_columns
                or not np.isfinite(raw).all()
            ):
                raise ValueError(
                    f"Invalid motion array in {path}; "
                    f"expected finite [frames, {expected_columns}]"
                )
            # Differentiate imported trajectories in float64 before storage to
            # avoid amplifying float32 position rounding by the 50 Hz interval.
            source = torch.tensor(raw, dtype=torch.float64, device=device)
            quaternion_norms = source[:, 3:7].norm(dim=-1, keepdim=True)
            if bool((quaternion_norms < 1e-8).any()):
                raise ValueError(f"Zero reference quaternion in {path}")
            source[:, 3:7] /= quaternion_norms
            count = int(np.floor((len(source) - 1) / source_rate / control_period)) + 1
            if count < 2:
                raise ValueError(f"Motion is shorter than one control interval: {path}")
            sample_times = torch.arange(
                count, dtype=torch.float64, device=device
            ) * (control_period * source_rate)
            lower = sample_times.floor().long().clamp(max=len(source) - 2)
            fraction = (sample_times - lower).unsqueeze(-1)
            positions = torch.lerp(source[lower, :3], source[lower + 1, :3], fraction)
            rotations = interpolate_quaternions(
                source[lower, 3:7], source[lower + 1, 3:7], fraction
            )
            joints = torch.lerp(source[lower, 7:], source[lower + 1, 7:], fraction)
            indices = torch.arange(count, device=device)
            before = (indices - 1).clamp_min(0)
            after = (indices + 1).clamp_max(count - 1)
            intervals = ((after - before) * control_period).unsqueeze(-1)
            linear_velocity = (positions[after] - positions[before]) / intervals
            angular_velocity = quaternion_to_rotation_vector(
                multiply(rotations[after], conjugate(rotations[before]))
            ) / intervals
            joint_velocity = (joints[after] - joints[before]) / intervals
            self.states.append(
                torch.cat(
                    (
                        positions,
                        rotations,
                        linear_velocity,
                        angular_velocity,
                        joints,
                        joint_velocity,
                    ),
                    dim=-1,
                ).float()
            )
            self.names.append(Path(path).stem)
            lengths.append(count)
        if not self.states:
            raise ValueError("Motion library is empty")
        self.states = torch.cat(self.states)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.starts = torch.cat((self.lengths.new_zeros(1), self.lengths.cumsum(0)[:-1]))
        self.ends = self.starts + self.lengths - 1
        self.clip_ids = torch.repeat_interleave(torch.arange(len(lengths), device=device), self.lengths)
        self.frame_ends = self.ends[self.clip_ids]
        self.bins = torch.cat([
            torch.arange(int(start), int(end), round(1 / control_period), device=device)
            for start, end in zip(self.starts, self.ends)
        ])

    @classmethod
    def from_directory(cls, directory, joint_count, device, **kwargs):
        return cls(sorted(Path(directory).glob("*.csv")), joint_count, device, **kwargs)

    def future_indices(self, current, count):
        future = current[..., None] + torch.arange(1, count + 1, device=self.device)
        return torch.minimum(future, self.frame_ends[current][..., None])

    def fingerprint(self):
        import hashlib

        digest = hashlib.sha256()
        digest.update("\n".join(self.names).encode())
        digest.update(self.states.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
