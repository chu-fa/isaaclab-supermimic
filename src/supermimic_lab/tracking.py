"""Floating-base motion tracking: analytic heads from paper Appendix A."""

from __future__ import annotations

import torch

from .contracts import ModelDimensions
from .geometry import (
    conjugate, heading, multiply, orientation_columns, orientation_error,
    quaternion_to_rotation_vector, rotate, rotation_vector_to_quaternion,
)


GOAL_FRAMES = ("robot", "reference")


class ReferenceTracking:
    """SI state with xyzw quaternions; no simulator imports or robot constants.

    The actor sees Table A.1 proprioception and future reference features only.
    ``goal_frame="robot"`` expresses future reference root poses and velocities
    in the robot's current position and heading frame, as BeyondMimic's anchor
    observations do, so the global root and world-velocity reward terms are
    observable. The horizontal robot-to-reference offset is saturated at
    ``goal_offset_limit`` metres: training episodes start on the reference, so
    multi-metre drift on long clips would otherwise drive the goal far outside
    the training distribution. ``"reference"`` reproduces the earlier encoding
    in the current reference's own frame, which hides heading and position
    drift from the actor and critic; it remains available to evaluate older
    checkpoints.
    """

    control_period = 0.02

    def __init__(
        self, kinematics, motions, default_positions, future_frames=10, *,
        goal_frame="robot", goal_offset_limit=1.0,
    ):
        if goal_frame not in GOAL_FRAMES:
            raise ValueError(f"goal_frame must be one of {GOAL_FRAMES}")
        if goal_offset_limit is not None and not goal_offset_limit > 0:
            raise ValueError("goal_offset_limit must be positive or None")
        self.kinematics = kinematics
        self.motions = motions
        self.goal_frame = goal_frame
        self.goal_offset_limit = goal_offset_limit
        self.control_period = motions.control_period
        self.device = kinematics.device
        self.joint_count = len(kinematics.joint_names)
        self.future_frames = future_frames
        self.default_positions = torch.as_tensor(default_positions, dtype=torch.float32, device=self.device)
        if self.default_positions.shape != (self.joint_count,):
            raise ValueError("default_positions must follow the configured joint order")
        goal_width = future_frames * (15 + 2 * self.joint_count)
        critic_state_width = 13 + 2 * self.joint_count if goal_frame == "reference" else 10 + 2 * self.joint_count
        self.dimensions = ModelDimensions(
            state=13 + 2 * self.joint_count,
            action=self.joint_count,
            context=self.joint_count + 4,
            observation=3 * self.joint_count + 6,
            goal=goal_width,
            model_features=14 + 4 * self.joint_count,
            model_outputs=12 + 2 * self.joint_count,
            state_errors=12 + 2 * self.joint_count,
            critic=critic_state_width + goal_width + self.joint_count + 4,
        )
        # Reference link kinematics do not depend on the policy; compute them
        # once instead of in every reward evaluation (about 13 floats per link
        # and frame, e.g. ~0.35 GB for 15 links over 2.45 h at 50 Hz).
        with torch.no_grad():
            chunks = [kinematics.forward(states) for states in motions.states.split(65536)]
            self.reference_links = tuple(torch.cat(parts) for parts in zip(*chunks))
        self.observation_noise = torch.tensor(
            [0.01] * self.joint_count + [0.5] * self.joint_count
            + [0.2] * 3 + [0.05] * 3 + [0.0] * self.joint_count,
            device=self.device,
        )
        midpoint = (kinematics.lower_limits + kinematics.upper_limits) / 2
        half_range = (kinematics.upper_limits - kinematics.lower_limits) * 0.45
        self.soft_lower_limits = midpoint - half_range
        self.soft_upper_limits = midpoint + half_range

    def _gravity(self, state):
        gravity = torch.zeros_like(state[..., :3])
        gravity[..., 2] = -1
        return rotate(conjugate(state[..., 3:7]), gravity)

    def _invariant_state(self, state):
        """Root height, gravity, body-frame velocities and joints; no XY or yaw."""
        inverse_rotation = conjugate(state[..., 3:7])
        return torch.cat((
            state[..., 2:3], self._gravity(state),
            rotate(inverse_rotation, state[..., 7:10]),
            rotate(inverse_rotation, state[..., 10:13]), state[..., 13:],
        ), dim=-1)

    def encode_dynamics(self, state, action, context):
        return torch.cat((self._invariant_state(state), action, context), dim=-1)

    def dynamics_target(self, state, next_state):
        inverse_rotation = conjugate(state[..., 3:7])
        return torch.cat((
            rotate(inverse_rotation, next_state[..., :3] - state[..., :3]),
            quaternion_to_rotation_vector(multiply(inverse_rotation, next_state[..., 3:7])),
            rotate(inverse_rotation, next_state[..., 7:10] - state[..., 7:10]),
            rotate(inverse_rotation, next_state[..., 10:13] - state[..., 10:13]),
            next_state[..., 13:] - state[..., 13:],
        ), dim=-1)

    def integrate_prediction(self, state, prediction):
        orientation = multiply(state[..., 3:7], rotation_vector_to_quaternion(prediction[..., 3:6]))
        orientation = orientation / orientation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.cat((
            state[..., :3] + rotate(state[..., 3:7], prediction[..., :3]), orientation,
            state[..., 7:10] + rotate(state[..., 3:7], prediction[..., 6:9]),
            state[..., 10:13] + rotate(state[..., 3:7], prediction[..., 9:12]),
            state[..., 13:] + prediction[..., 12:],
        ), dim=-1)

    def state_error(self, predicted, target):
        return torch.cat((
            predicted[..., :3] - target[..., :3],
            orientation_error(predicted[..., 3:7], target[..., 3:7]),
            predicted[..., 7:] - target[..., 7:],
        ), dim=-1)

    def observation(self, state, previous_action, context):
        joint_positions = state[..., 13 : 13 + self.joint_count]
        # Calibration offsets describe the encoder reading. The simulator uses
        # the inverse offset when converting a measured setpoint to joint angle.
        measured_positions = joint_positions + context[..., 1 : 1 + self.joint_count]
        return torch.cat((
            measured_positions - self.default_positions,
            state[..., 13 + self.joint_count :],
            rotate(conjugate(state[..., 3:7]), state[..., 10:13]),
            self._gravity(state), previous_action,
        ), dim=-1)

    def goal(self, state, reference_index):
        reference = self.motions.states[reference_index]
        future = self.motions.states[self.motions.future_indices(reference_index, self.future_frames)]
        frame = state if self.goal_frame == "robot" else reference
        inverse_heading = conjugate(heading(frame[..., 3:7]))[..., None, :]
        displacement = future[..., :3] - reference[..., None, :3]
        if self.goal_frame == "robot":
            offset = reference[..., :2] - state[..., :2]
            if self.goal_offset_limit is not None:
                distance = offset.norm(dim=-1, keepdim=True)
                offset = offset * (self.goal_offset_limit / distance.clamp_min(self.goal_offset_limit))
            displacement = displacement + torch.nn.functional.pad(offset, (0, 1))[..., None, :]
        relative_position = rotate(inverse_heading, displacement)
        relative_position = torch.cat((relative_position[..., :2], future[..., 2:3]), dim=-1)
        relative_rotation = multiply(inverse_heading.expand_as(future[..., 3:7]), future[..., 3:7])
        features = torch.cat((
            relative_position, orientation_columns(relative_rotation),
            rotate(inverse_heading, future[..., 7:10]),
            rotate(inverse_heading, future[..., 10:13]), future[..., 13:],
        ), dim=-1)
        return features.flatten(start_dim=-2)

    def reward_terms(self, state, action, previous_action, reference_index):
        reference = self.motions.states[reference_index]
        positions, orientations, linear_velocities, angular_velocities = self.kinematics.forward(state)
        (
            reference_positions,
            reference_orientations,
            reference_linear,
            reference_angular,
        ) = (values[reference_index] for values in self.reference_links)
        heading_alignment = multiply(
            heading(state[..., 3:7]), conjugate(heading(reference[..., 3:7]))
        )
        aligned_anchor = torch.cat((state[..., :2], reference[..., 2:3]), dim=-1)
        aligned_positions = aligned_anchor[..., None, :] + rotate(
            heading_alignment[..., None, :], reference_positions - reference[..., None, :3]
        )
        aligned_orientations = multiply(
            heading_alignment[..., None, :].expand_as(reference_orientations), reference_orientations
        )
        position_error = (positions - aligned_positions).square().sum(dim=-1).mean(dim=-1)
        body_orientation_error = (
            orientation_error(orientations, aligned_orientations)
            .square()
            .sum(dim=-1)
            .mean(dim=-1)
        )
        linear_error = (linear_velocities - reference_linear).square().sum(dim=-1).mean(dim=-1)
        angular_error = (angular_velocities - reference_angular).square().sum(dim=-1).mean(dim=-1)
        root_position_error = (state[..., :3] - reference[..., :3]).square().sum(dim=-1)
        root_orientation_error = orientation_error(state[..., 3:7], reference[..., 3:7]).square().sum(dim=-1)
        joint_positions = state[..., 13 : 13 + self.joint_count]
        limit_violation = (
            (self.soft_lower_limits - joint_positions).clamp_min(0)
            + (joint_positions - self.soft_upper_limits).clamp_min(0)
        ).sum(dim=-1)
        return {
            "body_position": torch.exp(-position_error / 0.3**2),
            "body_orientation": torch.exp(-body_orientation_error / 0.4**2),
            "body_linear_velocity": torch.exp(-linear_error),
            "body_angular_velocity": torch.exp(-angular_error / 3.14**2),
            "root_position": 0.5 * torch.exp(-root_position_error / 0.3**2),
            "root_orientation": 0.5 * torch.exp(-root_orientation_error / 0.4**2),
            "action_change": -0.1 * (action - previous_action).square().sum(dim=-1),
            "joint_limit": -2 * limit_violation,
        }

    def reward(self, state, action, previous_action, reference_index):
        return self.control_period * sum(
            self.reward_terms(state, action, previous_action, reference_index).values()
        )

    def tracking_metrics(self, state, reference_index):
        """Direct errors in SI units; global drift and aligned pose stay separate."""
        reference = self.motions.states[reference_index]
        positions, orientations, _, _ = self.kinematics.forward(state)
        reference_positions, reference_orientations = (
            values[reference_index] for values in self.reference_links[:2]
        )
        alignment = multiply(heading(state[..., 3:7]), conjugate(heading(reference[..., 3:7])))
        anchor = torch.cat((state[..., :2], reference[..., 2:3]), dim=-1)
        aligned_positions = anchor[..., None, :] + rotate(
            alignment[..., None, :], reference_positions - reference[..., None, :3]
        )
        aligned_orientations = multiply(
            alignment[..., None, :].expand_as(reference_orientations), reference_orientations
        )
        joint_error = state[..., 13:13 + self.joint_count] - reference[..., 13:13 + self.joint_count]
        return {
            "root_horizontal_error_m": (state[..., :2] - reference[..., :2]).norm(dim=-1),
            "root_height_error_m": (state[..., 2] - reference[..., 2]).abs(),
            "root_orientation_error_rad": orientation_error(state[..., 3:7], reference[..., 3:7]).norm(dim=-1),
            "root_heading_error_rad": orientation_error(heading(state[..., 3:7]), heading(reference[..., 3:7])).norm(dim=-1),
            "root_linear_velocity_error_m_s": (state[..., 7:10] - reference[..., 7:10]).norm(dim=-1),
            "joint_rmse_rad": joint_error.square().mean(dim=-1).sqrt(),
            "aligned_body_position_rms_m": (positions - aligned_positions).square().sum(dim=-1).mean(dim=-1).sqrt(),
            "aligned_body_orientation_rms_rad": orientation_error(orientations, aligned_orientations).square().sum(dim=-1).mean(dim=-1).sqrt(),
        }

    def tracking_errors(self, state, reference_index):
        reference = self.motions.states[reference_index]
        height_error = (state[..., 2] - reference[..., 2]).abs()
        tilt_error = (self._gravity(state)[..., 2] - self._gravity(reference)[..., 2]).abs()
        return height_error, tilt_error

    def terminated(self, state, reference_index):
        height_error, tilt_error = self.tracking_errors(state, reference_index)
        return (height_error > 0.25) | (tilt_error > 0.8)

    def termination_probability(self, state, reference_index):
        height_error, tilt_error = self.tracking_errors(state, reference_index)
        # Softening temperatures are engineering choices, not paper constants.
        height_probability = torch.sigmoid((height_error - 0.25) / 0.02)
        tilt_probability = torch.sigmoid((tilt_error - 0.8) / 0.04)
        return 1 - (1 - height_probability) * (1 - tilt_probability)

    def critic_features(self, state, reference_index, context):
        goal = self.goal(state, reference_index)
        if self.goal_frame == "reference":
            return torch.cat((state, goal, context), dim=-1)
        # The robot-frame goal already carries the root pose error that the
        # global reward terms depend on; absolute XY and yaw would only add
        # unbounded, reward-irrelevant inputs to the critic.
        return torch.cat((self._invariant_state(state), goal, context), dim=-1)

    def valid_starts(self, reference_index, horizon):
        return reference_index + horizon <= self.motions.frame_ends[reference_index]

    def advance_reference(self, reference_index):
        return reference_index + 1
