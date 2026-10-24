"""Kinematic and reward behavior independent of Isaac Lab and robot assets."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from supermimic_lab.geometry import (
    multiply, orientation_error, rotate, rotation_vector_to_quaternion,
)
from supermimic_lab.kinematics import KinematicTree
from supermimic_lab.motions import MotionLibrary
from supermimic_lab.metrics import TrackingMetricAccumulator
from supermimic_lab.sampling import ReturnSampler
from supermimic_lab.tracking import ReferenceTracking


URDF = """<robot name="pendulum">
<link name="base"/><link name="arm"/><link name="tip"/>
<joint name="hinge" type="revolute">
  <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
  <limit lower="-3" upper="3" effort="10" velocity="10"/>
</joint>
<joint name="tip_fixed" type="fixed">
  <parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/>
</joint></robot>"""


BRANCHED_URDF = """<robot name="branched">
<link name="base"/><link name="a"/><link name="a_tip"/><link name="b"/><link name="c"/><link name="d"/>
<joint name="hinge_a" type="revolute">
  <parent link="base"/><child link="a"/><origin xyz="0.1 0.2 0.3" rpy="0.3 -0.2 0.5"/>
  <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/>
</joint>
<joint name="a_fixed" type="fixed">
  <parent link="a"/><child link="a_tip"/><origin xyz="0.4 0 -0.1" rpy="0 0.7 0"/>
</joint>
<joint name="hinge_b" type="revolute">
  <parent link="base"/><child link="b"/><origin xyz="-0.2 0 0" rpy="0.1 0.2 0.3"/>
  <axis xyz="0 1 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/>
</joint>
<joint name="hinge_c" type="revolute">
  <parent link="b"/><child link="c"/><origin xyz="0 0.3 0.1"/>
  <axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="10" velocity="10"/>
</joint>
<joint name="hinge_d" type="revolute">
  <parent link="base"/><child link="d"/><origin xyz="0 0 -0.5"/>
  <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="10"/>
</joint></robot>"""


def sequential_forward(tree, state, tracked_links):
    """Reference link-by-link implementation of the documented URDF semantics."""
    joint_count = len(tree.joint_names)
    positions, rotations = [state[..., :3]], [state[..., 3:7]]
    velocities, angular_velocities = [state[..., 7:10]], [state[..., 10:13]]
    for parent, origin_position, origin_rotation, coordinate, axis in tree.operations:
        offset = rotate(rotations[parent], origin_position)
        rotation = multiply(rotations[parent], origin_rotation.expand_as(rotations[parent]))
        angular_velocity = angular_velocities[parent]
        if coordinate is not None:
            angular_velocity = angular_velocity + rotate(rotation, axis) * state[..., 13 + joint_count + coordinate, None]
            rotation = multiply(rotation, rotation_vector_to_quaternion(axis * state[..., 13 + coordinate, None]))
        positions.append(positions[parent] + offset)
        rotations.append(rotation)
        velocities.append(velocities[parent] + torch.cross(angular_velocities[parent], offset, dim=-1))
        angular_velocities.append(angular_velocity)
    indices = [tree.link_names.index(name) for name in tracked_links]
    return tuple(
        torch.stack([values[index] for index in indices], dim=-2)
        for values in (positions, rotations, velocities, angular_velocities)
    )


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        urdf = root / "robot.urdf"
        urdf.write_text(URDF)
        self.tree = KinematicTree(urdf, ["hinge"], ["base", "arm", "tip"], "cpu")
        samples = np.zeros((121, 8))
        samples[:, 0] = np.arange(121) / 30
        samples[:, 2] = 1.0
        samples[:, 6] = 1.0
        paths = [root / "first.csv", root / "second.csv"]
        for path in paths:
            np.savetxt(path, samples, delimiter=",")
        self.motions = MotionLibrary(paths, 1, "cpu")
        self.task = ReferenceTracking(self.tree, self.motions, [0.0], 2)

    def test_link_velocity_matches_rigid_rotation(self):
        state = self.motions.states[:1].clone()
        state[:, 7:13] = 0
        state[:, 13] = math.pi / 2
        state[:, 14] = 2
        positions, orientations, velocities, angular_velocities = self.tree.forward(state)
        torch.testing.assert_close(positions[0, -1], torch.tensor([0.0, 1.0, 1.0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(velocities[0, -1], torch.tensor([-2.0, 0.0, 0.0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(angular_velocities[0, -1], torch.tensor([0.0, 0.0, 2.0]))

    def test_resampling_and_clip_boundaries(self):
        torch.testing.assert_close(self.motions.states[:, 7], torch.ones(402))
        current = self.motions.ends[:1] - 1
        indices = self.motions.future_indices(current, 5)
        self.assertTrue(bool((indices == self.motions.ends[0]).all()))
        sampler = ReturnSampler(self.motions)
        selected, _ = sampler.sample(100)
        self.assertTrue(bool((selected < self.motions.frame_ends[selected]).all()))
        initial_probability = sampler.probabilities().clone()
        sampler.update(torch.tensor([0, 1]), torch.tensor([0.0, 10.0]))
        self.assertGreater(float(sampler.probabilities()[0]), float(sampler.probabilities()[1]))
        self.assertAlmostEqual(float(sampler.probabilities().sum()), 1.0, places=6)

    def test_prediction_coordinates_are_invertible_and_differentiable(self):
        initial = self.motions.states[:3].clone()
        increments = 0.01 * torch.randn(3, self.task.dimensions.model_outputs)
        increments.requires_grad_()
        predicted = self.task.integrate_prediction(initial, increments)
        recovered = self.task.dynamics_target(initial, predicted)
        torch.testing.assert_close(recovered, increments, atol=2e-6, rtol=1e-5)
        self.task.reward(predicted, torch.zeros(3, 1), torch.zeros(3, 1), torch.arange(3)).sum().backward()
        self.assertTrue(bool(torch.isfinite(increments.grad).all()))
        self.assertGreater(float(increments.grad.abs().sum()), 0)

    def test_reward_termination_and_actor_information_boundary(self):
        reference = self.motions.states[:1].clone()
        action = torch.zeros(1, 1)
        index = torch.zeros(1, dtype=torch.long)
        self.assertAlmostEqual(float(self.task.reward(reference, action, action, index)), 0.1, places=6)
        translated = reference.clone()
        translated[:, :2] += 10
        legacy = ReferenceTracking(self.tree, self.motions, [0.0], 2, goal_frame="reference")
        torch.testing.assert_close(legacy.goal(reference, index), legacy.goal(translated, index))
        context = torch.zeros(1, self.task.dimensions.context)
        torch.testing.assert_close(
            self.task.observation(reference, action, context),
            self.task.observation(translated, action, context),
        )
        translated[:, 2] += 0.251
        self.assertTrue(bool(self.task.terminated(translated, index)))
        self.assertFalse(bool(self.task.terminated(reference, index)))
        self.assertEqual(self.task.goal(reference, index).shape[-1], self.task.dimensions.goal)
        self.assertEqual(self.task.encode_dynamics(reference, action, context).shape[-1], self.task.dimensions.model_features)


    def test_robot_frame_goal_exposes_root_drift_to_actor_and_critic(self):
        reference = self.motions.states[:1].clone()
        index = torch.zeros(1, dtype=torch.long)
        context = torch.zeros(1, self.task.dimensions.context)
        legacy = ReferenceTracking(self.tree, self.motions, [0.0], 2, goal_frame="reference")
        # On the reference, both encodings describe the same future poses.
        torch.testing.assert_close(self.task.goal(reference, index), legacy.goal(reference, index))
        drifted = reference.clone()
        drifted[:, :2] += torch.tensor([0.6, -0.4])  # inside the 1 m offset limit
        drifted[:, 3:7] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
        frames = self.task.goal(drifted, index).reshape(2, -1)
        expected_offset = self.motions.states[1:3, :2] - drifted[:, :2]
        # A +90 degree heading maps world (x, y) to robot-heading (y, -x).
        torch.testing.assert_close(frames[:, :2], torch.stack((expected_offset[:, 1], -expected_offset[:, 0]), dim=-1))
        torch.testing.assert_close(frames[:, 2], self.motions.states[1:3, 2])
        critic = self.task.critic_features(drifted, index, context)
        self.assertEqual(critic.shape[-1], self.task.dimensions.critic)
        self.assertEqual(legacy.critic_features(drifted, index, context).shape[-1], legacy.dimensions.critic)
        self.assertFalse(torch.allclose(critic, self.task.critic_features(reference, index, context)))
        # Moving robot and reference together leaves the critic input unchanged.
        shifted = ReferenceTracking(self.tree, self.motions, [0.0], 2)
        shifted.motions = copy.copy(self.motions)
        shifted.motions.states = self.motions.states.clone()
        shifted.motions.states[:, :2] += 5
        moved = drifted.clone()
        moved[:, :2] += 5
        torch.testing.assert_close(shifted.critic_features(moved, index, context), critic)

    def test_robot_frame_goal_saturates_large_drift(self):
        reference = self.motions.states[:1].clone()
        index = torch.zeros(1, dtype=torch.long)
        unbounded = ReferenceTracking(self.tree, self.motions, [0.0], 2, goal_offset_limit=None)
        near = reference.clone()
        near[:, :2] += torch.tensor([0.3, 0.2])
        torch.testing.assert_close(self.task.goal(near, index), unbounded.goal(near, index))
        far = reference.clone()
        far[:, :2] += torch.tensor([30.0, -40.0])
        frames = self.task.goal(far, index).reshape(2, -1)
        # Only the direction of a 50 m offset remains, at the 1 m limit.
        expected = self.motions.states[1:3, :2] - reference[:, :2] + torch.tensor([-0.6, 0.8])
        torch.testing.assert_close(frames[:, :2], expected)
        self.assertGreater(float(unbounded.goal(far, index).abs().max()), 30)
        state = far.clone().requires_grad_()
        self.task.goal(state, index).sum().backward()
        self.assertTrue(bool(torch.isfinite(state.grad).all()))

    def test_batched_tree_matches_sequential_joint_evaluation(self):
        urdf = Path(self.directory.name) / "branched.urdf"
        urdf.write_text(BRANCHED_URDF)
        joints = ["hinge_a", "hinge_b", "hinge_c", "hinge_d"]
        tracked = ["c", "a_tip", "base", "b"]
        tree = KinematicTree(urdf, joints, tracked, "cpu")
        torch.manual_seed(3)
        state = torch.randn(5, 13 + 2 * len(joints), dtype=torch.float32)
        state[:, 3:7] = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True)
        state[0, 13:17] = 0  # fixed and revolute joints at zero angle
        state.requires_grad_()
        batched = tree.forward(state)
        for result, expected in zip(batched, sequential_forward(tree, state, tracked)):
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
        gradient = torch.autograd.grad(sum(value.sum() for value in batched), state)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        # The untracked branch is not evaluated, so its joint has no influence.
        torch.testing.assert_close(gradient[:, [16, 20]], torch.zeros(5, 2))

    def test_cached_reference_kinematics_match_direct_evaluation(self):
        for cached, direct in zip(self.task.reference_links, self.tree.forward(self.motions.states)):
            torch.testing.assert_close(cached, direct, rtol=0, atol=0)

    def test_direct_metrics_distinguish_global_drift_from_pose(self):
        reference = self.motions.states[:1].clone()
        index = torch.zeros(1, dtype=torch.long)
        for value in self.task.tracking_metrics(reference, index).values():
            torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-6, rtol=0)
        displaced = reference.clone()
        displaced[:, :2] += torch.tensor([3.0, 4.0])
        displaced[:, 3:7] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
        metrics = self.task.tracking_metrics(displaced, index)
        self.assertAlmostEqual(float(metrics["root_horizontal_error_m"]), 5.0, places=5)
        self.assertAlmostEqual(float(metrics["root_heading_error_rad"]), math.pi / 2, places=5)
        self.assertLess(float(metrics["aligned_body_position_rms_m"]), 1e-6)
        self.assertFalse(bool(self.task.terminated(displaced, index)))

    def test_metric_accumulation_excludes_finished_worlds(self):
        accumulator = TrackingMetricAccumulator(2, "cpu")
        accumulator.update({"distance_m": torch.tensor([2.0, 4.0])}, torch.tensor([True, True]))
        accumulator.update({"distance_m": torch.tensor([100.0, 6.0])}, torch.tensor([False, True]))
        self.assertEqual(accumulator.record(0)["distance_m"], {"mean": 2.0, "max": 2.0, "final": 2.0})
        self.assertEqual(accumulator.record(1)["distance_m"], {"mean": 5.0, "max": 6.0, "final": 6.0})


if __name__ == "__main__":
    unittest.main()
