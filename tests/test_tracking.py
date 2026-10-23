"""Kinematic and reward behavior independent of Isaac Lab and robot assets."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from supermimic_lab.geometry import orientation_error
from supermimic_lab.kinematics import KinematicTree
from supermimic_lab.motions import MotionLibrary
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
        torch.testing.assert_close(self.task.goal(reference, index), self.task.goal(translated, index))
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


if __name__ == "__main__":
    unittest.main()
