"""Numerical and learning behavior, without Isaac Lab or robot downloads."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from supermimic_lab.algorithm import SuperMimic
from supermimic_lab.config import RunnerConfig
from supermimic_lab.contracts import ModelDimensions, Transition
from supermimic_lab.geometry import orientation_error, rotation_vector_to_quaternion
from supermimic_lab.networks import FeatureStatistics
from supermimic_lab.objectives import imagined_objective, lambda_returns
from supermimic_lab.replay import SequenceReplay
from supermimic_lab.runner import PolicySession, SuperMimicRunner


class LinearTask:
    """A stable controlled system for testing the actual model/policy updates."""

    device = torch.device("cpu")
    dimensions = ModelDimensions(2, 1, 1, 3, 1, 4, 2, 2, 4)
    observation_noise = torch.zeros(3)

    def encode_dynamics(self, state, action, context):
        return torch.cat((state, action, context), dim=-1)

    def dynamics_target(self, state, next_state):
        return next_state - state

    def integrate_prediction(self, state, prediction):
        return state + prediction

    def state_error(self, predicted, target):
        return predicted - target

    def observation(self, state, previous_action, context):
        return torch.cat((state, previous_action), dim=-1)

    def goal(self, state, reference_index):
        return torch.ones_like(state[..., :1])

    def advance_reference(self, reference_index):
        # A constant command does not advance a motion timeline.
        return reference_index

    def reward(self, state, action, previous_action, reference_index):
        return -(state[..., 0] - 1).square() - 0.01 * action.square().sum(dim=-1)

    def termination_probability(self, state, reference_index):
        return state[..., 0] * 0

    def critic_features(self, state, reference_index, context):
        return torch.cat((state, self.goal(state, reference_index), context), dim=-1)

    def valid_starts(self, reference_index, horizon):
        return reference_index + horizon < 10000


class LinearEnvironment:
    def __init__(self, count=4):
        self.model_task = LinearTask()
        self.num_envs = count
        self.signature = {"fixture": "linear_control", "state": "position_velocity"}
        self.state = torch.randn(count, 2) * 0.1
        self.previous_action = torch.zeros(count, 1)
        self.context = torch.ones(count, 1)
        self.reference_index = torch.zeros(count, dtype=torch.long)
        self.episode_steps = torch.zeros(count, dtype=torch.long)

    def read_state(self):
        return self.state.clone()

    def step(self, action):
        velocity = 0.9 * self.state[:, 1] + 0.2 * action[:, 0]
        next_state = torch.stack((self.state[:, 0] + 0.05 * velocity, velocity), dim=-1)
        ended = self.episode_steps >= 19
        transition = Transition(
            self.state.clone(), next_state.clone(), action.clone(), self.previous_action.clone(),
            self.context.clone(), self.reference_index.clone(), self.episode_steps.clone(),
            self.model_task.reward(next_state, action, self.previous_action, self.reference_index),
            torch.zeros_like(ended), ended,
        )
        self.state = torch.where(ended[:, None], torch.zeros_like(next_state), next_state)
        self.previous_action = torch.where(ended[:, None], torch.zeros_like(action), action)
        self.episode_steps = torch.where(ended, 0, self.episode_steps + 1)
        self.reference_index += 1
        return transition


def small_configuration():
    configuration = RunnerConfig()
    configuration.from_dict({
        "max_iterations": 2, "replay_steps": 64, "initial_collection_steps": 8,
        "log_interval": 1, "save_interval": 1,
        "algorithm": {
            "model_widths": [32, 32], "encoder_widths": [16], "decoder_widths": [16],
            "critic_widths": [16], "model_window": 2, "model_batches": 1,
            "model_batch_size": 16, "policy_batch_size": 8, "imagination_steps": 3,
            "history_frames": 3, "future_frames": 2,
            "critic_epochs": 1, "critic_minibatches": 2, "learning_rate": 0.001,
        },
    })
    return configuration


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_terminal_transition_reward_and_lambda_limits(self):
        rewards = torch.tensor([[2.0], [100.0]])
        terminations = torch.tensor([[1.0], [0.0]])
        objective, survival = imagined_objective(rewards, terminations, torch.tensor([50.0]), 0.9)
        torch.testing.assert_close(objective, torch.tensor([2.0]))
        torch.testing.assert_close(survival, torch.tensor([0.0]))
        values = torch.tensor([[1.0], [3.0], [5.0]])
        no_termination = torch.zeros_like(terminations)
        one_step = lambda_returns(rewards, no_termination, values, 0.9, 0.0)
        torch.testing.assert_close(one_step, rewards + 0.9 * values[1:])
        complete = lambda_returns(rewards, no_termination, values, 0.9, 1.0)
        torch.testing.assert_close(complete[0], torch.tensor([2 + 0.9 * 100 + 0.9**2 * 5]))

    def test_rotation_error_has_finite_derivative_at_identity(self):
        tangent = torch.zeros(4, 3, requires_grad=True)
        quaternion = rotation_vector_to_quaternion(tangent)
        target = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand_as(quaternion)
        error = orientation_error(quaternion, target)
        gradient = torch.autograd.grad(error.sum(), tangent)[0]
        torch.testing.assert_close(gradient, torch.ones_like(tangent))

    def test_statistics_match_population_moments(self):
        samples = torch.randn(15, 3) + 2
        statistics = FeatureStatistics(3)
        statistics.update(samples[:4])
        statistics.update(samples[4:])
        torch.testing.assert_close(statistics.mean, samples.mean(dim=0))
        torch.testing.assert_close(statistics.variance, samples.var(dim=0, correction=0))

    def test_replay_keeps_terminal_state_and_masks_resets(self):
        environment = LinearEnvironment(2)
        replay = SequenceReplay(4, 2, environment.model_task.dimensions, "cpu")
        environment.episode_steps[:] = 19
        terminal = environment.step(torch.ones(2, 1))
        replay.append(terminal)
        replay.append(environment.step(torch.zeros(2, 1)))
        windows = replay.windows(8, 2)
        torch.testing.assert_close(windows["valid"], torch.tensor([[1.0, 0.0]]).expand(8, -1))
        self.assertTrue(bool((windows["next_state"][:, 0].abs().sum(dim=-1) > 0).all()))

    def test_model_learns_and_policy_updates_without_changing_model(self):
        configuration = small_configuration()
        environment = LinearEnvironment()
        replay = SequenceReplay(64, 4, environment.model_task.dimensions, "cpu")
        algorithm = SuperMimic(environment.model_task, replay, configuration.algorithm)
        for _ in range(64):
            transition = environment.step(torch.randn(4, 1))
            replay.append(transition)
            algorithm.observe_transition(transition)
        samples = replay.windows(32, 2)

        def error():
            prediction = algorithm.predict(samples["state"][:, 0], samples["action"][:, 0], samples["context"][:, 0])
            return float((prediction - samples["next_state"][:, 0]).square().mean().detach())

        initial_error = error()
        for _ in range(60):
            algorithm.fit_dynamics()
        self.assertLess(error(), initial_error * 0.75)
        model_before = {name: value.clone() for name, value in algorithm.dynamics.state_dict().items()}
        policy_before = [value.detach().clone() for value in algorithm.policy.parameters()]
        metrics = algorithm.improve_policy()
        self.assertGreater(float(metrics["policy_gradient_norm"]), 0)
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(policy_before, algorithm.policy.parameters())))
        for name, value in algorithm.dynamics.state_dict().items():
            torch.testing.assert_close(value, model_before[name], rtol=0, atol=0)

    def test_runner_resume_restores_policy_and_all_optimizers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = small_configuration()
            runner = SuperMimicRunner(LinearEnvironment(), configuration, root / "first")
            runner.learn()
            restored = SuperMimicRunner(LinearEnvironment(), configuration, root / "second")
            restored.load(root / "first/last.pt")
            self.assertEqual(restored.iteration, 2)
            observations = torch.randn(2, 3, 3)
            goals = torch.ones(2, 1)
            torch.testing.assert_close(runner.algorithm.policy(observations, goals), restored.algorithm.policy(observations, goals), rtol=0, atol=0)
            for name in ("dynamics_optimizer", "policy_optimizer", "critic_optimizer"):
                self.assertTrue(getattr(restored.algorithm, name).state)
            restored.learn(1)
            status = json.loads((root / "second/status.json").read_text())
            self.assertEqual(status["iteration"], 3)
            self.assertEqual(status["updates_this_run"], 1)

    def test_evaluation_reset_does_not_create_a_physics_transition(self):
        environment = LinearEnvironment(2)
        replay = SequenceReplay(8, 2, environment.model_task.dimensions, "cpu")
        replay.append(environment.step(torch.zeros(2, 1)))
        replay.mark_collection_reset()
        replay.append(environment.step(torch.zeros(2, 1)))
        windows = replay.windows(8, 2)
        torch.testing.assert_close(
            windows["valid"], torch.tensor([[1.0, 0.0]]).expand(8, -1)
        )

    def test_runner_callback_can_request_a_saved_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SuperMimicRunner(
                LinearEnvironment(), small_configuration(), directory
            )
            runner.learn(after_update=lambda current: "task_complete")
            status = json.loads((Path(directory) / "status.json").read_text())
            self.assertEqual(status["stop_reason"], "task_complete")
            self.assertEqual(status["iteration"], 1)
            self.assertTrue((Path(directory) / "last.pt").is_file())


if __name__ == "__main__":
    unittest.main()
