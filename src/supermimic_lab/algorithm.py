"""Independent implementation of SuperMimic Equations 1--3 and Algorithm A.1."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .networks import DynamicsNetwork, MotionPolicy, ValueNetwork
from .objectives import imagined_objective, lambda_returns


@contextmanager
def frozen_weights(*modules):
    """Freeze weights while retaining gradients with respect to module inputs."""
    previous_flags = [
        (parameter, parameter.requires_grad)
        for module in modules
        for parameter in module.parameters()
    ]
    try:
        for parameter, _ in previous_flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, was_trainable in previous_flags:
            parameter.requires_grad_(was_trainable)


class SuperMimic:
    def __init__(self, task, replay, configuration):
        configuration.validate()
        self.task = task
        self.replay = replay
        self.config = configuration
        self.device = torch.device(task.device)
        self.dynamics = DynamicsNetwork(task.dimensions, configuration.model_widths).to(self.device)
        self.policy = MotionPolicy(task.dimensions, configuration).to(self.device)
        self.critic = ValueNetwork(task.dimensions.critic, configuration.critic_widths).to(self.device)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=configuration.learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=configuration.learning_rate, betas=(0.9, 0.95)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=configuration.learning_rate)
        self.error_scales = torch.ones(task.dimensions.state_errors, device=self.device)
        if configuration.state_error_scales is not None:
            scales = torch.tensor(configuration.state_error_scales, device=self.device)
            if scales.shape != self.error_scales.shape:
                raise ValueError("state_error_scales must match the task's state-error dimension")
            self.error_scales = scales

    def predict(self, state, action, context):
        features = self.task.encode_dynamics(state, action, context)
        return self.task.integrate_prediction(state, self.dynamics(features))

    @torch.no_grad()
    def observe_transition(self, transition):
        self.dynamics.input_statistics.update(self.task.encode_dynamics(
            transition.state, transition.action, transition.context
        ))
        self.dynamics.target_statistics.update(self.task.dynamics_target(
            transition.state, transition.next_state
        ))

    def fit_dynamics(self):
        samples = self.replay.windows(self.config.model_batch_size, self.config.model_window)
        prediction = samples["state"][:, 0]
        accumulated = torch.zeros(prediction.shape[0], device=self.device)
        for step in range(self.config.model_window):
            prediction = self.predict(prediction, samples["action"][:, step], samples["context"][:, step])
            error = self.task.state_error(prediction, samples["next_state"][:, step])
            accumulated = accumulated + (
                (error / self.error_scales).square().sum(dim=-1)
                * samples["valid"][:, step]
            )
            # An auto-reset is not a dynamics transition. Invalid window suffixes
            # use logged states and carry no loss or gradient across the reset.
            prediction = torch.where(
                samples["valid"][:, step, None].bool(), prediction,
                samples["next_state"][:, step],
            )
        loss = accumulated.mean()
        self.dynamics_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.dynamics.parameters(), 1.0, error_if_nonfinite=True
        )
        self.dynamics_optimizer.step()
        return {"model_loss": loss.detach(), "model_gradient_norm": gradient_norm.detach()}

    def _valid_initial_states(self):
        selected_starts = []
        selected_histories = []
        remaining = self.config.policy_batch_size
        for _ in range(64):
            start, history = self.replay.starts(max(2 * remaining, 16), self.config.history_frames)
            valid = self.task.valid_starts(start["reference_index"], self.config.imagination_steps)
            selection = valid.nonzero(as_tuple=True)[0][:remaining]
            if selection.numel():
                selected_starts.append({name: value[selection] for name, value in start.items()})
                selected_histories.append({name: value[selection] for name, value in history.items()})
                remaining -= selection.numel()
            if remaining == 0:
                combine = lambda records: {
                    name: torch.cat([record[name] for record in records]) for name in records[0]
                }
                return combine(selected_starts), combine(selected_histories)
        raise ValueError("Replay does not contain enough reference starts for imagination_steps")

    def _noisy_observation(self, state, previous_action, context):
        observation = self.task.observation(state, previous_action, context)
        return observation + (2 * torch.rand_like(observation) - 1) * self.task.observation_noise

    def improve_policy(self):
        start, logged_history = self._valid_initial_states()
        state = start["state"]
        context = start["context"]
        reference_index = start["reference_index"]
        previous_action = start["previous_action"]
        with torch.no_grad():
            history = self._noisy_observation(
                logged_history["state"], logged_history["previous_action"], logged_history["context"]
            )
        rewards = []
        terminations = []
        critic_inputs = [self.task.critic_features(state, reference_index, context).detach()]
        with frozen_weights(self.dynamics, self.critic):
            for _ in range(self.config.imagination_steps):
                goal = self.task.goal(state, reference_index)
                action = self.policy(history, goal)
                state = self.predict(state, action, context)
                reference_index = self.task.advance_reference(reference_index)
                rewards.append(self.task.reward(state, action, previous_action, reference_index))
                terminations.append(self.task.termination_probability(state, reference_index))
                observation = self._noisy_observation(state, action, context)
                history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
                previous_action = action
                critic_inputs.append(self.task.critic_features(state, reference_index, context).detach())
            reward_tensor = torch.stack(rewards)
            termination_tensor = torch.stack(terminations)
            bootstrap = torch.zeros_like(reward_tensor[-1])
            if self.config.use_critic:
                bootstrap = self.critic(self.task.critic_features(state, reference_index, context))
            returns, survival = imagined_objective(
                reward_tensor, termination_tensor, bootstrap, self.config.discount
            )
            policy_loss = -returns.mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_gradient = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 1.0, error_if_nonfinite=True
            )
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
            self.policy_optimizer.step()
        value_loss = torch.zeros((), device=self.device)
        if self.config.use_critic:
            critic_inputs = torch.stack(critic_inputs)
            self.critic.statistics.update(critic_inputs)
            with torch.no_grad():
                values = self.critic(critic_inputs)
                targets = lambda_returns(
                    reward_tensor.detach(), termination_tensor.detach(), values,
                    self.config.discount, self.config.trace_decay,
                ).flatten()
            inputs = critic_inputs[:-1].flatten(0, 1)
            for _ in range(self.config.critic_epochs):
                permutation = torch.randperm(inputs.shape[0], device=self.device)
                for selection in permutation.tensor_split(self.config.critic_minibatches):
                    value_loss = (self.critic(inputs[selection]) - targets[selection]).square().mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, error_if_nonfinite=True)
                    self.critic_optimizer.step()
        return {
            "policy_loss": policy_loss.detach(), "critic_loss": value_loss.detach(),
            "policy_gradient_norm": policy_gradient.detach(),
            "imagined_reward": reward_tensor.detach().mean(),
            "imagined_survival": survival.detach().mean(),
        }

    def state_dict(self):
        return {
            "dynamics": self.dynamics.state_dict(), "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "dynamics_optimizer": self.dynamics_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, checkpoint):
        for name in (
            "dynamics", "policy", "critic", "dynamics_optimizer",
            "policy_optimizer", "critic_optimizer",
        ):
            getattr(self, name).load_state_dict(checkpoint[name])
