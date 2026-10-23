"""The simulator/differentiable-model boundary, independent of robot size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ModelDimensions:
    state: int
    action: int
    context: int
    observation: int
    goal: int
    model_features: int
    model_outputs: int
    state_errors: int
    critic: int


class DifferentiableTask(Protocol):
    """Analytic heads needed for BPTT; physics is never differentiated."""

    dimensions: ModelDimensions
    device: torch.device
    observation_noise: torch.Tensor

    def encode_dynamics(self, state, action, context): ...
    def dynamics_target(self, state, next_state): ...
    def integrate_prediction(self, state, prediction): ...
    def state_error(self, predicted, target): ...
    def observation(self, state, previous_action, context): ...
    def goal(self, state, reference_index): ...
    def advance_reference(self, reference_index): ...
    def reward(self, state, action, previous_action, reference_index): ...
    def termination_probability(self, state, reference_index): ...
    def critic_features(self, state, reference_index, context): ...
    def valid_starts(self, reference_index, horizon): ...


@dataclass
class Transition:
    """Labels captured before auto-reset, including the terminal next state."""

    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    context: torch.Tensor
    reference_index: torch.Tensor
    episode_step: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    episode_end: torch.Tensor

    def tensors(self):
        return vars(self)
