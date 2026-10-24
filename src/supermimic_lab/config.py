"""Paper defaults and explicitly named engineering choices."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields


@dataclass
class AlgorithmConfig:
    model_widths: tuple[int, ...] = (2048, 2048, 2048)
    encoder_widths: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_widths: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_widths: tuple[int, ...] = (256, 128)
    latent_size: int = 32
    history_frames: int = 10
    future_frames: int = 10
    model_window: int = 8
    imagination_steps: int = 32
    model_batch_size: int = 2048
    policy_batch_size: int = 4096
    model_batches: int = 10
    critic_epochs: int = 5
    critic_minibatches: int = 4
    learning_rate: float = 0.0002
    discount: float = 0.99
    trace_decay: float = 0.95
    use_critic: bool = True
    # The paper does not specify state preprocessing. Unit scaling is explicit
    # here; None means the task's raw SI/tangent state errors in Equation 1.
    state_error_scales: tuple[float, ...] | None = None
    # Execution option, not part of the objective: fuse each model-window and
    # imagination step with torch.compile. Eager tasks with analytic kinematics
    # launch thousands of small kernels per step; requires a working Inductor
    # backend (Triton on CUDA, a C++ compiler on CPU).
    compile_steps: bool = False

    def validate(self):
        for name in (
            "latent_size", "history_frames", "future_frames", "model_window",
            "imagination_steps", "model_batch_size", "policy_batch_size",
            "model_batches", "critic_epochs", "critic_minibatches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("model_widths", "encoder_widths", "decoder_widths", "critic_widths"):
            widths = getattr(self, name)
            if not widths or any(not isinstance(width, int) or width <= 0 for width in widths):
                raise ValueError(f"{name} must contain positive layer widths")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.discount < 1 or not 0 <= self.trace_decay <= 1:
            raise ValueError("Invalid discount or trace_decay")
        if not isinstance(self.compile_steps, bool):
            raise ValueError("compile_steps must be a boolean")
        if self.critic_minibatches > self.imagination_steps * self.policy_batch_size:
            raise ValueError("Critic minibatches would be empty")
        if self.state_error_scales is not None and any(
            not math.isfinite(value) or value <= 0 for value in self.state_error_scales
        ):
            raise ValueError("state_error_scales must be finite and positive")


@dataclass
class RunnerConfig:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    seed: int = 0
    max_iterations: int = 320000
    replay_steps: int = 1000
    updates_per_vector_step: float = 0.4
    action_noise_std: float = 0.1
    initial_collection_steps: int = 64
    model_pretraining_iterations: int = 0
    decay_iterations: int = 250000
    save_interval: int = 2000
    log_interval: int = 50
    max_hours: float | None = None
    experiment_name: str = "motion_tracking"
    tensorboard: bool = False

    def validate(self):
        self.algorithm.validate()
        for name in (
            "max_iterations", "replay_steps", "initial_collection_steps",
            "decay_iterations", "save_interval", "log_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.replay_steps, self.initial_collection_steps) < self.algorithm.model_window:
            raise ValueError("Replay and initial collection must cover model_window")
        if not math.isfinite(self.updates_per_vector_step) or self.updates_per_vector_step <= 0:
            raise ValueError("updates_per_vector_step must be finite and positive")
        if not math.isfinite(self.action_noise_std) or self.action_noise_std < 0:
            raise ValueError("action_noise_std must be finite and nonnegative")
        if self.model_pretraining_iterations < 0:
            raise ValueError("model_pretraining_iterations must be nonnegative")
        if self.max_hours is not None and (not math.isfinite(self.max_hours) or self.max_hours <= 0):
            raise ValueError("max_hours must be finite and positive")
        if not self.experiment_name or self.experiment_name in (".", "..") or any(
            separator in self.experiment_name for separator in ("/", "\\")
        ):
            raise ValueError("experiment_name must be one directory name")

    def to_dict(self):
        return asdict(self)

    def copy(self):
        return copy.deepcopy(self)

    def from_dict(self, overrides):
        candidate = self.copy()
        allowed = {item.name for item in fields(candidate)}
        if set(overrides) - allowed:
            raise ValueError(f"Unknown runner fields: {sorted(set(overrides) - allowed)}")
        for name, value in overrides.items():
            if name == "algorithm":
                current = asdict(candidate.algorithm)
                if set(value) - set(current):
                    raise ValueError(f"Unknown algorithm fields: {sorted(set(value) - set(current))}")
                current.update(value)
                candidate.algorithm = AlgorithmConfig(**current)
            else:
                setattr(candidate, name, value)
        candidate.validate()
        self.__dict__.update(candidate.__dict__)

    @classmethod
    def from_mapping(cls, values):
        configuration = cls()
        configuration.from_dict(values)
        return configuration
