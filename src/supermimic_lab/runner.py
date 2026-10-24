"""Learning-library-style runner with independent runs and explicit checkpoints."""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path

import torch

from .algorithm import SuperMimic
from .config import RunnerConfig
from .replay import SequenceReplay


class PolicySession:
    """An inference policy with explicit per-instance history reset."""

    def __init__(self, policy, history_frames):
        self.policy = policy.eval()
        self.history_frames = history_frames
        self.history = None

    def reset(self):
        self.history = None

    @torch.no_grad()
    def __call__(self, observation, goal, reset_mask=None):
        if self.history is None:
            self.history = observation[:, None].repeat(1, self.history_frames, 1)
        else:
            self.history = torch.cat((self.history[:, 1:], observation[:, None]), dim=1)
            if reset_mask is not None:
                self.history = torch.where(
                    reset_mask[:, None, None], observation[:, None].expand_as(self.history), self.history
                )
        return self.policy(self.history, goal)


class SuperMimicRunner:
    def __init__(self, environment, configuration, log_directory, *, metadata=None):
        self.environment = environment
        self.task = environment.model_task
        self.config = (
            RunnerConfig.from_mapping(configuration)
            if isinstance(configuration, dict)
            else configuration.copy()
        )
        self.config.validate()
        self.directory = Path(log_directory)
        self.metadata = dict(metadata or {})
        self.replay = SequenceReplay(
            self.config.replay_steps, environment.num_envs, self.task.dimensions, self.task.device
        )
        self.algorithm = SuperMimic(self.task, self.replay, self.config.algorithm)
        self.iteration = 0
        self.environment_steps = 0
        self.previous_training_seconds = 0.0
        self.started = None
        self.callback_seconds = 0.0
        self.source_checkpoint = None
        self.policy_session = PolicySession(self.algorithm.policy, self.config.algorithm.history_frames)

    def get_inference_policy(self, device=None):
        policy = copy.deepcopy(self.algorithm.policy)
        if device is not None:
            policy.to(device)
        return PolicySession(policy, self.config.algorithm.history_frames)

    def elapsed(self):
        return 0.0 if self.started is None else time.monotonic() - self.started

    def learning_seconds(self):
        """Wall-clock time of this run excluding task callbacks such as evaluation.

        ``max_hours`` limits this quantity, matching the paper's accounting in
        which checkpoint evaluations are not part of the training budget.
        """
        return 0.0 if self.started is None else self.elapsed() - self.callback_seconds

    def mark_collection_reset(self):
        """Call when a task evaluator resets physics between collection phases."""
        self.replay.mark_collection_reset()
        self.policy_session.reset()

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save({
            "format": "supermimic_lab_v1", "iteration": self.iteration,
            "environment_steps": self.environment_steps,
            "training_seconds": self.previous_training_seconds + self.elapsed(),
            "configuration": self.config.to_dict(),
            "metadata": self.metadata,
            "task_signature": self.environment.signature,
            "learner": self.algorithm.state_dict(),
        }, temporary)
        temporary.replace(target)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.task.device, weights_only=True)
        if checkpoint.get("format") != "supermimic_lab_v1":
            raise ValueError(
                "Checkpoint belongs to a different implementation; "
                "legacy weights are not imported"
            )
        if checkpoint["task_signature"] != self.environment.signature:
            raise ValueError(
                "Checkpoint robot/reference contract differs from the current task; "
                "tracking checkpoints from before goal_frame existed need "
                '{"goal_frame": "reference"} and the first robot-frame runs need '
                '{"goal_offset_limit": null} in the environment configuration'
            )
        self.algorithm.load_state_dict(checkpoint["learner"])
        self.iteration = checkpoint["iteration"]
        self.environment_steps = checkpoint["environment_steps"]
        self.previous_training_seconds = checkpoint["training_seconds"]
        self.source_checkpoint = str(Path(path).resolve())
        self.replay.length = 0
        self.replay.write_index = 0
        self.policy_session.reset()

    @torch.no_grad()
    def _collect(self, vector_steps):
        rewards = []
        failures = []
        for _ in range(vector_steps):
            state = self.environment.read_state()
            observation = self.task.observation(
                state, self.environment.previous_action, self.environment.context
            )
            observation = observation + (2 * torch.rand_like(observation) - 1) * self.task.observation_noise
            goal = self.task.goal(state, self.environment.reference_index)
            self.algorithm.policy.observation_statistics.update(observation)
            self.algorithm.policy.goal_statistics.update(goal)
            action = self.policy_session(observation, goal, self.environment.episode_steps == 0)
            action = action + self.config.action_noise_std * torch.randn_like(action)
            transition = self.environment.step(action)
            self.replay.append(transition)
            self.algorithm.observe_transition(transition)
            self.environment_steps += self.environment.num_envs
            rewards.append(float(transition.reward.mean()))
            failures.append(float(transition.terminated.float().mean()))
        return {
            "simulator_reward": sum(rewards) / max(len(rewards), 1),
            "terminated_fraction": sum(failures) / max(len(failures), 1),
        }

    def _status(self, starting_iteration, reason=None, error=None):
        record = {
            "phase": "failed" if error else "finished" if reason else "training",
            "iteration": self.iteration,
            "starting_iteration": starting_iteration,
            "updates_this_run": self.iteration - starting_iteration,
            "elapsed_seconds": self.elapsed(), "stop_reason": reason,
            "error": error,
        }
        temporary = self.directory / "status.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(self.directory / "status.json")

    def learn(
        self, num_learning_iterations=None, *, stop_requested=None,
        before_learning=None, after_update=None,
    ):
        additional_iterations = (
            self.config.max_iterations - self.iteration
            if num_learning_iterations is None else num_learning_iterations
        )
        if additional_iterations <= 0:
            raise ValueError("Training requires a positive number of additional updates")
        if any((self.directory / name).exists() for name in ("run.json", "last.pt", "metrics.csv")):
            raise FileExistsError("Use a fresh run directory, including when resuming")
        stop_requested = stop_requested or (lambda: False)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "run.json").write_text(json.dumps({
            "configuration": self.config.to_dict(), "metadata": self.metadata,
            "task_signature": self.environment.signature,
            "source_checkpoint": self.source_checkpoint,
            "resume_restores": ["networks", "feature_statistics", "optimizers", "counters"],
            "resume_reinitializes": ["replay", "physics", "history", "reset_sampler", "rng"],
        }, indent=2) + "\n")
        self.save(self.directory / "initial.pt")
        starting_iteration = self.iteration
        final_iteration = starting_iteration + additional_iterations
        writer = None
        try:
            if self.config.tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(str(self.directory / "tensorboard"))
            if before_learning is not None:
                before_learning(self)
            if stop_requested():
                self.save(self.directory / "last.pt")
                self._status(starting_iteration, "user_signal")
                return self.iteration
            self._collect(self.config.initial_collection_steps)
            self.started = time.monotonic()
            self.callback_seconds = 0.0
            self._status(starting_iteration)
            collection_remainder = 0.0
            with (self.directory / "metrics.csv").open("w", newline="") as log_file:
                csv_writer = None
                while self.iteration < final_iteration:
                    update_start = time.monotonic()
                    progress = min(self.iteration / self.config.decay_iterations, 1.0)
                    learning_rate = self.config.algorithm.learning_rate * (
                        0.03 + 0.485 * (1 + math.cos(math.pi * progress))
                    )
                    for optimizer in (
                        self.algorithm.dynamics_optimizer,
                        self.algorithm.policy_optimizer,
                        self.algorithm.critic_optimizer,
                    ):
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                    collection_remainder += 1 / self.config.updates_per_vector_step
                    vector_steps = int(collection_remainder)
                    collection_remainder -= vector_steps
                    metrics = self._collect(vector_steps)
                    model_metrics = []
                    for _ in range(self.config.algorithm.model_batches):
                        model_metrics.append(self.algorithm.fit_dynamics())
                    metrics.update({
                        name: sum(float(record[name]) for record in model_metrics) / len(model_metrics)
                        for name in model_metrics[0]
                    })
                    if self.iteration >= self.config.model_pretraining_iterations:
                        metrics.update({
                            name: float(value)
                            for name, value in self.algorithm.improve_policy().items()
                        })
                    else:
                        metrics.update({name: 0.0 for name in (
                            "policy_loss", "critic_loss", "policy_gradient_norm",
                            "imagined_reward", "imagined_survival"
                        )})
                    self.iteration += 1
                    metrics.update(
                        iteration=self.iteration, environment_steps=self.environment_steps,
                        elapsed_seconds=self.elapsed(), learning_rate=learning_rate,
                        update_seconds=time.monotonic() - update_start,
                    )
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError("A training metric became non-finite")
                    callback_reason = None
                    if after_update is not None and not stop_requested():
                        callback_start = time.monotonic()
                        callback_reason = after_update(self)
                        self.callback_seconds += time.monotonic() - callback_start
                    budget_done = (
                        self.config.max_hours is not None
                        and self.learning_seconds() >= 3600 * self.config.max_hours
                    )
                    reason = (
                        "user_signal" if stop_requested()
                        else callback_reason if callback_reason is not None
                        else "wall_clock_budget" if budget_done
                        else "iteration_limit" if self.iteration == final_iteration
                        else None
                    )
                    metrics["elapsed_seconds"] = self.elapsed()
                    metrics["learning_seconds"] = self.learning_seconds()
                    if self.iteration % self.config.log_interval == 0 or reason:
                        self._status(starting_iteration, reason)
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(log_file, fieldnames=list(metrics))
                            csv_writer.writeheader()
                        csv_writer.writerow(metrics)
                        log_file.flush()
                        if writer is not None:
                            for name, value in metrics.items():
                                writer.add_scalar(name, value, self.iteration)
                        print(json.dumps(metrics), flush=True)
                    if self.iteration % self.config.save_interval == 0 or reason:
                        self.save(self.directory / "last.pt")
                        self.save(self.directory / f"model_{self.iteration:06d}.pt")
                        self._status(starting_iteration, reason)
                    if reason:
                        return self.iteration
        except Exception as error:
            if hasattr(error, "evidence"):
                torch.save(error.evidence, self.directory / "numerical_failure.pt")
            self._status(starting_iteration, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if writer is not None:
                writer.close()
            self.previous_training_seconds += self.elapsed()
            self.started = None
