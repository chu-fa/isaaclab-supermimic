"""Deterministic full-clip evaluation; failed tails remain in the denominator."""

from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from .cli import (
    parser_for, resolve_entry_point, runner_configuration, simulation_environment,
    task_specification,
)
from .networks import MotionPolicy
from .metrics import TrackingMetricAccumulator
from .runner import PolicySession


def load_policy(path, environment, configuration):
    checkpoint = torch.load(path, map_location=environment.device, weights_only=True)
    if checkpoint["task_signature"] != environment.signature:
        raise ValueError(
            "Evaluation robot/reference contract differs from the checkpoint; "
            "tracking checkpoints from before goal_frame existed need "
            '{"goal_frame": "reference"} in the environment configuration'
        )
    policy = MotionPolicy(environment.model_task.dimensions, configuration.algorithm).to(environment.device)
    policy.load_state_dict(checkpoint["learner"]["policy"])
    return PolicySession(policy, configuration.algorithm.history_frames)


@torch.no_grad()
def evaluate_clips(
    environment, policy, clip_indices, max_steps=None, *,
    stop_requested=None, time_limit_seconds=1800,
):
    motions = environment.model_task.motions
    records = []
    stop_requested = stop_requested or (lambda: False)
    started = time.monotonic()
    for group_start in range(0, len(clip_indices), environment.num_envs):
        selection = clip_indices[group_start : group_start + environment.num_envs]
        actual_count = len(selection)
        padded = selection + [selection[-1]] * (environment.num_envs - actual_count)
        clips = torch.tensor(padded, device=environment.device)
        environment.reset_to_clips(clips)
        policy.reset()
        alive = torch.zeros(environment.num_envs, device=environment.device, dtype=torch.bool)
        alive[:actual_count] = True
        failed = torch.zeros_like(alive)
        completed = torch.zeros_like(alive)
        returns = torch.zeros(environment.num_envs, device=environment.device)
        tracked = torch.zeros_like(clips)
        failure_height = torch.zeros_like(returns)
        failure_tilt = torch.zeros_like(returns)
        metric_function = getattr(environment.model_task, "tracking_metrics", None)
        direct_metrics = TrackingMetricAccumulator(actual_count, environment.device)
        expected_steps = motions.lengths[clips] - 1
        limit = int(expected_steps.max())
        if max_steps is not None:
            limit = min(limit, max_steps)
        interruption = None
        for _ in range(limit):
            if stop_requested():
                interruption = "user_signal"
                break
            if time.monotonic() - started >= time_limit_seconds:
                interruption = "time_budget"
                break
            state = environment.read_state()
            observation = environment.model_task.observation(
                state[:actual_count], environment.previous_action[:actual_count],
                environment.context[:actual_count],
            )
            goal = environment.model_task.goal(
                state[:actual_count], environment.reference_index[:actual_count]
            )
            selected_actions = policy(observation, goal)
            # Finished worlds stay on their current pose while other clips finish.
            hold_action = (
                state[:, 13 : 13 + selected_actions.shape[-1]] - environment.default_positions
            ) / environment.action_scales
            action = hold_action.clone()
            action[:actual_count] = torch.where(
                alive[:actual_count, None], selected_actions, hold_action[:actual_count]
            )
            environment.step(action)
            transition = environment.last_transition
            if callable(metric_function):
                reference_after = torch.minimum(
                    transition.reference_index[:actual_count] + 1,
                    motions.frame_ends[transition.reference_index[:actual_count]],
                )
                direct_metrics.update(
                    metric_function(transition.next_state[:actual_count], reference_after),
                    alive[:actual_count],
                )
            returns += transition.reward * alive
            tracked += alive.long()
            just_failed = alive & transition.terminated
            height_error, tilt_error = environment.model_task.tracking_errors(
                transition.next_state, environment.reference_index
            )
            failure_height[just_failed] = height_error[just_failed]
            failure_tilt[just_failed] = tilt_error[just_failed]
            failed |= just_failed
            completed |= alive & ~transition.terminated & (tracked >= expected_steps)
            alive &= ~transition.episode_end
            if not bool(alive[:actual_count].any()):
                break
        for world_index in range(actual_count):
            clip_index = selection[world_index]
            records.append({
                "clip_index": clip_index, "clip": motions.names[clip_index],
                "expected_steps": int(expected_steps[world_index]),
                "tracked_steps": int(tracked[world_index]),
                "tracked_seconds": float(tracked[world_index]) * motions.control_period,
                "failed": bool(failed[world_index]), "complete": bool(completed[world_index]),
                "outcome": (
                    "failure" if failed[world_index]
                    else "complete" if completed[world_index]
                    else interruption or "step_budget"
                ),
                "return": float(returns[world_index]),
                "normalized_return": float(returns[world_index] / (0.1 * expected_steps[world_index])),
                "tracking_errors": direct_metrics.record(world_index),
                "failure_height_error_m": float(failure_height[world_index]),
                "failure_projected_gravity_z_error": float(failure_tilt[world_index]),
            })
        if interruption is not None:
            break
    metric_names = set().union(*(record["tracking_errors"] for record in records))
    metric_steps = sum(record["tracked_steps"] for record in records)
    aggregate_errors = {
        name: {
            "mean": sum(record["tracking_errors"][name]["mean"] * record["tracked_steps"]
                        for record in records if name in record["tracking_errors"]) / metric_steps,
            "max": max(record["tracking_errors"][name]["max"]
                       for record in records if name in record["tracking_errors"]),
        }
        for name in metric_names
    } if metric_steps else {}
    return {
        "tracking_errors": aggregate_errors,
        "tracking_error_protocol": "Executed transitions including the failure step; aggregate means weighted by tracked frames. Unexecuted tails have no error sample. Aligned body metrics remove root XY and heading drift, which are reported separately.",
        "task_signature": environment.signature,
        "protocol": "first frame; deterministic actor; nominal physics; full-clip denominator",
        "complete_evaluation": (
            len(records) == len(clip_indices)
            and all(record["outcome"] in ("failure", "complete") for record in records)
        ),
        "mean_normalized_return": sum(record["normalized_return"] for record in records) / len(records),
        "successes": sum(record["complete"] for record in records),
        "failures": sum(record["failed"] for record in records),
        "mean_tracking_seconds": sum(record["tracked_seconds"] for record in records) / len(records),
        "tracked_frame_fraction": (
            sum(record["tracked_steps"] for record in records)
            / sum(record["expected_steps"] for record in records)
        ),
        "evaluation_seconds": time.monotonic() - started,
        "clips": records,
    }


def evaluate_task(
    environment, policy, clip_indices=None, max_steps=None, *,
    stop_requested=None, time_limit_seconds=1800,
):
    """Task-registered evaluator; restore collection settings after full clips."""
    clips = (
        list(range(len(environment.model_task.motions.names)))
        if clip_indices is None else list(clip_indices)
    )
    if not clips or min(clips) < 0 or max(clips) >= len(environment.model_task.motions.names):
        raise ValueError("clips must index the loaded motion library")
    previous_settings = (
        environment.cfg.evaluation,
        environment.cfg.randomization,
        environment.cfg.episode_length_s,
    )
    environment.cfg.evaluation = True
    environment.cfg.randomization = "nominal"
    environment.cfg.episode_length_s = 1e9
    try:
        return evaluate_clips(
            environment, policy, clips, max_steps,
            stop_requested=stop_requested, time_limit_seconds=time_limit_seconds,
        )
    finally:
        (
            environment.cfg.evaluation,
            environment.cfg.randomization,
            environment.cfg.episode_length_s,
        ) = previous_settings
        environment._evaluation_starts = None
        environment.reset()


def main():
    arguments = parser_for("evaluate").parse_args()
    configuration = runner_configuration(arguments)
    output = arguments.output or Path("runs") / f"{arguments.checkpoint.stem}_evaluation.json"
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    with simulation_environment(arguments, configuration, evaluation=True) as environment:
        policy = load_policy(arguments.checkpoint, environment, configuration)
        specification = task_specification(arguments.task)
        entry_point = specification.kwargs.get("supermimic_eval_entry_point")
        if entry_point is None:
            raise ValueError("Register a supermimic_eval_entry_point for this task")
        report = resolve_entry_point(entry_point)(
            environment, policy, arguments.clips, arguments.max_steps
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
