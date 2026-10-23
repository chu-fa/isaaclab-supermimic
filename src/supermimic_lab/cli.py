"""Shared standalone entry points; no modification of Isaac Lab's dispatcher."""

from __future__ import annotations

import argparse
import importlib
import importlib.resources
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import RunnerConfig


def parser_for(operation):
    parser = argparse.ArgumentParser(description=f"Independent SuperMimic {operation}")
    parser.add_argument("--task", required=True, help="Registered task ID, optionally module:task")
    parser.add_argument("--asset-root", type=Path, help="External assets when required by the task")
    parser.add_argument("--motion-dir", type=Path, help="References when required by the task")
    parser.add_argument("--env-config", type=Path, help="Task configuration JSON overrides")
    parser.add_argument("--num-envs", type=int, default=64 if operation == "train" else 1)
    parser.add_argument("--config", type=Path, help="JSON runner configuration overrides")
    parser.add_argument("--preset", choices=("rtx3090",), help="Installed batch-size preset; paper network sizes remain unchanged")
    parser.add_argument("--checkpoint", type=Path, required=operation != "train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--physics",
        choices=("newton_mjwarp", "isaacsim_physx"),
        default="newton_mjwarp",
    )
    if operation == "train":
        parser.add_argument("--iterations", type=int, help="Additional updates, including after resume")
        parser.add_argument("--hours", type=float, help="Training wall-clock budget after warmup")
        parser.add_argument("--randomization", choices=("nominal", "paper"), default="nominal")
        parser.add_argument("--tensorboard", action="store_true")
        parser.add_argument("--eval-every", type=int, default=0)
        parser.add_argument("--stop-on-plateau", action="store_true")
    else:
        parser.add_argument(
            "--max-steps", type=int,
            help="Optional bounded evaluation; truncation is not success",
        )
        parser.add_argument("--clips", type=int, nargs="+", help="Zero-based motion indices; all by default")
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(physics="newton_mjwarp", visualizer="none", device="cuda:0")
    return parser


def task_specification(task_name):
    import gymnasium as gym

    if ":" in task_name:
        module, task_name = task_name.split(":", 1)
        importlib.import_module(module)
    return gym.spec(task_name)


def resolve_entry_point(entry_point):
    if callable(entry_point):
        return entry_point
    module, object_name = entry_point.split(":", 1)
    return getattr(importlib.import_module(module), object_name)


def runner_configuration(arguments):
    import torch

    specification = task_specification(arguments.task)
    configuration_factory = resolve_entry_point(
        specification.kwargs.get(
            "supermimic_cfg_entry_point", "supermimic_lab.config:RunnerConfig"
        )
    )
    configuration = configuration_factory()
    if arguments.checkpoint:
        saved = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "supermimic_lab_v1":
            raise ValueError("Only this independent implementation's checkpoints are supported")
        configuration = RunnerConfig.from_mapping(saved["configuration"])
    if arguments.preset:
        preset = importlib.resources.files("supermimic_lab").joinpath(
            "presets", arguments.preset + ".json"
        )
        configuration.from_dict(json.loads(preset.read_text()))
    if arguments.config:
        configuration.from_dict(json.loads(arguments.config.read_text()))
    if arguments.seed is not None:
        configuration.seed = arguments.seed
    if getattr(arguments, "hours", None) is not None:
        configuration.max_hours = arguments.hours
    if getattr(arguments, "tensorboard", False):
        configuration.tensorboard = True
    configuration.validate()
    if arguments.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    if getattr(arguments, "iterations", None) is not None and arguments.iterations <= 0:
        raise ValueError("iterations must be positive")
    if getattr(arguments, "max_steps", None) is not None and arguments.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if getattr(arguments, "eval_every", 0) < 0:
        raise ValueError("eval-every cannot be negative")
    if getattr(arguments, "stop_on_plateau", False) and not arguments.eval_every:
        raise ValueError("stop-on-plateau requires periodic task evaluation")
    torch.manual_seed(configuration.seed)
    return configuration


@contextmanager
def simulation_environment(arguments, runner_config, *, evaluation=False):
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg

    specification = task_specification(arguments.task)
    with launch_simulation(PhysicsCfg(), arguments) as physics_configuration:
        environment_config = resolve_entry_point(
            specification.kwargs["env_cfg_entry_point"]
        )()
        if arguments.env_config:
            environment_config.from_dict(json.loads(arguments.env_config.read_text()))
        if hasattr(environment_config, "configure_assets"):
            if arguments.asset_root is None:
                raise ValueError("This task requires --asset-root")
            environment_config.configure_assets(arguments.asset_root, arguments.physics)
        environment_config.sim.physics = physics_configuration
        if arguments.physics == "newton_mjwarp":
            from isaaclab_newton.physics import MJWarpSolverCfg

            physics_configuration.solver_cfg = MJWarpSolverCfg(
                integrator="implicitfast", njmax=512, nconmax=128
            )
            physics_configuration.num_substeps = 1
        environment_config.sim.device = arguments.device
        environment_config.scene.num_envs = arguments.num_envs
        environment_config.seed = runner_config.seed
        if arguments.motion_dir is not None:
            environment_config.motion_directory = str(arguments.motion_dir.resolve())
        for name, value in {
            "future_frames": runner_config.algorithm.future_frames,
            "discount": runner_config.algorithm.discount,
            "evaluation": evaluation,
            "randomization": "nominal" if evaluation else arguments.randomization,
        }.items():
            if hasattr(environment_config, name):
                setattr(environment_config, name, value)
        if evaluation and hasattr(environment_config, "episode_length_s"):
            environment_config.episode_length_s = 1e9
        environment = gym.make(specification.id, cfg=environment_config)
        try:
            yield environment.unwrapped
        finally:
            environment.close()


def new_run_directory(arguments, configuration):
    if arguments.output is not None:
        return arguments.output.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path("runs") / configuration.experiment_name / timestamp
