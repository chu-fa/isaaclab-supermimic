"""Run the A3 reference-motion demo in an existing Isaac Lab environment."""

import argparse
import math
import os
from pathlib import Path
import shlex
import sys


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_hours(value):
    hours = float(value)
    if not math.isfinite(hours) or hours <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return hours


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    for operation in ("train", "evaluate", "play"):
        command = operations.add_parser(operation)
        command.add_argument("--asset-root", type=Path, required=True)
        command.add_argument("--motion-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--physics",
            choices=("newton_mjwarp", "isaacsim_physx"),
            default="newton_mjwarp",
        )
        command.add_argument(
            "--num-envs", type=positive_integer,
            default=2048 if operation == "train" else 40,
        )
        command.add_argument("--device", default="cuda:0")
        command.add_argument(
            "--visualizer", default="newton" if operation == "play" else "none"
        )
        command.add_argument(
            "--dry-run", action="store_true",
            help="Print the command without importing Isaac Lab or launching it",
        )
        command.add_argument(
            "--checkpoint", type=Path, required=operation != "train"
        )
        if operation == "train":
            command.add_argument("--hours", type=positive_hours, default=3.0)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--eval-every", type=positive_integer, default=2000)
            command.add_argument("--stop-on-plateau", action="store_true")
        else:
            command.add_argument("--clips", type=int, nargs="+")
            command.add_argument("--max-steps", type=positive_integer)
    return parser, parser.parse_args()


def build_command(arguments, project):
    task_arguments = [
        "--task", "supermimic_a3:SuperMimic-Lab-Tracking-A3-v0",
        "--asset-root", str(arguments.asset_root.resolve()),
        "--motion-dir", str(arguments.motion_dir.resolve()),
        "--physics", arguments.physics,
        "--visualizer", arguments.visualizer,
        "--device", arguments.device,
        "--num-envs", str(arguments.num_envs),
    ]
    if arguments.checkpoint is not None:
        task_arguments.extend(["--checkpoint", str(arguments.checkpoint.resolve())])
    if arguments.operation == "train":
        task_arguments.extend([
            "--config", str(project / "configs/a3_rtx3090.json"),
            "--randomization", "nominal",
            "--seed", str(arguments.seed),
            "--hours", str(arguments.hours),
            "--eval-every", str(arguments.eval_every),
        ])
        if arguments.stop_on_plateau:
            task_arguments.append("--stop-on-plateau")
        return [
            sys.executable, str(project / "scripts/supervise_training.py"),
            "--source", str(project),
            "--task-source", str(project / "examples/a3/src"),
            "--output", str(arguments.output.resolve()),
            # Periodic evaluation runs outside the learner budget; leave room for it.
            "--process-timeout", str(arguments.hours * 7200 + 3600),
            "--", *task_arguments,
        ]
    task_arguments.extend(["--output", str(arguments.output.resolve())])
    if arguments.clips is not None:
        task_arguments.extend(["--clips", *map(str, arguments.clips)])
    if arguments.max_steps is not None:
        task_arguments.extend(["--max-steps", str(arguments.max_steps)])
    return [
        sys.executable, "-m", f"supermimic_lab.{arguments.operation}",
        *task_arguments,
    ]


def main():
    parser, arguments = parse_arguments()
    project = Path(__file__).resolve().parents[1]
    command = build_command(arguments, project)
    print(shlex.join(command), flush=True)
    if arguments.dry_run:
        return
    usd_relative_path = (
        "a3_usd_newton_v1/model.usda"
        if arguments.physics == "newton_mjwarp"
        else "a3_usd_physx_v2/model/model.usda"
    )
    required_paths = [
        arguments.asset_root / "a3_description/urdf/model.urdf",
        arguments.asset_root / "a3_description/runtime_profile.json",
        arguments.asset_root / usd_relative_path,
    ]
    if arguments.checkpoint is not None:
        required_paths.append(arguments.checkpoint)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        parser.error("Missing external files: " + ", ".join(missing_paths))
    if not any(arguments.motion_dir.glob("*.csv")):
        parser.error("motion-dir must contain A3-retargeted CSV files")
    if arguments.output.exists():
        parser.error("output already exists; select a new run directory or report file")
    # Preserve Isaac Sim's launcher-provided dependency paths. This only changes
    # the child process environment; it installs no package and edits no shell file.
    environment = dict(os.environ)
    runtime_paths = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join([
        str(project / "src"), str(project / "examples/a3/src")
    ]) + (
        os.pathsep + runtime_paths if runtime_paths else ""
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
