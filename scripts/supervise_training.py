"""Freeze a source run and supervise its explicit training command and timeout."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--task-source", type=Path, action="append", default=[],
        help="Extra task source directory to freeze and add to the child import path",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-timeout", type=float, required=True)
    parser.add_argument("training_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.process_timeout <= 0:
        parser.error("process-timeout must be positive")
    training_arguments = arguments.training_arguments
    if training_arguments[:1] == ["--"]:
        training_arguments = training_arguments[1:]
    project = arguments.source.resolve()
    task_source_directories = [path.resolve() for path in arguments.task_source]
    for task_source_directory in task_source_directories:
        if not task_source_directory.is_dir():
            parser.error(f"Task source directory does not exist: {task_source_directory}")
    directory = arguments.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    frozen = directory / "source_snapshot"
    frozen.mkdir()
    for name in ("src", "configs", "scripts"):
        shutil.copytree(
            project / name, frozen / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    for name in ("pyproject.toml", "README.md", "LICENSE", "PROVENANCE.md"):
        shutil.copyfile(project / name, frozen / name)
    frozen_task_directories = []
    for source_index, task_source_directory in enumerate(task_source_directories):
        frozen_task_directory = frozen / "task_sources" / f"task_{source_index:03d}"
        shutil.copytree(
            task_source_directory, frozen_task_directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        frozen_task_directories.append(frozen_task_directory)
    source_hashes = {
        str(path.relative_to(frozen)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(frozen.rglob("*")) if path.is_file()
    }
    (directory / "source_sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    for option in ("--config", "--env-config"):
        if option in training_arguments:
            position = training_arguments.index(option) + 1
            original = Path(training_arguments[position]).resolve()
            if original.is_relative_to(project):
                training_arguments[position] = str(frozen / original.relative_to(project))
    command = [
        sys.executable, "-m", "supermimic_lab.train", *training_arguments,
        "--output", str(directory),
    ]
    versions = {"python": sys.version}
    for name in ("torch", "numpy", "gymnasium", "isaaclab", "isaaclab_physx", "isaaclab_newton", "newton", "mujoco", "warp-lang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (directory / "runtime_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    environment = dict(os.environ)
    # Preserve the runtime paths supplied by Isaac Sim's python.sh launcher;
    # its bundled dependencies are not all installed in Python site-packages.
    runtime_paths = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join([
        str(frozen / "src"), *map(str, frozen_task_directories)
    ]) + (
        os.pathsep + runtime_paths if runtime_paths else ""
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    requested_stop = {"signal": None}
    with (directory / "stdout.log").open("x") as output:
        process = subprocess.Popen(
            command, env=environment, cwd=project,
            stdout=output, stderr=subprocess.STDOUT,
        )

        def forward_stop(signal_number, _frame):
            requested_stop["signal"] = signal_number
            if process.poll() is None:
                process.send_signal(signal_number)

        signal.signal(signal.SIGTERM, forward_stop)
        signal.signal(signal.SIGINT, forward_stop)
        record = {
            "phase": "running", "supervisor_pid": os.getpid(),
            "training_pid": process.pid, "command": command,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "process_timeout_seconds": arguments.process_timeout,
            "implementation": "independent-paper-reimplementation",
            "task_source_snapshots": [
                str(path.relative_to(frozen)) for path in frozen_task_directories
            ],
        }
        (directory / "controller.json").write_text(json.dumps(record, indent=2) + "\n")
        forced_kill = False
        timeout_reached = False
        try:
            exit_code = process.wait(timeout=arguments.process_timeout)
        except subprocess.TimeoutExpired:
            timeout_reached = True
            process.terminate()
            try:
                exit_code = process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                forced_kill = True
                process.kill()
                exit_code = process.wait()
        record.update(
            phase="finished" if exit_code == 0 else "failed", exit_code=exit_code,
            stopped_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            user_signal=requested_stop["signal"], external_timeout=timeout_reached,
            forced_kill=forced_kill,
        )
        (directory / "controller.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
