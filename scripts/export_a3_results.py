"""Export A3 metrics and configuration without copying private run paths or logs."""

import argparse
import csv
from datetime import datetime
from datetime import timezone
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def export_results(run_directory, output_directory, gpu_name="not recorded"):
    run = Path(run_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    recorded = read_json(run / "run.json")
    status = read_json(run / "status.json")
    controller = read_json(run / "controller.json")
    if not recorded or not status:
        raise ValueError("A run configuration and status are required")
    configuration = dict(recorded["configuration"])
    configuration["experiment_name"] = "a3_reference_tracking"
    metadata = recorded["metadata"]
    public_configuration = {
        "configuration": configuration,
        "num_envs": metadata["num_envs"],
        "randomization": metadata["randomization"],
        "robot": metadata["robot"],
        "task_signature": recorded["task_signature"],
        "initialized_from_checkpoint": recorded.get("source_checkpoint") is not None,
    }
    write_json(output / "training_configuration.json", public_configuration)
    write_json(output / "runtime_versions.json", read_json(run / "runtime_versions.json"))
    write_json(output / "training_source_sha256.json", read_json(run / "source_sha256.json"))
    evaluations = []
    evaluation_fields = (
        "iteration", "protocol", "task_signature", "complete_evaluation",
        "mean_normalized_return", "successes", "failures", "mean_tracking_seconds",
        "tracked_frame_fraction", "evaluation_seconds", "training_seconds",
    )
    clip_fields = (
        "clip_index", "clip", "expected_steps", "tracked_steps", "tracked_seconds",
        "failed", "complete", "outcome", "return", "normalized_return",
        "failure_height_error_m", "failure_projected_gravity_z_error",
    )
    for source in sorted(run.glob("evaluation_*.json")):
        if source.name == "evaluation_status.json":
            continue
        try:
            report = read_json(source)
        except json.JSONDecodeError:
            continue  # A live evaluation may still be writing its report.
        if "clips" not in report:
            continue
        public = {key: report[key] for key in evaluation_fields if key in report}
        public["clips"] = [
            {key: clip[key] for key in clip_fields if key in clip}
            for clip in report["clips"]
        ]
        checkpoint = run / Path(report["checkpoint"]).name
        public["checkpoint"] = checkpoint.name
        public["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        write_json(output / source.name, public)
        evaluations.append({
            "file": source.name,
            **{key: public[key] for key in evaluation_fields if key in public},
        })
    evaluations.sort(key=lambda record: (record["iteration"], record["file"]))
    latest_steps = 0
    metrics_path = run / "metrics.csv"
    if metrics_path.is_file():
        with metrics_path.open() as stream:
            for row in csv.DictReader(stream):
                latest_steps = int(row["environment_steps"])
    process_seconds = None
    if controller.get("started_at") and controller.get("stopped_at"):
        process_seconds = (
            datetime.strptime(controller["stopped_at"], "%Y-%m-%dT%H:%M:%S%z")
            - datetime.strptime(controller["started_at"], "%Y-%m-%dT%H:%M:%S%z")
        ).total_seconds()
    summary = {
        "experiment_id": "agibot-a3-official-newton-seed0",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "final_evaluation" if status["phase"] == "finished" and controller.get("phase") == "running" else status["phase"],
        "iteration": status["iteration"],
        "environment_steps_at_last_log": latest_steps,
        "learner_wall_seconds": status["elapsed_seconds"],
        "learner_wall_definition": "Includes periodic evaluation; excludes startup and initial evaluation.",
        "process_wall_seconds": process_seconds,
        "process_wall_definition": "Supervisor start to exit, including setup and final evaluation when present.",
        "stop_reason": status.get("stop_reason"),
        "robot": "AGIBOT A3",
        "physics": "newton_mjwarp",
        "gpu": gpu_name,
        "num_envs": metadata["num_envs"],
        "training_hours_budget": configuration["max_hours"],
        "seed": configuration["seed"],
        "evaluations": evaluations,
        "final_evaluation_available": (output / "evaluation_final.json").is_file(),
        "evaluation_split": "The same 40 reference clips used for training; no held-out split.",
    }
    write_json(output / "snapshot.json", summary)
    rows = [
        "# AGIBOT A3 training measurements", "",
        "Configuration and metrics for the official AGIBOT A3 example. This is an",
        "application test; clip completion alone does not establish tracking accuracy.", "",
        f"Snapshot: {summary['recorded_at_utc']}. State: **{summary['phase']}**.", "",
        "| Setting | Value |", "| --- | --- |",
        f"| Simulator | Isaac Lab / Newton |",
        f"| Compute | {summary['gpu']}, {summary['num_envs']} environments |",
        f"| Seed | {summary['seed']} |",
        f"| Initialization | {'Checkpoint' if public_configuration['initialized_from_checkpoint'] else 'From scratch'} |",
        f"| Budget | {configuration['max_hours']} hours |",
        f"| Learner updates | {status['iteration']} |",
        f"| Recorded learner wall time | {status['elapsed_seconds'] / 60:.2f} minutes |",
        f"| Interaction transitions at last log | {latest_steps} |", "",
        "Learner wall time includes periodic evaluation, but excludes startup and the",
        "initial evaluation. The final process wall time also includes final evaluation",
        "and is recorded in the summary once the supervisor exits.", "",
        "| Update | Completed | Failures | Mean tracking (s) | Normalized return | Coverage |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for report in evaluations:
        suffix = " (final)" if report["file"] == "evaluation_final.json" else ""
        rows.append(
            f"| {report['iteration']}{suffix} | {report['successes']}/40 | "
            f"{report['failures']} | {report['mean_tracking_seconds']:.4f} | "
            f"{report['mean_normalized_return']:.6f} | "
            f"{report['tracked_frame_fraction']:.2%} |"
        )
    rows.extend([
        "", "Incomplete evaluations are marked in each JSON report and must not be",
        "treated as full-library results. Training and evaluation use the same 40 clips.", "",
        "- [Summary and time definitions](snapshot.json)",
        "- [Training configuration](training_configuration.json)",
        "- [Runtime versions](runtime_versions.json)",
        "- [Frozen source hashes](training_source_sha256.json)",
        "- [Metric definitions](../README.md)", "",
        "## Video", "",
        "The recording protocol uses the final evaluated policy on three predetermined",
        "clips: walk1_subject1, dance1_subject1 and jumps1_subject1. Each starts at",
        "its first frame and is limited to 30 seconds; failures remain visible and",
        "are reported. Short previews do not replace the full-clip evaluations above.", "",
        "Video publication is pending. No video quality or successful rendering is",
        "claimed before an actual recording has been checked.", "",
    ])
    (output / "README.md").write_text("\n".join(rows))
    print(json.dumps({"phase": summary["phase"], "iteration": summary["iteration"],
                      "evaluations": len(evaluations)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-name", default="not recorded")
    arguments = parser.parse_args()
    export_results(arguments.run, arguments.output, arguments.gpu_name)


if __name__ == "__main__":
    main()
