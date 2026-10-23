"""Task-supplied evaluations, independent run records and optional plateau stop."""

import json
import math
from pathlib import Path


def relative_range(values):
    return (max(values) - min(values)) / max(max(abs(value) for value in values), 1e-9)


class EvaluationCallback:
    def __init__(
        self, environment, evaluator, directory, interval, *,
        stop_requested, stop_on_plateau=False,
    ):
        self.environment = environment
        self.evaluator = evaluator
        self.directory = Path(directory)
        self.interval = interval
        self.stop_requested = stop_requested
        self.stop_on_plateau = stop_on_plateau
        self.periodic_reports = []

    def evaluate(self, runner, label, time_limit_seconds=1800):
        runner.mark_collection_reset()
        checkpoint_path = self.directory / (
            "initial.pt" if label == "initial" else f"model_{runner.iteration:06d}.pt"
        )
        runner.save(checkpoint_path)
        status_path = self.directory / "evaluation_status.json"
        status_path.write_text(json.dumps({
            "phase": "evaluating", "iteration": runner.iteration, "label": label,
        }, indent=2) + "\n")
        report = self.evaluator(
            self.environment, runner.get_inference_policy(),
            stop_requested=self.stop_requested, time_limit_seconds=time_limit_seconds,
        )
        report.update(
            iteration=runner.iteration,
            checkpoint=str(checkpoint_path.resolve()),
            training_seconds=runner.previous_training_seconds + runner.elapsed(),
        )
        output = self.directory / f"evaluation_{label}.json"
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(report, indent=2) + "\n")
        status_path.write_text(json.dumps({
            "phase": "finished", "iteration": runner.iteration,
            "label": label, "report": str(output),
        }, indent=2) + "\n")
        runner.mark_collection_reset()
        print(json.dumps({
            "evaluation": label,
            **{name: value for name, value in report.items() if name != "clips"},
        }), flush=True)
        return report

    def before_learning(self, runner):
        self.evaluate(runner, "initial", time_limit_seconds=1800)

    def after_update(self, runner):
        if not self.interval or runner.iteration % self.interval:
            return None
        remaining_seconds = 1800
        if runner.config.max_hours is not None:
            remaining_seconds = min(
                remaining_seconds, 3600 * runner.config.max_hours - runner.elapsed()
            )
        if remaining_seconds <= 0:
            return None
        report = self.evaluate(
            runner, f"{runner.iteration:06d}", time_limit_seconds=remaining_seconds
        )
        self.periodic_reports.append(report)
        if not self.stop_on_plateau or runner.elapsed() < 3600:
            return None
        recent = self.periodic_reports[-3:]
        fields = ("failures", "mean_normalized_return", "mean_tracking_seconds")
        if len(recent) < 3 or not all(
            record.get("complete_evaluation")
            and all(name in record and math.isfinite(record[name]) for name in fields)
            for record in recent
        ):
            return None
        failures = [record["failures"] for record in recent]
        returns = [record["mean_normalized_return"] for record in recent]
        durations = [record["mean_tracking_seconds"] for record in recent]
        if (
            len(set(failures)) == 1
            and relative_range(returns) <= 0.02
            and relative_range(durations) <= 0.05
        ):
            (self.directory / "adaptive_stop.json").write_text(json.dumps({
                "reason": "evaluation_plateau", "evaluated_iterations": [
                    record["iteration"] for record in recent
                ],
                "failures": failures, "returns": returns, "tracking_seconds": durations,
                "note": "A stable plateau with failures is not successful motion tracking.",
            }, indent=2) + "\n")
            return "evaluation_plateau"
        return None

    def finish(self, runner):
        status = json.loads((self.directory / "status.json").read_text())
        if status["stop_reason"] == "user_signal" or self.stop_requested():
            return
        final = self.evaluate(runner, "final")
        initial = json.loads((self.directory / "evaluation_initial.json").read_text())
        (self.directory / "summary.json").write_text(json.dumps({
            "implementation": "independent-paper-reimplementation",
            "stop_reason": status["stop_reason"], "iteration": runner.iteration,
            "initial": {name: value for name, value in initial.items() if name != "clips"},
            "final": {name: value for name, value in final.items() if name != "clips"},
        }, indent=2) + "\n")
