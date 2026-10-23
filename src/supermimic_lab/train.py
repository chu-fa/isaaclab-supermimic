"""Train a registered task using SuperMimicRunner."""

import signal

from .cli import (
    new_run_directory, parser_for, resolve_entry_point, runner_configuration,
    simulation_environment, task_specification,
)
from .evaluation_callback import EvaluationCallback
from .runner import SuperMimicRunner


def main():
    arguments = parser_for("train").parse_args()
    configuration = runner_configuration(arguments)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    directory = new_run_directory(arguments, configuration)
    with simulation_environment(arguments, configuration) as environment:
        from .isaaclab.environment import TransitionAdapter

        environment.reset()
        runner = SuperMimicRunner(
            TransitionAdapter(environment), configuration, directory,
            metadata={
                "task": arguments.task,
                "robot": getattr(environment, "asset_contract", None),
                "num_envs": environment.num_envs, "randomization": arguments.randomization,
                "implementation": "independent-paper-reimplementation",
                "actor_inputs": "proprioception history and reference-heading future goals",
                "critic_and_world_model": "privileged simulator state and context",
                "long_training_validated": False,
                "evaluation_interval": arguments.eval_every,
                "plateau_stop": arguments.stop_on_plateau,
            },
        )
        if arguments.checkpoint:
            runner.load(arguments.checkpoint)
        callback = None
        if arguments.eval_every:
            evaluator_entry = task_specification(arguments.task).kwargs.get(
                "supermimic_eval_entry_point"
            )
            if evaluator_entry is None:
                raise ValueError("Periodic evaluation requires a task evaluator entry point")
            callback = EvaluationCallback(
                environment, resolve_entry_point(evaluator_entry), directory,
                arguments.eval_every, stop_requested=lambda: stop["requested"],
                stop_on_plateau=arguments.stop_on_plateau,
            )
        runner.learn(
            arguments.iterations, stop_requested=lambda: stop["requested"],
            before_learning=None if callback is None else callback.before_learning,
            after_update=None if callback is None else callback.after_update,
        )
        if callback is not None:
            callback.finish(runner)
    print(f"Run saved to {directory}", flush=True)


if __name__ == "__main__":
    main()
