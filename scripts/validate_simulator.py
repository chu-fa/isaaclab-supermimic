"""Bounded integration check, using a deliberately small test-only learner."""

import json
from pathlib import Path

import torch

from supermimic_lab.cli import parser_for, runner_configuration, simulation_environment
from supermimic_lab.evaluate import evaluate_clips, load_policy
from supermimic_lab.geometry import orientation_error
from supermimic_lab.runner import SuperMimicRunner


def main():
    parser = parser_for("train")
    parser.description = "Finite simulator/FK/gradient/checkpoint interface validation"
    arguments = parser.parse_args()
    if arguments.config is None or arguments.output is None:
        parser.error("Validation requires an explicit fixture --config and fresh --output")
    configuration = runner_configuration(arguments)
    if arguments.num_envs > 8 or configuration.max_iterations > 3:
        parser.error("Validation is limited to 8 worlds and 3 learner updates")
    with simulation_environment(arguments, configuration) as environment:
        from supermimic_lab.isaaclab.environment import TransitionAdapter

        environment.reset()
        state = environment.read_state()
        expected_positions, expected_orientations, expected_linear, expected_angular = environment.model_task.kinematics.forward(state)
        body_indices = [environment.robot.body_names.index(name) for name in environment.model_task.kinematics.tracked_links]
        poses = environment.robot.data.body_link_pose_w.torch[:, body_indices]
        velocities = environment.robot.data.body_link_vel_w.torch[:, body_indices]
        position_error = (poses[..., :3] - environment.scene.env_origins[:, None, :] - expected_positions).norm(dim=-1).max()
        rotation_error = orientation_error(poses[..., 3:], expected_orientations).norm(dim=-1).max()
        linear_error = (velocities[..., :3] - expected_linear).norm(dim=-1).max()
        angular_error = (velocities[..., 3:] - expected_angular).norm(dim=-1).max()
        geometry = {
            "maximum_position_error_m": float(position_error),
            "maximum_orientation_error_rad": float(rotation_error),
            "maximum_linear_velocity_error_m_s": float(linear_error),
            "maximum_angular_velocity_error_rad_s": float(angular_error),
        }
        print(json.dumps({"kinematics": geometry}), flush=True)
        if position_error > 0.002 or rotation_error > 0.002 or linear_error > 0.01 or angular_error > 0.01:
            raise AssertionError("URDF kinematics differs from the configured simulator asset")
        runner = SuperMimicRunner(
            TransitionAdapter(environment), configuration, arguments.output,
            metadata={"purpose": "finite_interface_check", "randomization": arguments.randomization},
        )
        runner.learn()
        environment.cfg.evaluation = True
        environment.cfg.randomization = "nominal"
        policy = load_policy(Path(arguments.output) / "last.pt", environment, configuration)
        result = evaluate_clips(environment, policy, [0], max_steps=4)
        result["kinematics"] = geometry
        result["learner_updates"] = runner.iteration
        result["not_a_tracking_benchmark"] = True
        (Path(arguments.output) / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
