"""Optional A3 task registration; importing it does not launch a simulator."""

import gymnasium as gym


TASK_ID = "SuperMimic-Lab-Tracking-A3-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="supermimic_lab.isaaclab.environment:TrackingEnvironment",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "supermimic_a3.configuration:A3EnvironmentConfig",
            "supermimic_cfg_entry_point": "supermimic_lab.config:RunnerConfig",
            "supermimic_eval_entry_point": "supermimic_lab.evaluate:evaluate_task",
        },
    )
