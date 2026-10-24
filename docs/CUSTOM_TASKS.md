# Train your own Isaac Lab task

The intended extension unit is an Isaac Lab task package. SuperMimic remains the
learning library; robot assets, state/action conventions, analytic rewards,
termination and evaluation belong to your task. You can use Isaac Lab throughout.
There is no requirement to implement a different simulator or a minimal toy task.

## Register the environment, runner defaults and evaluator

Install only `supermimic-lab` for your own task. The optional `supermimic-lab-a3`
distribution is an independent example; it is not a prerequisite. Register your
Gym task in the same way as that example:

```python
import gymnasium as gym

gym.register(
    id="MyProject-SuperMimic-Task-v0",
    entry_point="my_tasks.environment:MyEnvironment",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "my_tasks.configuration:MyEnvironmentConfig",
        "supermimic_cfg_entry_point": "my_tasks.configuration:MyRunnerConfig",
        "supermimic_eval_entry_point": "my_tasks.evaluation:evaluate_task",
    },
)
```

`MyRunnerConfig` can subclass `RunnerConfig` to supply task defaults. The training
CLI reads the registered class, then applies an optional installed `--preset` and
`--config` JSON overrides. Environment
configuration follows Isaac Lab's `configclass` convention; `--env-config` supplies
JSON overrides before construction. `--asset-root` and `--motion-dir` are required
only by tasks that need those inputs, not by the learner itself.

```bash
supermimic-train --task my_tasks:MyProject-SuperMimic-Task-v0 \
  --env-config /path/to/task.json --config /path/to/learner.json \
  --physics newton_mjwarp --num-envs 2048 --hours 3 \
  --eval-every 1000 --output runs/my_task_01
```

Changing `--physics` to `isaacsim_physx` selects the other adapter. Your task's
assets and simulator-specific settings still need to support that backend.

## Reuse the tracking task with another robot

For a floating-base reference-motion task, you can reuse the included environment
and analytic model/MDP functions. Install the core plus your own package, without
installing `supermimic-lab-a3`. The following package files are sufficient to
register that route; robot assets and compatible motions must still be prepared.

```text
my-project/
  pyproject.toml
  src/my_tasks/__init__.py
  src/my_tasks/configuration.py
```

A minimal `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "my-supermimic-tasks"
version = "0.1.0"
dependencies = ["supermimic-lab"]

[tool.setuptools.packages.find]
where = ["src"]
```

Register in `src/my_tasks/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="MyRobot-Tracking-v0",
    entry_point="supermimic_lab.isaaclab.environment:TrackingEnvironment",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "my_tasks.configuration:MyRobotConfig",
        "supermimic_cfg_entry_point": "supermimic_lab.config:RunnerConfig",
        "supermimic_eval_entry_point": "supermimic_lab.evaluate:evaluate_task",
    },
)
```

Implement `src/my_tasks/configuration.py`:

```python
import hashlib
import json
from pathlib import Path

from isaaclab.utils import configclass
from supermimic_lab.isaaclab.configuration import TrackingEnvironmentConfig
from supermimic_lab.isaaclab.configuration import configure_robot


@configclass
class MyRobotConfig(TrackingEnvironmentConfig):
    def configure_assets(self, directory, backend):
        root = Path(directory).resolve()
        metadata_path = root / "robot.json"
        metadata = json.loads(metadata_path.read_text())
        profile_path = root / metadata["profile"]
        urdf_path = root / metadata["urdf"]
        usd_path = root / metadata["usd"][backend]
        configure_robot(self, {
            "profile": json.loads(profile_path.read_text()),
            "urdf_path": str(urdf_path),
            "usd_path": str(usd_path),
            "tracked_links": metadata["tracked_links"],
            "collision_exclusions": metadata.get("collision_exclusions", []),
            "action_scale_effort_fraction": metadata["action_scale_effort_fraction"],
            "asset_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in {
                    "metadata": metadata_path, "profile": profile_path,
                    "urdf": urdf_path, "usd": usd_path,
                }.items()
            },
        }, backend)
```

Under your `--asset-root`, create `robot.json`. Replace the link names and paths
with those in your own robot; do not reuse A3 gains by default:

```json
{
  "profile": "runtime_profile.json",
  "urdf": "robot.urdf",
  "usd": {
    "newton_mjwarp": "robot_newton.usda",
    "isaacsim_physx": "robot_physx.usda"
  },
  "tracked_links": ["base_link", "torso_link", "left_foot_link", "right_foot_link"],
  "collision_exclusions": [],
  "action_scale_effort_fraction": 0.25
}
```

The profile has `root_body`, `torso_body`, `initial_height_m`, ordered `joint_names`
and a `joints` object keyed by those names. Each joint provides `default_position`,
`stiffness`, `damping`, `armature`, `effort_limit_sim`, `velocity_limit_sim`,
`friction` and `dynamic_friction`. The A3 [profile](../examples/a3/runtime_profile.json)
is a complete schema example, not a parameter recommendation for another robot.

The current kinematics implementation supports a single rooted tree of fixed and
bounded revolute joints. Reference CSVs contain root XYZ, root XYZW and joint
angles in profile order: `7 + number_of_joints` columns at `source_motion_rate`
(default 30 Hz). `configure_robot` does not convert a URDF or retarget human data.
A converted asset must preserve the tracked frames, joint order, inertia and
intended contact geometry. Keep USD payloads immutable and record their hashes
alongside the top-level file when distributing a prepared bundle.

After installing both the core and your task package in the existing runtime:

```bash
supermimic-train --task my_tasks:MyRobot-Tracking-v0 \
  --asset-root /data/my-robot --motion-dir /data/my-motions \
  --physics newton_mjwarp --num-envs 128 --hours 1 \
  --eval-every 1000 --output runs/my_robot_01
```

Use `--env-config` for supported tracking-configuration overrides. To freeze your
own source alongside the learner, use `scripts/supervise_training.py` with
`--task-source /path/to/my-project/src`; source directories are copied before
starting the child process. See the [Chinese integration guide](INTEGRATION.zh-CN.md)
for that full command and the distinction between this route and adapting an
existing task with different rewards or actions.

## Supply the SuperMimic model/MDP contract

SuperMimic cannot be attached to every existing PPO task through its observation
tensor alone. PPO can learn from scalar simulator rewards; imagined-rollout
backpropagation additionally needs a state transition representation and
differentiable reward/termination/observation functions. That task work is necessary
for meaningful reuse of this method.

Provide the methods in `DifferentiableTask`:

| Interface | Responsibility |
| --- | --- |
| `dimensions`, `device`, `observation_noise` | Tensor sizes and per-channel observation noise |
| `encode_dynamics` | Model input representation from state, action and physical context |
| `dynamics_target`, `integrate_prediction` | Supervised model target and state integration |
| `state_error` | State error coordinates for autoregressive model training |
| `observation`, `goal` | Differentiable actor inputs with an explicit information boundary |
| `advance_reference` | Advance a task condition handle during imagination; constant commands can return it unchanged |
| `reward`, `termination_probability` | Differentiable MDP heads on imagined next states |
| `critic_features` | Privileged value-function inputs |
| `valid_starts` | Whether a replayed condition supports the imagination horizon |

The `reference_index` field is a task condition handle. For motion tracking it is a
frame index; for another task it can index a stable command table. Keep command
data alive for as long as replay can reference them. It need not advance by one:
the algorithm calls your `advance_reference` method.

Reward and termination heads can be analytic, as in the included reference task.
Learned reward or termination heads would need their own supervised labels and
losses; they are not silently provided by the current implementation.

These methods run for every model-window and imagination step, so their eager
kernel count largely sets learner speed. Precompute anything that depends only on
the command data (the included task caches reference link kinematics), and batch
per-link loops. `compile_steps: true` fuses each step with `torch.compile`; the
task methods must then be traceable: no Python branching on tensor values, no
`.item()` and no host synchronization inside them.

`goal` and `critic_features` receive the live state. If a reward term depends on
the robot's pose relative to the command (for example global root position or
heading), expose that relation in the goal or observation; a term the actor cannot
observe only adds unexplained reward variance.

## Capture real transitions before reset

Your environment exposes `model_task`, `num_envs`, `signature`, `read_state()`,
`previous_action`, `context`, `reference_index`, `episode_steps` and a
`last_transition`. The normal Gym `step()` API remains intact;
`TransitionAdapter(env)` presents the learner-facing `step(action) -> Transition`.

Capture `next_state` before Isaac Lab resets terminated worlds. Keep hard physical
failures separate from episode time limits. The real executed action belongs in
replay, including exploration noise and any task-defined actuator conversion.
Include robot, observation/action conventions and reference identity in `signature`
so incompatible checkpoints are rejected.

The bundled `TrackingEnvironment` and `ReferenceTracking` supply this machinery for
floating-base motion imitation. A new robot can reuse `configure_robot` with its
URDF/USD, motor profile, joint order, tracked links and collision exclusions.
You can subclass `ReferenceTracking` for task-specific analytic heads, or implement
`DifferentiableTask` directly for a different task family. No algorithm file needs
robot names or joint counts.

## Evaluation and collection boundaries

The optional task evaluator receives `(environment, policy, clip_indices=None,
max_steps=None, *, stop_requested=None, time_limit_seconds=1800)` and returns a
JSON-serializable report. It should disable exploration/randomization, restore
collection settings on exit, and respect a user stop. The A3 evaluator is a
working full-reference example, not a required metric definition for every task.

`SuperMimicRunner.learn` exposes `before_learning` and `after_update` callbacks.
An after-update callback can request a named stop; the runner saves before exiting.
Call `runner.mark_collection_reset()` when evaluation resets the simulator, so
world-model windows cannot cross the discontinuity and policy history is cleared.
The supplied evaluation callback handles this for registered task evaluators.

`--stop-on-plateau` uses the supplied motion evaluator's complete-clip metrics.
It only applies after one training hour and three complete evaluations with equal
failure counts, return range ≤2%, and tracking-time range ≤5%. A custom task that
does not supply those metrics is never stopped by this heuristic. A stable plateau
with failures is reported as a plateau, not successful convergence.

The installed package has been checked without importing the simulator at core
import time. The production network sizes and objectives remain intact; small
fixtures under `tests/` are used only for bounded interface checks.
