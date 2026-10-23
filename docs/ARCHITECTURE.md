# Architecture

SuperMimic Lab is a single repository with a reusable core and an optional A3
application. Separate Python distributions allow other projects to install the
core without installing the example.

```text
src/supermimic_lab/              Learning library and Isaac Lab integration
examples/a3/src/supermimic_a3/   A3 configuration and task registration
examples/a3/EVALUATION.md       A3 evaluation protocol and metric definitions
configs/                       Example training configurations
scripts/                       Source-run and validation entry points
tests/                         CPU behavior checks and bounded-check fixtures
docs/                          Installation, interfaces and implementation details
```

## Dependencies

```text
Custom task package ──┐
                     ├──> supermimic-lab ──> Isaac Lab / PyTorch
supermimic-lab-a3 ────┘
```

`supermimic-lab` does not import, install or register A3. Task packages depend on
the core and own their robot configuration, task semantics and evaluation.

## Core learning library

The learner fits an autoregressive world model from real simulator transitions,
improves a policy through imagined rollouts, and trains a critic from fixed
lambda-return targets. Replay stores terminal next states before environment
reset and masks windows across episode boundaries.

`SuperMimicRunner` owns collection, optimizer updates, logging, checkpoints and
evaluation callbacks. Algorithm modules operate on Torch tensors and do not
import a simulator. Simulator differentiability is not required: gradients pass
through the learned model and task-defined differentiable functions.

## Isaac Lab integration

The adapter connects Isaac Lab's environment lifecycle to the learner transition
contract while retaining the Gym `step()` API. The reusable reference-tracking
environment supports Newton/MuJoCo-Warp and PhysX through backend-specific asset
and state-write handling.

Tasks register their environment, configuration, Runner defaults and evaluator.
Standalone train/play/evaluate commands load `module:TaskId`. This follows Isaac
Lab extension conventions; it does not add a native `--rl_library supermimic`
option to Isaac Lab's PPO launcher.

## Task packages

The optional `supermimic-lab-a3` package registers A3 and supplies its external
asset paths, joint/motor profile, tracked bodies and collision exclusions. It uses
the public core environment and learner. A custom task can reuse the tracking
adapter or supply the [differentiable task contract](CUSTOM_TASKS.md) directly.

Robot names, fixed joint counts and task-specific files belong in the task package.
The core contains no copy of the A3 task. Reference motion data, converted geometry
and trained weights remain external to both distributions.

## Reproducibility

Source runs can use `scripts/supervise_training.py`. It freezes source and
configuration, hashes the snapshot, records versions and bounds process runtime.
Repeatable `--task-source` arguments include external task source directories in
the snapshot and prioritize their frozen imports over working copies.

Checkpoint signatures include robot/data identity, state/action conventions and
backend. Resume restores networks, normalizers, optimizers and counters; replay,
physics and random state are reinitialized. These boundaries must be recorded
when comparing training runs. Metric definitions belong to the task, as shown by
the [A3 evaluation protocol](../examples/a3/EVALUATION.md).
