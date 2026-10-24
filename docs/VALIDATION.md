# Validation

This guide describes checks for the learner, simulator adapters and packaging.
Training measurements are not part of the initial source release. Interface
checks alone do not establish tracking quality or paper benchmark performance.

## CPU behavior

The checks cover world-model learning on a known system, gradients through frozen
models, terminal rewards, lambda returns, replay and history boundaries,
normalization, checkpoint restoration, kinematics, motion resampling and source
supervision. Run from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

These checks do not start Isaac Lab or require a GPU. CI runs this CPU suite.

## Simulator interfaces

Use `scripts/validate_simulator.py` with an existing Isaac Lab environment and
external robot and motion assets. The [setup guide](SETUP.md#bounded-interface-validation)
provides a bounded command. It compares URDF kinematics with simulator states,
performs small learner updates, reloads the checkpoint and evaluates a limited
number of steps. A truncated evaluation is not a successful full-clip result.

Newton and PhysX require their matching converted assets. Use separate output
directories and record resolved parameters, runtime versions and errors for each
backend. Passing one backend's checks does not establish physical equivalence.

## Package boundaries

The core should import without starting Isaac Lab or registering A3. The optional
A3 task should register through the public entry point and depend only on the
core interface. Neither wheel should include robot models, motion CSVs, generated
geometry, checkpoints or private experiment records.

## Training evaluation

Follow the [A3 protocol](../examples/a3/EVALUATION.md). Report complete and failed
clips, return and coverage, while distinguishing completion from tracking accuracy.
Convergence, backend equivalence, held-out generalization and hardware transfer
require their own evidence. Multi-interface joint training remains unimplemented.

## Tracking diagnostics

Direct-error tests verify a 3-4-5 m global displacement remains visible after
body-pose alignment and that finished worlds stop contributing.
