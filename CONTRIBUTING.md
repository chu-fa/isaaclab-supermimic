# Contributing

Contributions can improve the reusable SuperMimic integration, add task examples,
or strengthen reproducibility. The core and A3 example are maintained in one
repository.

## Development setup

Use a compatible Isaac Lab environment for simulator integration. CPU learner
checks require Python, PyTorch, NumPy and Gymnasium; they do not launch Isaac Lab.

From the repository root:

```bash
python -m pip install --no-deps --no-build-isolation . ./examples/a3
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run checks relevant to the change. Documentation-only changes need link, command
and data-consistency review; they do not require a simulator run. Backend changes
should include a bounded check using `scripts/validate_simulator.py`, with the
backend, versions, seed, assets and limits recorded. CI runs CPU checks only.

## Package boundaries

- Keep learning algorithms, generic contracts and reusable adapters in
  `src/supermimic_lab/`.
- Put robot-specific assets, configuration and registration in a task package.
  The A3 example lives in `examples/a3/src/supermimic_a3/`.
- Do not introduce a core dependency on an example task or a fixed joint count.
- Preserve Gym's environment API and capture terminal states before reset.
- Document coordinate frames, units, quaternion order and privileged inputs.

Use descriptive Python names, four-space indentation and readable expressions.
Document behavioral changes together with their configuration and compatibility
implications. Add tests for changed behavior rather than repeating implementation
details in assertions.

## Issues and pull requests

For a reproducible issue, include the package revision, runtime versions, backend,
command, seed, relevant configuration, expected behavior and actual behavior.
Use data hashes or minimal permitted fixtures when the original assets cannot be
shared. Remove credentials and machine-specific paths from attached logs.

A pull request should explain the concrete behavior changed, relevant validation
and remaining limitations. Include third-party attribution when required. Keep
datasets, generated models, checkpoints, caches and large videos outside the Git
source tree.

## Reporting training results

Identify the configuration, source revision or hashes, robot/data identity,
initialization, seed, compute budget and evaluation protocol. Report all evaluated
clips, including failures and truncations. Distinguish clip completion, tracking
quality and held-out generalization. A short interface check or a plateau is not
evidence of convergence.
