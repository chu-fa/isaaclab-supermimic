# Setup

## Runtime requirements

Run training through the Python launcher of a compatible Isaac Lab installation.
The integration uses Isaac Lab 3 development APIs, including `launch_simulation`,
indexed articulation writes and XYZW pose buffers. Isaac Lab 2 compatibility has
not been established.

The integration was developed with Python 3.12, PyTorch 2.11, Newton 1.5.2,
MuJoCo/MuJoCo-Warp 3.11 and Warp 1.16. These identify a compatible environment,
not a universal installation lock. Source runs record their own resolved versions.

Isaac Lab and the selected physics backend must already be available. Package
metadata lists the core Python dependencies, but minimum versions alone do not
establish simulator compatibility. TensorBoard is required when logging is enabled
by the Runner configuration.

## Install from this repository

```bash
# Core only, for another project's tasks.
python -m pip install --no-deps --no-build-isolation .

# Optional A3 task, maintained in the same repository.
python -m pip install --no-deps --no-build-isolation ./examples/a3

supermimic-train --help
supermimic-evaluate --help
```

Use `/isaac-sim/python.sh` instead of `python` in an Isaac Sim container. All data
and output paths passed to a container command must be visible inside that
container. `--no-deps` leaves the environment's existing runtime dependencies in
place; it does not install missing dependencies.

The core imports without starting a simulator or registering a robot. The A3
package owns an `isaaclab.tasks` entry point. The explicit task name
`supermimic_a3:SuperMimic-Lab-Tracking-A3-v0` imports that package before resolving
the registered task.

## Source-based demo

The [A3 demo](../examples/a3/README.md) can run directly from a checkout using
`scripts/a3_demo.py`. Its process import path includes both packages, and training
freezes both source directories. `--dry-run` prints the command without launching
simulation or creating a training run.

External A3 data requirements and the CSV contract are specified in
[DATA.md](../examples/a3/DATA.md). Pinned official-model download and USD conversion
scripts are included in the checkout; they use the existing runtime. Retargeting
tools and motion CSVs are not installed by the package.

## Physics backends

| CLI backend | Asset requirement | Validation scope |
| --- | --- | --- |
| `newton_mjwarp` | Newton-compatible USD | Bounded interface check |
| `isaacsim_physx` | PhysX-compatible USD | Bounded interface check |

Changing the backend requires its matching assets and a separate run. Task
signatures prevent silently loading a checkpoint under a different backend.

The reference adapter uses a procedural 1000 m × 1000 m ground with its upper
surface at z=0. The checked PhysX API requires explicit link-origin-to-COM velocity
conversion at reset: `v_com = v_link + omega × R * offset_com`. Newton does not
enforce all actuator properties identically to PhysX; nonzero dynamic joint
friction is unsupported in the checked Newton runtime.

Nominal physics is the default. The optional paper randomization configuration
includes friction, encoder offsets and torso COM offsets. Newton runtime COM
changes have an SDK mass-matrix update caveat and require further physical
validation. Backend availability does not imply equivalent dynamics or results.

## Bounded interface validation

With both packages installed and external data available:

```bash
python scripts/validate_simulator.py \
  --task supermimic_a3:SuperMimic-Lab-Tracking-A3-v0 \
  --asset-root /path/to/external-data \
  --motion-dir /path/to/external-data/lafan1_a3 \
  --physics newton_mjwarp --visualizer none --num-envs 4 \
  --config tests/fixtures/tiny_runner.json \
  --output runs/a3_interface_check_01
```

This compares URDF forward kinematics with simulator states, performs two small
learner updates, reloads the policy and evaluates four control steps. It is an
interface check, not a tracking benchmark. The tiny fixture is separate from
production training configuration. Choose a fresh output directory and available
GPU resources. Check coverage is described in [VALIDATION.md](VALIDATION.md).
