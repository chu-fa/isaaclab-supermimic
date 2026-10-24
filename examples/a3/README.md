# AGIBOT A3 reference-motion example

[中文](README.zh-CN.md) · [Project](../../README.md) ·
[Data](DATA.md) · [Evaluation](EVALUATION.md)

This example applies the reusable SuperMimic Lab interface to AGIBOT A3 reference
motion tracking in Isaac Lab. It is an application test maintained in the same
repository as the core library.

The optional `supermimic-lab-a3` package registers the task and provides A3-specific
configuration. Its learner, Runner, tracking environment and evaluator come from
`supermimic-lab`. Other projects can use the core without this example.

## Requirements

- A compatible Isaac Lab environment; see [setup](../../docs/SETUP.md).
- The official A3 model, included runtime profile and prepared A3-retargeted
  references as specified in [DATA.md](DATA.md). Model binaries and CSVs are external.
- GPU resources appropriate for the selected environment count. The example
  configuration targets one RTX 3090 with 2048 environments.

From the repository root, install both packages using the environment's Python:

```bash
python -m pip install --no-deps --no-build-isolation . ./examples/a3
supermimic-train --task supermimic_a3:SuperMimic-Lab-Tracking-A3-v0 --help
```

The source demo below can also run without package installation. It adds the core
and task source directories to the process import path. In an Isaac Sim container,
use `/isaac-sim/python.sh` instead of `python` and container-visible data paths.

## Prepare the official model

The default is the original AGIBOT A3 model, pinned by commit and file hashes.
Use the existing Isaac Lab runtime:

```bash
python scripts/fetch_a3_assets.py --output /data/a3_official
python scripts/convert_a3_assets.py --asset-root /data/a3_official --visualizer none
```

These scripts install no dependencies. The motor profile is included as example
configuration; retargeted motion CSVs remain external. See [DATA.md](DATA.md) for
license and conversion details, and the
[Chinese training walkthrough](TRAINING.zh-CN.md) for PD values and the action path.
Use a new run directory when changing robot models or physics backends.

## Train

Preview the command from the repository root:

```bash
python scripts/a3_demo.py train \
  --asset-root /data/a3_official \
  --motion-dir /data/lafan1_a3 \
  --physics newton_mjwarp --num-envs 2048 --hours 3 \
  --stop-on-plateau --output runs/a3_demo_01 --dry-run
```

Remove `--dry-run` to start. The [configuration](../../configs/a3_rtx3090.json)
preserves full network widths and uses model/policy batch sizes of 512, an eight-step
world-model window, 32-step imagination and seed 0. The demo uses nominal physics.
Networks start from scratch unless `--checkpoint` is explicitly provided.

The supervisor freezes source/configuration and records runtime versions,
checksums, metrics and checkpoints in a new output directory. Initial evaluation
and every 2000 updates evaluate all supplied clips. The three-hour learner budget
starts after warmup and initial evaluation and excludes periodic evaluation; the
supervisor has a separate 7-hour process timeout with a 90-second termination grace period. Ctrl+C requests
saving at an update boundary and cancels the final automatic evaluation.

Optional `--stop-on-plateau` can stop after at least one hour and three complete
evaluations with unchanged failure counts, return variation at most 2%, and mean
tracking-time variation at most 5%. This reports a plateau, not convergence.

For PhysX, select `--physics isaacsim_physx`, provide the corresponding USD and use
a new output directory. Checkpoints enforce backend-specific task signatures.
See the [validation guide](../../docs/VALIDATION.md) for bounded interface checks.

## Evaluate

```bash
python scripts/a3_demo.py evaluate \
  --asset-root /data/a3_official \
  --motion-dir /data/lafan1_a3 \
  --physics newton_mjwarp --num-envs 40 \
  --checkpoint runs/a3_demo_01/last.pt \
  --output runs/a3_demo_01/evaluation.json
```

Use an existing checkpoint with its original robot, motion library and backend.
Evaluation starts each clip at its first frame with a deterministic policy and
nominal physics. Failure criteria and metric definitions are specified in the
[evaluation guide](EVALUATION.md).

For visual playback, replace `evaluate` with `play`, add `--num-envs 1 --clips 0`,
and use `--visualizer newton` for Newton or `--visualizer kit` for PhysX. A
`--max-steps` limit can bound a preview; truncation is not counted as completion.
Interactive window playback has not been validated. Bounded offscreen A3 video
recording is described below.

## Scope and limitations

This release contains source code, example configuration and usage documentation.
Training measurements, pretrained weights and demonstration videos are not included.
Clip completion alone does not establish accurate tracking, convergence or held-out
generalization.

The exact prepared data bundle and its preparation pipeline are not included, so
the example requires external data preparation. A3 morphology, actuator settings,
reference retargeting and nominal physics differ from the paper benchmark.
Multi-interface joint training remains unimplemented. See
[implementation coverage](../../docs/IMPLEMENTATION.md) before using this example
as a reproduction claim.

## Export metrics and record a video

Create a report with configuration, elapsed time, update count and full-clip metrics:

```bash
python scripts/export_a3_results.py --run runs/a3_demo_01 \
  --output results/a3_demo_01 --gpu-name "Your GPU model"
```

The exporter selects public fields; it excludes absolute run paths, logs and
checkpoints. Review task-specific labels before sharing records from another project.
Learner wall time includes periodic evaluation. Full process time includes setup
and the final evaluation and is only available once the supervisor exits.

After training releases the GPU, use the existing Isaac Sim runtime to record:

```bash
PYTHONPATH=src:examples/a3/src python scripts/record_a3_video.py \
  --task supermimic_a3:SuperMimic-Lab-Tracking-A3-v0 \
  --asset-root /data/a3_official --motion-dir /data/lafan1_a3 \
  --physics newton_mjwarp --visualizer newton --num-envs 1 \
  --checkpoint runs/a3_demo_01/last.pt \
  --video-seconds 30 --output artifacts/a3_video_01
```

This helper requires the existing Isaac Sim `omni.videoencoding` extension,
Newton GL visualizer and Pillow. It uses Kit for the bundled MP4 encoder; physics
remains Newton. Three predetermined clips cover walking, dancing and jumping.
The recording ends on failure and adds an outcome card instead of showing an
automatic reset. Metadata records the checkpoint hash and actual outcome.
Bounded A3 recording has been checked with the existing Newton renderer and
Isaac Sim encoder. The helper finalizes MP4 duration fields from the encoded
frame count; this avoids an unknown-duration sentinel being displayed as an
excessively long video. Review the actual player timeline and motion before
publishing. Host MP4 files as release assets or other external media and link
them from the report.
