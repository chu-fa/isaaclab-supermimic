# Sources and attribution

## Algorithm

The method is based on Liao et al.,
[*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model*](https://arxiv.org/abs/2610.28480), arXiv:2610.28480v1, 2026-10-20.
Equations 1–3, Algorithm A.1 and Appendices A/C/D inform the implementation.
Coverage and implementation choices are documented in
[IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

Reference PDF SHA256:
`fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`.
The PDF is not distributed with this repository.

This is an independently written, paper-based implementation. Development involved
prior exposure to [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed);
no algorithm source or pretrained checkpoints from that repository are included.
No audited clean-room provenance is claimed.

## Dependencies

Isaac Lab's public SDK APIs and source inform the launcher, articulation, cloning,
material and task-registration adapters. Isaac Lab, Isaac Sim, Newton,
MuJoCo-Warp, PyTorch and other runtime dependencies remain separate projects under
their respective licenses.

## A3 assets and motion data

The current robot source is AgibotTech's
[A3-A3U-robot-model](https://github.com/AgibotTech/A3-A3U-robot-model), variant
`a3_t2d5`, commit `589f508ff357447c610a3f3004419035ddc8f153`, under Mulan PSL v2.
The model downloader retains the original files and license; the Isaac Lab importer
produces separate derived USD files. Model assets are not bundled or relicensed
by the project's MIT license. Runtime motor parameters are supplied as example
configuration, with their configuration-source digest retained; they are not a
controller distributed by that official model repository.

A3-retargeted reference motions remain external. Preparation requirements,
format and availability are described in [DATA.md](examples/a3/DATA.md).

Collision exclusion pairs in the A3 task are physical assembly metadata derived
from asset diagnostics. They are robot-specific configuration, not algorithm
constants or a claim of calibrated contact physics.

Future published test records should identify their source, configuration and
data separately. No endorsement by the paper authors, robot manufacturer or
Isaac Lab maintainers is implied.
