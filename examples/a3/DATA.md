# AGIBOT A3 data specification

## Availability

The current example uses the official base **AGIBOT A3 T2.5** model from
[AgibotTech/A3-A3U-robot-model](https://github.com/AgibotTech/A3-A3U-robot-model),
pinned to commit `589f508ff357447c610a3f3004419035ddc8f153`. The
[official asset manifest](official_asset_manifest.json) records every downloaded
file's SHA256. The source model is licensed under Mulan PSL v2; the downloader
retains its license and original URDF/meshes without editing them.

From this checkout, using the existing Isaac Lab environment's Python:

```bash
python scripts/fetch_a3_assets.py --output /data/a3_official
python scripts/convert_a3_assets.py --asset-root /data/a3_official --visualizer none
```

The output directory must be new. An offline copy of the pinned upstream tree can
be supplied with `--source /path/to/upstream-tree`. The downloader verifies all
files before they are used. Conversion uses Isaac Lab's existing URDF importer;
these commands install no packages. The simulator importer may use Kit even when
training uses Newton. The supplied [runtime motor profile](runtime_profile.json)
is an example configuration, not an official controller release.

Models, generated USD and motion CSVs are not bundled in wheels. Compatible
A3-retargeted motions remain a separate required input under their applicable
licenses. No public download for the exact prepared motion bundle or a complete
A3 retargeting pipeline is provided. Supply permitted, compatible reference
motions and retain their source and checksums with each experiment.

## Robot files

The example uses A3 T2.5 with 31 actuated joints. Under `--asset-root`, supply:

```text
a3_description/
  runtime_profile.json
  urdf/model.urdf
  meshes/...
a3_usd_newton_v1/model.usda
a3_usd_physx_v2/model/model.usda
```

Only the selected backend's USD is needed. Keep all referenced payload files for
the PhysX asset. The URDF, USD and profile must agree on geometry, articulation and
joint names. The profile supplies:

- Root and torso body names, initial height and ordered joint names.
- Per-joint default positions, stiffness, damping and armature.
- Effort/velocity limits and friction parameters.

The example preserves these parameters. The official conversion merges all seven
fixed joints while preserving their mass and geometry on the parent bodies; 31
active joints remain. Physics variants use convex decomposition for PhysX and
convex hulls for Newton. `conversion.json` records the mass check, merged links and
output hashes. Keep the complete converted directory when using PhysX.

The default task rejects a URDF with a different SHA256. Historical runs can be
reproduced with their frozen source and original assets; using an alternative
`expected_urdf_sha256` through `--env-config` is an explicit task change and does
not make an arbitrary USD compatible. See the [training explanation](TRAINING.zh-CN.md)
for PD values, reset behavior and model differences.

## Motion format

Each file in `--motion-dir` is a headerless CSV:

| Columns, zero-based | Values | Units/convention |
| --- | --- | --- |
| 0–2 | Root position | Metres, world frame |
| 3–6 | Root orientation | Quaternion XYZW |
| 7–37 | 31 joint positions | Radians, profile joint order |

Source rate is 30 Hz. The task resamples to 50 Hz using linear position/joint
interpolation and shortest-arc quaternion interpolation, then estimates velocities
by finite differences. Motion lookup respects clip boundaries. G1 references
cannot be supplied directly without A3 retargeting.

## Preparing reference motions

Human motion datasets such as LAFAN1 require retargeting to the A3 morphology and
joint convention. A preparation pipeline may use body targets, constrained inverse
kinematics, contact correction and smoothing. Validate joint limits and physical
feasibility before using a prepared library for training.

The loader uses all compatible CSV files in the supplied directory. Record data
provenance, licensing, preprocessing and file hashes with each run. The evaluator
does not automatically create a held-out split; prepare separate evaluation data
when studying generalization. Replacing the reference library changes checkpoint
compatibility.
