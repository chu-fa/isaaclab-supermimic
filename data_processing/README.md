# Direct LAFAN1 to AGIBOT A3

These optional data preparation tools convert the original human BVH sequences
directly to the pinned official AGIBOT A3 model. They do not read G1 trajectories
or pretrained policies. The reusable SuperMimic learner is unchanged.

Use the existing Isaac Lab Python environment with NumPy, SciPy, PyTorch and
Trimesh available. The tools do not install dependencies or start a simulator.
Run from the repository root, replacing `python` with the environment's launcher
where needed:

```bash
python data_processing/fetch_lafan1.py --output /data/lafan1_original
PYTHONPATH=src python data_processing/retarget_lafan1.py \
  --source /data/lafan1_original \
  --asset-root /data/a3_official \
  --clips configs/a3_lafan1_clips.json \
  --output /data/lafan1_a3_direct \
  --iterations 1000 --device cuda:0
```

Output directories must be new. The downloader pins Ubisoft commit
`94084601bacdf9cc3764b5c73daaeccae6035fac`, verifies the archive SHA256 and preserves
the upstream license. It extracts all 77 original BVH sequences. The supplied
clip list selects the 40 action names used by the A3 example.

## Conversion

- Read each joint's declared intrinsic Euler rotation order. Root position
  channels replace the LAFAN1 root OFFSET. Convert centimetres to metres and
  `(x, y, z)` to `(x, -z, y)` for a right-handed Z-up world.
- Build morphology-adjusted body targets from human bone directions and A3
  segment lengths. Fit the rigid torso to the hip-to-neck chord and shoulder
  lateral axis, rather than extending a terminal vertebra orientation over the
  complete torso. Infer pelvis, head, sole and wrist orientations from their
  anatomical axes. The rigid A3 foot uses the human toe frame for its
  orientation; it cannot reproduce a human's independent toe articulation.
- Initialize joint angles from human limb planes. Solve the full sequence jointly
  under official joint limits, bounded pelvis translation/orientation corrections,
  temporal penalties and foot support constraints. Pelvis correction limits are
  0.18 m per translation axis and 0.6 rad per rotation-vector component.
- Lock a support point only when both human ankle and toe have 3D speed below
  0.15 m/s and the toe is within 4.5 cm of the estimated floor. This distinguishes
  a stationary sole from a heel/toe pivot. The floor estimate is the second
  percentile of human toe height for the clip.
- Include whole-body ground constraints using mesh support vertices. Check exact
  mesh convex-hull extrema afterwards and record any remaining vertical lift.
  Convex-hull extrema preserve each original mesh's lowest point under rotation.
- Keep temporal smoothing inside the joint solve. Independently filtering joint
  positions afterwards would break the coupled foot/base contact solution.
- Retime only intervals that would exceed approximately 85% of a runtime joint
  speed limit. A conservative smooth envelope spreads transitions locally. Root
  motion and joint motion use the same monotone time map; quaternions use
  shortest-arc interpolation. Reject a total clip-duration increase above 3x,
  and record the largest local interval stretch separately.

The retargeted headerless CSV has root XYZ, quaternion XYZW and 31 joint angles,
at 30 Hz in the runtime profile's joint order. Before training, finalize it on
the actual 50 Hz control grid. This uses the training loader's interpolation,
then checks the complete mesh extrema again and applies a smooth, bounded floor
clearance correction. The correction must stay below 2 cm; its magnitude and the
minimum height before/after are recorded. Training then reads the 50 Hz files
with `source_motion_rate=50`, avoiding a subsequent change of time grid. `manifest.json` records source/output hashes, joint order, time scaling,
fit error, support-point slip, floor clearance and code hashes.

## Quality gates and scope

The preparation command exits unsuccessfully if any clip has pooled body-fit
95th-percentile error above 0.12 m, joint velocity above 86% of its runtime limit,
stationary-support sole-centre speed above 0.20 m/s at the 95th percentile, or
stationary-support foot clearance above 0.05 m at the 95th percentile. Fit errors
compare pelvis-relative body positions with morphology-adjusted human targets;
they are not direct errors against unscaled human joint positions. Per-body
errors are recorded so a pooled statistic does not conceal difficult body parts.

These are engineering screening criteria. They do not certify torque feasibility,
balance, self-collision clearance or successful dynamic tracking. Raw and prepared
motions remain external and retain the original data license; the repository's
MIT license applies to these tools, not to the motion data.

Finalize the full passing preparation directory before training:

```bash
PYTHONPATH=src python data_processing/finalize_reference.py \
  --source /data/lafan1_a3_direct --output /data/lafan1_a3_control50 \
  --asset-root /data/a3_official --clips configs/a3_lafan1_clips.json
```

The final manifest keeps the 30 Hz human-fit/contact statistics under
`preparation_30hz` and records control-grid geometry/velocity checks separately
under `runtime_50hz`. It does not relabel source-grid measurements as new
50 Hz measurements. Both stages must pass for every clip.

For a fresh A3 training run after the complete manifest passes:

```bash
supermimic-train \
  --task supermimic_a3:SuperMimic-Lab-Tracking-A3-v0 \
  --asset-root /data/a3_official --motion-dir /data/lafan1_a3_control50 \
  --physics newton_mjwarp --visualizer none --device cuda:0 \
  --num-envs 4096 --config configs/a3_direct_4096.json \
  --env-config configs/a3_direct_50hz_env.json \
  --randomization nominal --seed 0 --hours 3 --eval-every 1000 \
  --output runs/a3_direct_4096
```

This command supplies no checkpoint. The actor, world model, critic, normalizers
and optimizers start fresh. Environment count and reference data differ from the
earlier 2048-environment setup, so this is not a controlled data-only ablation.
Clip durations can also change; compare normalized coverage and per-clip results,
not just mean tracking seconds. The 40 clips are training references, not a held-out
generalization set.

Sources: [original LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
and [official A3 model](https://github.com/AgibotTech/A3-A3U-robot-model).


For generation in separate batches, use `--only` with explicit clip names.
Combine batches only when every requested clip passes and all source/settings
match:

```bash
python data_processing/assemble_motions.py \
  --batch /data/a3_batch_1 --batch /data/a3_batch_2 \
  --clips configs/a3_lafan1_clips.json --output /data/lafan1_a3_complete
```

Assembly rejects missing, duplicate, mismatched or failing clips. It does not
filter difficult motions out of the benchmark. Candidate batches remain intact.
