# Algorithm and task contract

The normative source is
[SuperMimic v1](https://arxiv.org/pdf/2610.28480). The following distinguishes
implemented behavior from details the paper leaves unspecified and missing
extensions. Passing an interface check is not evidence of tracking convergence.

| Paper component | Implementation | Boundary |
| --- | --- | --- |
| Executed-action replay | `replay.py`, `runner.py` | Terminal next state is captured before reset; windows never train across resets |
| Eq. 1 autoregressive dynamics | `algorithm.py:fit_dynamics` | Eight-step default; sum of state errors; analytic reward/termination have no learned-head loss |
| Eq. 2 imagined policy objective | `objectives.py`, `algorithm.py:improve_policy` | 32-step default; discounted survival; terminal-transition reward retained; terminal critic bootstrap |
| Frozen dynamics/critic in actor update | `algorithm.py:frozen_weights` | Weights frozen while gradients through inputs remain enabled |
| Eq. 3 critic targets | `objectives.py:lambda_returns` | Fixed detached lambda targets, recomputed once per update; five epochs/four minibatches |
| Table C.1 network/optimizer defaults | `config.py`, `networks.py` | Model 2048×3; encoder/decoder 2048/1024/512/256; latent 32; critic 256/128 |
| Policy optimization | `algorithm.py` | Adam 2e-4, betas .9/.95, gradient norm 1 then element clip .1 |
| Replay/update schedule | `runner.py` | 1000 steps/world; UTD .4; ten model batches; cosine learning rate to .03 of initial |
| Proprioception and history | `tracking.py`, `runner.py` | Ten frames, joint offsets/rates, gyro, projected gravity, previous action; paper uniform noise |
| Reference goal | `tracking.py:goal` | Ten future frames; exact feature encoding is an engineering choice |
| Appendix A reward and failures | `tracking.py` | Tracking, action-change and soft-limit terms; ×.02; height >.25 m or gravity-z error >.8 |
| One-second return sampler | `sampling.py` | .5 uniform + .4 difficult + .1 historical; temperature 25; EMA rates are explicit choices |
| Simulator context | Isaac Lab adapter | Nominal default; opt-in ranges from paper; Newton COM caveat below |
| Full-clip deterministic evaluation | `evaluate.py` | First frame, no noise/DR, complete-clip denominator; bounded checks marked incomplete |
| Multiple encoders/interfaces | Not implemented | No velocity-interface training, alignment, decoder stop-gradient ablation or joint batch allocation |
| Paper benchmarks/hardware transfer | Not performed | No claim of paper-level reproduction or sim-to-real success |

## State, action and observability

For J joints, the state is `[root xyz, root quaternion xyzw, world-frame root-link
linear velocity, world-frame angular velocity, q, qdot]`, width `13 + 2J`.
All lengths, times, masses and angles use SI units. Quaternions describe link
orientation, not a center-of-mass principal-inertia frame. Differentiable URDF
kinematics returns link-origin positions, orientations and spatial velocities.

The model predicts increments: root translation and rotation tangent in the
current body frame, linear/angular velocity increments in that frame, and joint
position/rate increments. Model inputs omit absolute root XY and yaw. Model
outputs are integrated directly; no unknown simulator state is used in imagination.

The actor receives proprioception history and reference-only future goals. Its
goal features contain relative future XY, absolute reference Z, orientation
columns, velocities and joint states, all expressed relative to the **current
reference heading**. They do not add live true root pose or velocity. This choice
preserves the input boundary but leaves global heading/position errors
unobservable directly; tracking quality must be measured. It is not a proven
replacement for the paper authors' unspecified goal encoding.

True simulator state and per-episode context are privileged inputs to the world
model and critic and are used in reward/evaluation. Context is friction, J encoder
zero offsets, and torso COM offsets. It is not appended to the actor observation.
Encoder offset modifies sensed joint position; the commanded measured setpoint
is converted to physical angle by subtracting that offset.

For the A3 example, position target = default angle + action scale × action −
encoder offset. Action scale is `.25 × motor effort limit / motor stiffness`.
The external motor profile supplies gains, damping, limits and armature without
retuning. This scale and the tracked-link list are A3 engineering choices, not
the paper's G1 constants. Actuator torque/velocity enforcement differs by backend.

## Explicit implementation choices

The paper does not fully prescribe activations, normalizers, residual state
coordinates, exact goal features, termination smoothing, EMA rates, initial
collection or simulator solver settings. This implementation uses SiLU, online
population mean/variance, normalized model outputs, tangent rotations, the goal
encoding above, height/tilt sigmoid temperatures .02 m/.04, return/history EMA
rates .2/.001, and 64 initial vector steps. These are testable choices, not claims
about private author code.

Equation 1 uses raw SI/tangent errors by default. `state_error_scales` can specify
per-coordinate scaling but changes that default and must be recorded. Large
joint-velocity errors can dominate unscaled loss. Normalizing network features
does not eliminate that relative weighting.

The real simulator uses hard failure thresholds. The model-based actor objective
uses a differentiable soft probability near those thresholds. No NaN replacement
is used; invalid actions or states stop the run and record raw evidence. The
fixed-size history repeats the reset observation rather than inventing a preceding
motion trajectory.

Nominal dynamics uses friction 1 with no offsets. Opt-in `paper` randomization uses
friction U[.3,1.2], encoder offsets U[-.01,.01] rad and torso COM offsets
x ±.025 m, y/z ±.05 m, sampled per reset and held during an episode. Isaac Lab's
Newton/MJWarp SDK warns that runtime COM writes can leave mass-matrix updates
incomplete. That mode requires dedicated physical validation; a nominal check
does not validate it. Material-pair combination rules can also differ by backend.

## Adding a task

`DifferentiableTask` in `contracts.py` specifies the pure-Torch interface: dimension
declarations, model inputs/targets/integration, state error, observation, goal,
reward, termination probability, condition-handle advancement, critic features and
valid rollout starts. Full extension instructions are in [CUSTOM_TASKS.md](CUSTOM_TASKS.md).

The collection environment provides `model_task`, `num_envs`, `signature`,
`read_state()`, `previous_action`, `context`, `reference_index`, `episode_steps`
and `step(action) -> Transition`. `Transition` carries the true next state before
auto-reset, the executed action, reward, hard termination and episode-end flags.

Isaac Lab's Gym API remains unchanged. `TransitionAdapter` exposes this contract
to the learner. A new robot can reuse `TrackingEnvironmentConfig` and
`configure_robot`, providing its external profile, assets, link list and collision
exclusions. Other task families can implement the pure-Torch contract directly.
The algorithm modules do not import Isaac Lab or assume A3/G1 joint counts.
The optional `supermimic-lab-a3` distribution owns A3 registration and configuration;
`supermimic-lab` owns the generic Isaac Lab adapter and imports no A3 package.

## Checkpoints and evaluation

Checkpoints use the `supermimic_lab_v1` format. Other implementations' checkpoint
formats are not supported. Task signatures include robot/model/profile hashes,
joint/link order, motion fingerprint, backend, period, action scaling and collision
exclusions. Incompatible weights are not silently mapped into the networks.

Resume restores networks, population statistics, all optimizers, update/step
counters and cumulative training time. Replay, RNG, reset sampler, history and
physics are reinitialized and explicitly recorded in `run.json`. Output directories
cannot overwrite previous runs. SIGINT/SIGTERM request saving at the next complete
learner-update boundary.

Evaluation counts a success only when a clip completes without a hard failure.
A step budget is an incomplete evaluation, never a success. Normalized return
divides actual accumulated reward by `.1 × all expected transitions`, including
the untracked tail after failure. Positive return alone is not success. Active
worlds continue until their clip finishes; completed worlds are excluded from
further statistics.
