# A3 evaluation protocol

[Example](README.md) · [中文训练说明](TRAINING.zh-CN.md)

This document defines evaluation behavior. Training measurements and demonstration
videos are not included in the initial source release.

## Full-clip evaluation

Each clip starts at its first reference frame, with a deterministic actor and
nominal physics. Evaluation continues to clip end or hard failure. Hard failure is
root-height tracking error greater than **0.25 m**, or projected-gravity-z tracking
error greater than **0.8**.

| Metric | Definition |
| --- | --- |
| Completed clips | Reached the clip end without triggering hard failure |
| Failures | Triggered a hard failure before completion |
| Mean tracked duration | Mean executed control steps multiplied by the control period, 0.02 s for this example |
| Normalized return | Accumulated reward divided by `0.1 × expected control transitions` for the complete reference clip, then averaged across clips |
| Reference coverage | Total tracked transitions divided by total expected transitions across all clips |

The normalized-return denominator includes untracked portions after a failure.
Timeouts and step-limited previews are incomplete outcomes, not successful clips.
`complete_evaluation: true` means every requested clip has a completion/failure
outcome; it does not mean every clip completed successfully.

Horizontal root drift, limb-pose error and velocity error affect tracking reward
but do not independently trigger the hard failure rule. Completion counts and
durations must therefore be interpreted together with return, direct tracking
errors and visual assessment.

## Reporting an experiment

Record the robot and reference identity, source revision, resolved configuration,
seed, backend, hardware, initialization, learner updates and elapsed time. Report
all clips, including failures and incomplete evaluations. Distinguish evaluations
on training references from held-out motions; the evaluator does not create a
data split automatically.

Learner wall time includes periodic evaluation but excludes initialization and
initial evaluation. Total process time also includes final evaluation. A checkpoint
continuation restores networks, normalization, optimizers and counters; replay,
simulator state, history, reset sampling and RNG are reinitialized.

Use `scripts/export_a3_results.py` to prepare a report from local run artifacts.
Review the exported fields before publication. Keep raw trajectories, model
weights and videos outside Git source history.

New evaluations additionally record `tracking_errors` per clip (mean, maximum
and final) and globally (executed-frame-weighted mean and maximum). Angles are
in radians; body pose is aligned in root XY and heading, while global drift is
reported separately. Errors include the failure transition but exclude unexecuted
tails. Historical evaluation files are preserved and are not backfilled.
