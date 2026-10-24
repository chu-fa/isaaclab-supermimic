"""Record bounded A3 policy previews using the existing Isaac Sim MP4 encoder."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch

from mp4_timing import finalize_mp4_timing

from supermimic_lab.cli import parser_for
from supermimic_lab.cli import runner_configuration
from supermimic_lab.cli import simulation_environment
from supermimic_lab.evaluate import load_policy


def labelled_frame(frame, label):
    picture = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3])
    picture = picture.resize((960, 540))
    drawing = ImageDraw.Draw(picture)
    drawing.rectangle((0, 0, 960, 38), fill=(20, 24, 30))
    drawing.text((12, 12), label, fill=(245, 245, 245))
    return np.ascontiguousarray(picture.convert("RGBA"), dtype=np.uint8)


@torch.no_grad()
def record_clip(environment, policy, encoder, clip_name, seconds, directory, iteration):
    motions = environment.model_task.motions
    clip_index = motions.names.index(clip_name)
    environment.reset_to_clips(torch.tensor([clip_index], device=environment.device))
    policy.reset()
    visualizers = environment.sim.visualizers
    visualizer = next(
        visualizer for visualizer in visualizers
        if visualizer.cfg.visualizer_type in ("newton", "newton_gl")
    )
    frame_stride = 2
    frame_rate = 1.0 / (environment.step_dt * frame_stride)
    maximum_steps = min(
        round(seconds / environment.step_dt), int(motions.lengths[clip_index]) - 1
    )
    video_path = directory / f"{clip_name}.mp4"
    if video_path.exists():
        raise FileExistsError(video_path)
    if not encoder.start_encoding(str(video_path), frame_rate, 0, False):
        raise RuntimeError("Isaac Sim could not initialize MP4 encoding")
    started = time.monotonic()
    outcome = "preview_limit"
    rendered_frames = 0
    executed_steps = 0
    last_frame = None
    try:
        for control_step in range(maximum_steps):
            state = environment.read_state()
            if control_step % frame_stride == 0:
                root = state[0, :3].cpu().tolist()
                # Camera follows the root; these coordinates never enter the actor.
                environment.sim.set_camera_view(
                    (root[0] + 3.2, root[1] - 3.2, root[2] + 1.1),
                    (root[0], root[1], root[2]),
                )
                environment.sim.render()
                frame = visualizer.render_rgb_array()
                if frame is None:
                    raise RuntimeError("Newton returned no rendered frame")
                label = (
                    f"AGIBOT A3 | Newton | update {iteration} | {clip_name} | "
                    f"{control_step * environment.step_dt:.2f} s"
                )
                last_frame = labelled_frame(frame, label)
                if not encoder.encode_next_frame_from_buffer(last_frame):
                    raise RuntimeError("MP4 encoder rejected a frame")
                rendered_frames += 1
                if rendered_frames == 1:
                    Image.fromarray(last_frame).convert("RGB").save(
                        directory / f"{clip_name}_first_frame.png"
                    )
            observation = environment.model_task.observation(
                state, environment.previous_action, environment.context
            )
            goal = environment.model_task.goal(state, environment.reference_index)
            environment.step(policy(observation, goal))
            executed_steps += 1
            transition = environment.last_transition
            if bool(transition.episode_end[0]):
                outcome = "failure" if bool(transition.terminated[0]) else "clip_complete"
                break
        if last_frame is not None:
            # Do not record Isaac Lab's automatic reset as if it were continued tracking.
            label = f"{outcome} at {executed_steps * environment.step_dt:.2f} s | bounded preview"
            end_frame = labelled_frame(last_frame, label)
            for _ in range(round(frame_rate)):
                if not encoder.encode_next_frame_from_buffer(end_frame):
                    raise RuntimeError("MP4 encoder rejected the outcome card")
                rendered_frames += 1
    finally:
        encoder.finalize_encoding()
    if not video_path.is_file() or video_path.stat().st_size < 1024:
        raise RuntimeError("No usable MP4 file was produced")
    timing = finalize_mp4_timing(video_path, rendered_frames, frame_rate)
    return {
        "clip": clip_name,
        "file": video_path.name,
        "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "bytes": video_path.stat().st_size,
        "frame_rate": frame_rate,
        "container_timing": timing,
        "frames": rendered_frames,
        "policy_seconds": executed_steps * environment.step_dt,
        "outcome": outcome,
        "recording_wall_seconds": time.monotonic() - started,
        "outcome_card_seconds": 1.0,
        "terminal_pose_note": "Auto-reset is excluded; outcome card repeats the last pre-step rendered frame.",
    }


def main():
    parser = parser_for("evaluate")
    parser.description = __doc__
    parser.add_argument("--video-seconds", type=float, default=30.0)
    parser.add_argument(
        "--clip-names", nargs="+",
        default=["walk1_subject1", "dance1_subject1", "jumps1_subject1"],
    )
    parser.set_defaults(num_envs=1, visualizer="newton")
    arguments = parser.parse_args()
    if arguments.output is None or arguments.output.exists():
        parser.error("Specify a fresh --output directory")
    if not 0 < arguments.video_seconds <= 120:
        parser.error("video-seconds must be in (0, 120]")
    if arguments.num_envs != 1 or arguments.physics != "newton_mjwarp":
        parser.error("This recording helper currently supports one Newton environment")
    # Kit supplies the already installed NVENC extension; no dependency is installed.
    arguments.require_kit = True
    configuration = runner_configuration(arguments)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    iteration = checkpoint["iteration"]
    del checkpoint
    arguments.output.mkdir(parents=True)
    with simulation_environment(arguments, configuration, evaluation=True) as environment:
        from isaaclab.sim.utils import enable_extension

        enable_extension("omni.videoencoding")
        import video_encoding

        encoder = video_encoding.get_video_encoding_interface()
        if encoder is None:
            raise RuntimeError("The installed Isaac Sim video encoding extension is unavailable")
        policy = load_policy(arguments.checkpoint, environment, configuration)
        records = []
        for clip_name in arguments.clip_names:
            if clip_name not in environment.model_task.motions.names:
                raise ValueError(f"Unknown reference clip: {clip_name}")
            records.append(record_clip(
                environment, policy, encoder, clip_name, arguments.video_seconds,
                arguments.output, iteration,
            ))
            (arguments.output / "videos.json").write_text(json.dumps({
                "robot": "AGIBOT A3", "physics": "newton_mjwarp",
                "iteration": iteration,
                "checkpoint": arguments.checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(arguments.checkpoint.read_bytes()).hexdigest(),
                "protocol": "Predetermined clips, first frame, deterministic policy, nominal physics; failures retained.",
                "not_a_full_clip_benchmark": True,
                "videos": records,
            }, indent=2) + "\n")
    print(json.dumps({"recorded_clips": len(records), "iteration": iteration}))


if __name__ == "__main__":
    main()
