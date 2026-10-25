"""Record a split-screen A3 preview: reference motion beside the trained policy.

World 0 runs the deterministic policy under nominal Newton physics. World 1 is a
kinematic puppet rewritten to the current reference frame before every rendered
step, so the left panel shows exactly the motion the policy must track. Each
segment resets both worlds to the reference state at its start time.

Outputs an MP4 at the control-rate/2 frame rate and a smaller animated WebP for
documentation pages.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch

from mp4_timing import finalize_mp4_timing

from supermimic_lab.cli import parser_for
from supermimic_lab.cli import runner_configuration
from supermimic_lab.cli import simulation_environment
from supermimic_lab.evaluate import load_policy

PANEL = (960, 540)
FRAME_STRIDE = 2
# The puppet stands 40 m along both follow cameras' lateral axis, outside either view.
PUPPET_OFFSET = (28.0, 28.0, 0.0)
CAMERA_OFFSET = (2.4, -2.4, 0.6)


def load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


TITLE_FONT = load_font(30)
TEXT_FONT = load_font(24)


def parse_segment(text):
    """clip@start+duration, in seconds."""
    clip, timing = text.split("@", 1)
    start, duration = timing.split("+", 1)
    return clip, float(start), float(duration)


def panel(frame, title, subtitle, color):
    picture = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3]).resize(PANEL)
    drawing = ImageDraw.Draw(picture)
    drawing.rectangle((0, 0, PANEL[0], 46), fill=(20, 24, 30))
    drawing.text((16, 8), title, fill=color, font=TITLE_FONT)
    drawing.rectangle((0, PANEL[1] - 38, PANEL[0], PANEL[1]), fill=(20, 24, 30))
    drawing.text((16, PANEL[1] - 33), subtitle, fill=(235, 235, 235), font=TEXT_FONT)
    return picture


def side_by_side(left, right):
    canvas = Image.new("RGB", (2 * PANEL[0], PANEL[1]))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (PANEL[0], 0))
    return canvas


def place_puppet(environment, reference_index, offset):
    """Write world 1 to the reference frame; it is never policy controlled."""
    reference = environment.model_task.motions.states[reference_index][None]
    joints = len(environment.joint_indices)
    world = torch.tensor([1], dtype=torch.long, device=environment.device)
    pose = reference[:, :7].clone()
    pose[:, :3] += environment.scene.env_origins[:1] + offset
    environment.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=world)
    environment.robot.write_root_link_velocity_to_sim_index(
        root_velocity=reference[:, 7:13].contiguous(), env_ids=world
    )
    environment.robot.write_joint_position_to_sim_index(
        position=reference[:, 13 : 13 + joints].contiguous(),
        joint_ids=environment.joint_indices, env_ids=world,
    )
    environment.robot.write_joint_velocity_to_sim_index(
        velocity=reference[:, 13 + joints :].contiguous(),
        joint_ids=environment.joint_indices, env_ids=world,
    )
    environment.scene.write_data_to_sim()
    environment.sim.forward()
    return pose[0, :3]


def capture(environment, visualizer, target):
    x, y, z = target.tolist()
    environment.sim.set_camera_view(
        (x + CAMERA_OFFSET[0], y + CAMERA_OFFSET[1], z + CAMERA_OFFSET[2]), (x, y, z)
    )
    environment.sim.render()
    frame = visualizer.render_rgb_array()
    if frame is None:
        raise RuntimeError("Newton returned no rendered frame")
    return frame


@torch.no_grad()
def record_segment(environment, visualizer, policy, clip_name, start, duration, iteration):
    motions = environment.model_task.motions
    clip = motions.names.index(clip_name)
    first = int(motions.starts[clip]) + round(start / environment.step_dt)
    steps = min(round(duration / environment.step_dt), int(motions.ends[clip]) - first)
    if steps <= 0:
        raise ValueError(f"Segment starts after the end of {clip_name}")
    environment.reset_to_frames(torch.tensor([first, first], device=environment.device))
    policy.reset()
    offset = torch.tensor(PUPPET_OFFSET, device=environment.device)
    joints = len(environment.joint_indices)
    frames, errors, outcome = [], [], "segment_complete"
    for step in range(steps):
        index = environment.reference_index[:1].clone()
        puppet_root = place_puppet(environment, index[0], offset)
        robot_state = environment.read_state()[:1]
        metrics = environment.model_task.tracking_metrics(robot_state, index)
        errors.append((float(metrics["joint_rmse_rad"][0]), float(metrics["root_horizontal_error_m"][0])))
        if step % FRAME_STRIDE == 0:
            clip_seconds = start + step * environment.step_dt
            subtitle = f"{clip_name}   t = {clip_seconds:5.1f} s"
            frames.append(side_by_side(
                panel(capture(environment, visualizer, puppet_root),
                      "Reference motion", subtitle, (255, 196, 110)),
                panel(capture(environment, visualizer, robot_state[0, :3] + environment.scene.env_origins[0]),
                      f"AGIBOT A3 - SuperMimic policy (update {iteration})", subtitle, (120, 200, 255)),
            ))
        observation = environment.model_task.observation(
            robot_state, environment.previous_action[:1], environment.context[:1]
        )
        goal = environment.model_task.goal(robot_state, index)
        actions = torch.zeros(2, joints, device=environment.device)
        actions[:1] = policy(observation, goal)
        # The puppet is rewritten before every step; holding the reference keeps
        # its 20 ms of free physics small.
        puppet_joints = motions.states[index, 13 : 13 + joints]
        actions[1:] = (puppet_joints - environment.default_positions) / environment.action_scales
        environment.step(actions)
        transition = environment.last_transition
        if bool(transition.episode_end[0]):
            outcome = "fell" if bool(transition.terminated[0]) else "clip_end"
            break
    return frames, {
        "clip": clip_name, "start_seconds": start, "recorded_seconds": len(errors) * environment.step_dt,
        "outcome": outcome,
        "mean_joint_rmse_rad": float(np.mean([error[0] for error in errors])),
        "max_root_horizontal_offset_m": float(np.max([error[1] for error in errors])),
    }


def main():
    parser = parser_for("evaluate")
    parser.description = __doc__
    parser.add_argument(
        "--segments", nargs="+", required=True, type=parse_segment,
        help="clip@start+duration in seconds, e.g. walk1_subject5@178+5",
    )
    parser.add_argument("--webp-scale", type=float, default=0.5)
    parser.add_argument("--webp-stride", type=int, default=2, help="Keep every n-th MP4 frame")
    parser.add_argument("--webp-quality", type=int, default=60)
    parser.set_defaults(num_envs=2, visualizer="newton")
    arguments = parser.parse_args()
    if arguments.output is None or arguments.output.exists():
        parser.error("Specify a fresh --output directory")
    if arguments.num_envs != 2 or arguments.physics != "newton_mjwarp":
        parser.error("Use --num-envs 2 with Newton: one policy world and one reference puppet")
    # Kit supplies the already installed NVENC extension; no dependency is installed.
    arguments.require_kit = True
    configuration = runner_configuration(arguments)
    iteration = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)["iteration"]
    arguments.output.mkdir(parents=True)
    started = time.monotonic()
    with simulation_environment(arguments, configuration, evaluation=True) as environment:
        from isaaclab.sim.utils import enable_extension

        enable_extension("omni.videoencoding")
        import video_encoding

        encoder = video_encoding.get_video_encoding_interface()
        if encoder is None:
            raise RuntimeError("The installed Isaac Sim video encoding extension is unavailable")
        policy = load_policy(arguments.checkpoint, environment, configuration)
        visualizer = next(
            visualizer for visualizer in environment.sim.visualizers
            if visualizer.cfg.visualizer_type in ("newton", "newton_gl")
        )
        frame_rate = 1.0 / (environment.step_dt * FRAME_STRIDE)
        frames, records = [], []
        for clip_name, start, duration in arguments.segments:
            if clip_name not in environment.model_task.motions.names:
                raise ValueError(f"Unknown reference clip: {clip_name}")
            segment_frames, record = record_segment(
                environment, visualizer, policy, clip_name, start, duration, iteration
            )
            frames.extend(segment_frames)
            records.append(record)
            print(json.dumps(record), flush=True)
        video = arguments.output / "reference_vs_policy.mp4"
        if not encoder.start_encoding(str(video), frame_rate, 0, False):
            raise RuntimeError("Isaac Sim could not initialize MP4 encoding")
        try:
            for frame in frames:
                if not encoder.encode_next_frame_from_buffer(
                    np.ascontiguousarray(frame.convert("RGBA"), dtype=np.uint8)
                ):
                    raise RuntimeError("MP4 encoder rejected a frame")
        finally:
            encoder.finalize_encoding()
        # Closing the simulation app ends the process, so finish all outputs here.
        timing = finalize_mp4_timing(video, len(frames), frame_rate)
        size = tuple(round(dimension * arguments.webp_scale) for dimension in frames[0].size)
        preview = [frame.resize(size, Image.LANCZOS) for frame in frames[:: arguments.webp_stride]]
        webp = arguments.output / "reference_vs_policy.webp"
        preview[0].save(
            webp, save_all=True, append_images=preview[1:], loop=0,
            duration=round(1000 * arguments.webp_stride / frame_rate),
            quality=arguments.webp_quality, method=6,
        )
        report = {
            "checkpoint": str(arguments.checkpoint), "iteration": iteration,
            "layout": "left: kinematic reference puppet; right: policy-driven robot",
            "protocol": "segments reset to the reference state at their start; deterministic policy; nominal Newton physics",
            "mp4": {"file": video.name, "frame_rate": frame_rate, "frames": len(frames),
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "container_timing": timing},
            "webp": {"file": webp.name, "size": list(size), "frames": len(preview),
                     "bytes": webp.stat().st_size},
            "segments": records, "wall_seconds": time.monotonic() - started,
        }
        (arguments.output / "video.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mp4": str(video), "webp": str(webp), "webp_bytes": webp.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
