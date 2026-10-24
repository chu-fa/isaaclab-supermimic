"""Isaac Lab 3 task configuration with external robot assets."""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass


@configclass
class TrackingEnvironmentConfig(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 1
    observation_space = 1
    state_space = 0
    sim = SimulationCfg(dt=0.005, render_interval=4)
    scene = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True
    )
    robot_cfg: ArticulationCfg | None = None
    robot_spec: dict | None = None
    motion_directory: str = ""
    source_motion_rate: float = 30.0
    future_frames: int = 10
    # "robot": future reference poses relative to the robot's position/heading.
    # "reference": the earlier self-relative encoding, for older checkpoints.
    goal_frame: str = "robot"
    # Nominal physics is an explicit development setting, not paper-equivalent DR.
    randomization: str = "nominal"
    evaluation: bool = False
    discount: float = 0.99


def configure_robot(configuration, specification, backend):
    """Populate the simulator from a robot contract without changing its gains."""
    profile = specification["profile"]
    joint_names = profile["joint_names"]
    parameters = profile["joints"]
    if backend == "newton_mjwarp":
        from isaaclab_newton.sim.schemas import NewtonArticulationCfg

        articulation_properties = [NewtonArticulationCfg(self_collision_enabled=True)]
    elif backend == "isaacsim_physx":
        from isaaclab_physx.sim.schemas import PhysxArticulationCfg

        articulation_properties = [PhysxArticulationCfg(enabled_self_collisions=True)]
    else:
        raise ValueError(f"Backend has no adapter in this release: {backend}")
    actuators = {}
    for joint_name in joint_names:
        motor = parameters[joint_name]
        if motor["stiffness"] <= 0:
            raise ValueError(f"Positive motor stiffness is required: {joint_name}")
        actuators[joint_name] = ImplicitActuatorCfg(
            joint_names_expr=[joint_name],
            stiffness=motor["stiffness"],
            damping=motor["damping"],
            armature=motor["armature"],
            joint_effort_limit=motor["effort_limit_sim"],
            joint_velocity_limit=motor["velocity_limit_sim"],
            friction=motor["friction"],
            dynamic_friction=motor["dynamic_friction"],
        )
    configuration.robot_spec = specification
    configuration.robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=specification["usd_path"],
            articulation_props=articulation_properties,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, profile["initial_height_m"]),
            joint_pos={name: parameters[name]["default_position"] for name in joint_names},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    )
    configuration.action_space = len(joint_names)
    configuration.observation_space = 3 * len(joint_names) + 6
