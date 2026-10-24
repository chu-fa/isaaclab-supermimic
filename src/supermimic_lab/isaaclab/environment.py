"""Real-simulator collection with explicit pre-reset transition capture."""

from __future__ import annotations

import hashlib
import json

import torch
from pxr import Usd
from pxr import UsdPhysics

from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import CuboidCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg

from supermimic_lab.contracts import Transition
from supermimic_lab.geometry import rotate
from supermimic_lab.kinematics import KinematicTree
from supermimic_lab.motions import MotionLibrary
from supermimic_lab.sampling import ReturnSampler
from supermimic_lab.tracking import ReferenceTracking


class PhysicsStateError(FloatingPointError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


class TrackingEnvironment(DirectRLEnv):
    """Gym environment; the runner consumes it through TransitionAdapter."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.robot_spec is None or not cfg.motion_directory:
            raise ValueError("Configure external robot assets and motion_directory first")
        if cfg.randomization not in ("nominal", "paper"):
            raise ValueError("randomization must be nominal or paper")
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        profile = cfg.robot_spec["profile"]
        joint_names = profile["joint_names"]
        self.joint_indices = [self.robot.joint_names.index(name) for name in joint_names]
        self.torso_index = self.robot.body_names.index(profile["torso_body"])
        self.root_index = self.robot.body_names.index(profile["root_body"])
        self.nominal_com = self.robot.data.body_com_pose_b.torch[:, self.torso_index : self.torso_index + 1].clone()
        self._com_has_been_randomized = False
        self.default_positions = torch.tensor(
            [profile["joints"][name]["default_position"] for name in joint_names], device=self.device
        )
        self.action_scales = torch.tensor([
            cfg.robot_spec["action_scale_effort_fraction"]
            * profile["joints"][name]["effort_limit_sim"]
            / profile["joints"][name]["stiffness"]
            for name in joint_names
        ], device=self.device)
        kinematics = KinematicTree(
            cfg.robot_spec["urdf_path"], joint_names,
            cfg.robot_spec["tracked_links"], self.device,
        )
        motions = MotionLibrary.from_directory(
            cfg.motion_directory, len(joint_names), self.device,
            source_rate=cfg.source_motion_rate, control_period=self.step_dt,
        )
        self.model_task = ReferenceTracking(
            kinematics, motions, self.default_positions, cfg.future_frames,
            goal_frame=cfg.goal_frame, goal_offset_limit=cfg.goal_offset_limit,
        )
        self.sampler = ReturnSampler(motions, cfg.discount)
        self.previous_action = torch.zeros(self.num_envs, len(joint_names), device=self.device)
        self.context = torch.zeros(self.num_envs, self.model_task.dimensions.context, device=self.device)
        self.context[:, 0] = 1.0
        self.reference_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_bins = torch.zeros_like(self.reference_index)
        self.discounted_returns = torch.zeros(self.num_envs, device=self.device)
        self.last_transition = None
        self._evaluation_starts = None
        contract = {
            "assets": cfg.robot_spec["asset_sha256"],
            "joint_names": joint_names, "tracked_links": cfg.robot_spec["tracked_links"],
            "motions": motions.fingerprint(), "future_frames": cfg.future_frames,
            "period": self.step_dt, "backend": self.scene.physics_backend,
            "action_scales": self.action_scales.tolist(),
            "collision_exclusions": cfg.robot_spec["collision_exclusions"],
            "state_contract": "xyzw_link_origin_world_velocity_v1",
        }
        if cfg.goal_frame != "reference":
            # Omitted for the original encoding so existing signatures still match.
            contract["goal_frame"] = cfg.goal_frame
            if cfg.goal_offset_limit is not None:
                contract["goal_offset_limit"] = cfg.goal_offset_limit
        self.signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.asset_contract = contract

    @property
    def episode_steps(self):
        return self.episode_length_buf

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # A local procedural surface keeps the package independent of Nucleus
        # downloads. Its upper face is z=0; it is a static finite collider.
        ground = CuboidCfg(
            size=(1000.0, 1000.0, 0.1),
            collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)],
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )
        ground.func("/World/ground", ground, translation=(0.0, 0.0, -0.05))
        root = self.scene.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        rigid_links = {
            prim.GetName(): prim for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        for first, second in self.cfg.robot_spec["collision_exclusions"]:
            if first not in rigid_links or second not in rigid_links:
                raise ValueError(f"Collision exclusion names missing rigid bodies: {first}, {second}")
            relationship = UsdPhysics.FilteredPairsAPI.Apply(rigid_links[first]).CreateFilteredPairsRel()
            relationship.AddTarget(rigid_links[second].GetPath())
        origins = cloner.grid_transforms(self.num_envs, self.cfg.scene.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0(
            "/World/envs/env_0", "/World/envs/env_{}", self.num_envs,
            self.device, origins, global_paths=("/World/ground",),
        )
        cloner.replicate(plan, stage=self.scene.stage)
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot

    def read_state(self):
        pose = self.robot.data.root_link_pose_w.torch.clone()
        pose[:, :3] -= self.scene.env_origins
        return torch.cat((
            pose, self.robot.data.root_link_vel_w.torch,
            self.robot.data.joint_pos.torch[:, self.joint_indices],
            self.robot.data.joint_vel.torch[:, self.joint_indices],
        ), dim=-1)

    def _pre_physics_step(self, actions):
        if not bool(torch.isfinite(actions).all()):
            raise PhysicsStateError("Non-finite policy action; simulation was not advanced", {"action": actions.detach().cpu()})
        self._state_before = self.read_state()
        self._previous_before = self.previous_action.clone()
        self._context_before = self.context.clone()
        self._reference_before = self.reference_index.clone()
        self._steps_before = self.episode_steps.clone()
        self.actions = actions.clone()
        self._targets = (
            self.default_positions + self.action_scales * self.actions
            - self.context[:, 1 : 1 + self.actions.shape[-1]]
        )

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self._targets, joint_ids=self.joint_indices)

    def _get_dones(self):
        next_state = self.read_state()
        if not bool(torch.isfinite(next_state).all()):
            raise PhysicsStateError("Non-finite simulator state", {
                "state": self._state_before.detach().cpu(),
                "next_state": next_state.detach().cpu(), "action": self.actions.detach().cpu(),
            })
        self.reference_index = torch.minimum(
            self.reference_index + 1, self.model_task.motions.frame_ends[self.reference_index]
        )
        failed = self.model_task.terminated(next_state, self.reference_index)
        clip_end = self.reference_index >= self.model_task.motions.frame_ends[self.reference_index]
        timed_out = clip_end | (self.episode_steps >= self.max_episode_length)
        reward = self.model_task.reward(next_state, self.actions, self._previous_before, self.reference_index)
        self.discounted_returns += self.cfg.discount ** self._steps_before * reward
        self.last_transition = Transition(
            state=self._state_before, next_state=next_state, action=self.actions.clone(),
            previous_action=self._previous_before, context=self._context_before,
            reference_index=self._reference_before, episode_step=self._steps_before,
            reward=reward, terminated=failed.clone(), episode_end=(failed | timed_out).clone(),
        )
        self.previous_action.copy_(self.actions)
        return failed, timed_out

    def _get_rewards(self):
        return self.last_transition.reward

    def _get_observations(self):
        return {"policy": self.model_task.observation(self.read_state(), self.previous_action, self.context)}

    def _reset_envs_from_buffer(self):
        if self.cfg.evaluation:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        return super()._reset_envs_from_buffer()

    def _apply_materials(self, env_ids, friction):
        if "newton" in self.scene.physics_backend:
            import warp as wp
            from newton import ModelFlags
            from isaaclab_newton.physics.newton_manager import NewtonManager

            model = NewtonManager.get_model()
            # Isaac Lab 3 currently exposes per-shape material writes through this
            # backend binding. Use one coefficient for every robot shape per world.
            binding = self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0]
            wp.to_torch(binding)[env_ids, :] = friction[:, None]
            NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        else:
            import warp as wp

            materials = wp.to_torch(self.robot.root_view.get_material_properties())
            host_ids = env_ids.to(device="cpu", dtype=torch.int32)
            materials[host_ids, :, :2] = friction.cpu()[:, None, None]
            materials[host_ids, :, 2] = 0.0
            self.robot.root_view.set_material_properties(
                wp.from_torch(materials), wp.from_torch(host_ids)
            )

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        finished = self.episode_steps[env_ids] > 0
        if bool(finished.any()) and not self.cfg.evaluation:
            self.sampler.update(self.reset_bins[env_ids][finished], self.discounted_returns[env_ids][finished])
        super()._reset_idx(env_ids)
        if self._evaluation_starts is not None:
            references = self._evaluation_starts[env_ids]
        else:
            references, bins = self.sampler.sample(len(env_ids))
            self.reset_bins[env_ids] = bins
        self.reference_index[env_ids] = references
        self.discounted_returns[env_ids] = 0
        self.previous_action[env_ids] = 0
        self.context[env_ids] = 0
        self.context[env_ids, 0] = 1
        if self.cfg.randomization == "paper" and not self.cfg.evaluation:
            self.context[env_ids, 0] = 0.3 + 0.9 * torch.rand(len(env_ids), device=self.device)
            self.context[env_ids, 1:-3] = 0.02 * torch.rand(len(env_ids), len(self.joint_indices), device=self.device) - 0.01
            extent = torch.tensor([0.025, 0.05, 0.05], device=self.device)
            self.context[env_ids, -3:] = (2 * torch.rand(len(env_ids), 3, device=self.device) - 1) * extent
            self._com_has_been_randomized = True
        if self._com_has_been_randomized:
            # Switching an existing environment to evaluation must also restore
            # physical COM values; zeroing only the model's context is insufficient.
            com = self.nominal_com[env_ids].clone()
            com[:, 0, :3] += self.context[env_ids, -3:]
            if "newton" in self.scene.physics_backend:
                com = com[..., :3].contiguous()
            self.robot.set_coms_index(coms=com, body_ids=[self.torso_index], env_ids=env_ids)
        self._apply_materials(env_ids, self.context[env_ids, 0])
        reference = self.model_task.motions.states[references]
        pose = reference[:, :7].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        if "physx" in self.scene.physics_backend:
            # The checked PhysX SDK link writer forwards its link buffer to the
            # COM-velocity tensor API. Use an explicit link-to-COM conversion and
            # the COM writer to preserve the same physical reset contract.
            center_of_mass_offset = rotate(
                pose[:, 3:7],
                self.robot.data.body_com_pose_b.torch[env_ids, self.root_index, :3],
            )
            center_of_mass_velocity = reference[:, 7:13].clone()
            center_of_mass_velocity[:, :3] += torch.cross(
                reference[:, 10:13], center_of_mass_offset, dim=-1
            )
            self.robot.write_root_com_velocity_to_sim_index(
                root_velocity=center_of_mass_velocity, env_ids=env_ids
            )
        else:
            self.robot.write_root_link_velocity_to_sim_index(
                root_velocity=reference[:, 7:13].contiguous(), env_ids=env_ids
            )
        joint_count = len(self.joint_indices)
        self.robot.write_joint_position_to_sim_index(
            position=reference[:, 13 : 13 + joint_count].contiguous(),
            joint_ids=self.joint_indices, env_ids=env_ids,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=reference[:, 13 + joint_count :].contiguous(),
            joint_ids=self.joint_indices, env_ids=env_ids,
        )

    def reset_to_clips(self, clip_indices):
        if len(clip_indices) != self.num_envs or not self.cfg.evaluation:
            raise ValueError("Explicit clip reset requires evaluation mode and one clip per world")
        self._evaluation_starts = self.model_task.motions.starts[clip_indices]
        return self.reset()


class TransitionAdapter:
    """Keep Gym's API intact while exposing the learner's transition contract."""

    def __init__(self, environment):
        self.environment = environment.unwrapped

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def step(self, action):
        self.environment.step(action)
        return self.environment.last_transition
