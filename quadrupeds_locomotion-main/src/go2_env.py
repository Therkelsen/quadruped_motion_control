import random
import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def gs_rand_gaussian(mean, min, max, n_std, shape, device):
    mean_tensor = mean.expand(shape).to(device)
    std_tensor = torch.full(shape, (max - min)/ 4.0 * n_std, device=device)
    return torch.clamp(torch.normal(mean_tensor, std_tensor), min, max)

def gs_additive(base, increment):
    return base + increment



class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda", add_camera = False):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.5, 0.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, show_world_frame=False),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        # self.cam_0 : gs.Camera = None
        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2.5, 0.5, 3.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=True,
            )

        # build
        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["lin_vel"], self.obs_scales["lin_vel"]] ,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        
        self.jump_toggled_buf = torch.zeros((self.num_envs,), device=self.device)
        self.jump_target_height = torch.zeros((self.num_envs,), device=self.device)
        
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        # self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 0] =  gs_additive(self.last_actions[envs_idx, 0], self.command_cfg["lin_vel_x_range"][0] + (self.command_cfg["lin_vel_x_range"][1] - self.command_cfg["lin_vel_x_range"][0]) * torch.sin(2 * math.pi * self.episode_length_buf[envs_idx] / 300))
        self.commands[envs_idx, 0] =  gs_rand_gaussian(self.last_actions[envs_idx, 0], *self.command_cfg["lin_vel_x_range"],  2.0, (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] =  gs_rand_gaussian(self.last_actions[envs_idx, 1], *self.command_cfg["lin_vel_y_range"],  2.0, (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] =  gs_rand_gaussian(self.last_actions[envs_idx, 2], *self.command_cfg["ang_vel_range"],  2.0, (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] =  gs_rand_gaussian(self.last_actions[envs_idx, 3], *self.command_cfg["height_range"],  0.5,(len(envs_idx),), self.device)
        self.commands[envs_idx, 4] = 0.0
        
        # scale lin_vel and ang_vel proportionally to the height difference between the target and default height
        height_diff_scale = 0.5 + abs(self.commands[envs_idx, 3] - self.reward_cfg["base_height_target"])/ (self.command_cfg["height_range"][1] - self.reward_cfg["base_height_target"]) * 0.5
        self.commands[envs_idx, 0] *= height_diff_scale
        self.commands[envs_idx, 1] *= height_diff_scale
        self.commands[envs_idx, 2] *= height_diff_scale
        
    def _sample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["height_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 4] = 0.0
        
        # scale lin_vel and ang_vel proportionally to the height difference between the target and default height
        height_diff_scale = 0.5 + abs(self.commands[envs_idx, 3] - self.reward_cfg["base_height_target"])/ (self.command_cfg["height_range"][1] - self.reward_cfg["base_height_target"]) * 0.5
        self.commands[envs_idx, 0] *= height_diff_scale
        self.commands[envs_idx, 1] *= height_diff_scale
        self.commands[envs_idx, 2] *= height_diff_scale
    
    def _sample_jump_commands(self, envs_idx):
        self.commands[envs_idx, 4] = gs_rand_float(*self.command_cfg["jump_range"], (len(envs_idx),), self.device)
        
    def step(self, actions, is_train=True):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands, it is a variable that holds the indices of environments that need to be resampled or reset. 
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if is_train:
            # self._resample_commands(all_envs_idx)
            self._sample_commands(envs_idx)
            # Idxs with probability of 5% to sample random commands
            ranomd_idxs_1 = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
            self._sample_commands(ranomd_idxs_1)
            
            random_idxs_2 = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
            self._sample_jump_commands(random_idxs_2)
            
        # Update jump_toggled_buf if command 4 goes from 0 -> non-zero
        jump_cmd_now = (self.commands[:, 4] > 0.0).float()
        toggle_mask = ((self.jump_toggled_buf == 0.0) & (jump_cmd_now > 0.0)).float()
        self.jump_toggled_buf += toggle_mask * self.reward_cfg["jump_reward_steps"]  # stay 'active' for n steps, for example
        self.jump_toggled_buf = torch.clamp(self.jump_toggled_buf - 1.0, min=0.0)
        # Update jump_target_height if command 4 goes from 0 -> non-zero
        self.jump_target_height = torch.where(jump_cmd_now > 0.0, self.commands[:, 4], self.jump_target_height)
        
        # print(f'jump_toggled_buf: {self.jump_toggled_buf}, jump_target_height: {self.jump_target_height}, commands: {self.commands}')
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())    
        
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 5
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                (self.jump_toggled_buf / self.reward_cfg["jump_reward_steps"]).unsqueeze(-1),  # 1
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        # Reset jump command
        self.commands[:, 4] = 0.0

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.jump_toggled_buf[envs_idx] = 0.0
        self.jump_target_height[envs_idx] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._sample_commands(envs_idx)
        
        # set target height command to default height
        self.commands[envs_idx, 3] = self.reward_cfg["base_height_target"]
        

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_pos[:, 2] - self.commands[:, 3])
    
    # def _reward_jump(self):
    #     # Reward if jump_toggled_buf > 0, even if command is now 0
    #     target_height = self.jump_target_height
    #     # Reward is active if jump_toggled_buf is active and some steps have passed (in order to prepare for jump)
    #     active_mask = (self.jump_toggled_buf > 0.0).float() * (self.jump_toggled_buf < (1.0/3.0 * self.reward_cfg["jump_reward_steps"])).float()
    #     active_mask_speed = (self.jump_toggled_buf > 1.0/3.0 * self.reward_cfg["jump_reward_steps"]).float() * (self.jump_toggled_buf < (2.0/3.0 * self.reward_cfg["jump_reward_steps"])).float()
    #     # Reward for reaching the target height
    #     height_reward = torch.exp(-torch.square(self.base_pos[:, 2] - target_height))
        
    #     # Reward for having a significant upward velocity
    #     upward_velocity_reward = 5 * torch.exp(-torch.square(self.base_lin_vel[:, 2] - self.reward_cfg["jump_upward_velocity"]))
        
    #     stay_penalty = -torch.square(self.base_pos[:, 2] - target_height) * (self.jump_toggled_buf > (2.0/3.0 * self.reward_cfg["jump_reward_steps"])).float()

    #     return active_mask * height_reward + active_mask_speed * upward_velocity_reward + stay_penalty * 0.1

    # def _reward_jump(self):
    #     target_height = self.jump_target_height
        
    #     # Target speed the robot should have to reach the target height in half the available time, considering the gravity (uniform acceleration)
    #     delta_height = target_height - self.base_pos[:, 2]
    #     available_time = self.reward_cfg["jump_reward_steps"] * self.dt * 0.6 * 0.5
    #     target_speed = torch.sqrt(2 * torch.abs(delta_height) * 9.81) * torch.sign(delta_height)
        
    #     # Phase 2: near peak height
    #     phase2_mask = (self.jump_toggled_buf >= (0.3 * self.reward_cfg["jump_reward_steps"])) & (self.jump_toggled_buf < (0.6 * self.reward_cfg["jump_reward_steps"]))
    #     target_height_reward = torch.exp(-torch.square(self.base_pos[:, 2] - target_height))
    #     # upward_speed_reward = torch.exp(-torch.square(self.base_lin_vel[:, 2] - target_speed))
    #     upward_speed_reward = torch.exp(self.base_lin_vel[:, 2]) * 0.2
    #     binary_reward_close_to_target = (torch.abs(self.base_pos[:, 2] - target_height) < 0.2).float() * 6.0

        
    #     # # Phase 1: descend
    #     phase1_mask = (self.jump_toggled_buf >= (0.6 * self.reward_cfg["jump_reward_steps"]))
    #     phase1_penalty = -torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    #     return (
    #         phase2_mask.float() * (target_height_reward * 2 + upward_speed_reward + binary_reward_close_to_target) +
    #         phase1_mask.float() * phase1_penalty * 0.08
    #     )
    
    def _reward_jump_height_tracking(self):
        """Continuous reward for minimizing distance to target height during peak phase"""
        mask = ((self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"]) & 
                (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"]))
        target_height = self.jump_target_height
        height_diff = torch.exp(-torch.square(self.base_pos[:, 2] - target_height))
        return mask.float() * height_diff

    def _reward_jump_height_achievement(self):
        """Binary reward for reaching close to target height during peak phase"""
        mask = ((self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"]) & 
                (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"]))
        target_height = self.jump_target_height
        binary_bonus = (torch.abs(self.base_pos[:, 2] - target_height) < 0.2).float()
        return mask.float() * binary_bonus

    def _reward_jump_speed(self):
        """Reward for upward velocity during peak phase"""
        mask = ((self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"]) & 
                (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"]))
        return mask.float() * torch.exp(self.base_lin_vel[:, 2]) * 0.2

    def _reward_jump_landing(self):
        """Penalty for deviation from base height during landing"""
        mask = (self.jump_toggled_buf >= 0.6 * self.reward_cfg["jump_reward_steps"])
        height_error = -torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        return mask.float() * height_error 
