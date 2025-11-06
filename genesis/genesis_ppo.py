import time
import torch
import gymnasium as gym
import numpy as np
import genesis as gs

from genesis.utils.geom import quat_to_rotvec
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# --------------------------------------------------------------------
# Initialize Genesis
# --------------------------------------------------------------------
try:
    gs.init(backend=gs.gpu)
except Exception:
    print("⚠️ GPU backend not available — falling back to CPU.")
    gs.init(backend=gs.cpu)


class Go2GenesisEnv(gym.Env):
    """Quadruped Go2-like environment using Genesis v0.9+."""
    def __init__(self, render=False, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.steps_taken = 0
        self.max_steps = 10000000
        self.target = np.array([2.0, 2.0], dtype=np.float32)
        self.last_action = None
        self.dt = 1 / 240.0
        self.dt = 0.02  # control frequency on real robot is 50hz
        
        self.start_pos = (0.0, 0.0, 0.5)
        self.start_quat = (0.0, 0.0, 0.0, 1.0)

        # -----------------------------
        # Scene
        # -----------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt, 
                substeps=10,
                gravity=[0, 0, -9.81]
            ),
            viewer_options=gs.options.ViewerOptions(max_FPS=int(1 / self.dt)),
            vis_options=gs.options.VisOptions(show_world_frame=False),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            show_viewer=render
        )

        self.scene.add_entity(gs.morphs.URDF(file="../objects/plane/plane.urdf", fixed=True))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="../objects/go2/urdf/go2.urdf",
                pos=self.start_pos,
                quat=self.start_quat,
                fixed=False,
            )
        )

        self.scene.build(n_envs=1)


        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        
        self.joint_ids = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        self.num_joints = len(self.joint_ids)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + (2 * self.num_joints) + 4 + 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        

    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0

        # Safe base start (start slightly above ground to avoid initial penetration)
        self.robot.set_pos(self.start_pos)
        self.robot.set_quat(self.start_quat)

        # Zero base and joint velocities if API available
        self.robot.zero_all_dofs_velocity()
        self.robot.set_dofs_velocity(torch.zeros(self.num_joints), self.joint_ids)

        # Initialize a reasonable standing joint pose (12 DOFs: 3 per leg)
        standing_pose = torch.tensor([0.0, 0.8, -1.5] * 4, dtype=torch.float32)

        self.robot.set_dofs_position(standing_pose, self.joint_ids)
        self.robot.set_dofs_velocity(torch.zeros_like(standing_pose), self.joint_ids)

        # Let physics settle so no NaNs and stable contact states
        for _ in range(60):
            self.scene.step()

        # Randomize target and reset last action
        self.target = np.random.uniform(low=-5.0, high=5.0, size=2).astype(np.float32)
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)

        return self._get_obs(), {}

    # ==========================================================
    def step(self, action):
        self.steps_taken += 1
        action = np.clip(action, -1.0, 1.0)
        action_t = torch.tensor(action, dtype=torch.float32)

        # Small delta to joint positions
        max_delta = np.deg2rad(15)
        current_positions = torch.tensor(self.robot.get_dofs_position(self.joint_ids), dtype=torch.float32)
        target_positions = torch.clamp(current_positions + action_t * max_delta, -1.0, 1.0)

        # Control DOFs
        self.robot.control_dofs_position(target_positions, self.joint_ids)

        # Step simulation
        self.scene.step()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        self.last_action = action.copy()
        return obs, float(reward), terminated, truncated, {}

    # ==========================================================
    def _compute_reward(self, action):
        base_pos = np.asarray(self.robot.get_pos()).reshape(-1)
        dist = np.linalg.norm(base_pos[:2] - self.target)
        reward = -dist - 0.01 * np.sum(np.square(action))
        return reward

    # ==========================================================
    def _get_obs(self):
        base_pos = np.asarray(self.robot.get_pos(), dtype=np.float32).ravel()
        base_quat = np.asarray(self.robot.get_quat(), dtype=np.float32).ravel()
        base_lin_vel = np.asarray(self.robot.get_vel(), dtype=np.float32).ravel()
        base_ang_vel = np.asarray(self.robot.get_ang(), dtype=np.float32).ravel()
        euler = quat_to_rotvec(base_quat).astype(np.float32)
        joint_pos = np.asarray(self.robot.get_dofs_position(self.joint_ids), dtype=np.float32).ravel()
        joint_vel = np.asarray(self.robot.get_dofs_velocity(self.joint_ids), dtype=np.float32).ravel()
        contact_vec = np.zeros(4, dtype=np.float32)
        # for i, link_name in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
        #     link_id = self.robot.get_link(link_name).idx
        #     contact_vec[i] = 1.0 if self.robot.get_contacts(link_id) else 0.0
        list = np.concatenate([euler, base_pos, base_lin_vel, base_ang_vel, joint_pos, joint_vel, contact_vec, self.target])
        print("obs: ", list)
        return list


    # ==========================================================
    def close(self):
        self.scene.destroy()


# ============================================================
# Training loop
# ============================================================
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def make_env():
        env = Go2GenesisEnv(render=True, device=device)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path="./models/",
        name_prefix="ppo_go2_genesis"
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", device=device)
    total_timesteps = 1_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_go2_genesis_latest")

    env.close()
