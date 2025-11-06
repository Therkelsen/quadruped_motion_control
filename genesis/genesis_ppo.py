import torch
import gymnasium as gym
import numpy as np
import genesis as gs

from genesis.utils.geom import quat_to_rotvec
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------
# Genesis backend (GPU for simulation)
# ---------------------------
use_gpu = gs.gpu if torch.cuda.is_available() else gs.cpu
gs.init(backend=use_gpu)

class Go2GenesisEnv(gym.Env):
    """Quadruped Go2 environment — PyTorch stays on CPU, Genesis can use GPU."""
    def __init__(self, render=False):
        super().__init__()
        self.steps_taken = 0
        self.max_steps = 10_000_000
        self.dt = 0.02  # 50 Hz

        # Start pose
        self.start_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.start_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=10,
                gravity=[0, 0, -9.81]
            ),
            viewer_options=gs.options.ViewerOptions(max_FPS=int(1/self.dt)),
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

        # Joints
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.joint_ids = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        self.num_joints = len(self.joint_ids)

        # Action & observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        obs_dim = 3 + 3 + 3 + 3 + (2*self.num_joints) + 4 + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0

        # Reset robot
        self.robot.set_pos(self.start_pos)
        self.robot.set_quat(self.start_quat)
        self.robot.zero_all_dofs_velocity()

        # Step simulation to settle
        for _ in range(60):
            self.scene.step()

        # Random target
        self.target = np.random.uniform(-5.0, 5.0, size=2).astype(np.float32)
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)

        return self._get_obs(), {}

    # -----------------------------
    def step(self, action):
        self.steps_taken += 1
        action = np.clip(action, -1.0, 1.0)

        # Small delta to joint positions
        max_delta = np.deg2rad(15.0)
        current_positions = self.robot.get_dofs_position(self.joint_ids).astype(np.float32)
        target_positions = np.clip(current_positions + action * max_delta, -1.0, 1.0)

        self.robot.control_dofs_position(target_positions, self.joint_ids)

        # Step simulation
        self.scene.step()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self.steps_taken >= self.max_steps
        truncated = False
        self.last_action = action.copy()

        return obs, float(reward), terminated, truncated, {}

    # -----------------------------
    def _compute_reward(self, action):
        base_pos = self.robot.get_pos()[:2].astype(np.float32)
        dist = np.linalg.norm(base_pos - self.target)
        reward = -dist - 0.01 * np.sum(action**2)
        return reward

    # -----------------------------
    def _get_obs(self):
        base_pos = self.robot.get_pos().astype(np.float32)
        base_quat = self.robot.get_quat().astype(np.float32)

        base_lin_vel = self.robot.get_vel()[:3].astype(np.float32)
        base_ang_vel = self.robot.get_ang()[:3].astype(np.float32)

        euler = quat_to_rotvec(base_quat)

        joint_pos = self.robot.get_dofs_position(self.joint_ids).astype(np.float32)
        joint_vel = self.robot.get_dofs_velocity(self.joint_ids).astype(np.float32)

        contact_vec = np.zeros(4, dtype=np.float32)

        obs_list = [
            euler.flatten(),
            base_pos.flatten(),
            base_lin_vel.flatten(),
            base_ang_vel.flatten(),
            joint_pos.flatten(),
            joint_vel.flatten(),
            contact_vec.flatten(),
            self.target.flatten()
        ]

        return np.concatenate(obs_list).astype(np.float32)

    # -----------------------------
    def close(self):
        self.scene.destroy()

# -----------------------------
# Training loop
# -----------------------------
if __name__ == "__main__":

    def make_env():
        env = Go2GenesisEnv(render=True)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./models/",
        name_prefix="ppo_go2_genesis"
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", device="cpu")
    total_timesteps = 1_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_go2_genesis_latest")

    env.close()
