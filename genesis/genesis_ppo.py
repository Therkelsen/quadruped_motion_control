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
# Device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_device = torch.device(device)

# ---------------------------
# Initialize Genesis
# ---------------------------
try:
    gs.init(backend=gs.gpu if device == "cuda" else gs.cpu)
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
        self.dt = 0.02  # 50 Hz control frequency

        # -----------------------------
        # Genesis requires CPU/NumPy for URDF init
        # -----------------------------
        self.start_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.start_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

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

        # -----------------------------
        # Joints
        # -----------------------------
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.joint_ids = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        self.num_joints = len(self.joint_ids)

        # -----------------------------
        # Spaces
        # -----------------------------
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        obs_dim = 3 + 3 + 3 + 3 + (2 * self.num_joints) + 4 + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0

        # Reset robot to start position and orientation
        self.robot.set_pos(self.start_pos)
        self.robot.set_quat(self.start_quat)

        # Zero velocities
        self.robot.zero_all_dofs_velocity()
        self.robot.set_dofs_velocity(torch.zeros(self.num_joints, device=self.device), self.joint_ids)

        # Standing pose
        #standing_pose = torch.tensor([0.0, 0.8, -1.5] * 4, device=self.device)
        #self.robot.set_dofs_position(standing_pose, self.joint_ids)
        #self.robot.set_dofs_velocity(torch.zeros_like(standing_pose), self.joint_ids)

        # Step simulation to settle contacts
        for _ in range(60):
            self.scene.step()

        # Random target
        self.target = torch.tensor(np.random.uniform(-5.0, 5.0, size=2), device=self.device, dtype=torch.float32)
        self.last_action = torch.zeros(self.num_joints, device=self.device, dtype=torch.float32)

        return self._get_obs(), {}

    # ==========================================================
    def step(self, action):
        self.steps_taken += 1
        action = torch.tensor(np.clip(action, -1.0, 1.0), device=self.device, dtype=torch.float32)

        # Small delta to joint positions
        max_delta = torch.deg2rad(torch.tensor(15.0, device=self.device))
        current_positions = self.robot.get_dofs_position(self.joint_ids).to(self.device)
        target_positions = torch.clamp(current_positions + action * max_delta, -1.0, 1.0)

        self.robot.control_dofs_position(target_positions, self.joint_ids)

        # Step simulation
        self.scene.step()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        self.last_action = action.clone()
        return obs, float(reward), terminated, truncated, {}

    # ==========================================================
    def _compute_reward(self, action):
        base_pos = torch.tensor(self.robot.get_pos(), device=self.device)
        dist = torch.norm(base_pos[:2] - self.target)
        reward = -dist - 0.01 * torch.sum(action**2)
        return reward.item()

    # ==========================================================
    def _get_obs(self):
        base_pos = torch.tensor(self.robot.get_pos(), device=self.device)
        base_quat = torch.tensor(self.robot.get_quat(), device=self.device)
        
        # Only base link linear/angular velocity
        base_lin_vel = torch.tensor(self.robot.get_vel(), device=self.device)[:3]
        base_ang_vel = torch.tensor(self.robot.get_ang(), device=self.device)[:3]

        euler = torch.tensor(quat_to_rotvec(base_quat.cpu().numpy()), device=self.device, dtype=torch.float32)
        
        joint_pos = torch.tensor(self.robot.get_dofs_position(self.joint_ids), device=self.device)
        joint_vel = torch.tensor(self.robot.get_dofs_velocity(self.joint_ids), device=self.device)

        contact_vec = torch.zeros(4, device=self.device)

        # Ensure all tensors are 1D and same dtype
        obs_list = [
            euler.flatten(),
            base_pos.flatten(),
            base_lin_vel.flatten(),
            base_ang_vel.flatten(),
            joint_pos.flatten(),
            joint_vel.flatten(),
            contact_vec.flatten(),
            torch.tensor(self.target, device=self.device, dtype=torch.float32).flatten()
        ]

        obs = torch.cat(obs_list)
        return obs.cpu().numpy().astype(np.float32)


    # ==========================================================
    def close(self):
        self.scene.destroy()


# ============================================================
# Training loop
# ============================================================
if __name__ == "__main__":

    def make_env():
        env = Go2GenesisEnv(render=True, device=device)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./models/",
        name_prefix="ppo_go2_genesis"
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", device=device)
    total_timesteps = 1_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_go2_genesis_latest")

    env.close()
