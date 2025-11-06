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

# ---------------------------
# Initialize Genesis
# ---------------------------
try:
    gs.init(backend=gs.gpu)
except Exception:
    print("⚠️ GPU backend not available — falling back to CPU.")
    gs.init(backend=gs.cpu)


# ---------------------------
# Go2 Gym Environment
# ---------------------------
class Go2GenesisEnv(gym.Env):
    """Gym environment for Go2 robot using Genesis."""
    def __init__(self, render=False):
        super().__init__()
        self.dt = 0.02
        self.steps_taken = 0
        self.max_steps = 1000

        # -------------------------
        # Create Genesis scene
        # -------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt, substeps=10, gravity=[0, 0, -9.81]
            ),
            viewer_options=gs.options.ViewerOptions(max_FPS=int(1/self.dt)),
            vis_options=gs.options.VisOptions(show_world_frame=False),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            show_viewer=render
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=[0, 0, 0.6],
                quat=[0, 0, 0, 1],
                fixed=False
            )
        )

        self.scene.build(n_envs=1)

        # -------------------------
        # Joints
        # -------------------------
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.joint_ids = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        self.num_joints = len(self.joint_ids)

        # -------------------------
        # Action / Observation spaces
        # -------------------------
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + (2 * self.num_joints) + 4 + 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Random target
        self.target = np.random.uniform(-5.0, 5.0, size=2)

    # -------------------------
    # Get robot state
    # -------------------------
    def get_robot_state(self):
        pos = self.robot.get_pos().cpu().numpy()
        quat = self.robot.get_quat().cpu().numpy()
        lin_vel = self.robot.get_vel().cpu().numpy()[:3]
        ang_vel = self.robot.get_ang().cpu().numpy()[:3]
        joint_pos = self.robot.get_dofs_position(self.joint_ids).cpu().numpy()
        joint_vel = self.robot.get_dofs_velocity(self.joint_ids).cpu().numpy()
        euler = quat_to_rotvec(quat)
        return {
            "pos": pos,
            "quat": quat,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "euler": euler
        }

    # -------------------------
    # Reset
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0

        # Reset robot
        self.robot.set_pos([0,0,0.6])
        self.robot.set_quat([0,0,0,1])
        self.robot.zero_all_dofs_velocity()
        self.robot.set_dofs_velocity(torch.zeros(self.num_joints), self.joint_ids)

        # Random target
        self.target = np.random.uniform(-5.0, 5.0, size=2)

        # Step simulation to settle
        for _ in range(60):
            self.scene.step()

        return self._get_obs(), {}

    # -------------------------
    # Step
    # -------------------------
    def step(self, action):
        self.steps_taken += 1
        action = np.clip(action, -1.0, 1.0)

        # Apply small delta
        max_delta = np.deg2rad(15.0)
        current_pos = self.robot.get_dofs_position(self.joint_ids).cpu().numpy()
        target_pos = np.clip(current_pos + action * max_delta, -1.0, 1.0)
        self.robot.control_dofs_position(torch.tensor(target_pos), self.joint_ids)

        # Step physics
        self.scene.step()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        return obs, float(reward), terminated, truncated, {}

    # -------------------------
    # Observation
    # -------------------------
    def _get_obs(self):
        base_pos = self.robot.get_pos().cpu().numpy()
        base_quat = self.robot.get_quat().cpu().numpy()
        base_lin_vel = self.robot.get_vel().cpu().numpy()[:3]
        base_ang_vel = self.robot.get_ang().cpu().numpy()[:3]
        joint_pos = self.robot.get_dofs_position(self.joint_ids).cpu().numpy()
        joint_vel = self.robot.get_dofs_velocity(self.joint_ids).cpu().numpy()
        contact_vec = np.zeros(4, dtype=np.float32)
        target_np = self.target.cpu().numpy()
        
        obs = np.concatenate([
            quat_to_rotvec(base_quat).flatten(),
            base_pos.flatten(),
            base_lin_vel.flatten(),
            base_ang_vel.flatten(),
            joint_pos.flatten(),
            joint_vel.flatten(),
            contact_vec.flatten(),
            target_np.flatten()
        ]).astype(np.float32)
        
        return obs


    # -------------------------
    # Reward
    # -------------------------
    def _compute_reward(self, action):
        # Get robot position as numpy on CPU
        base_pos = self.robot.get_pos().cpu().numpy()  # shape (3,)
        
        # Only x-y coordinates for target distance
        pos_xy = base_pos[:2]  # shape (2,)
        
        # Ensure target and action are also on CPU as numpy
        target_np = self.target.cpu().numpy()
        action_np = action.cpu().numpy()
        
        # Compute reward: negative distance to target minus small action penalty
        reward = -np.linalg.norm(pos_xy - target_np) - 0.01 * np.sum(action_np**2)
        return reward



    # -------------------------
    # Close
    # -------------------------
    def close(self):
        self.scene.destroy()


# ---------------------------
# Training loop
# ---------------------------
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
    total_timesteps = 10000  # reduce for testing

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_go2_genesis_latest")

    env.close()
