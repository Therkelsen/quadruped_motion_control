import time
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class MyCustomEnv(gym.Env):
    """
    Continuous control environment for a quadruped robot (Go2-like).
    Compatible with Stable Baselines3 PPO.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, human_friendly=False):
        super().__init__()

        # ---- Define physical & control limits ----
        self.joint_ids = [2, 3, 4, 11, 12, 13, 20, 21, 22, 29, 30, 31]
        self.foot_links = [7, 16, 25, 34]

        self.joint_lower_limits = np.array([-1.0] * len(self.joint_ids), dtype=np.float32)
        self.joint_upper_limits = np.array([1.0] * len(self.joint_ids), dtype=np.float32)
        self.effort_limits = np.array([
            23.7, 23.7, 45.43,   # Front Left
            23.7, 23.7, 45.43,   # Front Right
            23.7, 23.7, 45.43,   # Rear Left
            23.7, 23.7, 45.43    # Rear Right
        ], dtype=np.float32)

        # Continuous actions: desired joint deltas or normalized torques
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.joint_ids),), dtype=np.float32)

        # Observation: orientation(3) + pos(3) + lin_vel(3) + ang_vel(3) + joint_pos(12) + joint_vel(12) + contact(4)
        obs_dim = 3 + 3 + 3 + 3 + len(self.joint_ids) + len(self.joint_ids) + len(self.foot_links)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ---- Environment state ----
        self.human_friendly = human_friendly
        self.max_steps = 1000
        self.steps_taken = 0
        self.desired_speed = 0.3  # target forward velocity
        self.last_action = np.zeros(len(self.joint_ids), dtype=np.float32)

        # ---- Connect to PyBullet ----
        if human_friendly:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)
        
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.startPos = [0, 0, 0.45]
        self.robot = p.loadURDF("go2_description/urdf/go2.urdf", self.startPos, startOrientation)

        for link in self.foot_links:
            p.changeDynamics(self.robot, link, lateralFriction=1.2)

    # --------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.last_action = np.zeros(len(self.joint_ids), dtype=np.float32)

        # Reset robot state
        for j in self.joint_ids:
            p.resetJointState(self.robot, j, targetValue=0.0, targetVelocity=0.0)

        p.resetBasePositionAndOrientation(self.robot, self.startPos, p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        for _ in range(10):
            p.stepSimulation()
            if self.human_friendly:
                time.sleep(1.0 / 240.0)

        return self._get_obs(), {}

    # --------------------------------------------------------

    def step(self, action):
        self.steps_taken += 1
        action = np.clip(action, -1.0, 1.0)

        # Scale to target joint angle deltas
        max_delta = np.deg2rad(15)
        joint_states = [p.getJointState(self.robot, j) for j in self.joint_ids]
        current_positions = np.array([s[0] for s in joint_states])
        new_positions = current_positions + action * max_delta

        # Clip to mechanical joint limits
        new_positions = np.clip(new_positions, self.joint_lower_limits, self.joint_upper_limits)

        # Apply control
        for idx, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=new_positions[idx],
                force=self.effort_limits[idx]
            )

        # Step simulation
        p.stepSimulation()
        if self.human_friendly:
            time.sleep(1.0 / 240.0)

        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward(action)

        terminated = self.steps_taken >= self.max_steps
        truncated = False

        self.last_action = action.copy()
        return obs, float(reward), terminated, truncated, {}

    # --------------------------------------------------------

    def _compute_reward(self, action):
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot)
        euler = np.array(p.getEulerFromQuaternion(base_orient))

        v_forward = base_lin_vel[0]
        v_des = self.desired_speed
        r_vel = 1.0 - abs(v_forward - v_des) / max(1e-3, abs(v_des))
        r_orient = -np.linalg.norm(euler[:2])  # penalize pitch/roll

        # action smoothness & energy
        r_action_smooth = -0.1 * np.sum(np.square(action - self.last_action))
        r_energy = -0.001 * np.sum(np.square(action))

        reward = 1.0 * r_vel + r_orient + r_action_smooth + r_energy
        return reward

    # --------------------------------------------------------

    def _get_obs(self):
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot)
        euler = np.array(p.getEulerFromQuaternion(base_orient), dtype=np.float32)

        joint_states = [p.getJointState(self.robot, j) for j in self.joint_ids]
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        contact = np.zeros(len(self.foot_links), dtype=np.float32)
        for i, foot_link in enumerate(self.foot_links):
            pts = p.getContactPoints(bodyA=self.robot, linkIndexA=foot_link, bodyB=self.plane_id)
            contact[i] = 1.0 if len(pts) > 0 else 0.0

        obs = np.concatenate([
            euler, base_pos, base_lin_vel, base_ang_vel,
            joint_pos, joint_vel, contact
        ])
        return obs.astype(np.float32)

    # --------------------------------------------------------

    def close(self):
        p.disconnect()


# ============================================================
# PPO TRAINING LOOP (same structure as BallEnv)
# ============================================================

if __name__ == "__main__":
    def make_env():
        env = MyCustomEnv(human_friendly=False)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/",
        name_prefix="ppo_go2"
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    total_timesteps = 1_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_go2_latest")

    env.close()
