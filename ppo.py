import array
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import time


class MyCustomEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, human_friendly):
        super().__init__()

        self.joint_lower_limits = np.array([
            -1.0472, -1.5708, -2.7227,
            -1.0472, -1.5708, -2.7227,
            -1.0472, -0.5236, -2.7227,
            -1.0472, -0.5236, -2.7227
        ], dtype=np.float32)

        self.joint_upper_limits = np.array([
            1.0472, 3.4907, -0.83776,
            1.0472, 3.4907, -0.83776,
            1.0472, 4.5379, -0.83776,
            1.0472, 4.5379, -0.83776
        ], dtype=np.float32)
        
        self.effort_limits = np.array([
            23.7, 23.7, 45.43,
            23.7, 23.7, 45.43,
            23.7, 23.7, 45.43,
            23.7, 23.7, 45.43
        ], dtype=np.float32)
        
        self.action_space = gym.spaces.Box(low=self.joint_lower_limits,
                                           high=self.joint_upper_limits,
                                           dtype=np.float32)

        # observation: orientation(3) + lin_vel(3) + ang_vel(3) + joint_pos(12) + joint_vel(12) + contact(4)
        obs_dim = 3 + 3 + 3 + 12 + 12 + 4
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.state = None
        self.max_steps = 1000
        self.steps_taken = 0

        # PyBullet setup
        self.human_friendly = human_friendly
        if self.human_friendly:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.82)
        self.plane_id = p.loadURDF("plane.urdf")
        self.startPos = [0, 0, 0.5]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF("go2_description/urdf/go2.urdf", self.startPos, startOrientation)
        self.joint_ids = [2, 3, 4, 11, 12, 13, 20, 21, 22, 29, 30, 31]

        # Indices of foot links for contact detection
        self.foot_links = [7, 16, 25, 34]

        self.desired_speed = 0.5  # desired forward speed in m/s 
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_steps = [[0, 0, 0] for _ in range(5)]  # fixed: no shared references

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        # Reset joints
        for joint_idx, effort in zip(self.joint_ids, self.effort_limits):
            p.resetJointState(self.robot, joint_idx, targetValue=0.0, targetVelocity=0.0)
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                force=effort
            )

        # Reset base
        p.resetBasePositionAndOrientation(self.robot, self.startPos, p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.robot, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        for _ in range(25):
            p.stepSimulation()
            if self.human_friendly:
                time.sleep(1. / 240)

        self.last_steps = [self.startPos.copy() for _ in range(5)]
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1

        for idx, joint_idx in enumerate(self.joint_ids):
            p.setJointMotorControl2(self.robot, joint_idx, p.POSITION_CONTROL, targetPosition=action[idx], force=30)
        p.stepSimulation()
        
        if self.human_friendly:
            time.sleep(1. / 240)

        self.last_action = action

        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        if len(self.last_steps) >= 5:
            self.last_steps.pop(0)
        self.last_steps.append(list(base_pos))

        reward = self._compute_reward(base_pos)
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self, base_pos):
        dist_now = np.linalg.norm(np.array(base_pos) - np.array(self.startPos))
        dist_prev = np.linalg.norm(np.array(self.last_steps[-2]) - np.array(self.startPos))
        progress = dist_now - dist_prev

        energy_penalty = 0.001 * np.linalg.norm(self.last_action)
        return float(progress - energy_penalty)

    def _get_obs(self):
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot)
        joint_states = [p.getJointState(self.robot, i) for i in self.joint_ids]
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        # Contact detection
        contact = np.zeros(len(self.foot_links), dtype=np.float32)
        for i, foot_link in enumerate(self.foot_links):
            pts = p.getContactPoints(bodyA=self.robot, linkIndexA=foot_link, bodyB=self.plane_id)
            contact[i] = 1.0 if len(pts) > 0 else 0.0

        euler = np.array(p.getEulerFromQuaternion(base_orient), dtype=np.float32)
        obs = np.concatenate([euler, base_lin_vel, base_ang_vel, joint_pos, joint_vel, contact])
        return obs

    def render(self):
        pass

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = MyCustomEnv(human_friendly=False)

    log_dir = "./tensorboard/"
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/', name_prefix='ppo_go2')

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=10000000, callback=checkpoint_callback)

    env.close()
