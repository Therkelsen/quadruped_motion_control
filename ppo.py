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

        self.joint_lower_limits = np.array([-1.0472, -1.5708, -2.7227,
                                            -1.0472, -1.5708, -2.7227,
                                            -1.0472, -0.5236, -2.7227,
                                            -1.0472, -0.5236, -2.7227], dtype=np.float32)
        self.joint_upper_limits = np.array([1.0472, 3.4907, -0.83776,
                                            1.0472, 3.4907, -0.83776,
                                            1.0472, 4.5379, -0.83776,
                                            1.0472, 4.5379, -0.83776], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=self.joint_lower_limits,
                                           high=self.joint_upper_limits,
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        
        # Reset all controllable joints to zero position and zero velocity
        for joint_idx in self.joint_ids:
            # Reset position and velocity
            p.resetJointState(self.robot, joint_idx, targetValue=0.0, targetVelocity=0.0)
            # Disable motors temporarily so physics doesn't fight the reset
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )

        # Reset base position and orientation
        p.resetBasePositionAndOrientation(self.robot, self.startPos, p.getQuaternionFromEuler([0,0,0]))
        p.resetBaseVelocity(self.robot, linearVelocity=[0,0,0], angularVelocity=[0,0,0])

        # Step simulation a few times to let it settle
        for _ in range(25):
            p.stepSimulation()
            if self.human_friendly: 
                time.sleep(1./240)

        # Update internal state
        joint_states = [p.getJointState(self.robot, i)[0] for i in self.joint_ids]
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        self.state = np.array(list(base_pos) + joint_states, dtype=np.float32)

        return self.state, {}


    def step(self, action):
        self.steps_taken += 1

        # Apply actions in PyBullet
        for idx, joint_idx in enumerate(self.joint_ids):
            p.setJointMotorControl2(self.robot, joint_idx, p.POSITION_CONTROL, targetPosition=action[idx])
        p.stepSimulation()
        
        if self.human_friendly: 
                time.sleep(1./240)

        # Update state
        joint_states = [p.getJointState(self.robot, i)[0] for i in self.joint_ids]
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        self.state = np.array(list(base_pos) + joint_states, dtype=np.float32)

        # Compute reward
        reward = np.linalg.norm(self.state[:3] - self.startPos)  # example: negative distance to origin
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = MyCustomEnv(human_friendly=True)

    # TensorBoard log directory
    log_dir = "./tensorboard/"
    
    # Save checkpoints every 5000 steps 
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./models/', name_prefix='ppo_go2')

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=50000, callback=checkpoint_callback)

    env.close()
