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

        self.joint_lower_limits = np.array([-1.0472, -1.5708, -2.7227,
                                            -1.0472, -1.5708, -2.7227,
                                            -1.0472, -0.5236, -2.7227,
                                            -1.0472, -0.5236, -2.7227], dtype=np.float32)
        self.joint_upper_limits = np.array([1.0472, 3.4907, -0.83776,
                                            1.0472, 3.4907, -0.83776,
                                            1.0472, 4.5379, -0.83776,
                                            1.0472, 4.5379, -0.83776], dtype=np.float32)
        
        self.effort_limits = np.array([23.7, 23.7, 45.43, 
                                       23.7, 23.7, 45.43, 
                                       23.7, 23.7, 45.43, 
                                       23.7, 23.7, 45.43], dtype=np.float32)
        
        self.action_space = gym.spaces.Box(low=self.joint_lower_limits,
                                           high=self.joint_upper_limits,
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32) # Joints, xyz, velocities

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

        self.desired_speed = 0.5  # desired forward speed in m/s 
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_steps = [[0]*3]*5


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        
        # Reset all controllable joints to zero position and zero velocity
        for joint_idx, effort in zip(self.joint_ids, self.effort_limits):
            # Reset position and velocity
            p.resetJointState(self.robot, joint_idx, targetValue=0.0, targetVelocity=0.0)
            # Disable motors temporarily so physics doesn't fight the reset
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                force=effort
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
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        
        self.state = np.array(list(base_pos) + list(lin_vel) + joint_states, dtype=np.float32)

        return self.state, {}


    def step(self, action):
        self.steps_taken += 1

        # Apply actions in PyBullet
        for idx, joint_idx in enumerate(self.joint_ids):
            p.setJointMotorControl2(self.robot, joint_idx, p.POSITION_CONTROL, targetPosition=action[idx], force=30)
        p.stepSimulation()
        
        if self.human_friendly: 
                time.sleep(1./240)

        # Update state
        joint_states = [p.getJointState(self.robot, i)[0] for i in self.joint_ids]
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        
        self.state = np.array(list(base_pos) + list(lin_vel) + joint_states, dtype=np.float32)
        self.last_action = action
        while len(self.last_steps) > 5:
            self.last_steps.pop(0)    
        self.last_steps.append(list(base_pos))

        # Compute reward
        reward = self._compute_reward()
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def _compute_reward(self):
        # Reward
        coefficients = np.array([1000.0, -10.0, -1.0, -1.0, -0.01, -1.0])
        
        P = np.array(self.state[:3] - self.startPos)     # displacement vector (3,)
        V = np.linalg.norm(self.state[3:6])              # scalar speed
        D_cur = np.mean(np.array(self.last_steps), axis=0)  # mean direction vector (3,)

        reward = np.array([np.linalg.norm(P),                                # encourage displacement
                  abs(V - self.desired_speed),                               # penalize speed error
                  self.last_action.dot(self.last_action),                    # energy penalty
                  self.compute_height_punishment(self.state[2]),             # penalize bad height
                  self.steps_taken,                                          # time penalty
                  abs(np.arccos(np.dot(P, D_cur) / ((np.linalg.norm(P) * np.linalg.norm(D_cur)) + 1e-6)))                                             # direction penalty
        ])
        
        return np.dot(coefficients, reward)
    
    def compute_height_punishment(self, height):
        
        if height <= 0:
            return 100
        
        h,k = (40, 0) # (height_desired, cost)
        x,y = (0, 50) # (no_height, cost_at_no_height)
        
        a = (y-k)/((x-h)**2)
        
        return a*(height-h)**2 + k
        
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
