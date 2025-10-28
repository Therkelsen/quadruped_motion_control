import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


class BallEnv(gym.Env):
    """
    A simple environment where an agent controls x,y force on a rolling ball to reach a target.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, human_friendly=False):
        super().__init__()

        # RL controls: x,y force (2D)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: position (x,y), linear velocity (x,y), target position (x,y)
        obs_dim = 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.max_steps = 500
        self.human_friendly = human_friendly
        self.steps = 0
        self.force_scale = 15.0  # Try slightly higher for better motion
        
        self._debug_target_ids = []

        # Physics setup
        if self.human_friendly:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane_id = p.loadURDF("plane.urdf")
        #p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)
        
        self.ball_id = p.loadURDF("ball.urdf", basePosition=[0, 0, 0.3])  # simpler URDF if available
        self.target = np.array([2.0, 2.0])
        self.start_pos = np.array([0.0, 0.0])
        self.use_terminate = False
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        p.resetBasePositionAndOrientation(self.ball_id, [0, 0, 0.1], [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

        # Randomize target position a bit
        self.target = np.random.uniform(low=-5.0, high=5.0, size=2)
        obs = self._get_obs()
        
        if self.human_friendly:
            self._draw_target(self.target, radius=0.12, segments=36, color=[1, 0, 0], lifeTime=0)

        return obs, {}

    def step(self, action):
        self.steps += 1
        action = np.clip(action, -1, 1)

        # Map action → external force (in XY plane)
        force = self.force_scale * np.array([action[0], action[1], 0.0])
        p.applyExternalForce(self.ball_id, -1, force, [0, 0, 0], p.WORLD_FRAME)

        p.stepSimulation()
        

        obs = self._get_obs()
        pos = obs[0:2]
        vel = obs[2:4]

        # Distance to target
        dist = np.linalg.norm(pos - self.target)

        # Reward: move toward target, penalize distance and control effort
        reward = -dist - 0.01 * np.sum(np.square(action))

        # Done if close to target or too long
        #terminated = dist < 0.05
        truncated = self.steps >= self.max_steps
        terminated = False
        if self.use_terminate:
            if dist < 0.06:
                terminated = True
        
        return obs, float(reward), terminated, truncated, {}
    
    def compute_reward(self, action):
        obs = self._get_obs()
        pos = obs[0:2]

        # Distance to target
        dist = np.linalg.norm(pos - self.target)

        # Reward: move toward target, penalize distance and control effort
        reward = -dist - 0.01 * np.sum(np.square(action))
        return reward
    
    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        vel, _ = p.getBaseVelocity(self.ball_id)
        return np.concatenate([pos[0:2], vel[0:2], self.target])

    def render(self):
        pass

    def close(self):
        p.disconnect()
        
    def _draw_target(self, center=None, radius=0.15, segments=24, color=[1, 0, 0], lifeTime=0):
        """
        Draw a small circle in the plane z = 0 at position `center` (x,y).
        - Removes previously drawn circle (stored in self._debug_target_ids).
        - `lifeTime=0` persists until removed manually; set >0 to auto-expire.
        """
        # Remove old debug items
        for dbg_id in self._debug_target_ids:
            try:
                p.removeUserDebugItem(dbg_id)
            except Exception:
                pass
        self._debug_target_ids = []

        if center is None:
            center = self.target
        cx, cy = float(center[0]), float(center[1])
        z = 0.01  # draw slightly above ground to be visible

        points = []
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            points.append((x, y, z))

        # draw line segments between consecutive points
        for i in range(len(points)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            dbg_id = p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=2.0, lifeTime=lifeTime)
            self._debug_target_ids.append(dbg_id)

        # optionally add a small cross at the center
        cross_scale = radius * 0.5
        dbg_id1 = p.addUserDebugLine((cx - cross_scale, cy, z), (cx + cross_scale, cy, z), lineColorRGB=color, lineWidth=2.0, lifeTime=lifeTime)
        dbg_id2 = p.addUserDebugLine((cx, cy - cross_scale, z), (cx, cy + cross_scale, z), lineColorRGB=color, lineWidth=2.0, lifeTime=lifeTime)
        self._debug_target_ids.extend([dbg_id1, dbg_id2])

if __name__ == "__main__":

    def make_env():
        env = BallEnv(human_friendly=False)
        env = Monitor(env)
        return env

    # Wrap in DummyVecEnv only
    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='ppo_ball'
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    total_timesteps = 1_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save("./models/ppo_ball_latest")

    env.close()
