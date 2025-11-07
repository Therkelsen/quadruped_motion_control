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

        # Observation: orientation(3) + pos(3) + lin_vel(3) + ang_vel(3) + joint_pos(12) + joint_vel(12) + contact(4) + TARGET
        self.target = np.array([2.0, 2.0])
        obs_dim = 3 + 3 + 3 + 3 + len(self.joint_ids) + len(self.joint_ids) + len(self.foot_links) + 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ---- Environment state ----
        self.human_friendly = human_friendly
        self._debug_target_ids = []
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
        p.changeDynamics(self.plane_id, -1, lateralFriction=2.0)
        
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.startPos = [0, 0, 0.55]
        self.robot = p.loadURDF("go2_description/urdf/go2.urdf", self.startPos, startOrientation)

        #for link in self.foot_links:
        #    p.changeDynamics(self.robot, link, lateralFriction=1.2)

        num_joints = p.getNumJoints(self.robot)
        for link_index in range(-1, num_joints):  # -1 includes the base
            p.changeDynamics(
                self.robot, 
                link_index, 
                lateralFriction=1.2,      # good grip for walking
                spinningFriction=0.1,     # helps with yaw stability
                rollingFriction=0.05,     # prevents sliding
                restitution=0.0,           # no bounciness
            )

    # --------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.last_action = np.zeros(len(self.joint_ids), dtype=np.float32)

        # ----------------------------------------------------
        # Reset base slightly higher to avoid ground penetration
        base_start = [self.startPos[0], self.startPos[1], 0.65]
        p.resetBasePositionAndOrientation(
            self.robot,
            base_start,
            p.getQuaternionFromEuler([0, 0, 0])
        )
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        # ----------------------------------------------------
        # Set joint states directly to standing pose (no PD yet)
        standing_pose = [0.0, 0.8, -1.5,
                        0.0, 0.8, -1.5,
                        0.0, 0.8, -1.5,
                        0.0, 0.8, -1.5]

        for i, j in enumerate(self.joint_ids):
            p.resetJointState(self.robot, j, standing_pose[i], targetVelocity=0.0)

        # ----------------------------------------------------
        # Let simulation settle with motors OFF (no forces)
        for j in self.joint_ids:
            p.setJointMotorControl2(
                self.robot, j,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )

        for _ in range(100):
            p.stepSimulation()
            if self.human_friendly:
                time.sleep(1 / 240)

        # ----------------------------------------------------
        # Re-enable PD motors *after* settling
        for i, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.robot,
                j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=standing_pose[i],
                force=self.effort_limits[i],
                positionGain=0.3,
                velocityGain=1.0
            )

        # ----------------------------------------------------
        # Randomize target for training
        self.target = np.random.uniform(low=-5.0, high=5.0, size=2)
        if self.human_friendly:
            self._draw_target(self.target, radius=0.12, segments=36, color=[1, 0, 0], lifeTime=0)

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
                force=self.effort_limits[idx],
                positionGain=0.3,  # less aggressive
                velocityGain=1.0
            )

        # Step simulation
        p.stepSimulation()
        if self.human_friendly:
            time.sleep(1.0 / 240.0)

        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward(action)

        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
        euler = np.array(p.getEulerFromQuaternion(base_orient))
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        self.last_action = action.copy()
        return obs, float(reward), terminated, truncated, {}

    # --------------------------------------------------------

    def _compute_reward(self, action):
        pos = self._get_obs()[3:5]
        dist = np.linalg.norm(pos - self.target)
        # Reward: move toward target, penalize distance and control effort
        reward = -dist - 0.01 * np.sum(np.square(action))
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

        target = self.target

        obs = np.concatenate([
            euler, base_pos, base_lin_vel, base_ang_vel,
            joint_pos, joint_vel, contact, target
        ])
        return obs.astype(np.float32)

    # --------------------------------------------------------

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


# ============================================================
# PPO TRAINING LOOP (same structure as BallEnv)
# ============================================================

if __name__ == "__main__":
    def make_env():
        env = MyCustomEnv(human_friendly=True)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models/",
        name_prefix="ppo_go2"
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    total_timesteps = 1_000_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_go2_latest")

    env.close()
